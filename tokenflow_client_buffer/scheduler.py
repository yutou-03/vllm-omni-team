"""Opt-in synchronous scheduler adapter using only gateway-validated client credit.

The gateway owns W across the whole audio pipeline. This adapter never reads
connector queues for buffer credit and never changes KV ownership. It masks
running requests for one scheduling iteration, then restores their identity.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from .core import ClientFeedback, number
from .stream import atomic_json


CONFIG_ENV = "VLLM_OMNI_CLIENT_BUFFER_CONFIG"
FORBIDDEN_ENV = (
    "VLLM_OMNI_RECEIVER_BUFFER_CONFIG", "VLLM_OMNI_TEXT_DELIVERY_CONFIG",
    "VLLM_OMNI_SMALL_HOLD_CONTROLLER", "VLLM_OMNI_CONTROLLED_PREEMPTION_FILE",
)


class ClientBufferScheduler:
    def __init__(self, scheduler, config: dict, emit, *, clock=time.monotonic):
        self.scheduler, self.emit, self.clock = scheduler, emit, clock
        self.config = config
        self.stage = int(scheduler._omni_stage_id_for_trace())
        self.cap = int(config["execution_cap"])
        self.resident_cap = int(scheduler.max_num_running_reqs)
        if not 1 <= self.cap <= self.resident_cap:
            raise ValueError("client-buffer requires 1 <= execution cap <= resident cap")
        self.snapshot_path = Path(config["snapshot_path"])
        self.max_age = number(config.get("max_feedback_age_s", .5), "feedback age", positive=True)
        self.snapshot_age = number(config.get("max_snapshot_age_s", .5), "snapshot age", positive=True)
        self.resume_s = number(config["resume_to_client_s"], "replenishment bound", positive=True)
        self.margin_s = number(config.get("margin_s", .2), "margin", positive=True)
        self.min_slice = number(config.get("min_slice_s", .05), "min slice", positive=True)
        self.max_slice = number(config.get("max_slice_s", .1), "max slice", positive=True)
        if self.min_slice > self.max_slice:
            raise ValueError("min slice exceeds max slice")
        self.last_snapshot = {}
        self.last_mode = None
        self.held_since: dict[str, float] = {}
        self.resume_deadlines: dict[str, float] = {}
        self.actual_slices = []
        self.last_tick = None
        self.last_selected = set()
        self.last_log = 0.0
        self.latched_fault = None
        self.telemetry = dict(steps=0, full_steps=0, held_steps=0, scheduled_tokens=0)
        self.last_telemetry = 0.
        self.seen_contracts = set()
        self.active_ids = None
        self.admission_guard = number(config.get("admission_guard_s", .5), "admission guard", positive=True)

    @classmethod
    def from_environment(cls, scheduler, emit):
        path = os.environ.get(CONFIG_ENV)
        if not path:
            return None
        for key in FORBIDDEN_ENV:
            if os.environ.get(key):
                raise ValueError(f"client-only buffer is incompatible with {key}")
        for key in ("VLLM_OMNI_STAGE_BUFFER_POLICY", "VLLM_OMNI_SMALL_HOLD_PROBE",
                    "VLLM_OMNI_URGENT_KEEP_ADMISSION", "VLLM_OMNI_SLO_ACTIVE_PREEMPTION"):
            if os.environ.get(key, "0").lower() not in {"0", "false", ""}:
                raise ValueError(f"client-only buffer is incompatible with {key}")
        if os.environ.get("VLLM_OMNI_SMALL_HOLD_EXECUTION_CAP"):
            raise ValueError("client-only adapter owns execution capacity")
        config = json.loads(Path(path).read_text())
        stage = int(scheduler._omni_stage_id_for_trace())
        if stage != config["target_stage"]:
            return None
        expected_stage = 0 if config["modality"] == "text" else 1
        if stage != expected_stage:
            raise ValueError("pilot targets text stage 0 or audio stage 1")
        if type(scheduler).__name__ != "OmniARScheduler" or scheduler.scheduler_config.async_scheduling:
            raise ValueError("client-only pilot requires synchronous OmniARScheduler")
        if getattr(scheduler, "num_spec_tokens", 0) or getattr(scheduler, "use_pp", False):
            raise ValueError("speculative decoding/pipeline parallelism not supported")
        if getattr(scheduler, "connector", None) is not None:
            raise ValueError("remote KV connectors are outside this pilot")
        if getattr(getattr(scheduler, "_baseline_policy", None), "value", None) != "native_fcfs":
            raise ValueError("native_fcfs baseline required for real fallback")
        return cls(scheduler, config, emit)

    def metadata(self, request):
        # Injected test metadata avoids loading the inference stack in CPU tests.
        if hasattr(request, "client_buffer_test_metadata"):
            return request.client_buffer_test_metadata
        from vllm_omni.scheduling.policy import request_scheduling_metadata
        return request_scheduling_metadata(request)

    def source_id(self, request):
        return self.metadata(request).get("sched_source_request_id")

    def order(self, request):
        return (self.metadata(request).get("sched_ingress_order", float("inf")), request.arrival_time)

    def read_snapshot(self, now):
        try:
            data = json.loads(self.snapshot_path.read_text())
            if (data["schema_version"] != 1 or data["session"] != self.config["session"]
                    or data["modality"] != self.config["modality"]
                    or not 0 <= now - data["written_at"] <= self.snapshot_age):
                return {}, "snapshot_invalid"
            # Demand is independently recomputed at the scheduling boundary.
            demand = sum(number(r["rate"], "member rate", positive=True)
                         for r in data["members"].values())
            if abs(demand - data["demand"]) > 1e-7:
                return {}, "working_set_inconsistent"
            if data["mode"] != "buffer":
                return data, data.get("reason", "fcfs")
            if data.get("gamma") is None or demand > data["gamma"] + 1e-9:
                return data, "overload"
            if now > data["capacity_valid_until"]:
                return data, "capacity_unknown"
            return data, None
        except (OSError, ValueError, KeyError, TypeError):
            return {}, "snapshot_unavailable"

    def coverage(self, request, snapshot, now):
        rid = self.source_id(request)
        member = snapshot.get("members", {}).get(rid)
        raw = snapshot.get("feedback", {}).get(rid)
        if not member:
            return None
        field = "sched_text_consume_rate_tps" if self.config["modality"] == "text" else "sched_audio_consume_rate"
        observed_rate = self.metadata(request).get(field)
        if observed_rate != member["rate"]:
            self.latched_fault = "request_rate_not_propagated"
            return None
        if request.request_id not in self.seen_contracts:
            self.emit("client_buffer_contract", stage_id=self.stage, request_id=request.request_id,
                source_request_id=rid, requested_rate=member["rate"], observed_rate=observed_rate,
                modality=self.config["modality"])
            self.seen_contracts.add(request.request_id)
        if not raw:
            return None
        try:
            feedback = ClientFeedback(**raw)
            if (feedback.session != self.config["session"] or feedback.request_id != rid
                    or feedback.modality != self.config["modality"] or feedback.rate != member["rate"]):
                return None
            return feedback.coverage(now, self.max_age, self.config.get("reserve_units", 0.0))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def decode_boundary(r):
        return (r.num_computed_tokens >= r.num_prompt_tokens and r.num_output_tokens > 0
                and r.num_output_placeholders == 0 and not r.spec_token_ids
                and r.num_tokens - r.num_computed_tokens == 1)

    def _resident_count(self):
        # Resource accounting only. Connector contents are never buffer credit.
        ids = {r.request_id for r in self.scheduler.running}
        coordinator = getattr(self.scheduler.kv_cache_manager, "coordinator", None)
        for manager in getattr(coordinator, "single_type_managers", ()):
            ids.update(rid for rid, blocks in manager.req_to_blocks.items() if blocks)
        return len(ids)

    @contextmanager
    def scheduling_context(self):
        s, now = self.scheduler, self.clock()
        original = list(s.running)
        original_cap = s.max_num_running_reqs
        if original_cap != self.resident_cap:
            raise RuntimeError("resident cap changed")
        if self.last_tick is not None:
            self.actual_slices.append(now - self.last_tick)
            self.actual_slices = self.actual_slices[-64:]
        if not original and not self.held_since:
            self.actual_slices.clear()
        self.last_tick = now
        snapshot, fallback = self.read_snapshot(now)
        fallback = self.latched_fault or fallback
        cover = {r.request_id: self.coverage(r, snapshot, now) for r in original}
        fallback = self.latched_fault or fallback
        mode = "fcfs" if fallback else "buffer"
        guard = self.resume_s + self.margin_s + max(self.actual_slices, default=self.min_slice)
        ordered = sorted(original, key=self.order)
        chosen = ordered[:self.cap]
        newly_paused = set()
        if mode == "buffer":
            # Retain execution-slot owners until a safe, deliberate swap. Merely
            # having a lower priority never pauses a client without buffer credit.
            if self.active_ids is None:
                self.active_ids = {r.request_id for r in chosen}
            chosen = [r for r in ordered if r.request_id in self.active_ids][:self.cap]
            held = [r for r in ordered if r.request_id in self.held_since]
            # Fresh client receipts may include in-flight output after a pause.
            # Recompute deadlines from current client credit, not a frozen lease.
            for r in held:
                if cover[r.request_id] is not None:
                    self.resume_deadlines[r.request_id] = now + cover[r.request_id] - guard
            held.sort(key=lambda r: self.resume_deadlines[r.request_id])
            step_guard = max(self.admission_guard,
                             max(self.actual_slices, default=self.min_slice) + self.min_slice)
            due = [r for r in held if cover[r.request_id] is None
                   or now + step_guard >= self.resume_deadlines[r.request_id]
                   or cover[r.request_id] <= guard + step_guard]

            def safe_victims():
                return [r for r in chosen if r not in due and self.decode_boundary(r)
                        and cover[r.request_id] is not None
                        and cover[r.request_id] > guard + self.admission_guard + step_guard]

            for r in due:
                if len(chosen) >= self.cap:
                    victims = safe_victims()
                    if victims:
                        victim = max(victims, key=lambda x: cover[x.request_id])
                        chosen.remove(victim)
                        newly_paused.add(victim.request_id)
                    else:
                        # An early warning may wait for credit; a critical client
                        # must abandon optimization rather than lose its buffer.
                        if cover[r.request_id] is None or cover[r.request_id] <= guard:
                            fallback = "no_safe_resume_slot"
                            mode = "fcfs"
                        break
                chosen.append(r)
            # Fill free slots with existing residents before admitting more work.
            for r in ordered:
                if len(chosen) == self.cap:
                    break
                if r not in chosen:
                    chosen.append(r)
            resident_waiters = [r for r in ordered if r not in chosen
                                and r.request_id not in self.held_since]
            can_admit = resident_waiters or (s.waiting and self._resident_count() < self.resident_cap)
            if mode == "buffer" and not due and can_admit and len(chosen) == self.cap:
                victims = safe_victims()
                if victims:
                    victim = max(victims, key=lambda r: cover[r.request_id])
                    chosen.remove(victim)
                    newly_paused.add(victim.request_id)
                    # Native chunk arrivals may already be resident/runnable.
                    # They are ordinary waiters, not policy-paused requests.
                    if resident_waiters:
                        chosen.append(resident_waiters[0])
            if mode == "fcfs":
                chosen = ordered[:self.cap]
        selected_ids = {r.request_id for r in chosen}
        skipped = {r.request_id for r in original} - selected_ids
        for r in original:
            rid = r.request_id
            if mode == "buffer" and rid in skipped and rid in newly_paused:
                if rid not in self.held_since:
                    self.held_since[rid] = now
                    self.resume_deadlines[rid] = now + max(0, cover[rid] - guard)
                    self.emit("client_buffer_pause", stage_id=self.stage, request_id=rid,
                              source_request_id=self.source_id(r), coverage_s=cover[rid],
                              resume_by=self.resume_deadlines[rid], inventory_source="client_consumption_buffer",
                              mode=mode, decision_monotonic=now, demand=snapshot.get("demand"), gamma=snapshot.get("gamma"))
        if mode != self.last_mode or now - self.last_log >= 1:
            self.emit("client_buffer_mode", stage_id=self.stage, mode=mode,
                      reason=fallback or "capacity_feasible", demand=snapshot.get("demand"),
                      gamma=snapshot.get("gamma"), members=list(snapshot.get("members", {})),
                      selected=[r.request_id for r in chosen])
            self.last_log, self.last_mode = now, mode
        if mode == "fcfs":
            # No policy queue overrides, leases, or borrowed-slot reservations survive.
            for rid in self.held_since:
                self.emit("client_buffer_fallback_release", stage_id=self.stage,
                          request_id=rid, reason=fallback)
            self.held_since.clear()
            self.resume_deadlines.clear()
        free = max(0, self.resident_cap - self._resident_count())
        s.max_num_running_reqs = min(self.cap, len(chosen) + free)
        s.running[:] = chosen
        self.last_selected = selected_ids
        try:
            yield
        finally:
            visible = list(s.running)
            self.active_ids = {r.request_id for r in visible}
            s.max_num_running_reqs = original_cap
            restored = []
            seen = set()
            for r in original + visible:
                if (r.request_id not in seen and s.requests.get(r.request_id) is r
                        and getattr(r.status, "name", "") == "RUNNING"):
                    restored.append(r)
                    seen.add(r.request_id)
            s.running[:] = sorted(restored, key=self.order)
            for rid in list(self.held_since):
                if rid not in s.requests:
                    self.held_since.pop(rid)
                    self.resume_deadlines.pop(rid, None)

    def record_scheduled(self, output):
        now = self.clock()
        if len(output.num_scheduled_tokens) > self.cap:
            raise RuntimeError("client-buffer execution cap exceeded")
        self.telemetry["steps"] += 1
        self.telemetry["full_steps"] += int(len(output.num_scheduled_tokens) >= self.cap)
        self.telemetry["held_steps"] += int(bool(self.held_since))
        self.telemetry["scheduled_tokens"] += sum(output.num_scheduled_tokens.values())
        if self.config.get("execution_telemetry_path") and now-self.last_telemetry >= .1:
            atomic_json(self.config["execution_telemetry_path"], dict(self.telemetry,
                written_at=now, session=self.config["session"], stage=self.stage,
                last_scheduled=len(output.num_scheduled_tokens)))
            self.last_telemetry = now
        for rid, tokens in output.num_scheduled_tokens.items():
            if rid in self.held_since:
                self.emit("client_buffer_resumed", stage_id=self.stage, request_id=rid,
                          held_s=now - self.held_since.pop(rid), scheduled_tokens=tokens)
                self.resume_deadlines.pop(rid, None)


def create_adapter(scheduler, emit):
    return ClientBufferScheduler.from_environment(scheduler, emit)
