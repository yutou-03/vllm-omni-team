"""Synchronous scheduler adapter using the configured end-user output buffer.

The gateway owns W across the whole audio pipeline. This adapter never reads
connector queues for buffer credit and never changes KV ownership. PauseRegistry
owns explicit pause transitions. Policy decisions are applied by a single
execution adapter which preserves native queue and KV ownership.
"""
from __future__ import annotations

from dataclasses import asdict
from .virtual_output import VirtualOutput
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from .core import ClientFeedback, buffer_conservativeness, number
from .capacity_signals import CapacityExecutionSignals
from .engine_queue import engine_queue_admission_context
from .stream import atomic_json
from .pause import PausedRequest, PauseRegistry
from .candidate_priority import CandidatePriority
from .recovery_ranking import rank_recovery
from .execution import ExecutionDecision, apply_execution_decision


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
        self.native_limit = "execution_cap" not in config
        self.resident_cap = int(scheduler.max_num_running_reqs)
        self.cap = int(config.get("execution_cap", self.resident_cap))
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
        # Opt-in so the existing per-step policy remains an explicit control.
        # This clock gates scheduling decisions/new prefill, never client playback.
        self.policy_interval = number(config.get("policy_interval_s", 0.), "policy interval")
        # Cadence controls proactive swaps, not the use of vacant native slots.
        self.first_output_refill = bool(config.get("first_output_refill", False))
        if self.first_output_refill and not self.native_limit:
            raise ValueError("first-output refill requires native max_num_seqs")
        self.ranked_choices = []
        self.deadline_first = bool(config.get("deadline_first_recovery", False))
        self.next_policy_at = float('-inf')
        self.policy_tick = True
        self.executed_requests = {}
        self.phase_started = {}
        self.virtual_text_output = {}
        self.feedback_cache = {}
        self.snapshot_cache_key = None
        self.snapshot_cache_data = None
        self.last_snapshot = {}
        self.admission_hidden_ids = set()
        self.admission_blocked_resumes = set()
        self.last_mode = None
        self.pauses = PauseRegistry()
        self.last_lease_payload = None
        self.actual_slices = []
        self.last_tick = None
        self.last_opportunistic_pause = float('-inf')
        self.last_scheduled_ids = set()
        self.native_execution_seen = set()
        self.last_selected = set()
        self.decision_counts = Counter()
        self.last_log = 0.0
        self.latched_fault = None
        self.telemetry = dict(steps=0, full_steps=0, held_steps=0, scheduled_tokens=0,
                              buffer_steps=0, fcfs_steps=0, resource_decode_tokens=0)
        self.capacity_signals = CapacityExecutionSignals(scheduler, config) if config.get("resource_telemetry") else None
        self.recovery_pending = False
        self.last_telemetry = 0.
        self.seen_contracts = set()
        self.metadata_cache = {}
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
        if 'by_stage' in config:
            config = config['by_stage'].get(str(stage))
            if config is None:
                return None
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
        raw = getattr(request, 'additional_information', None)
        cached = self.metadata_cache.get(request.request_id)
        if cached is not None and cached[0] is request and cached[1] is raw:
            return cached[2]
        from vllm_omni.scheduling.policy import request_scheduling_metadata
        metadata = request_scheduling_metadata(request)
        self.metadata_cache[request.request_id] = (request, raw, metadata)
        return metadata

    def source_id(self, request):
        return self.metadata(request).get("sched_source_request_id")

    def order(self, request):
        return (self.metadata(request).get("sched_ingress_order", float("inf")), request.arrival_time)

    def read_snapshot(self, now=None):
        """Validate against a time sampled AFTER reading, including cached reads."""
        try:
            if self.policy_interval:
                stat = self.snapshot_path.stat()
                key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if key != self.snapshot_cache_key:
                    self.snapshot_cache_data = json.loads(self.snapshot_path.read_text())
                    self.snapshot_cache_key = key
                    self.decision_counts['snapshot_parses'] += 1
                data = self.snapshot_cache_data
            else:
                data = json.loads(self.snapshot_path.read_text())
            self.snapshot_read_at = self.clock() if now is None else now
            now = self.snapshot_read_at
            if (data["schema_version"] != 1 or data["session"] != self.config["session"]
                    or data["modality"] != self.config["modality"]
                    or not 0 <= now - data["written_at"] <= self.snapshot_age):
                return {}, "snapshot_invalid"
            if "buffer_conservativeness" in data:
                data["buffer_conservativeness"] = buffer_conservativeness(data["buffer_conservativeness"])
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
            self.snapshot_read_at = self.clock() if now is None else now
            return {}, "snapshot_unavailable"

    def coverage(self, request, snapshot, now):
        rid = self.source_id(request)
        member = (snapshot.get("members", {}).get(rid)
                  or snapshot.get("paused_members", {}).get(rid))
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
            cached = self.feedback_cache.get(request.request_id) if self.policy_interval else None
            if cached is not None and cached[0] is request and cached[1] == raw:
                feedback = cached[2]
            else:
                feedback = ClientFeedback(**raw)
                if self.policy_interval:
                    self.feedback_cache[request.request_id] = (request, raw, feedback)
                    self.decision_counts['feedback_parses'] += 1
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

    def _waiting_requests(self):
        return list(self.scheduler.waiting) + list(getattr(self.scheduler, 'skipped_waiting', ()))

    def _new_prefill_waiters(self, snapshot):
        if self.first_output_refill or not self.policy_interval or self.policy_tick:
            return set()
        registered = set(snapshot.get('registered_ids', ()))
        registered.update(snapshot.get('members', ()))
        registered.update(snapshot.get('paused_members', ()))
        blocked = set()
        for r in self._waiting_requests():
            if self.executed_requests.get(r.request_id) is r or r.request_id in self.pauses:
                continue
            metadata = self.metadata(r)
            if (metadata.get('sched_source_request_id') in registered
                    or metadata.get('sched_request_path') == self.config['modality']):
                blocked.add(r.request_id)
        return blocked

    def _continue_between_policy_ticks(self, snapshot):
        """Execute native steps while retaining P and coalescing new prefill."""
        self.ranked_choices = []
        s = self.scheduler
        original = list(s.running)
        eligible = [r for r in original if r.request_id not in self.pauses]
        owners = [r for r in eligible if self.active_ids is None or r.request_id in self.active_ids]
        pending = [r for r in eligible if self.active_ids is not None and r.request_id not in self.active_ids]
        # A resident restored ahead of an owner is not permission to preempt
        # that owner. Preserve execution ownership, then fill genuine vacancies.
        chosen = (owners + pending)[:self.cap]
        blocked = self._new_prefill_waiters(snapshot)
        blocked.update(r.request_id for r in self._waiting_requests() if r.request_id in self.pauses)
        self.admission_blocked_resumes = blocked
        self.last_selected = {r.request_id for r in chosen}
        self.decision_counts['cadence_cached_steps'] += 1
        self.decision_counts['buffer_steps'] += 1
        return ExecutionDecision(running=tuple(chosen), blocked_waiting=frozenset(blocked),
                                 sort_restored=False)

    @contextmanager
    def scheduling_context(self):
        snapshot, fallback = self.read_snapshot()
        now = self.snapshot_read_at
        self._gc_request_state()
        if snapshot:
            self.last_snapshot = snapshot
        # An invalid control snapshot freezes the last known queue admission;
        # it must never release all backend waiters by accident.
        gate_snapshot = snapshot or self.last_snapshot
        if gate_snapshot.get('engine_queue_admission'):
            # Requests sent after the last control snapshot still fail closed.
            # Unmanaged audio through the text stage remains native.
            known = set(gate_snapshot.get('registered_ids', ()))
            for r in list(self.scheduler.running) + self._waiting_requests():
                if self.metadata(r).get('sched_request_path') == self.config['modality']:
                    known.add(self.source_id(r))
            gate_snapshot = dict(gate_snapshot, registered_ids=list(known))
        with engine_queue_admission_context(self.scheduler, gate_snapshot, self.source_id) as mask:
            self.admission_hidden_ids = set(mask['hidden_running_ids']) | set(mask['hidden_waiting_ids'])
            decision = self.decide(now, snapshot, fallback)
            self.last_decision = decision
            result = None
            try:
                with apply_execution_decision(self.scheduler, decision, self.source_id, self.order) as result:
                    yield
            finally:
                if result is not None and not decision.native_passthrough:
                    self.active_ids = result.active_ids
                self._gc_request_state()

    def _recover_ack_timeout(self, snapshot, now):
        """A missed control deadline is recoverable after exact-epoch ACKs.

        Called only with a snapshot validated by read_snapshot. Keep contract
        faults latched. Never infer restored quota from elapsed time, a stale
        snapshot, or a different pause epoch. Capacity fallback is evaluated
        independently after clearing this communication fault.
        """
        if self.latched_fault != 'pause_resume_ack_timeout' or not snapshot:
            return
        acknowledged = []
        pending = self.pauses.requested_resumes
        for rid in pending:
            request = self.scheduler.requests.get(rid)
            if request is None or rid not in self.pauses or not self.pauses.releases_quota(rid):
                continue
            if getattr(request.status, 'name', '').startswith('FINISHED'):
                continue
            source = self.source_id(request)
            ack = snapshot.get('pause_acknowledgements', {}).get(source, {})
            if (ack.get('epoch') != self.pauses[rid].epoch or ack.get('state') != 'active'
                    or source not in snapshot.get('members', {})):
                return
            acknowledged.append(source)
        self.latched_fault = None
        self.emit('client_buffer_resume_ack_recovered', stage_id=self.stage,
                  source_request_ids=acknowledged, decision_monotonic=now)

    def decide(self, now, snapshot, fallback):
        """Select execution and pause actions without mutating native queues."""
        self.ranked_choices = []
        s = self.scheduler
        original = list(s.running)
        original_cap = s.max_num_running_reqs
        if original_cap != self.resident_cap:
            raise RuntimeError("resident cap changed")
        if self.last_tick is not None:
            self.actual_slices.append(now - self.last_tick)
            self.actual_slices = self.actual_slices[-64:]
        if not original and not self.pauses:
            self.actual_slices.clear()
        self.last_tick = now
        quota_protocol = snapshot.get('pause_admission_protocol') == 1
        def resume_acknowledged(r):
            pause = self.pauses.get(r.request_id)
            if pause is None:
                return False
            source = self.source_id(r)
            ack = snapshot.get('pause_acknowledgements', {}).get(source, {})
            return pause.acknowledge(ack, charged=source in snapshot.get('members', {}))
        def resume_deferred(r):
            ack = snapshot.get('pause_acknowledgements', {}).get(self.source_id(r), {})
            pause = self.pauses.get(r.request_id)
            return (pause is not None and ack.get('epoch') == pause.epoch
                    and ack.get('state') == 'resume_deferred')
        self._recover_ack_timeout(snapshot, now)
        fallback = self.latched_fault or fallback
        mode = "fcfs" if fallback else "buffer"
        self.policy_tick = (not self.policy_interval or now >= self.next_policy_at
                            or mode != self.last_mode)
        if self.policy_interval and self.policy_tick:
            self.next_policy_at = now + self.policy_interval
            self.executed_requests = {rid:r for rid,r in self.executed_requests.items()
                                      if s.requests.get(rid) is r}
            self.feedback_cache = {rid:entry for rid,entry in self.feedback_cache.items()
                                   if s.requests.get(rid) is entry[0]}

        # Keep the physical replenishment estimate separate from the extra
        # protection used to choose donors and set their return deadlines.
        conservativeness = snapshot.get("buffer_conservativeness", 1.0)
        guard = conservativeness * (self.resume_s + self.margin_s
                                    + max(self.actual_slices, default=self.min_slice))
        step_guard = max(self.admission_guard,
                         max(self.actual_slices, default=self.min_slice) + self.min_slice)
        cover = {}
        urgent = bool(self.pauses.requested_resumes)
        if self.policy_interval and not self.policy_tick and mode == "buffer":
            # Only paused obligations need per-step safety checks. Fresh client
            # receipts can extend credit; ACK/control changes never wait .5 s.
            for rid in self.pauses:
                r = s.requests.get(rid)
                if r is None:
                    urgent = True
                    continue
                credit = self.coverage(r, snapshot, now)
                cover[rid] = credit
                if credit is not None:
                    self.pauses[rid].resume_by = now + credit - guard
                if (credit is None or now + step_guard >= self.pauses[rid].resume_by
                        or resume_acknowledged(r)):
                    urgent = True
            for r in original:
                if r.request_id not in self.seen_contracts:
                    cover[r.request_id] = self.coverage(r, snapshot, now)
            if not urgent and not self.latched_fault and self.active_ids is not None:
                return self._continue_between_policy_ticks(snapshot)
        if self.policy_interval:
            self.decision_counts['periodic_policy_decisions' if self.policy_tick else
                                 'urgent_policy_decisions'] += 1
        cover.update({r.request_id: self.coverage(r, snapshot, now) for r in original
                      if r.request_id not in cover})
        fallback = self.latched_fault or fallback
        mode = "fcfs" if fallback else "buffer"
        # At the rate limit, ordinary work runs native FCFS. Buffer credit is
        # still inspected: a confirmed safe exchange replaces demand rather
        # than expanding the full set. Saturation must not disable that check.
        bound = snapshot.get('admission_gamma') or snapshot.get('gamma')
        at_capacity = (mode == 'buffer' and bound is not None and
                       snapshot.get('demand', 0.) >= bound-1e-9)
        if self.native_limit and at_capacity and not self.pauses:
            progress = snapshot.get('startup_feedback', snapshot.get('feedback', {}))
            owners = (self.active_ids if self.active_ids is not None else
                      {r.request_id for r in sorted(original, key=self.order)[:self.cap]})
            replenishment_waiter = any(
                r.request_id not in owners
                and progress.get(self.source_id(r), {}).get('started', False)
                and cover.get(r.request_id) is not None
                and cover[r.request_id] <= guard + step_guard + self.policy_interval
                for r in original)
            waiter = (snapshot.get('admission_waiter') or self._waiting_requests()
                      or replenishment_waiter or
                      any(r.num_computed_tokens > 0 and
                          not progress.get(self.source_id(r), {}).get('started', False)
                          for r in original))
            safe_donor = any(cover.get(r.request_id) is not None and
                cover[r.request_id] > guard+self.admission_guard+step_guard and
                self.decode_boundary(r) for r in original)
            if not waiter or not safe_donor:
                mode, fallback = 'fcfs', 'capacity_full'
        if (self.native_limit and mode == "fcfs" and not self.pauses
                and (self.active_ids is None or len(original) <= self.cap)):
            # Only native-owned queues can be passed through unchanged. After
            # buffer pauses end, retained KV owners may still exceed the native
            # execution limit. Keep selecting current owners until that excess
            # drains; otherwise native schedule sees >max_num_seqs RUNNING and
            # asserts even though no pause lease remains.
            self.recovery_pending = False
            self.active_ids = None
            self.last_selected = {r.request_id for r in original}
            self.decision_counts["fcfs_steps"] += 1
            if mode != self.last_mode or now - self.last_log >= 1:
                self.emit("client_buffer_mode", stage_id=self.stage, mode=mode,
                    reason=fallback, demand=snapshot.get("demand"), gamma=snapshot.get("gamma"),
                    members=list(snapshot.get("members", {})),
                    selected=[r.request_id for r in original],
                    concurrency_source="max_num_seqs", execution_limit=self.cap,
                    retained_kv_requests=self._resident_count(),
                    decision_counts=dict(self.decision_counts), pending_pauses=0)
                self.last_log = now
            self.last_mode = mode
            return ExecutionDecision(native_passthrough=True)
        ordered = sorted(original, key=self.order)
        def fcfs_continuation():
            # Native FCFS is non-preemptive except for resource pressure. A
            # retained KV owner is not necessarily an executing slot owner.
            # Continue the current owners, then fill vacancies by arrival.
            # Sorting all residents and truncating silently evicts borrowers
            # without client credit and is NOT native FCFS continuation.
            if self.active_ids is None:
                return ordered[:self.cap]
            owners = [r for r in original if r.request_id in self.active_ids]
            pending = [r for r in ordered if r.request_id not in self.active_ids]
            return (owners + pending)[:self.cap]

        chosen = fcfs_continuation()
        newly_paused = {}
        recovering = mode == "fcfs" and bool(self.pauses)
        self.recovery_pending = recovering
        can_admit = False
        resume_from_waiting = []
        beneficiary_waiters = []
        due_order = {}
        scored_step = False
        if mode == "buffer" or recovering:
            # Retain execution-slot owners until a safe, deliberate swap. Merely
            # having a lower priority never pauses a client without buffer credit.
            if self.active_ids is None:
                self.active_ids = {r.request_id for r in chosen}
            chosen = [r for r in ordered if r.request_id in self.active_ids][:self.cap]
            waiting_requests = self._waiting_requests()
            waiting_ids = {r.request_id for r in waiting_requests}
            scored_step = (mode == 'buffer' and self.policy_tick
                           and snapshot.get('recovery_ranking') == 'paper'
                           and not snapshot.get('admission_waiter')
                           and all(self.metadata(r).get('sched_request_path') == self.config['modality']
                                   for r in original + waiting_requests))
            priority = CandidatePriority(self, snapshot, now, cover) if scored_step else None
            held = [r for r in ordered + waiting_requests if r.request_id in self.pauses]
            # A chunk boundary may move a policy pause into native waiting.
            # It remains the same recovery obligation, including quota ACK.
            for r in held:
                if r.request_id not in cover:
                    cover[r.request_id] = self.coverage(r, snapshot, now)
            # Fresh client receipts may include in-flight output after a pause.
            # Recompute deadlines from current client credit, not a frozen lease.
            for r in held:
                if cover[r.request_id] is not None:
                    self.pauses[r.request_id].resume_by = now + cover[r.request_id] - guard
            held.sort(key=lambda r: self.pauses[r.request_id].resume_by)
            step_guard = max(self.admission_guard,
                             max(self.actual_slices, default=self.min_slice) + self.min_slice)
            # Batch returns which would become urgent before the next tick.
            # Out-of-band urgent/ACK decisions use the original safety horizon.
            return_horizon = step_guard + (self.policy_interval if self.policy_tick else 0.)
            due = [r for r in held if cover[r.request_id] is None
                   or now + return_horizon >= self.pauses[r.request_id].resume_by
                   or cover[r.request_id] <= guard + return_horizon
                   or self.pauses.is_resume_requested(r.request_id) or resume_acknowledged(r)]

            if self.deadline_first:
                due_order = {r.request_id: i for i, r in enumerate(due)
                             if cover[r.request_id] is None or cover[r.request_id] <= guard + return_horizon}

            def safe_victims():
                return [r for r in chosen if r not in due and r.request_id not in self.pauses
                        and self.decode_boundary(r)
                        and cover[r.request_id] is not None
                        and cover[r.request_id] > guard + self.admission_guard + step_guard]

            planned_demand = snapshot.get('demand', 0.)
            # Existing paused promises remain obligations after a capacity drop.
            # Restoring them may overload the ledger (which freezes new admission);
            # it must not be blocked a second time by the new, smaller Gamma.
            recovery_bound = float('inf') if recovering else snapshot.get('gamma', float('inf'))
            def active_rate(r):
                return snapshot.get('members', {}).get(self.source_id(r), {}).get('rate', 0.)
            def paused_rate(r):
                return snapshot.get('paused_members', {}).get(self.source_id(r), {}).get('rate', 0.)
            for r in due:
                if r in chosen:
                    continue  # selected but not yet executed is not a second slot
                if (self.pauses.is_resume_requested(r.request_id) and not resume_acknowledged(r)
                        and not resume_deferred(r) and now < self.pauses[r.request_id].resume_by):
                    continue  # an atomic quota exchange is already in flight
                cost = paused_rate(r) if self.pauses.releases_quota(r.request_id) else 0.
                # Both an execution slot AND active demand budget are required.
                # A spare GPU sequence slot does not fund a returning promise.
                while (len(chosen) + len(resume_from_waiting) >= self.cap or
                       planned_demand + cost > recovery_bound + 1e-9):
                    victims = safe_victims()
                    if planned_demand + cost > recovery_bound + 1e-9:
                        victims = [v for v in victims if cover[v.request_id] > guard +
                                   snapshot.get('pause_release_min_s', .5) + step_guard]
                    if not victims:
                        break
                    victim = (min(victims, key=priority.score) if priority else
                              max(victims, key=lambda x: cover[x.request_id]))
                    chosen.remove(victim)
                    planned_demand -= active_rate(victim)
                    source = self.source_id(r)
                    progress = snapshot.get('startup_feedback', snapshot.get('feedback', {})).get(source, {})
                    newly_paused[victim.request_id] = PausedRequest(
                        now, now + max(0, cover[victim.request_id] - guard), guard,
                        releases_quota=quota_protocol, beneficiary=source,
                        beneficiary_received=progress.get('received', 0.))
                blocked = (len(chosen) + len(resume_from_waiting) >= self.cap or
                           planned_demand + cost > recovery_bound + 1e-9)
                if blocked and cover[r.request_id] is not None and cover[r.request_id] > guard:
                    continue  # early warning: wait for a safe exchange or completion
                if blocked:
                    fallback, mode = "no_safe_resume_slot", "fcfs"
                    self.recovery_pending = True
                    # Do not discard a successfully planned safe exchange, and
                    # never evict an unbuffered owner to conceal infeasibility.
                    # A spare physical slot can repay existing restoration debt.
                    if len(chosen) + len(resume_from_waiting) >= self.cap:
                        self.decision_counts['recovery_blocked_no_safe_slot'] += 1
                        continue
                if r.request_id in waiting_ids:
                    resume_from_waiting.append(r)
                else:
                    chosen.append(r)
                planned_demand += cost
            startup_members = snapshot.get('startup_members', snapshot.get('members', {}))
            startup_feedback = snapshot.get('startup_feedback', snapshot.get('feedback', {}))
            local_sources = {self.source_id(r) for r in original}
            local_sources.update(self.source_id(r) for r in s.waiting)

            def beneficiary_pending(r):
                pause = self.pauses[r.request_id]
                source = pause.beneficiary
                progress = startup_feedback.get(source, {})
                waiting = not progress.get('started', False)
                if pause.beneficiary_received is not None:
                    waiting = progress.get('received', 0.) <= pause.beneficiary_received
                return ((self.pauses.releases_quota(r.request_id) or source in local_sources)
                        and source in startup_members and waiting)

            # A useful pause below max_num_seqs must not be cancelled merely
            # because a sequence slot is free. Let its actual beneficiary reach
            # first output, or restore at the safety deadline, whichever is first.
            first_output_waiting = self.first_output_refill and any(
                r.request_id not in self.pauses
                and not startup_feedback.get(self.source_id(r), {}).get('started', False)
                for r in waiting_requests)
            scored_step = scored_step and mode == 'buffer'
            for r in (() if scored_step else ordered):
                if len(chosen) + len(resume_from_waiting) >= self.cap:
                    break
                quota_needed = (self.pauses.releases_quota(r.request_id) and
                    planned_demand + paused_rate(r) > recovery_bound + 1e-9)
                if (r not in chosen and r.request_id not in newly_paused and not quota_needed and not
                    (self.pauses.is_resume_requested(r.request_id) and not resume_acknowledged(r))
                    and not (r in held and r not in due and beneficiary_pending(r))
                    and not (first_output_waiting and r in held and r not in due)
                    and not (recovering and r in held and r not in due)):
                    chosen.append(r)
                    planned_demand += paused_rate(r)
            resident_waiters = [r for r in ordered if r not in chosen
                                and r.request_id not in self.pauses]
            # In native-limit mode max_num_seqs limits execution, not the number
            # of retained KV owners. Native allocation enforces physical memory;
            # W/Gamma bounds active admitted demand. Confirmed pauses in P
            # retain state/KV and use client credit until quota is restored.
            startup_waiters = [r for r in chosen
                if r.num_computed_tokens > 0
                and self.source_id(r) in startup_members
                and not startup_feedback.get(self.source_id(r), {}).get('started', False)]
            execution_waiter = resident_waiters or (s.waiting and
                (self.native_limit or self._resident_count() < self.resident_cap)) or startup_waiters
            admission_waiter = snapshot.get('admission_waiter') if quota_protocol else None
            # A gateway-blocked request is not in the engine queues yet. Its
            # absence must not prevent the safe pause that funds its admission.
            def frees_head(r):
                if not admission_waiter:
                    return False
                eligible = admission_waiter.get('eligible_pause_sources')
                if eligible is not None and self.source_id(r) not in eligible:
                    return False
                rate = snapshot.get('members', {}).get(self.source_id(r), {}).get('rate', 0.)
                costs = snapshot.get('pause_release_cost_by_request', {}).get(
                    self.source_id(r), snapshot.get('pause_release_cost_per_rate', {}))
                return all(deficit <= rate * costs.get(key, 0.) + 1e-9
                           for key, deficit in admission_waiter['resource_deficit'].items())
            admission_victims = [r for r in safe_victims() if frees_head(r)
                and cover[r.request_id] > guard + max(snapshot.get('pause_release_min_s', .5),
                    admission_waiter.get('startup_latency_estimate_s', 0.) if admission_waiter else 0.) + step_guard]
            can_admit = execution_waiter or admission_victims
            if scored_step and mode == 'buffer':
                def eligible_return(r):
                    if self.pauses.is_resume_requested(r.request_id) and not resume_acknowledged(r):
                        return False
                    return r not in due and not beneficiary_pending(r)

                can_admit = priority.fill(chosen, resume_from_waiting, beneficiary_waiters,
                                          waiting_requests, original, newly_paused, guard,
                                          safe_victims, planned_demand, quota_protocol, eligible_return)
            else:
                scored_step = False
            # Buffer-safe requests may yield compute to actual waiting work
            # even below max_num_seqs. Full sequence occupancy is not required
            # for compute contention (in particular during a new prefill).
            if (not scored_step and mode == "buffer" and self.policy_tick and not due_order
                    and (not due or self.first_output_refill) and can_admit and chosen
                    and now - self.last_opportunistic_pause >= self.min_slice):
                self.decision_counts['waiting_work_checks'] += 1
                victims = admission_victims if admission_victims else safe_victims()
                if not victims:
                    self.decision_counts['waiting_without_safe_victim'] += 1
                    threshold = guard + self.admission_guard + step_guard
                    for candidate in chosen:
                        credit = cover[candidate.request_id]
                        reason = ('no_valid_client_credit' if credit is None else
                                  'insufficient_client_credit' if credit <= threshold else
                                  'not_decode_boundary' if not self.decode_boundary(candidate) else
                                  'resume_has_priority')
                        self.decision_counts['ineligible_' + reason] += 1
                if victims:
                    # At a policy tick, offer the admitted waiting cohort a
                    # batch of independently safe donors, not one prefill per
                    # decode step. No extra concurrency/residency cap is added.
                    beneficiaries = []
                    assigned = {pause.beneficiary for pause in self.pauses.values()
                                if pause.beneficiary is not None}
                    if admission_victims and admission_waiter['request_id'] not in assigned:
                        beneficiaries.append((admission_waiter['request_id'], None, True))
                    seen_beneficiaries = assigned | {source for source, _, _ in beneficiaries}
                    candidates = resident_waiters + self._waiting_requests() + startup_waiters
                    # A started stream can still need replenishment. First
                    # output is not evidence that its future client buffer is
                    # safe. Rank urgent continuing streams by depletion time;
                    # retain arrival order for new streams. Use only validated
                    # client credit, never stage-to-stage inventory.
                    def needs_replenishment(candidate):
                        source = self.source_id(candidate)
                        if not startup_feedback.get(source, {}).get('started', False):
                            return False
                        if candidate.request_id not in cover:
                            cover[candidate.request_id] = self.coverage(candidate, snapshot, now)
                        credit = cover[candidate.request_id]
                        return credit is not None and credit <= guard + return_horizon

                    def beneficiary_order(candidate):
                        if self.first_output_refill and needs_replenishment(candidate):
                            return (0, cover[candidate.request_id], self.order(candidate))
                        return (1, 0., self.order(candidate))

                    for candidate in sorted(candidates, key=beneficiary_order):
                        source = self.source_id(candidate)
                        started = startup_feedback.get(source, {}).get('started', False)
                        if (candidate.request_id not in self.pauses
                                and candidate not in resume_from_waiting
                                and (not self.first_output_refill or not started
                                     or needs_replenishment(candidate))
                                and source not in seen_beneficiaries):
                            beneficiaries.append((source, candidate, False))
                            seen_beneficiaries.add(source)
                    if self.first_output_refill:
                        beneficiaries.sort(key=lambda item: beneficiary_order(item[1])
                                           if item[1] is not None else (1, 0., (float('-inf'), 0.)))
                    if not self.policy_interval:
                        beneficiaries = beneficiaries[:1]
                    pool = list(safe_victims())
                    # A free sequence slot does not imply spare compute. A
                    # safe donor can relieve competition for an actual newcomer
                    # even below max_num_seqs. Existing pause/ACK rules still
                    # protect the donor and bound its absence.
                    for source, beneficiary, funds_admission in beneficiaries:
                        available = [v for v in pool if v in chosen
                                     and (not funds_admission or v in admission_victims)]
                        if not available:
                            break
                        victim = max(available, key=lambda r: cover[r.request_id])
                        pool.remove(victim)
                        chosen.remove(victim)
                        pause = PausedRequest(
                            now, now + max(0, cover[victim.request_id] - guard), guard,
                            releases_quota=quota_protocol, beneficiary=source)
                        newly_paused[victim.request_id] = pause
                        if source is not None:
                            progress = startup_feedback.get(source, {})
                            if progress.get('started', False):
                                # A continuing beneficiary must receive NEW
                                # output; its old started flag cannot repay a
                                # newly created pause immediately.
                                pause.beneficiary_received = progress.get('received', 0.)
                                self.decision_counts['replenishment_exchanges'] += 1
                        if funds_admission:
                            self.decision_counts['admission_funded_pauses'] += 1
                        self.last_opportunistic_pause = now
                        if beneficiary is not None and beneficiary in resident_waiters:
                            chosen.append(beneficiary)
                        elif beneficiary is not None and beneficiary in self._waiting_requests():
                            beneficiary_waiters.append(beneficiary)
        waiting_resume_ack = any(
            rid in self.pauses and rid in s.requests
            and not resume_acknowledged(s.requests[rid])
            for rid in self.pauses.requested_resumes)
        for r in list(chosen) + list(resume_from_waiting):
            rid = r.request_id
            if not self.pauses.releases_quota(rid):
                continue
            self.pauses[rid].request_resume(now)
            if not resume_acknowledged(r):
                # resume_by is the replenishment deadline, not permission to
                # execute without capacity. Even a late return must first be
                # charged by the gateway. Keep publishing resume_requested.
                if r in chosen:
                    chosen.remove(r)
                if r in resume_from_waiting:
                    resume_from_waiting.remove(r)
                waiting_resume_ack = True
                since = self.pauses[rid].ack_wait_since
                if resume_deferred(r) and now < self.pauses[rid].resume_by:
                    # Explicit, fresh coordination response, not a lost ACK.
                    # Retry safe exchanges until the client-safe deadline.
                    self.pauses[rid].ack_wait_since = now
                elif now - since > self.snapshot_age and not self.latched_fault:
                    self.latched_fault = 'pause_resume_ack_timeout'
                    fallback, mode = self.latched_fault, 'fcfs'
                    self.emit('client_buffer_resume_ack_timeout', stage_id=self.stage, request_id=rid,
                              source_request_id=self.source_id(r), decision_monotonic=now,
                              acknowledgement_wait_s=now - since)
            else:
                self.pauses[rid].ack_wait_since = None
        self.decision_counts[mode + '_steps'] += 1
        if mode == 'buffer' and not can_admit:
            self.decision_counts['no_scheduler_waiter'] += 1
        # Paused native waiters must not sneak through admission, even when
        # their turn has not reached the recovery deadline yet.
        blocked_paused_waiters = {r.request_id for r in self._waiting_requests()
            if r.request_id in self.pauses and r not in resume_from_waiting}
        if resume_from_waiting:
            ready_returns = {r.request_id for r in resume_from_waiting}
            if not self.first_output_refill:
                blocked_paused_waiters.update(r.request_id for r in self._waiting_requests() if r.request_id not in ready_returns)
        if scored_step:
            # The native skipped and regular queues have separate orderings.
            # Enforce the same score winners even when they occupy different queues.
            allowed = {r.request_id for r in resume_from_waiting + beneficiary_waiters}
            blocked_paused_waiters.update(r.request_id for r in waiting_requests if r.request_id not in allowed)
        if mode == "buffer":
            blocked_paused_waiters.update(self._new_prefill_waiters(snapshot))
        if self.first_output_refill and (waiting_resume_ack or resume_from_waiting or beneficiary_waiters):
            # Reserve only actual outstanding returns. A single quota ACK must
            # not close every other free slot to already-admitted newcomers.
            pending_ack = {rid for rid in self.pauses.requested_resumes
                if rid in self.pauses and rid in s.requests
                and not resume_acknowledged(s.requests[rid])}
            ready_returns = {r.request_id for r in resume_from_waiting}
            selected_beneficiaries = {r.request_id for r in beneficiary_waiters}
            reserved_waiters = ready_returns | selected_beneficiaries
            free_for_new = max(0, self.cap - len(chosen)
                               - len(pending_ack) - len(reserved_waiters))
            candidates = [r for r in self._waiting_requests()
                if r.request_id not in blocked_paused_waiters
                and r.request_id not in reserved_waiters]
            admitted_new = {r.request_id for r in candidates[:free_for_new]}
            blocked_paused_waiters.update(r.request_id for r in candidates
                                          if r.request_id not in admitted_new)
        if due_order:
            # Native decode processes running requests before waiting requests.
            # Give returning residents first access to the iteration token budget.
            chosen.sort(key=lambda r: due_order.get(r.request_id, len(due)))
            if (resume_from_waiting and len(chosen) + len(waiting_requests) > self.cap):
                # Native skipped_waiting and waiting have independent ordering.
                # Across queues, expose only waiting returns until their first
                # service step. Existing running owners continue decoding.
                returns = {r.request_id for r in resume_from_waiting}
                blocked_paused_waiters.update(r.request_id for r in waiting_requests
                                              if r.request_id not in returns)
            self.decision_counts['deadline_first_steps'] += 1
        if (due_order and mode == 'buffer' and snapshot.get('recovery_ranking') == 'weighted'):
            self.ranked_choices = rank_recovery(self, snapshot, now, due_order, chosen,
                resume_from_waiting, beneficiary_waiters, blocked_paused_waiters,
                waiting_requests, original, cover, newly_paused)
        self.admission_blocked_resumes = blocked_paused_waiters
        selected_ids = {r.request_id for r in chosen}
        skipped = {r.request_id for r in original} - selected_ids
        for r in original:
            rid = r.request_id
            if rid in skipped and rid in newly_paused:
                if rid not in self.pauses:
                    self.decision_counts["pauses"] += 1
                    self.pauses[rid] = newly_paused[rid]
                    self.emit("client_buffer_pause", stage_id=self.stage, request_id=rid,
                              source_request_id=self.source_id(r), coverage_s=cover[rid],
                              resume_by=self.pauses[rid].resume_by, inventory_source=snapshot.get("buffer_source", "client_consumption_buffer"),
                              mode=mode, decision_monotonic=now, demand=snapshot.get("demand"), gamma=snapshot.get("gamma"),
                              trigger="resume_due" if due else "waiting_work",
                              beneficiary_source_request_id=self.pauses[rid].beneficiary,
                              pause_releases_admission=quota_protocol, lease_epoch=now,
                              selected_before_pause=len(chosen) + len(newly_paused),
                              execution_limit=self.cap, buffer_conservativeness=conservativeness, safety_threshold_s=guard + self.admission_guard + step_guard)
        if mode != self.last_mode or now - self.last_log >= 1:
            self.metadata_cache = {rid:entry for rid,entry in self.metadata_cache.items()
                                   if s.requests.get(rid) is entry[0]}
            self.emit("client_buffer_mode", stage_id=self.stage, mode=mode,
                      reason=fallback or "capacity_feasible", demand=snapshot.get("demand"),
                      gamma=snapshot.get("gamma"), members=list(snapshot.get("members", {})),
                      selected=[r.request_id for r in chosen],
                      concurrency_source="max_num_seqs" if self.native_limit else "legacy_execution_cap",
                      execution_limit=self.cap, retained_kv_requests=self._resident_count(),
                      decision_counts=dict(self.decision_counts),
                      pending_pauses=len(self.pauses))
            self.last_log, self.last_mode = now, mode
        # A mode change is not execution: keep paused identities and deadlines
        # until record_scheduled observes their actual return. FCFS fills freed
        # slots in arrival order; it does not promise buffer-safe resumption
        # under a genuine sustained capacity loss.
        if self.native_limit and mode == "fcfs" and len(original) <= self.cap and not self.pauses and not newly_paused and not waiting_resume_ack:
            # Preserve native running order and admission exactly for the clean
            # FCFS arm. A fallback with retained excess owners drains by arrival.
            self.recovery_pending = False
            self.active_ids = None
            self.last_selected = {r.request_id for r in original}
            return ExecutionDecision(native_passthrough=True)
        free = 0 if self.native_limit else max(0, self.resident_cap - self._resident_count())
        self.last_selected = selected_ids
        return ExecutionDecision(
            running=tuple(chosen), blocked_waiting=frozenset(blocked_paused_waiters),
            waiting_priority=tuple(resume_from_waiting + beneficiary_waiters),
            hide_all_waiting=waiting_resume_ack and not self.first_output_refill,
            max_running=original_cap if self.native_limit else min(self.cap, len(chosen) + free))

    def _gc_request_state(self):
        requests = self.scheduler.requests
        self.pauses.discard_terminal(requests)
        live = {rid for rid, r in requests.items()
                if not getattr(r.status, 'name', '').startswith('FINISHED')}
        for states in (self.executed_requests, self.phase_started, self.virtual_text_output, self.feedback_cache, self.metadata_cache):
            for rid in states.keys() - live:
                states.pop(rid)
        self.seen_contracts.intersection_update(live)
        self.native_execution_seen.intersection_update(live)

    def record_final_text(self, request, units):
        if (self.stage != 0 or units <= 0
                or self.last_snapshot.get('buffer_source') != 'server_virtual_output'):
            return
        metadata = self.metadata(request)
        if metadata.get('sched_request_path') != 'text':
            return  # Thinker output on an audio path is not playable output.
        rid = request.request_id
        now = self.clock()
        if rid not in self.virtual_text_output:
            self.virtual_text_output[rid] = VirtualOutput(
                self.source_id(request), metadata['sched_text_consume_rate_tps'], now, now)
        self.virtual_text_output[rid].produce(units, now)

    def record_phase_completed(self, request, finish_reason):
        path = self.config.get('execution_telemetry_path')
        started = self.phase_started.pop(request.request_id, None)
        reason = str(finish_reason)
        if not path or self.stage != 0 or reason not in ('stop', 'length') or started is None:
            return
        metadata = self.metadata(request)
        record = dict(event='completed', stage=0, session=self.config['session'],
            request_id=metadata.get('sched_source_request_id'),
            route=metadata.get('sched_request_path'), backend_request_id=request.request_id,
            started_at=started, completed_at=self.clock(), finish_reason=reason)
        # Append at terminal completion, including when the engine goes idle
        # immediately afterwards; not dependent on a subsequent schedule step.
        with Path(str(path) + '.completed.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def record_scheduled(self, output):
        now = self.clock()
        if len(output.num_scheduled_tokens) > self.cap:
            raise RuntimeError("client-buffer execution cap exceeded")
        self.last_scheduled_ids = set(output.num_scheduled_tokens)
        for old, new, chose_new in self.ranked_choices:
            self.emit('client_buffer_recovery_rank_result', stage_id=self.stage, request_id=old,
                      newcomer=new, chose_new=chose_new, old_executed=old in self.last_scheduled_ids,
                      new_executed=new in self.last_scheduled_ids)
        self.ranked_choices = []
        path = self.config.get('execution_telemetry_path')
        if path:
            for request_id in output.num_scheduled_tokens:
                if request_id in self.native_execution_seen:
                    continue
                request = self.scheduler.requests.get(request_id)
                if request is None:
                    continue
                metadata = self.metadata(request)
                if metadata.get('sched_request_path') == self.config['modality']:
                    with Path(str(path) + '.admissions.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(event='first_execution',
                            session=self.config['session'], stage=self.stage,
                            request_id=self.source_id(request), started_at=now,
                            backend_request_id=request_id)) + '\n')
                    self.native_execution_seen.add(request_id)
        if self.stage == 0:
            for request_id in output.num_scheduled_tokens:
                if request_id not in self.phase_started and request_id in self.scheduler.requests:
                    self.phase_started[request_id] = now
                    path = self.config.get('execution_telemetry_path')
                    if path:
                        metadata = self.metadata(self.scheduler.requests[request_id])
                        with Path(str(path) + '.completed.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(event='started', stage=0,
                                session=self.config['session'], started_at=now,
                                request_id=metadata.get('sched_source_request_id'),
                                route=metadata.get('sched_request_path'),
                                backend_request_id=request_id)) + '\n')
            # Aborted/preempted attempts cannot supply a completion event.
            self.phase_started = {rid: start for rid, start in self.phase_started.items()
                                  if rid in self.scheduler.requests}
        if self.policy_interval:
            self.executed_requests.update((rid, self.scheduler.requests[rid])
                for rid in output.num_scheduled_tokens if rid in self.scheduler.requests)
        if self.capacity_signals is not None:
            self.telemetry.update(self.capacity_signals.record(output, now, paused_ids=set(self.pauses) | self.admission_hidden_ids | self.admission_blocked_resumes))
        if self.config.get('resource_telemetry'):
            # schedule() has already advanced num_computed_tokens. Count only
            # scheduled decode positions, excluding prompt/prefill positions.
            # This is COMPUTE accounting, never client-visible buffer credit.
            for rid, tokens in output.num_scheduled_tokens.items():
                request = self.scheduler.requests.get(rid)
                if request is not None:
                    before = request.num_computed_tokens - tokens
                    prompt = max(0, min(tokens, request.num_prompt_tokens - before))
                    self.telemetry['resource_decode_tokens'] += tokens - prompt
        self.telemetry["steps"] += 1
        self.telemetry[self.last_mode + "_steps"] += 1
        self.telemetry["full_steps"] += int(len(output.num_scheduled_tokens) >= self.cap)
        self.telemetry["held_steps"] += int(bool(self.pauses))
        self.telemetry["scheduled_tokens"] += sum(output.num_scheduled_tokens.values())
        for rid, tokens in output.num_scheduled_tokens.items():
            if tokens > 0 and rid in self.pauses:
                pause = self.pauses.pop(rid)
                self.emit("client_buffer_resumed", stage_id=self.stage, request_id=rid,
                          held_s=now - pause.epoch, scheduled_tokens=tokens)
        leases = {}
        for rid in self.pauses.quota_pauses:
            request = self.scheduler.requests.get(rid)
            if request is not None:
                leases[self.source_id(request)] = self.pauses[rid].lease(now)
        signature = {rid:{k:v for k,v in lease.items() if k != 'not_scheduled_at'}
                     for rid,lease in leases.items()}
        if self.config.get("execution_telemetry_path") and (now-self.last_telemetry >= .1
                or signature != self.last_lease_payload):
            atomic_json(self.config["execution_telemetry_path"], dict(self.telemetry,
                written_at=now, session=self.config["session"], stage=self.stage,
                last_scheduled=len(output.num_scheduled_tokens), execution_limit=self.cap,
                recovery_pending=self.recovery_pending and bool(self.pauses),
                virtual_text_output=[asdict(state) for state in self.virtual_text_output.values()],
                pause_leases=leases))
            self.last_telemetry = now
            self.last_lease_payload = signature


def create_adapter(scheduler, emit):
    return ClientBufferScheduler.from_environment(scheduler, emit)
