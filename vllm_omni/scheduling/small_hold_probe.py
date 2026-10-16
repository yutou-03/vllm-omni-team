"""Opt-in synchronous, KV-preserving hold for a stage-local experiment.

Set ``VLLM_OMNI_SMALL_HOLD_PROBE=1`` on a synchronous OmniARScheduler.
Call ``scheduler.request_hold(request_id, duration_s, reason, evidence)`` from
the scheduler thread, before scheduling. It returns whether the hold was
accepted. This primitive does not choose an inventory policy.

For a one-shot correctness probe, set ``VLLM_OMNI_SMALL_HOLD_CONTROL_FILE`` to
a JSON file read at scheduler initialization::

    {"schema_version": 1, "forced_hold": {
        "request_id": "*", "min_output_tokens": 8, "duration_s": 0.025,
        "reason": "forced_correctness", "evidence": {"run": "canary"}
    }}

``*`` selects the first eligible request, once for the entire scheduler;
otherwise request_id must match the physical scheduler ID exactly. The
minimum output count is checked only at a confirmed decode boundary. A hold is checked against its monotonic deadline at the next scheduling
boundary; synchronous execution cannot be interrupted at the deadline.
The buffer-policy mode blocks new prefills in its legacy hold path; urgent
keep admission intentionally allows the urgent request to enter. It leaves
the configured batch token budget unchanged and stops further holds after a
large deadline overrun. Events named
``small_hold_requested``, ``small_hold_released`` and
``small_hold_first_rescheduled`` expose these separate times.

Alternatively ``VLLM_OMNI_SMALL_HOLD_CONTROLLER=package.module:function``
loads an experimental callback once. It receives ``(scheduler, monotonic_now)``
after input chunks/policy ordering are refreshed and may call request_hold.
The controller and forced control file are mutually exclusive. No controller
is imported, and no control file is read, when the probe is disabled.

Experimental execution/residency split (default behavior is unchanged): set
``VLLM_OMNI_SMALL_HOLD_EXECUTION_CAP=K`` and optionally
``VLLM_OMNI_SMALL_HOLD_STAGES=0`` (allowed stages: 0 and 1). The configured
``max_num_seqs`` remains resident cap R, with 1 <= K <= R <= 64. K also applies
when the hold probe is disabled, so FCFS can use the same execution capacity.
Held and input-blocked requests retain resident slots. Released holds await an
execution slot, ahead of new waiting requests, and finish their pause only at
``small_hold_first_rescheduled``. This mode requires a fixed resident cap and excludes native remote-KV connectors.
Omni stage chunk transfers remain supported and count as resident while blocked.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

HOLD_ENV = "VLLM_OMNI_SMALL_HOLD_PROBE"
CONTROL_FILE_ENV = "VLLM_OMNI_SMALL_HOLD_CONTROL_FILE"
CONTROLLER_ENV = "VLLM_OMNI_SMALL_HOLD_CONTROLLER"
EXECUTION_CAP_ENV = "VLLM_OMNI_SMALL_HOLD_EXECUTION_CAP"
STAGES_ENV = "VLLM_OMNI_SMALL_HOLD_STAGES"
URGENT_KEEP_ENV = "VLLM_OMNI_URGENT_KEEP_ADMISSION"
MAX_HOLD_S = 0.10
MAX_RESIDENT_REQUESTS = 64
# A late synchronous step disables subsequent holds; it cannot interrupt that step.
MAX_DEADLINE_LATENESS_S = 0.05


def _duration(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("hold duration_s must be in (0, 0.10]")
    duration = float(value)
    if not math.isfinite(duration) or not 0 < duration <= MAX_HOLD_S:
        raise ValueError("hold duration_s must be in (0, 0.10]")
    return duration


def _running(request: Any) -> bool:
    return getattr(getattr(request, "status", None), "name", "") == "RUNNING"


@dataclass
class _Hold:
    request: Any
    requested_monotonic_s: float
    deadline_monotonic_s: float
    reason: str
    evidence: dict[str, Any]
    released_monotonic_s: float | None = None
    held_schedule_steps: int = 0


class SmallHoldProbe:
    """Scheduler-thread-only hold state; never owns or frees request KV."""

    def __init__(
        self,
        scheduler: Any,
        *,
        emit: Callable[..., None],
        clock: Callable[[], float] = time.monotonic,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.emit = emit
        self.clock = clock
        environment = os.environ if environ is None else environ
        self.stage_id = int(scheduler._omni_stage_id_for_trace())
        self.buffer_policy_mode = environment.get("VLLM_OMNI_STAGE_BUFFER_POLICY", "0") == "1"
        self.buffer_policy_suspended = False
        raw_urgent_keep = environment.get(URGENT_KEEP_ENV, "0")
        if raw_urgent_keep not in {"0", "1"}:
            raise ValueError(f"{URGENT_KEEP_ENV} must be 0 or 1")
        self.urgent_keep_mode = raw_urgent_keep == "1"
        stages = {0, 1} if self.buffer_policy_mode else {1}
        if STAGES_ENV in environment:
            raw_stages = environment[STAGES_ENV].split(",")
            if not raw_stages or any(value.strip() not in {"0", "1"} for value in raw_stages):
                raise ValueError(f"{STAGES_ENV} must contain only Stage 0 and/or 1")
            stages = {int(value.strip()) for value in raw_stages}
        if self.urgent_keep_mode and (STAGES_ENV not in environment or len(stages) != 1):
            raise ValueError("urgent keep requires exactly one target stage")
        self.execution_cap: int | None = None
        if EXECUTION_CAP_ENV in environment:
            raw_cap = environment[EXECUTION_CAP_ENV]
            if not raw_cap.isdecimal() or int(raw_cap) < 1:
                raise ValueError(f"{EXECUTION_CAP_ENV} must be a positive integer")
            if self.stage_id in stages:
                self.execution_cap = int(raw_cap)
        self.resident_cap = int(scheduler.max_num_running_reqs)
        # Persistent execution-slot owners, not GPU batch members.
        self._execution_active_ids: set[str] | None = None
        self._urgent_pending: dict[str, str] = {}
        self._urgent_pending_logged: set[str] = set()
        self._last_held_ids: set[str] = set()
        self.enabled = (
            environment.get(HOLD_ENV, "0") == "1"
            and self.stage_id in stages
        )
        self.holds: dict[str, _Hold] = {}
        self.accepted_request_ids: set[str] = set()
        self.forced_hold: dict[str, Any] | None = None
        self.forced_hold_fired = False
        self.controller: Callable[[Any, float], None] | None = None
        if self.execution_cap is not None:
            self._validate_scheduler()
            if self.execution_cap > self.resident_cap:
                raise ValueError("execution cap must not exceed configured resident cap")
            # Native remote-KV loads can allocate without joining running, so
            # their admission needs a separate budget not implemented here.
            if getattr(scheduler, "connector", None) is not None:
                raise ValueError("split execution cap does not support native remote KV connectors")
            self.emit("small_hold_capacity_configured", stage_id=self.stage_id,
                      execution_cap=self.execution_cap, resident_cap=self.resident_cap,
                      hold_enabled=self.enabled, mode="execution_resident_split")
        if self.urgent_keep_mode and self.stage_id in stages:
            if not self.enabled or not self.buffer_policy_mode:
                raise ValueError("urgent keep requires enabled stage buffer holds")
            if self.execution_cap is None or self.execution_cap >= self.resident_cap:
                raise ValueError("urgent keep requires execution K < resident R")
            if getattr(getattr(scheduler, "_baseline_policy", None), "value", None) != "stage_deadline_edf_np":
                raise ValueError("urgent keep requires stage_deadline_edf_np waiting order")
            if environment.get(CONTROLLER_ENV) != "stage_buffer_controller_budget:decide":
                raise ValueError("urgent keep requires the budget controller")
        if not self.enabled:
            return
        self._validate_scheduler()
        if (
            getattr(scheduler, "_slo_active_preemption_enabled", False)
            or environment.get("VLLM_OMNI_SLO_ACTIVE_PREEMPTION", "0").lower()
            in {"1", "true"}
            or environment.get("VLLM_OMNI_CONTROLLED_PREEMPTION_FILE")
            or getattr(scheduler, "_controlled_preemption_rules", ())
        ):
            raise ValueError("small hold cannot coexist with active/controlled preemption")
        path = environment.get(CONTROL_FILE_ENV)
        controller_path = environment.get(CONTROLLER_ENV)
        if path and controller_path:
            raise ValueError("small hold controller and forced control file are mutually exclusive")
        if controller_path:
            module_name, separator, function_name = controller_path.partition(":")
            if not separator or not module_name or not function_name:
                raise ValueError("small hold controller requires package.module:function")
            self.controller = getattr(importlib.import_module(module_name), function_name)
            if not callable(self.controller):
                raise ValueError("small hold controller must be callable")
        if path:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            if document.get("schema_version") != 1:
                raise ValueError("small hold control file requires schema_version=1")
            rule = document.get("forced_hold")
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("request_id"), str)
                or not rule["request_id"]
            ):
                raise ValueError("forced_hold requires a non-empty request_id")
            rule = dict(rule)
            rule["duration_s"] = _duration(rule["duration_s"])
            minimum = rule.get("min_output_tokens", 1)
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError("forced_hold min_output_tokens must be a positive integer")
            rule["min_output_tokens"] = minimum
            rule.setdefault("reason", "forced_correctness")
            rule.setdefault("evidence", {})
            self._validate_evidence(rule["reason"], rule["evidence"])
            self.forced_hold = rule

    def _validate_scheduler(self) -> None:
        scheduler = self.scheduler
        # In particular, reject OmniARAsyncScheduler's extra in-flight state.
        if type(scheduler).__name__ != "OmniARScheduler" or getattr(
            scheduler.scheduler_config, "async_scheduling", False
        ):
            raise ValueError("small hold requires synchronous OmniARScheduler")
        if getattr(scheduler, "num_spec_tokens", 0) or getattr(scheduler, "use_pp", False):
            raise ValueError("small hold requires no speculative decoding or pipeline parallelism")
        if not 1 <= scheduler.max_num_running_reqs <= MAX_RESIDENT_REQUESTS:
            raise ValueError("small hold requires resident cap <=64")

    @staticmethod
    def _validate_evidence(reason: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("hold reason must be a non-empty string")
        if not isinstance(evidence, Mapping):
            raise ValueError("hold evidence must be a JSON object")
        # Detach caller-owned state and reject non-JSON/NaN evidence early.
        return json.loads(json.dumps(dict(evidence), allow_nan=False))

    def _resident_fields(self) -> dict[str, int]:
        scheduler = self.scheduler
        running_ids = {request.request_id for request in scheduler.running}
        adapter = getattr(scheduler, "chunk_transfer_adapter", None)
        blocked_ids = {
            request.request_id
            for request in getattr(adapter, "waiting_for_chunk_running_requests", ())
        }
        fields = {
            "resident_count": len(running_ids | blocked_ids),
            "input_blocked_resident_count": len(blocked_ids - running_ids),
            "resident_cap": int(scheduler.max_num_running_reqs),
        }
        if self.execution_cap is not None:
            fields.update(
                resident_count=len(self._split_resident_ids()),
                resident_cap=self.resident_cap,
                execution_cap=self.execution_cap,
                execution_slot_owners=len(self._execution_active_ids or ()),
                resume_pending_count=sum(
                    hold.released_monotonic_s is not None for hold in self.holds.values()
                ),
            )
        return fields

    def _split_resident_ids(self) -> set[str]:
        """Include input waits and protected KV; inspect requests, not blocks."""
        scheduler = self.scheduler
        ids = {request.request_id for request in scheduler.running}
        adapter = getattr(scheduler, "chunk_transfer_adapter", None)
        ids.update(request.request_id for request in getattr(
            adapter, "waiting_for_chunk_running_requests", ()
        ))
        coordinator = getattr(scheduler.kv_cache_manager, "coordinator", None)
        for manager in getattr(coordinator, "single_type_managers", ()):
            ids.update(rid for rid, blocks in manager.req_to_blocks.items() if blocks)
        return ids

    @staticmethod
    def _at_decode_boundary(request: Any) -> bool:
        return (
            _running(request)
            and request.num_computed_tokens >= request.num_prompt_tokens
            and request.num_output_tokens > 0
            and request.num_output_placeholders == 0
            and not request.spec_token_ids
            and request.num_tokens - request.num_computed_tokens == 1
        )

    def _event(self, event: str, request_id: str, hold: _Hold, **fields: Any) -> None:
        request = hold.request
        blocks = self.scheduler.kv_cache_manager.get_block_ids(request_id)
        self.emit(
            event,
            stage_id=self.stage_id,
            request_id=request_id,
            requested_monotonic_s=hold.requested_monotonic_s,
            deadline_monotonic_s=hold.deadline_monotonic_s,
            released_monotonic_s=hold.released_monotonic_s,
            held_schedule_steps=hold.held_schedule_steps,
            reason=hold.reason,
            evidence=hold.evidence,
            num_computed_tokens=request.num_computed_tokens,
            num_output_tokens=request.num_output_tokens,
            num_output_placeholders=request.num_output_placeholders,
            kv_block_ids=blocks,
            **self._resident_fields(),
            **fields,
        )

    def request_hold(
        self, request_id: str, duration_s: float, reason: str, evidence: Mapping[str, Any]
    ) -> bool:
        if not self.enabled:
            return False
        duration = _duration(duration_s)
        evidence_copy = self._validate_evidence(reason, evidence)
        request = self.scheduler.requests.get(request_id)
        peers = [item for item in self.scheduler.running
                 if item is not request and item.request_id not in self.holds]
        if (
            request is None
            or (self.buffer_policy_mode and self.buffer_policy_suspended)
            or (self.buffer_policy_mode and not self.urgent_keep_mode and (
                not peers or any(not self._at_decode_boundary(item) for item in peers)
            ))
            or request_id in self.holds
            or ((not self.buffer_policy_mode or self.urgent_keep_mode)
                and request_id in self.accepted_request_ids)
            or not any(item is request for item in self.scheduler.running)
            or not self._at_decode_boundary(request)
            or self._resident_fields()["resident_count"] > self.scheduler.max_num_running_reqs
            or (self.execution_cap is not None
                and self._execution_active_ids is not None
                and request_id not in self._execution_active_ids)
        ):
            return False
        now = self.clock()
        hold = _Hold(request, now, now + duration, reason, evidence_copy)
        self.holds[request_id] = hold
        self.accepted_request_ids.add(request_id)
        if self.urgent_keep_mode and reason == "urgent_inversion":
            self._urgent_pending[request_id] = str(evidence_copy["urgent_request_id"])
        self._event("small_hold_requested", request_id, hold)
        return True

    def release_hold(self, request_id: str, reason: str) -> bool:
        hold = self.holds.get(request_id)
        if hold is None or hold.released_monotonic_s is not None:
            return False
        hold.deadline_monotonic_s = min(hold.deadline_monotonic_s, self.clock())
        self._event("small_hold_resume_requested", request_id, hold, resume_reason=reason)
        return True

    def maybe_force_hold(self) -> None:
        if not self.enabled or self.forced_hold is None or self.forced_hold_fired:
            return
        rule = self.forced_hold
        for request in self.scheduler.running:
            if rule["request_id"] not in ("*", request.request_id):
                continue
            if request.num_output_tokens < rule["min_output_tokens"]:
                continue
            if self.request_hold(
                request.request_id, rule["duration_s"], rule["reason"], rule["evidence"]
            ):
                self.forced_hold_fired = True
                return

    def before_schedule(self) -> None:
        if not self.enabled:
            return
        self.maybe_force_hold()
        if self.controller is not None:
            self.controller(self.scheduler, self.clock())

    @contextmanager
    def scheduling_context(self) -> Iterator[None]:
        if self.execution_cap is not None:
            with self._execution_limited_context():
                yield
            return
        if not self.enabled or not self.holds:
            yield
            return
        scheduler = self.scheduler
        now = self.clock()
        if self.buffer_policy_mode:
            # A resident peer can receive another input chunk after a hold was
            # accepted. Do not keep the victim held while that peer prefills.
            peers = [request for request in scheduler.running
                     if request.request_id not in self.holds]
            if not peers or any(not self._at_decode_boundary(request) for request in peers):
                for request_id, hold in list(self.holds.items()):
                    if hold.released_monotonic_s is None:
                        self.release_hold(request_id, "resident_left_decode")
                now = self.clock()
        held_ids: set[str] = set()
        for request_id, hold in list(self.holds.items()):
            if scheduler.requests.get(request_id) is not hold.request or not _running(hold.request):
                # Cancelled/finished requests must never be restored.
                self._event("small_hold_cancelled", request_id, hold)
                self.holds.pop(request_id)
                continue
            if hold.released_monotonic_s is not None:
                continue
            if now >= hold.deadline_monotonic_s:
                hold.released_monotonic_s = now
                lateness = max(0.0, now - hold.deadline_monotonic_s)
                self._event("small_hold_released", request_id, hold,
                            deadline_lateness_s=lateness)
                if (self.buffer_policy_mode and lateness > MAX_DEADLINE_LATENESS_S
                        and not self.buffer_policy_suspended):
                    # Synchronous execution cannot interrupt an in-flight GPU
                    # batch. Stop issuing new holds after a large miss.
                    self.buffer_policy_suspended = True
                    self.emit("stage_buffer_policy_suspended", stage_id=self.stage_id,
                              request_id=request_id, deadline_lateness_s=lateness,
                              reason="hold_deadline_overrun")
            else:
                held_ids.add(request_id)
        original_running = list(scheduler.running)
        held = [request for request in original_running if request.request_id in held_ids]
        if not held:
            yield
            return
        original_cap = scheduler.max_num_running_reqs
        resident = self._resident_fields()
        if resident["resident_count"] > min(original_cap, MAX_RESIDENT_REQUESTS):
            raise RuntimeError("small hold resident count exceeds configured cap")
        # Reserve both held slots and slots hidden by the chunk adapter.
        # Otherwise admission could exceed the original resident cap.
        scheduler.max_num_running_reqs = (
            original_cap - len(held) - resident["input_blocked_resident_count"]
        )
        scheduler.running[:] = [request for request in original_running if request.request_id not in held_ids]
        if self.buffer_policy_mode:
            # Legacy buffer holds still block new waiting prefills.
            scheduler.max_num_running_reqs = min(
                scheduler.max_num_running_reqs, len(scheduler.running))
        for request in held:
            self.holds[request.request_id].held_schedule_steps += 1
        try:
            yield
        finally:
            scheduler.max_num_running_reqs = original_cap
            current = {request.request_id: request for request in scheduler.running}
            for request in held:
                if scheduler.requests.get(request.request_id) is request and _running(request):
                    current[request.request_id] = request
            restored = []
            for request in original_running:
                if current.get(request.request_id) is request:
                    restored.append(current.pop(request.request_id))
            restored.extend(current.values())
            scheduler.running[:] = restored

    def _refresh_split_holds(self, now: float) -> None:
        for victim_id, urgent_id in list(self._urgent_pending.items()):
            urgent = self.scheduler.requests.get(urgent_id)
            if urgent is None or getattr(getattr(urgent, "status", None), "name", "") not in {"WAITING", "RUNNING"}:
                self.release_hold(victim_id, "urgent_request_retired")
                self._urgent_pending.pop(victim_id, None)
                self._urgent_pending_logged.discard(victim_id)
        for request_id, hold in list(self.holds.items()):
            status = getattr(getattr(hold.request, "status", None), "name", "")
            if (self.scheduler.requests.get(request_id) is not hold.request
                    or status not in {"RUNNING", "WAITING_FOR_CHUNK"}):
                self._event("small_hold_cancelled", request_id, hold,
                            cancel_reason="request_retired_or_preempted")
                self.holds.pop(request_id)
                self._urgent_pending.pop(request_id, None)
                self._urgent_pending_logged.discard(request_id)
            elif hold.released_monotonic_s is None and now >= hold.deadline_monotonic_s:
                hold.released_monotonic_s = now
                lateness = max(0.0, now - hold.deadline_monotonic_s)
                self._event("small_hold_released", request_id, hold,
                            deadline_lateness_s=lateness)
                if (self.buffer_policy_mode and lateness > MAX_DEADLINE_LATENESS_S
                        and not self.buffer_policy_suspended):
                    self.buffer_policy_suspended = True
                    self.emit("stage_buffer_policy_suspended", stage_id=self.stage_id,
                              request_id=request_id, deadline_lateness_s=lateness,
                              reason="hold_deadline_overrun")

    @contextmanager
    def _execution_limited_context(self) -> Iterator[None]:
        """Opt-in execution K / resident R split, also when holds are disabled.

        Omitted requests stay RUNNING and retain request/worker state. Existing
        execution owners continue first; released holds get a free slot before
        other returning residents or native waiting requests. Native allocation
        and reactive preemption remain responsible for memory safety.
        """
        scheduler = self.scheduler
        original_cap = scheduler.max_num_running_reqs
        if original_cap != self.resident_cap:
            raise RuntimeError("split execution/resident caps require a fixed resident cap")
        self._refresh_split_holds(self.clock())
        original_running = list(scheduler.running)
        if len({request.request_id for request in original_running}) != len(original_running):
            raise RuntimeError("duplicate resident request in running queue")
        resident_count = len(self._split_resident_ids())
        if resident_count > self.resident_cap:
            raise RuntimeError("split execution cap resident count exceeds configured cap")
        active_ids = self._execution_active_ids
        continuing = [request for request in original_running
                      if _running(request) and request.request_id not in self.holds
                      and (active_ids is None or request.request_id in active_ids)]
        if active_ids is not None and len(continuing) > self.execution_cap:
            raise RuntimeError("execution-slot owner count exceeds execution cap")
        selected = continuing[:self.execution_cap]
        ready_holds = sorted(
            (hold for hold in self.holds.values()
             if hold.released_monotonic_s is not None and _running(hold.request)
             and any(request is hold.request for request in original_running)),
            key=lambda hold: (hold.released_monotonic_s, hold.requested_monotonic_s),
        )
        selected_ids = {request.request_id for request in selected}
        candidates = [hold.request for hold in ready_holds]
        candidates.extend(request for request in original_running
                          if _running(request) and request.request_id not in self.holds)
        for request in candidates:
            if len(selected) == self.execution_cap:
                break
            if request.request_id not in selected_ids:
                selected.append(request)
                selected_ids.add(request.request_id)
        for request in original_running:
            hold = self.holds.get(request.request_id)
            if hold is not None and request.request_id not in selected_ids:
                hold.held_schedule_steps += 1
        # A new request needs both an execution slot and a resident slot.
        free_resident = self.resident_cap - resident_count
        if self.urgent_keep_mode and self.holds:
            # Exactly one eligible urgent waiter may use a newly opened slot.
            free_resident = min(free_resident, 1 if self._urgent_pending else 0)
        scheduler.max_num_running_reqs = min(
            self.execution_cap, len(selected) + free_resident
        )
        scheduler.running[:] = selected
        self._last_held_ids = {
            request.request_id for request in original_running
            if request.request_id in self.holds and request.request_id not in selected_ids
        }
        try:
            yield
        finally:
            visible = list(scheduler.running)
            self._execution_active_ids = {
                request.request_id for request in visible
                if scheduler.requests.get(request.request_id) is request and _running(request)
            }
            scheduler.max_num_running_reqs = original_cap
            current = {request.request_id: request for request in visible
                       if scheduler.requests.get(request.request_id) is request and _running(request)}
            for request in original_running:
                if scheduler.requests.get(request.request_id) is request and _running(request):
                    current[request.request_id] = request
            restored = []
            for request in original_running:
                if current.get(request.request_id) is request:
                    restored.append(current.pop(request.request_id))
            restored.extend(current.values())
            scheduler.running[:] = restored
            self._refresh_split_holds(self.clock())
            if len(self._execution_active_ids) > self.execution_cap:
                raise RuntimeError("native scheduler exceeded execution cap")
            if len(self._split_resident_ids()) > self.resident_cap:
                raise RuntimeError("native scheduler exceeded resident cap")

    def record_scheduled(self, scheduler_output: Any) -> None:
        if self.execution_cap is not None and len(scheduler_output.num_scheduled_tokens) > self.execution_cap:
            raise RuntimeError("scheduled batch exceeds execution cap")
        if not self.enabled:
            return
        now = self.clock()
        scheduled_ids = set(scheduler_output.num_scheduled_tokens)
        for victim_id, urgent_id in list(self._urgent_pending.items()):
            if urgent_id in scheduled_ids and victim_id in self._last_held_ids:
                self.emit("urgent_keep_admitted", stage_id=self.stage_id,
                          urgent_request_id=urgent_id, victim_request_id=victim_id,
                          victim_kv_block_ids=self.scheduler.kv_cache_manager.get_block_ids(victim_id),
                          urgent_kv_block_ids=self.scheduler.kv_cache_manager.get_block_ids(urgent_id),
                          num_scheduled_tokens=scheduler_output.num_scheduled_tokens[urgent_id],
                          **self._resident_fields())
                self._urgent_pending.pop(victim_id, None)
                self._urgent_pending_logged.discard(victim_id)
            elif victim_id not in self.holds or victim_id in scheduled_ids:
                self._urgent_pending.pop(victim_id, None)
                self._urgent_pending_logged.discard(victim_id)
            elif victim_id not in self._urgent_pending_logged:
                self.emit("urgent_keep_admission_pending", stage_id=self.stage_id,
                          urgent_request_id=urgent_id, victim_request_id=victim_id,
                          reason="not_scheduled_this_round")
                self._urgent_pending_logged.add(victim_id)
        for request_id in scheduler_output.num_scheduled_tokens:
            hold = self.holds.get(request_id)
            if hold is None:
                continue
            if hold.released_monotonic_s is None:
                raise RuntimeError("request scheduled before its small hold was released")
            self._event(
                "small_hold_first_rescheduled",
                request_id,
                hold,
                first_rescheduled_monotonic_s=now,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens[request_id],
            )
            self.holds.pop(request_id)
            self._urgent_pending.pop(request_id, None)
            self._urgent_pending_logged.discard(request_id)
