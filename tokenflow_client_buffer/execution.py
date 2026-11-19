"""Apply a scheduling decision without changing request identity or KV ownership.

The native scheduler has no compute-only pause API. All compatibility queue
changes therefore live here, with exception-safe restoration; the policy only
returns a decision. Native memory preemption remains owned by the engine.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

from .engine_queue import engine_queue_admission_context, _replace_waiting


@dataclass(frozen=True)
class ExecutionDecision:
    running: tuple = ()
    blocked_waiting: frozenset[str] = frozenset()
    waiting_priority: tuple = ()
    hide_all_waiting: bool = False
    max_running: int | None = None
    native_passthrough: bool = False
    sort_restored: bool = True


@dataclass
class ExecutionResult:
    active_ids: set[str] = field(default_factory=set)


@contextmanager
def apply_execution_decision(scheduler, decision, source_id, order):
    result = ExecutionResult()
    if decision.native_passthrough:
        yield result
        return
    original = list(scheduler.running)
    original_cap = scheduler.max_num_running_reqs
    scheduler.running[:] = decision.running
    if decision.max_running is not None:
        scheduler.max_num_running_reqs = decision.max_running
    try:
        with engine_queue_admission_context(
            scheduler, {}, source_id,
            blocked_waiting_ids=decision.blocked_waiting,
            hide_all_waiting=decision.hide_all_waiting,
        ):
            priority = {r.request_id: i for i, r in enumerate(decision.waiting_priority)}
            if priority:
                seen = set()
                for queue in (scheduler.waiting, getattr(scheduler, "skipped_waiting", ())):
                    if queue and id(queue) not in seen:
                        seen.add(id(queue))
                        _replace_waiting(queue, sorted(queue, key=lambda r:
                            priority.get(r.request_id, len(priority))))
            yield result
    finally:
        visible = list(scheduler.running)
        result.active_ids = {r.request_id for r in visible}
        scheduler.max_num_running_reqs = original_cap
        restored, seen = [], set()
        for request in original + visible:
            rid = request.request_id
            if (rid not in seen and scheduler.requests.get(rid) is request
                    and getattr(request.status, "name", "") == "RUNNING"):
                restored.append(request)
                seen.add(rid)
        scheduler.running[:] = sorted(restored, key=order) if decision.sort_restored else restored
