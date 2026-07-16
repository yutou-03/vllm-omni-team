from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

from vllm_omni.profiling.nvtx import nvtx_range

_DEFAULT_TRACE_DIR = os.path.join(
    tempfile.gettempdir(),
    "vllm_omni_stage_queue_trace",
)
_TRACE_DIR = os.environ.get("STAGE_QUEUE_TRACE_DIR", _DEFAULT_TRACE_DIR)
_RUN_ID = os.environ.get("STAGE_QUEUE_RUN_ID", "server")
_PID = os.getpid()


class ConformanceIneligibleReason(str, Enum):
    """Closed vocabulary for explaining why a request was not runnable."""

    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_CHUNK = "waiting_for_chunk"
    FINISHED = "finished"
    NONPREEMPTIVE_RUNNING_CAPACITY = "nonpreemptive_running_capacity"
    SEQUENCE_SLOT_LIMIT = "sequence_slot_limit"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    KV_ALLOCATION_FAILED = "kv_allocation_failed"
    ALREADY_SELECTED = "already_selected"
    UNKNOWN = "unknown"


def _enabled() -> bool:
    return os.environ.get("STAGE_QUEUE_TRACE_DISABLE", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def conformance_trace_enabled() -> bool:
    """Return whether replayable per-iteration policy details are requested."""

    return os.environ.get("VLLM_OMNI_CONFORMANCE_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_jsonl(filename: str, payload: dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        os.makedirs(_TRACE_DIR, exist_ok=True)
        path = os.path.join(_TRACE_DIR, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=_json_default, sort_keys=True))
            f.write("\n")
    except Exception:
        # Trace should never affect serving correctness.
        return


def emit_stage_event(
    event: str,
    *,
    stage_id: int | str | None,
    request_id: Any | None = None,
    **fields: Any,
) -> None:
    payload = {
        "run_id": _RUN_ID,
        "pid": _PID,
        "event": event,
        "stage_id": stage_id,
        "request_id": None if request_id is None else str(request_id),
        "timestamp": time.time(),
        "timestamp_ns": time.time_ns(),
        "monotonic": time.monotonic(),
    }
    payload.update(fields)
    _write_jsonl("stage_events.jsonl", payload)


def emit_stage_duration_event(
    event: str,
    *,
    stage_id: int | str | None,
    request_id: Any | None = None,
    timestamp_start: float,
    timestamp_end: float,
    monotonic_start: float,
    monotonic_end: float,
    **fields: Any,
) -> None:
    payload = {
        "run_id": _RUN_ID,
        "pid": _PID,
        "event": event,
        "stage_id": stage_id,
        "request_id": None if request_id is None else str(request_id),
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "duration_ms": (monotonic_end - monotonic_start) * 1000.0,
        "timestamp_ns": time.time_ns(),
        "monotonic_start": monotonic_start,
        "monotonic_end": monotonic_end,
    }
    payload.update(fields)
    _write_jsonl("stage_events.jsonl", payload)


@contextmanager
def stage_trace_span(
    event: str,
    *,
    stage_id: int | str | None,
    request_id: Any | None = None,
    nvtx_name: str | None = None,
    nvtx_color: str | None = None,
    **fields: Any,
) -> Iterator[None]:
    timestamp_start = time.time()
    monotonic_start = time.monotonic()
    with nvtx_range(nvtx_name or f"{event}:stage={stage_id}", color=nvtx_color):
        try:
            yield
        finally:
            monotonic_end = time.monotonic()
            emit_stage_duration_event(
                event,
                stage_id=stage_id,
                request_id=request_id,
                timestamp_start=timestamp_start,
                timestamp_end=time.time(),
                monotonic_start=monotonic_start,
                monotonic_end=monotonic_end,
                **fields,
            )


def emit_iteration_event(
    *,
    stage_id: int | str | None,
    iteration_id: int,
    timestamp_start: float,
    timestamp_end: float,
    **fields: Any,
) -> None:
    payload = {
        "run_id": _RUN_ID,
        "pid": _PID,
        "event": "iteration",
        "stage_id": stage_id,
        "iteration_id": iteration_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "duration_ms": (timestamp_end - timestamp_start) * 1000.0,
        "timestamp_ns": time.time_ns(),
        "monotonic": time.monotonic(),
    }
    payload.update(fields)
    _write_jsonl("iteration_events.jsonl", payload)


def emit_connector_event(
    event: str,
    *,
    from_stage: int | str | None,
    to_stage: int | str | None,
    request_id: Any | None = None,
    **fields: Any,
) -> None:
    payload = {
        "run_id": _RUN_ID,
        "pid": _PID,
        "event": event,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "request_id": None if request_id is None else str(request_id),
        "timestamp": time.time(),
        "timestamp_ns": time.time_ns(),
        "monotonic": time.monotonic(),
    }
    payload.update(fields)
    _write_jsonl("connector_events.jsonl", payload)
