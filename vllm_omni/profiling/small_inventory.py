"""Opt-in observations for the single-hold pilot; no scheduling decisions."""

from __future__ import annotations

import os
from typing import Any

from vllm_omni.profiling.stage_queue_trace import emit_stage_event
from vllm_omni.profiling import stage_study

_PREFIX = "small_hold_"
_FIELDS = (
    "chunk_id",
    "connector_request_id",
    "logical_request_id",
    "codec_total_frames",
    "codec_payload_frames",
    "left_context_frames",
    "new_frames",
    "raw_codec_shape",
    "stream_finished",
    "execution_id",
    "pipeline_attempt_id",
    "stage_attempt_id",
)


def small_inventory_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_SMALL_HOLD_PROBE") == "1" or stage_study.enabled()


def inventory_fields(info: Any) -> dict[str, Any]:
    """Read only explicitly tagged chunk metadata, never infer from tokens."""
    if not isinstance(info, dict):
        return {}
    meta = info.get("meta", {})
    if not isinstance(meta, dict) or _PREFIX + "chunk_id" not in meta:
        return {}
    return {
        name: meta[_PREFIX + name]
        for name in _FIELDS
        if _PREFIX + name in meta
    }


def emit_inventory_event(
    event: str,
    *,
    stage_id: int,
    request_id: str,
    fields: dict[str, Any],
    **observations: Any,
) -> None:
    if not small_inventory_enabled() or not fields:
        return
    if stage_study.enabled():
        event = event.replace("small_hold_", "stage_study_", 1)
    emit_stage_event(
        event,
        stage_id=stage_id,
        request_id=request_id,
        inventory_schema_version=1,
        **fields,
        **observations,
    )


def scheduled_inventory_fields(scheduler_output: Any) -> dict[str, dict[str, Any]]:
    """Snapshot the exact metadata shipped with this execution batch."""
    if not small_inventory_enabled() or scheduler_output is None:
        return {}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    infos = dict(getattr(cached, "additional_information", {}) or {})
    for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        infos[request.req_id] = getattr(request, "additional_information", None)
    return {
        str(request_id): fields
        for request_id in getattr(scheduler_output, "num_scheduled_tokens", {})
        if (fields := inventory_fields(infos.get(request_id)))
    }
