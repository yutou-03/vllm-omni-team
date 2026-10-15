"""Read-only, opt-in observations; never enable a scheduling intervention.

All counts are metadata on the CPU. No tensors are copied or synchronized.
Consumption is the model-input selection boundary, not forward completion.
"""
from __future__ import annotations

import os
import time
from typing import Any
from vllm_omni.profiling.stage_queue_trace import emit_stage_event


def enabled() -> bool:
    return os.environ.get("VLLM_OMNI_STAGE_STUDY_TRACE") == "1"


def identity(info: Any, request_id: Any) -> dict[str, Any]:
    meta = info.get("meta", {}) if isinstance(info, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    stored = meta.get("stage_study_identity", {})
    if isinstance(stored, dict) and stored:
        return dict(stored)
    physical = str(request_id)
    return {
        "logical_request_id": str(meta.get("omni_logical_request_id", meta.get("sched_logical_request_id", physical))),
        "execution_id": str(meta.get("omni_execution_id", physical)),
        "pipeline_attempt_id": int(meta.get("omni_pipeline_attempt_id", 0)),
        "stage_attempt_id": int(meta.get("omni_stage_attempt_id", 0)),
    }


def emit(event: str, *, stage_id: int, request_id: Any = None, info: Any = None, **fields: Any) -> None:
    if not enabled():
        return
    try:
        payload = identity(info, request_id) if request_id is not None else {}
        payload.update(fields)
        emit_stage_event("stage_study_" + event, stage_id=stage_id, request_id=request_id,
                         study_schema_version=1, **payload)
    except Exception:
        # Instrumentation must not change request execution.
        return


def text_ready_fields(payload: Any) -> dict[str, Any]:
    """Legacy ID-derived readiness; not evidence that tensor rows are present.

    Keep this field for trace compatibility. Independent entity observations
    come from text_entity_fields and must never be backfilled into old traces.
    """
    if not isinstance(payload, dict):
        return {}
    ids = payload.get("ids", {})
    embed = payload.get("embed", {})
    if not isinstance(ids, dict) or not isinstance(embed, dict):
        return {}
    output_ids = ids.get("output")
    if output_ids is None:
        all_ids, prompt_ids = ids.get("all"), ids.get("prompt")
        if all_ids is None or prompt_ids is None:
            return {}
        total = max(len(all_ids) - len(prompt_ids), 0)
    else:
        total = len(output_ids)
    return {"generated_output_tokens_total": total,
            "conditioning_ready_index_exclusive": max(total - 1, 0),
            "conditioning_ready_source": "ids_count_minus_one",
            "conditioning_ready_entity_verified": False,
            "unit": "thinker_output_embedding_position"}



def text_entity_fields(payload: Any, *, prefix: str = "") -> dict[str, Any]:
    """Read only tensor shape metadata; never inspect/copy/synchronize data.

    Expected sequence tensors are [rows, hidden]. Missing tensors retain rows
    None plus present=False; malformed layouts retain rows=None, valid=False.
    The caller must identify raw packet versus merged/model state with prefix.
    Only raw packet rows can be accumulated, once per execution/chunk/boundary.
    cached_decode is model state and is deliberately excluded from packet rows.
    """
    valid_payload = isinstance(payload, dict)
    payload = payload if valid_payload else {}
    embed = payload.get("embed", {})
    hidden = payload.get("hidden_states", {})
    ids = payload.get("ids", {})
    valid_containers = all(isinstance(value, dict) for value in (embed, hidden, ids))
    fields: dict[str, Any] = {
        "entity_count_schema_version": 1,
        "entity_observation_basis": "tensor_shape_metadata_only",
        "entity_payload_valid": valid_payload and valid_containers,
    }
    embedding_rows = []
    embedding_present = False
    embedding_valid = valid_payload and valid_containers
    for name, container, key in (
        ("embedding_prefill", embed, "prefill"),
        ("embedding_decode", embed, "decode"),
        ("embedding_cached_decode", embed, "cached_decode"),
        ("hidden_output", hidden, "output"),
    ):
        value = container.get(key) if isinstance(container, dict) else None
        present = value is not None
        shape = None
        layout_valid = None
        rows = None
        if present:
            layout_valid = False
            try:
                raw_shape = value.shape
                if isinstance(raw_shape, (tuple, list)) and all(type(dim) is int and dim >= 0 for dim in raw_shape):
                    shape = list(raw_shape)
                    layout_valid = len(shape) == 2 and shape[1] > 0
                    if layout_valid:
                        rows = shape[0]
            except Exception:
                pass
        fields.update({name + "_present": present, name + "_shape": shape,
                       name + "_layout_valid": layout_valid, name + "_rows": rows})
        if name in {"embedding_prefill", "embedding_decode"} and present:
            embedding_present = True
            embedding_valid = embedding_valid and bool(layout_valid)
            if rows is not None:
                embedding_rows.append(rows)
    for key in ("prompt", "all", "output"):
        value = ids.get(key) if isinstance(ids, dict) else None
        try:
            fields["ids_" + key + "_count"] = len(value) if value is not None else None
        except (TypeError, ValueError):
            fields["ids_" + key + "_count"] = None
    fields["entity_embedding_layout_valid"] = embedding_valid if embedding_present else None
    fields["entity_embedding_rows_total"] = sum(embedding_rows) if embedding_present and embedding_valid else None
    prefill_rows, prompt_rows = fields["embedding_prefill_rows"], fields["ids_prompt_count"]
    fields["entity_prefill_minus_prompt_rows_signed"] = (
        prefill_rows - prompt_rows if prefill_rows is not None and prompt_rows is not None else None
    )
    return {prefix + key: value for key, value in fields.items()}


def text_consumed(payload: Any, *, start_index: int, count: int, kind: str, phase: str,
                  **entity_details: Any) -> None:
    if not enabled() or not isinstance(payload, dict):
        return
    request_id = payload.get("request_id")
    if request_id is None:
        return
    emit("text_conditioning_selected", stage_id=1, request_id=request_id, info=payload,
         start_index=int(start_index), end_index=int(start_index) + int(count),
         selected_output_embeddings=int(count), conditioning_kind=kind, phase=phase,
         observation_point="model_input_prepared", **text_ready_fields(payload),
         **text_entity_fields(payload, prefix="model_"), **entity_details)


def kv_counts(scheduler: Any) -> dict[str, Any]:
    try:
        manager = getattr(scheduler, "kv_cache_manager", None)
        pool = getattr(manager, "block_pool", None)
        config = getattr(scheduler, "kv_cache_config", None)
        total = getattr(pool, "num_gpu_blocks", getattr(config, "num_blocks", None))
        get_free = getattr(pool, "get_num_free_blocks", None)
        free = get_free() if callable(get_free) else None
        groups = getattr(config, "kv_cache_groups", ())
        sizes = [getattr(getattr(group, "kv_cache_spec", None), "block_size", None) for group in groups]
        return {"kv_total_blocks": total, "kv_free_blocks": free,
                "kv_used_blocks": (total - free) if total is not None and free is not None else None,
                "kv_group_block_sizes": sizes,
                "kv_cache_usage": getattr(manager, "usage", None)}
    except Exception:
        return {"kv_observation_error": True}


def request_progress(scheduler: Any, request: Any) -> dict[str, Any]:
    try:
        request_id = str(request.request_id)
        manager = getattr(scheduler, "kv_cache_manager", None)
        get_blocks = getattr(manager, "get_block_ids", None)
        block_ids = get_blocks(request_id) if callable(get_blocks) else ()
        if block_ids and isinstance(block_ids[0], int):
            block_ids = (block_ids,)
        null_block = getattr(getattr(manager, "block_pool", None), "null_block", None)
        null_id = getattr(null_block, "block_id", None)
        return {"request_id": request_id,
                "num_prompt_tokens": int(request.num_prompt_tokens),
                "num_computed_tokens": int(request.num_computed_tokens),
                "num_output_tokens": len(getattr(request, "output_token_ids", ())),
                "num_output_placeholders": int(getattr(request, "num_output_placeholders", 0)),
                "num_preemptions": int(getattr(request, "num_preemptions", 0)),
                "block_table_slots_per_group": [len(ids) for ids in block_ids],
                "allocated_blocks_per_group": [len({block for block in ids if block != null_id}) for ids in block_ids],
                "null_block_id": null_id,
                "status": str(getattr(getattr(request, "status", None), "name", "unknown"))}
    except Exception:
        return {"request_id": str(getattr(request, "request_id", "unknown")), "observation_error": True}


def gpu_memory(runner: Any, torch_module: Any, *, point: str, force: bool = False, encoder_request_ids: Any = None) -> None:
    """Total process allocator counters; not isolated encoder workspace or KV."""
    if not enabled():
        return
    try:
        now = time.monotonic()
        if not force and now - getattr(runner, "_study_memory_time", -1.0) < 0.05:
            return
        runner._study_memory_time = now
        cuda = torch_module.cuda
        device = runner.device
        emit("gpu_memory", stage_id=int(runner.model_config.stage_id),
             observation_point=point, device=str(device),
             torch_allocated_bytes=cuda.memory_allocated(device),
             torch_reserved_bytes=cuda.memory_reserved(device),
             torch_lifetime_peak_allocated_bytes=cuda.max_memory_allocated(device),
             torch_lifetime_peak_reserved_bytes=cuda.max_memory_reserved(device),
             memory_scope="process_torch_allocator_not_isolated_encoder_workspace",
             encoder_request_ids=list(encoder_request_ids) if encoder_request_ids is not None else None,
             request_ids_scope="entire_input_batch",
             request_ids=list(getattr(getattr(runner, "input_batch", None), "req_ids", ())))
    except Exception:
        return
