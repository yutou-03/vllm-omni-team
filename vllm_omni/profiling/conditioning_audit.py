"""Opt-in first-16-step value/RNG observations; unsuitable for timing claims.

Snapshots copy tensors to CPU and therefore synchronize GPU work. No tensor,
sampling state or model state is modified; both diagnostic arms must enable it.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from vllm_omni.profiling.stage_queue_trace import emit_stage_event

_ENABLED = os.environ.get("VLLM_OMNI_CONDITIONING_AUDIT") == "1"
_MINIMAL = os.environ.get("VLLM_OMNI_CONDITIONING_AUDIT_MODE") == "minimal"
_DEFERRED = _MINIMAL or os.environ.get("VLLM_OMNI_CONDITIONING_AUDIT_MODE") == "deferred"
_LIMIT = int(os.environ.get("VLLM_OMNI_CONDITIONING_AUDIT_STEPS", "16"))
if not 1 <= _LIMIT <= 512:
    raise ValueError("conditioning audit steps must be in 1..512")
_COUNTS: dict[tuple[str, str], int] = defaultdict(int)
_PENDING: dict[str, list] = defaultdict(list)


def _retain(value: Any) -> Any:
    """Own the snapshot on its current device; do not read CUDA values yet."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _retain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_retain(v) for v in value]
    return value


def _tick(phase: str, request_id: str) -> int | None:
    if not _ENABLED:
        return None
    key = (phase, request_id)
    index = _COUNTS[key]
    _COUNTS[key] += 1
    limit = _LIMIT - 1 if _DEFERRED and phase == "mtp" else _LIMIT
    return index if index < limit else None


def snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cpu = value.detach().to("cpu").contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        folder = Path(os.environ["STAGE_QUEUE_TRACE_DIR"]) / "conditioning_tensors"
        folder.mkdir(exist_ok=True)
        path = folder / (digest + ".bin")
        if not path.exists():
            path.write_bytes(raw)
        return {"tensor_sha256": digest, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                "saved_file": str(path), "numel": cpu.numel()}
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def emit(phase: str, request_id: str, index: int, **values: Any) -> None:
    if _DEFERRED:
        _PENDING[request_id].append((phase, index, _retain(values)))
        return
    emit_stage_event("conditioning_audit", stage_id=1, request_id=request_id,
                     phase=phase, step_index=index, values=snapshot(values),
                     diagnostic_gpu_synchronization=True)


def before_preprocess(stage_id, request_id, payload, input_ids, span_len):
    if str(stage_id) != "1":
        return None
    index = _tick("preprocess", request_id)
    if index is not None:
        emit("preprocess_input", request_id, index, span_len=span_len, input_ids=input_ids,
             payload={k: payload.get(k) for k in (("meta", "ids") if _MINIMAL else ("meta", "ids", "embed", "hidden_states"))})
    return index


def after_preprocess(request_id, index, input_ids, embeds, update):
    if index is not None:
        emit("preprocess_output", request_id, index, input_ids=input_ids,
             input_embeds=embeds, update=({k: update.get(k) for k in ("meta", "mtp_inputs")} if _MINIMAL else update))


def after_mtp(request_ids, input_ids, codes, embeds, sampling_kwargs, padded_size, graph_wrapped):
    if not _ENABLED:
        return
    for row, rid in enumerate(request_ids):
        index = _tick("mtp", rid)
        if index is not None:
            emit("mtp_output", rid, index, batch_request_ids=request_ids, row=row,
                 input_ids=input_ids[row], codes=codes[row], embeds=embeds[row],
                 sampling_kwargs=sampling_kwargs, padded_size=padded_size, graph_wrapped=graph_wrapped)


def before_sample(runner, logits):
    if not _ENABLED or str(getattr(runner.model_config, "stage_id", "")) != "1" or logits is None:
        return []
    selected = []
    metadata = runner.input_batch.sampling_metadata
    for row, rid in enumerate(runner.input_batch.req_ids):
        index = _tick("sample", rid)
        if index is None:
            continue
        generator = metadata.generators.get(row)
        request = runner.requests[rid]
        sp = getattr(request, "sampling_params", None)
        observed_logits = logits[row].detach()
        if _DEFERRED:
            top_ids, top_values = None, None
        else:
            observed_logits = observed_logits.to("cpu").contiguous()
            top = observed_logits.float().topk(min(10, observed_logits.numel()))
            top_ids, top_values = top.indices.tolist(), top.values.tolist()
        emit("sample_input", rid, index, batch_request_ids=list(runner.input_batch.req_ids), row=row,
             sampling={k: getattr(sp, k, None) for k in
                       ("seed", "temperature", "top_k", "top_p", "repetition_penalty", "ignore_eos")},
             generator_present=generator is not None,
             rng_state=None if _MINIMAL else (generator.get_state() if generator is not None else torch.cuda.get_rng_state(logits.device)),
             logits=observed_logits, top_logit_ids=top_ids, top_logit_values=top_values,
             output_history_length=len(getattr(request, "output_token_ids", ())))
        selected.append((row, rid, index))
    return selected


def after_sample(selected, sampled_token_ids):
    for row, rid, index in selected:
        emit("sample_output", rid, index, sampled_token_ids=sampled_token_ids[row])
        if _DEFERRED and index == _LIMIT - 1:
            for phase, step, values in _PENDING.pop(rid, []):
                emit_stage_event("conditioning_audit", stage_id=1, request_id=rid,
                                 phase=phase, step_index=step, values=snapshot(values),
                                 diagnostic_gpu_synchronization=True, deferred_until_sample_step=index)
