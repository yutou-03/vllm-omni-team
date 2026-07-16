"""Shared policy definitions and deterministic keys for baseline experiments."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.scheduling.metadata import extract_scheduling_metadata

BASELINE_POLICY_ENV = "VLLM_OMNI_BASELINE_POLICY"


class BaselineSchedulingPolicy(str, Enum):
    """Scheduling policies compared by the multi-stage baseline study."""

    NATIVE_FCFS = "native_fcfs"
    SRPF_LOCAL_NP = "srpf_local_np"
    FINAL_DEADLINE_EDF_NP = "final_deadline_edf_np"
    S2_ONLY_EDF_NP = "s2_only_edf_np"


class SchedulingMetadataError(ValueError):
    """Raised when an enabled baseline policy lacks required metadata."""


PolicyKey = tuple[float | int | str, ...]


def get_baseline_scheduling_policy(
    environ: Mapping[str, str] | None = None,
) -> BaselineSchedulingPolicy:
    """Read and validate the process-wide baseline policy selection."""

    source = os.environ if environ is None else environ
    raw_policy = source.get(
        BASELINE_POLICY_ENV,
        BaselineSchedulingPolicy.NATIVE_FCFS.value,
    )
    normalized = raw_policy.strip().lower()
    try:
        return BaselineSchedulingPolicy(normalized)
    except ValueError as error:
        supported = [policy.value for policy in BaselineSchedulingPolicy]
        raise ValueError(
            f"unknown {BASELINE_POLICY_ENV}={raw_policy!r}; "
            f"expected one of {supported!r}"
        ) from error


def policy_applies_to_stage(
    policy: BaselineSchedulingPolicy,
    stage_id: int,
) -> bool:
    """Return whether ``policy`` is allowed to reorder the given stage."""

    if stage_id < 0:
        raise ValueError(f"stage_id must be non-negative, got {stage_id}")
    if policy is BaselineSchedulingPolicy.NATIVE_FCFS:
        return False
    if policy is BaselineSchedulingPolicy.S2_ONLY_EDF_NP:
        return stage_id == 2
    return True


def request_scheduling_metadata(request: Any) -> dict[str, Any]:
    """Decode canonical scheduling metadata from a scheduler request."""

    raw_payload = getattr(request, "additional_information", None)
    decoded = deserialize_additional_information(raw_payload)
    metadata = extract_scheduling_metadata(decoded)
    if not metadata:
        request_id = getattr(request, "request_id", "<unknown>")
        raise SchedulingMetadataError(
            f"request {request_id!r} has no scheduling metadata"
        )
    return metadata


def predict_remaining_ms(
    *,
    total_predicted_ms: float,
    total_work_units: float,
    completed_work_units: float,
) -> float:
    """Scale a frozen total-time prediction by monotonic remaining work."""

    if not math.isfinite(total_predicted_ms) or total_predicted_ms < 0:
        raise ValueError("total_predicted_ms must be non-negative")
    if not math.isfinite(total_work_units) or total_work_units <= 0:
        raise ValueError("total_work_units must be positive")
    if not math.isfinite(completed_work_units):
        raise ValueError("completed_work_units must be finite")
    completed = min(max(float(completed_work_units), 0.0), total_work_units)
    return total_predicted_ms * (total_work_units - completed) / total_work_units


def _required_field(
    metadata: Mapping[str, Any],
    field: str,
    *,
    request_id: str,
) -> Any:
    value = metadata.get(field)
    if value is None:
        raise SchedulingMetadataError(
            f"request {request_id!r} is missing required field {field!r}"
        )
    return value


def _required_stage_value(
    metadata: Mapping[str, Any],
    field: str,
    *,
    stage_id: int,
    request_id: str,
) -> float:
    values = _required_field(metadata, field, request_id=request_id)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SchedulingMetadataError(
            f"request {request_id!r} field {field!r} must be a stage sequence"
        )
    if stage_id >= len(values) or values[stage_id] is None:
        raise SchedulingMetadataError(
            f"request {request_id!r} field {field!r} has no value for stage {stage_id}"
        )
    return float(values[stage_id])


def policy_key(
    request: Any,
    *,
    policy: BaselineSchedulingPolicy,
    stage_id: int,
    data_ready_time: float,
    now: float,
) -> PolicyKey:
    """Compute a deterministic total-order key for one eligible request.

    ``native_fcfs`` and stages excluded by the stage-2-only EDF ablation must
    never call this function. Keeping that path explicit prevents accidental
    replacement of vLLM's native queue.
    """

    del now  # Reserved for future age-aware policies; keys remain time-stable.
    if not policy_applies_to_stage(policy, stage_id):
        raise ValueError(
            f"policy {policy.value!r} does not reorder stage {stage_id}"
        )

    request_id = str(getattr(request, "request_id", "<unknown>"))
    metadata = request_scheduling_metadata(request)
    source_request_id = str(
        _required_field(
            metadata,
            "sched_source_request_id",
            request_id=request_id,
        )
    )

    if policy in (
        BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
        BaselineSchedulingPolicy.S2_ONLY_EDF_NP,
    ):
        deadline = float(
            _required_field(
                metadata,
                "sched_deadline_monotonic_s",
                request_id=request_id,
            )
        )
        if not math.isfinite(deadline):
            raise SchedulingMetadataError(
                f"request {request_id!r} has a non-finite deadline"
            )
        ingress_order = int(
            _required_field(
                metadata,
                "sched_ingress_order",
                request_id=request_id,
            )
        )
        return deadline, ingress_order, source_request_id

    if policy is BaselineSchedulingPolicy.SRPF_LOCAL_NP:
        total_ms = _required_stage_value(
            metadata,
            "sched_predicted_stage_ms",
            stage_id=stage_id,
            request_id=request_id,
        )
        total_work = _required_stage_value(
            metadata,
            "sched_predicted_stage_work_units",
            stage_id=stage_id,
            request_id=request_id,
        )
        completed_work = float(getattr(request, "num_computed_tokens", 0))
        remaining_ms = predict_remaining_ms(
            total_predicted_ms=total_ms,
            total_work_units=total_work,
            completed_work_units=completed_work,
        )
        ready_time = float(data_ready_time)
        if not math.isfinite(ready_time):
            raise ValueError("data_ready_time must be finite")
        return remaining_ms, ready_time, source_request_id

    raise AssertionError(f"unhandled baseline policy {policy!r}")
