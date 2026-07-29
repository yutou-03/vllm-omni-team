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
ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS_ENV = (
    "VLLM_OMNI_ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS"
)
ACTIVE_PREEMPTION_MAX_PER_REQUEST_ENV = (
    "VLLM_OMNI_ACTIVE_PREEMPTION_MAX_PER_REQUEST"
)
ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS_ENV = (
    "VLLM_OMNI_ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS"
)


class BaselineSchedulingPolicy(str, Enum):
    """Scheduling policies compared by the multi-stage baseline study."""

    NATIVE_FCFS = "native_fcfs"
    SRPF_LOCAL_NP = "srpf_local_np"
    FINAL_DEADLINE_EDF_NP = "final_deadline_edf_np"
    STAGE_DEADLINE_EDF_NP = "stage_deadline_edf_np"
    FINAL_DEADLINE_EDF_P = "final_deadline_edf_p"
    STAGE_DEADLINE_EDF_P = "stage_deadline_edf_p"


class SchedulingMetadataError(ValueError):
    """Raised when an enabled baseline policy lacks required metadata."""


PolicyKey = tuple[float | int | str, ...]


def policy_uses_active_preemption(policy: BaselineSchedulingPolicy) -> bool:
    """Return whether a policy may replace a running request."""

    return policy in (
        BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_P,
        BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_P,
    )


def policy_is_deadline_edf(policy: BaselineSchedulingPolicy) -> bool:
    """Return whether the leading policy key is an absolute deadline."""

    return policy in (
        BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
        BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP,
        BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_P,
        BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_P,
    )


def _read_non_negative_number(
    environ: Mapping[str, str],
    name: str,
    *,
    default: float,
    integer: bool = False,
) -> float | int:
    raw_value = environ.get(name)
    if raw_value is None:
        return int(default) if integer else default
    try:
        value = int(raw_value) if integer else float(raw_value)
    except ValueError as error:
        expected = "integer" if integer else "number"
        raise ValueError(
            f"invalid {name}={raw_value!r}; expected a non-negative {expected}"
        ) from error
    if value < 0:
        raise ValueError(f"invalid {name}={raw_value!r}; expected a non-negative value")
    return value


def get_active_preemption_config(
    environ: Mapping[str, str] | None = None,
) -> dict[str, float | int]:
    """Read conservative guards for the opt-in preemptive EDF policies.

    vLLM recompute preemption discards the victim's KV cache and resets its
    ``num_computed_tokens`` to zero. The token cap therefore bounds the direct
    wasted work, while the per-request cap prevents oscillation. A deployment
    can additionally require a minimum deadline improvement.
    """

    source = os.environ if environ is None else environ
    return {
        "max_recompute_tokens": _read_non_negative_number(
            source,
            ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS_ENV,
            default=256,
            integer=True,
        ),
        "max_per_request": _read_non_negative_number(
            source,
            ACTIVE_PREEMPTION_MAX_PER_REQUEST_ENV,
            default=1,
            integer=True,
        ),
        "min_deadline_gain_ms": _read_non_negative_number(
            source,
            ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS_ENV,
            default=0.0,
        ),
    }


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


def remaining_prefill_tokens(request: Any) -> int:
    """Return Sarathi-style local remaining work for one scheduler request.

    Sarathi's SRPF baseline uses unprocessed prompt tokens as a direct proxy
    for remaining processing time. Output length and downstream-stage work are
    deliberately excluded. Once prefill is complete the key stays at zero, so
    decode requests remain ahead of requests with unfinished local prefill.
    """

    request_id = str(getattr(request, "request_id", "<unknown>"))
    total = getattr(request, "num_prompt_tokens", None)
    completed = getattr(request, "num_computed_tokens", None)
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise SchedulingMetadataError(
            f"request {request_id!r} has invalid num_prompt_tokens={total!r}"
        )
    if not isinstance(completed, int) or isinstance(completed, bool):
        raise SchedulingMetadataError(
            f"request {request_id!r} has invalid num_computed_tokens={completed!r}"
        )
    completed_prefill = min(max(completed, 0), total)
    return total - completed_prefill


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

    ``native_fcfs`` must never call this function. Keeping that path explicit
    prevents accidental replacement of vLLM's native queue.
    """

    # Reserved for future age-aware policies; current keys remain time-stable.
    del now, data_ready_time
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
        BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP,
        BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_P,
        BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_P,
    ):
        if policy in (
            BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
            BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_P,
        ):
            deadline = float(
                _required_field(
                    metadata,
                    "sched_deadline_monotonic_s",
                    request_id=request_id,
                )
            )
        else:
            deadline = _required_stage_value(
                metadata,
                "sched_stage_deadline_monotonic_s",
                stage_id=stage_id,
                request_id=request_id,
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
        return deadline, ingress_order, source_request_id, request_id

    if policy is BaselineSchedulingPolicy.SRPF_LOCAL_NP:
        ingress_order = int(
            _required_field(
                metadata,
                "sched_ingress_order",
                request_id=request_id,
            )
        )
        return (
            remaining_prefill_tokens(request),
            ingress_order,
            source_request_id,
            request_id,
        )

    raise AssertionError(f"unhandled baseline policy {policy!r}")
