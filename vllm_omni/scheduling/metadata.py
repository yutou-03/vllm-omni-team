"""Versioned client contract for multi-stage scheduling metadata.

The benchmark sends this object as the top-level ``omni_scheduling`` field in
an OpenAI-compatible request. The serving layer will later enrich the client
fields with server-owned ingress timestamps and an absolute deadline before
placing them in ``additional_information``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

CLIENT_SCHEDULING_FIELD = "omni_scheduling"
SCHEDULING_SCHEMA_VERSION = 1
NUM_BASELINE_STAGES = 3

_CLIENT_FIELDS = frozenset(
    {
        "schema_version",
        "source_request_id",
        "ingress_order",
        "slo_ms",
        "request_path",
        "predicted_stage_ms",
        "predicted_stage_work_units",
    }
)
_REQUEST_PATHS = frozenset({"text", "audio"})


def _finite_number(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: {field} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: {field} must be finite, got {value!r}")
    return number


def _nonnegative_int(value: Any, *, field: str, context: str) -> int:
    number = _finite_number(value, field=field, context=context)
    integer = int(number)
    if integer != number or integer < 0:
        raise ValueError(
            f"{context}: {field} must be a non-negative integer, got {value!r}"
        )
    return integer


def normalize_stage_vector(
    value: Any,
    *,
    field: str,
    context: str = CLIENT_SCHEDULING_FIELD,
    strictly_positive: bool = False,
) -> list[float | None]:
    """Normalize a stage mapping or sequence to a three-element JSON list."""

    normalized: list[Any]
    if isinstance(value, Mapping):
        normalized = [None] * NUM_BASELINE_STAGES
        for raw_stage_id, stage_value in value.items():
            stage_key = str(raw_stage_id)
            if stage_key not in {"0", "1", "2"}:
                raise ValueError(
                    f"{context}: {field} has invalid stage {raw_stage_id!r}"
                )
            normalized[int(stage_key)] = stage_value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = list(value)
        if len(normalized) != NUM_BASELINE_STAGES:
            raise ValueError(
                f"{context}: {field} must contain exactly "
                f"{NUM_BASELINE_STAGES} stage entries"
            )
    else:
        raise ValueError(
            f"{context}: {field} must be a stage mapping or a "
            f"{NUM_BASELINE_STAGES}-element sequence"
        )

    result: list[float | None] = []
    for stage_id, stage_value in enumerate(normalized):
        if stage_value is None:
            result.append(None)
            continue
        number = _finite_number(
            stage_value,
            field=f"{field}[{stage_id}]",
            context=context,
        )
        if number < 0 or (strictly_positive and number <= 0):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(
                f"{context}: {field}[{stage_id}] must be {qualifier}, "
                f"got {stage_value!r}"
            )
        result.append(number)

    if all(stage_value is None for stage_value in result):
        raise ValueError(f"{context}: {field} must contain at least one value")
    return result


def normalize_client_scheduling_metadata(
    raw_metadata: Mapping[str, Any],
    *,
    context: str = CLIENT_SCHEDULING_FIELD,
) -> dict[str, Any]:
    """Validate and normalize the version-1 client scheduling contract."""

    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"{context} must be an object")
    unknown_fields = set(raw_metadata) - _CLIENT_FIELDS
    if unknown_fields:
        raise ValueError(
            f"{context}: unknown fields {sorted(unknown_fields)!r}"
        )

    schema_version = _nonnegative_int(
        raw_metadata.get("schema_version"),
        field="schema_version",
        context=context,
    )
    if schema_version != SCHEDULING_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: unsupported schema_version {schema_version}; "
            f"expected {SCHEDULING_SCHEMA_VERSION}"
        )

    if "source_request_id" not in raw_metadata:
        raise ValueError(f"{context}: missing source_request_id")
    raw_source_request_id = raw_metadata["source_request_id"]
    if raw_source_request_id is None:
        raise ValueError(f"{context}: source_request_id must not be empty")
    source_request_id = str(raw_source_request_id)
    if not source_request_id.strip():
        raise ValueError(f"{context}: source_request_id must not be empty")

    if "ingress_order" not in raw_metadata:
        raise ValueError(f"{context}: missing ingress_order")
    ingress_order = _nonnegative_int(
        raw_metadata["ingress_order"],
        field="ingress_order",
        context=context,
    )

    normalized: dict[str, Any] = {
        "schema_version": schema_version,
        "source_request_id": source_request_id,
        "ingress_order": ingress_order,
    }

    slo_ms = raw_metadata.get("slo_ms")
    if slo_ms is not None:
        normalized_slo = _finite_number(slo_ms, field="slo_ms", context=context)
        if normalized_slo <= 0:
            raise ValueError(f"{context}: slo_ms must be positive")
        normalized["slo_ms"] = normalized_slo

    request_path = raw_metadata.get("request_path")
    if request_path is not None:
        normalized_path = str(request_path).lower()
        if normalized_path not in _REQUEST_PATHS:
            raise ValueError(
                f"{context}: request_path must be one of "
                f"{sorted(_REQUEST_PATHS)!r}, got {request_path!r}"
            )
        normalized["request_path"] = normalized_path

    predicted_stage_ms = raw_metadata.get("predicted_stage_ms")
    if predicted_stage_ms is not None:
        normalized["predicted_stage_ms"] = normalize_stage_vector(
            predicted_stage_ms,
            field="predicted_stage_ms",
            context=context,
        )

    predicted_stage_work_units = raw_metadata.get("predicted_stage_work_units")
    if predicted_stage_work_units is not None:
        normalized["predicted_stage_work_units"] = normalize_stage_vector(
            predicted_stage_work_units,
            field="predicted_stage_work_units",
            context=context,
            strictly_positive=True,
        )

    if "predicted_stage_ms" in normalized and "predicted_stage_work_units" in normalized:
        ms_presence = [value is not None for value in normalized["predicted_stage_ms"]]
        work_presence = [
            value is not None
            for value in normalized["predicted_stage_work_units"]
        ]
        if ms_presence != work_presence:
            raise ValueError(
                f"{context}: predicted_stage_ms and "
                "predicted_stage_work_units must cover the same stages"
            )

    return normalized


def build_client_scheduling_metadata(
    *,
    source_request_id: str,
    ingress_order: int,
    slo_ms: float | None = None,
    request_path: str | None = None,
    predicted_stage_ms: Mapping[str | int, Any] | Sequence[Any] | None = None,
    predicted_stage_work_units: Mapping[str | int, Any]
    | Sequence[Any]
    | None = None,
) -> dict[str, Any]:
    """Build the JSON object attached to one benchmark request."""

    raw_metadata: dict[str, Any] = {
        "schema_version": SCHEDULING_SCHEMA_VERSION,
        "source_request_id": source_request_id,
        "ingress_order": ingress_order,
    }
    optional_fields = {
        "slo_ms": slo_ms,
        "request_path": request_path,
        "predicted_stage_ms": predicted_stage_ms,
        "predicted_stage_work_units": predicted_stage_work_units,
    }
    raw_metadata.update(
        {
            field: value
            for field, value in optional_fields.items()
            if value is not None
        }
    )
    return normalize_client_scheduling_metadata(raw_metadata)
