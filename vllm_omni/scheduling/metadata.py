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
_SERVER_FIELDS = frozenset(
    {
        "sched_schema_version",
        "sched_source_request_id",
        "sched_ingress_order",
        "sched_ingress_monotonic_s",
        "sched_ingress_wall_s",
        "sched_deadline_monotonic_s",
        "sched_slo_ms",
        "sched_request_path",
        "sched_predicted_stage_ms",
        "sched_predicted_stage_work_units",
    }
)


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


def build_server_scheduling_metadata(
    raw_metadata: Mapping[str, Any],
    *,
    ingress_monotonic_s: float,
    ingress_wall_s: float,
    context: str = CLIENT_SCHEDULING_FIELD,
) -> dict[str, Any]:
    """Validate client metadata and add server-owned ingress timestamps.

    Absolute deadlines are deliberately computed from the server's monotonic
    clock. Clients can specify an SLO duration, but cannot choose the absolute
    deadline used by the scheduler.
    """

    normalized = normalize_client_scheduling_metadata(
        raw_metadata,
        context=context,
    )
    monotonic_s = _finite_number(
        ingress_monotonic_s,
        field="ingress_monotonic_s",
        context=context,
    )
    wall_s = _finite_number(
        ingress_wall_s,
        field="ingress_wall_s",
        context=context,
    )
    if monotonic_s < 0 or wall_s < 0:
        raise ValueError(
            f"{context}: server ingress timestamps must be non-negative"
        )

    server_metadata: dict[str, Any] = {
        "sched_schema_version": normalized["schema_version"],
        "sched_source_request_id": normalized["source_request_id"],
        "sched_ingress_order": normalized["ingress_order"],
        "sched_ingress_monotonic_s": monotonic_s,
        "sched_ingress_wall_s": wall_s,
    }
    for field in (
        "slo_ms",
        "request_path",
        "predicted_stage_ms",
        "predicted_stage_work_units",
    ):
        if field in normalized:
            server_metadata[f"sched_{field}"] = normalized[field]

    slo_ms = normalized.get("slo_ms")
    if slo_ms is not None:
        server_metadata["sched_deadline_monotonic_s"] = (
            monotonic_s + slo_ms / 1000.0
        )
    return server_metadata


def extract_scheduling_metadata(
    additional_information: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the scheduling subset from a decoded Omni payload."""

    if not isinstance(additional_information, Mapping):
        return None
    meta = additional_information.get("meta")
    if not isinstance(meta, Mapping):
        return None
    scheduling = {
        str(key): value
        for key, value in meta.items()
        if str(key) in _SERVER_FIELDS
    }
    return scheduling or None


def merge_scheduling_metadata_into_additional_information(
    additional_information: Mapping[str, Any] | None,
    scheduling_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy an Omni payload and merge canonical ``sched_*`` meta fields."""

    merged = dict(additional_information or {})
    if not scheduling_metadata:
        return merged
    invalid_fields = [
        str(field)
        for field in scheduling_metadata
        if str(field) not in _SERVER_FIELDS
    ]
    if invalid_fields:
        raise ValueError(
            "scheduling metadata contains non-canonical fields "
            f"{sorted(invalid_fields)!r}"
        )
    meta = dict(merged.get("meta") or {})
    meta.update(scheduling_metadata)
    merged["meta"] = meta
    return merged


def merge_scheduling_metadata_into_prompt(
    prompt: Any,
    scheduling_metadata: Mapping[str, Any] | None,
) -> Any:
    """Attach scheduling metadata to a prompt without discarding other data."""

    if not scheduling_metadata or not isinstance(prompt, dict):
        return prompt
    merged_prompt = dict(prompt)
    merged_prompt["additional_information"] = (
        merge_scheduling_metadata_into_additional_information(
            prompt.get("additional_information"),
            scheduling_metadata,
        )
    )
    return merged_prompt
