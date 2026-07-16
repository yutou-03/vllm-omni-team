"""Validation helpers for timestamped ServeGen benchmark traces.

The workload generator and the benchmark client are separate programs.  This
module keeps their shared JSONL contract explicit so malformed traces fail
before any expensive model startup or GPU work begins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from vllm_omni.scheduling.metadata import normalize_stage_vector


_VALID_OUTPUT_MODALITIES = {"text", "audio"}


def _finite_number(value: Any, *, field: str, line_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ServeGen line {line_number}: {field} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"ServeGen line {line_number}: {field} must be finite, got {value!r}"
        )
    return number


def _positive_int(value: Any, *, field: str, line_number: int) -> int:
    number = _finite_number(value, field=field, line_number=line_number)
    integer = int(number)
    if integer != number or integer <= 0:
        raise ValueError(
            f"ServeGen line {line_number}: {field} must be a positive integer, "
            f"got {value!r}"
        )
    return integer


def _nonnegative_int(value: Any, *, field: str, line_number: int) -> int:
    number = _finite_number(value, field=field, line_number=line_number)
    integer = int(number)
    if integer != number or integer < 0:
        raise ValueError(
            f"ServeGen line {line_number}: {field} must be a non-negative "
            f"integer, got {value!r}"
        )
    return integer


def normalize_output_modalities(value: Any, *, line_number: int) -> list[str] | None:
    """Normalize optional per-request output modalities.

    ``None`` remains valid for backward compatibility and means that the
    benchmark-wide ``extra_body`` controls the output path.
    """

    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"ServeGen line {line_number}: output_modalities must be a non-empty list"
        )
    modalities = [str(item).lower() for item in value]
    unknown = sorted(set(modalities) - _VALID_OUTPUT_MODALITIES)
    if unknown:
        raise ValueError(
            f"ServeGen line {line_number}: unsupported output modalities {unknown}"
        )
    if len(set(modalities)) != len(modalities):
        raise ValueError(
            f"ServeGen line {line_number}: output_modalities contains duplicates"
        )
    return modalities


def normalize_servegen_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """Return one validated, normalized ServeGen record."""

    if "request_id" not in record:
        raise ValueError(f"ServeGen line {line_number}: missing request_id")
    request_id = str(record["request_id"])
    if not request_id:
        raise ValueError(f"ServeGen line {line_number}: request_id must not be empty")

    timestamp = _finite_number(
        record.get("timestamp"), field="timestamp", line_number=line_number
    )
    if timestamp < 0:
        raise ValueError(
            f"ServeGen line {line_number}: timestamp must be non-negative"
        )

    output_tokens = _positive_int(
        record.get("output_tokens"),
        field="output_tokens",
        line_number=line_number,
    )

    mm_items = record.get("mm_items")
    if mm_items is None:
        mm_items = []
    if not isinstance(mm_items, list):
        raise ValueError(
            f"ServeGen line {line_number}: mm_items must be a list when present"
        )
    for item_index, item in enumerate(mm_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"ServeGen line {line_number}: mm_items[{item_index}] must be an object"
            )
        if item.get("modality") not in {"image", "video", "audio"}:
            raise ValueError(
                f"ServeGen line {line_number}: mm_items[{item_index}] has unsupported "
                f"modality {item.get('modality')!r}"
            )

    normalized = dict(record)
    normalized["request_id"] = request_id
    normalized["timestamp"] = timestamp
    normalized["output_tokens"] = output_tokens
    normalized["mm_items"] = mm_items
    normalized["output_modalities"] = normalize_output_modalities(
        record.get("output_modalities"), line_number=line_number
    )

    ingress_order = record.get("ingress_order")
    if ingress_order is not None:
        normalized["ingress_order"] = _nonnegative_int(
            ingress_order,
            field="ingress_order",
            line_number=line_number,
        )

    slo_ms = record.get("slo_ms")
    if slo_ms is not None:
        slo_ms = _finite_number(slo_ms, field="slo_ms", line_number=line_number)
        if slo_ms <= 0:
            raise ValueError(f"ServeGen line {line_number}: slo_ms must be positive")
        normalized["slo_ms"] = slo_ms

    predicted_stage_ms = record.get("predicted_stage_ms")
    if predicted_stage_ms is not None:
        normalized["predicted_stage_ms"] = normalize_stage_vector(
            predicted_stage_ms,
            field="predicted_stage_ms",
            context=f"ServeGen line {line_number}",
        )

    predicted_stage_work_units = record.get("predicted_stage_work_units")
    if predicted_stage_work_units is not None:
        normalized["predicted_stage_work_units"] = normalize_stage_vector(
            predicted_stage_work_units,
            field="predicted_stage_work_units",
            context=f"ServeGen line {line_number}",
            strictly_positive=True,
        )

    request_path = record.get("request_path")
    if request_path is not None:
        normalized_path = str(request_path).lower()
        if normalized_path not in {"text", "audio"}:
            raise ValueError(
                f"ServeGen line {line_number}: request_path must be text or audio"
            )
        normalized["request_path"] = normalized_path

    if "predicted_stage_ms" in normalized and "predicted_stage_work_units" in normalized:
        ms_presence = [value is not None for value in normalized["predicted_stage_ms"]]
        work_presence = [
            value is not None
            for value in normalized["predicted_stage_work_units"]
        ]
        if ms_presence != work_presence:
            raise ValueError(
                f"ServeGen line {line_number}: predicted_stage_ms and "
                "predicted_stage_work_units must cover the same stages"
            )

    return normalized


def load_servegen_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a trace while enforcing stable IDs and nondecreasing timestamps."""

    records: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    previous_timestamp = -1.0
    previous_ingress_order = -1

    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"ServeGen line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw_record, dict):
                raise ValueError(
                    f"ServeGen line {line_number}: each JSONL row must be an object"
                )
            record = normalize_servegen_record(raw_record, line_number=line_number)
            if "ingress_order" not in record:
                record["ingress_order"] = previous_ingress_order + 1
            request_id = record["request_id"]
            if request_id in request_ids:
                raise ValueError(
                    f"ServeGen line {line_number}: duplicate request_id {request_id!r}"
                )
            if record["timestamp"] < previous_timestamp:
                raise ValueError(
                    f"ServeGen line {line_number}: timestamps must be nondecreasing"
                )
            if record["ingress_order"] <= previous_ingress_order:
                raise ValueError(
                    f"ServeGen line {line_number}: ingress_order must be "
                    "strictly increasing"
                )
            request_ids.add(request_id)
            previous_timestamp = record["timestamp"]
            previous_ingress_order = record["ingress_order"]
            records.append(record)

    if not records:
        raise ValueError(f"ServeGen trace {path} contains no requests")
    return records


def mm_item_config(item: dict[str, Any], *, request_id: str) -> tuple[int, int | float, int]:
    """Translate a normalized multimodal item to the synthetic generator tuple."""

    modality = item["modality"]
    try:
        if modality == "image":
            return int(item["h"]), int(item["w"]), 1
        if modality == "video":
            # ``t`` is already the generated frame count.  Do not multiply by fps.
            return int(item["h"]), int(item["w"]), int(item["t"])
        if modality == "audio":
            return 0, float(item["duration_s"]), int(item["num_channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"ServeGen request {request_id}: malformed {modality} item {item!r}"
        ) from exc
    raise ValueError(
        f"ServeGen request {request_id}: unsupported modality {modality!r}"
    )
