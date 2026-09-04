# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable request contracts shared by SLO-aware scheduling components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


class RequestPath(str, Enum):
    TEXT = "text"
    AUDIO = "audio"


class SLOMilestone(str, Enum):
    TTFT = "ttft"
    TTFP = "ttfp"
    TEXT_GAP = "text_gap"
    AUDIO_GAP = "audio_gap"
    E2E = "e2e"


_APPLICABLE_MILESTONES = {
    RequestPath.TEXT: frozenset(
        {SLOMilestone.TTFT, SLOMilestone.TEXT_GAP, SLOMilestone.E2E}
    ),
    RequestPath.AUDIO: frozenset(SLOMilestone),
}
_REQUIRED_MILESTONES = {
    RequestPath.TEXT: frozenset({SLOMilestone.TTFT, SLOMilestone.E2E}),
    RequestPath.AUDIO: frozenset(
        {SLOMilestone.TTFT, SLOMilestone.TTFP, SLOMilestone.E2E}
    ),
}


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _request_path(value: RequestPath | str) -> RequestPath:
    if isinstance(value, RequestPath):
        return value
    try:
        return RequestPath(str(value).lower())
    except ValueError as error:
        choices = sorted(path.value for path in RequestPath)
        raise ValueError(f"request_path must be one of {choices!r}") from error


def normalize_milestone_values(
    raw_values: Mapping[SLOMilestone | str, Any],
    *,
    request_path: RequestPath | str,
    field_name: str,
) -> dict[SLOMilestone, float]:
    """Normalize milestone keys and enforce path-specific applicability."""

    if not isinstance(raw_values, Mapping):
        raise ValueError(f"{field_name} must be an object")
    path = _request_path(request_path)
    normalized: dict[SLOMilestone, float] = {}
    for raw_milestone, raw_value in raw_values.items():
        if isinstance(raw_milestone, SLOMilestone):
            milestone = raw_milestone
        else:
            try:
                milestone = SLOMilestone(str(raw_milestone).lower())
            except ValueError as error:
                raise ValueError(
                    f"{field_name} has unknown milestone {raw_milestone!r}"
                ) from error
        if milestone in normalized:
            raise ValueError(f"{field_name} repeats milestone {milestone.value}")
        value = _finite_number(
            raw_value,
            field=f"{field_name}.{milestone.value}",
        )
        if value <= 0:
            raise ValueError(
                f"{field_name}.{milestone.value} must be positive"
            )
        normalized[milestone] = value

    missing = _REQUIRED_MILESTONES[path] - normalized.keys()
    if missing:
        names = sorted(milestone.value for milestone in missing)
        raise ValueError(
            f"{field_name}: missing required milestones for {path.value}: {names}"
        )
    unexpected = normalized.keys() - _APPLICABLE_MILESTONES[path]
    if unexpected:
        names = sorted(milestone.value for milestone in unexpected)
        raise ValueError(
            f"{field_name}: milestones not applicable to {path.value}: {names}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RequestSLOContract:
    """Server-owned, deeply immutable schema-v2 scheduling contract."""

    schema_version: int
    logical_request_id: str
    ingress_monotonic_s: float
    request_path: RequestPath
    service_class: str
    milestone_deadlines_s: Mapping[SLOMilestone, float]
    text_consume_rate_tps: float | None
    audio_consume_rate: float | None
    max_preemptions: int

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("RequestSLOContract schema_version must be 2")

        logical_request_id = str(self.logical_request_id)
        if not logical_request_id.strip():
            raise ValueError("logical_request_id must not be empty")
        object.__setattr__(self, "logical_request_id", logical_request_id)

        ingress = _finite_number(
            self.ingress_monotonic_s,
            field="ingress_monotonic_s",
        )
        if ingress < 0:
            raise ValueError("ingress_monotonic_s must be non-negative")
        object.__setattr__(self, "ingress_monotonic_s", ingress)

        path = _request_path(self.request_path)
        object.__setattr__(self, "request_path", path)

        service_class = str(self.service_class)
        if not service_class.strip():
            raise ValueError("service_class must not be empty")
        object.__setattr__(self, "service_class", service_class)

        deadlines = normalize_milestone_values(
            self.milestone_deadlines_s,
            request_path=path,
            field_name="milestone_deadlines_s",
        )
        for milestone, deadline in deadlines.items():
            if deadline <= ingress:
                raise ValueError(
                    f"{milestone.value} deadline must be after ingress"
                )
        e2e_deadline = deadlines[SLOMilestone.E2E]
        later_than_e2e = sorted(
            milestone.value
            for milestone, deadline in deadlines.items()
            if milestone is not SLOMilestone.E2E and deadline > e2e_deadline
        )
        if later_than_e2e:
            raise ValueError(
                "milestone deadlines must not exceed e2e: "
                f"{later_than_e2e}"
            )
        object.__setattr__(
            self,
            "milestone_deadlines_s",
            MappingProxyType(dict(deadlines)),
        )

        for field_name in ("text_consume_rate_tps", "audio_consume_rate"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized_rate = _finite_number(value, field=field_name)
            if normalized_rate <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, normalized_rate)

        max_preemptions = self.max_preemptions
        if (
            isinstance(max_preemptions, bool)
            or not isinstance(max_preemptions, int)
            or max_preemptions < 0
        ):
            raise ValueError("max_preemptions must be a non-negative integer")

    @classmethod
    def from_server_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> RequestSLOContract:
        required = {
            "sched_schema_version",
            "sched_logical_request_id",
            "sched_ingress_monotonic_s",
            "sched_request_path",
            "sched_service_class",
            "sched_milestone_deadlines_monotonic_s",
            "sched_max_preemptions",
        }
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValueError(f"server SLO contract missing fields: {missing}")
        return cls(
            schema_version=metadata["sched_schema_version"],
            logical_request_id=metadata["sched_logical_request_id"],
            ingress_monotonic_s=metadata["sched_ingress_monotonic_s"],
            request_path=metadata["sched_request_path"],
            service_class=metadata["sched_service_class"],
            milestone_deadlines_s=metadata[
                "sched_milestone_deadlines_monotonic_s"
            ],
            text_consume_rate_tps=metadata.get(
                "sched_text_consume_rate_tps"
            ),
            audio_consume_rate=metadata.get("sched_audio_consume_rate"),
            max_preemptions=metadata["sched_max_preemptions"],
        )

    def to_server_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "sched_schema_version": self.schema_version,
            "sched_logical_request_id": self.logical_request_id,
            "sched_ingress_monotonic_s": self.ingress_monotonic_s,
            "sched_request_path": self.request_path.value,
            "sched_service_class": self.service_class,
            "sched_milestone_deadlines_monotonic_s": {
                milestone.value: deadline
                for milestone, deadline in self.milestone_deadlines_s.items()
            },
            "sched_max_preemptions": self.max_preemptions,
        }
        if self.text_consume_rate_tps is not None:
            metadata["sched_text_consume_rate_tps"] = (
                self.text_consume_rate_tps
            )
        if self.audio_consume_rate is not None:
            metadata["sched_audio_consume_rate"] = self.audio_consume_rate
        return metadata
