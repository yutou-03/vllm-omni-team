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


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_nonnegative_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, field=field)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _optional_ratio(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    ratio = _finite_number(value, field=field)
    if ratio < 0 or ratio > 1:
        raise ValueError(f"{field} must be in [0, 1]")
    return ratio


@dataclass(frozen=True, slots=True)
class StageRuntimeSnapshot:
    """One immutable input snapshot for a stage control decision."""

    schema_version: int
    stage_id: int
    captured_monotonic_s: float
    control_epoch: int
    running_requests: int
    waiting_requests: int
    input_waiting_requests: int
    physical_max_num_seqs: int
    effective_max_num_seqs: int
    token_budget_total: int
    token_budget_available: int
    token_budget_utilization_ratio: float | None = None
    kv_cache_usage_ratio: float | None = None
    gpu_active_ratio: float | None = None
    gpu_active_sample_age_ms: float | None = None
    gpu_active_source: str | None = None
    gpu_active_device_count: int | None = None
    service_time_ewma_ms: float | None = None
    service_time_p95_ms: float | None = None
    queue_growth_rate_rps: float | None = None
    downstream_waiting_requests: int | None = None
    downstream_effective_cap: int | None = None
    connector_backlog: int | None = None
    deadline_miss_risk: float | None = None
    deadline_miss_risk_population: int | None = None
    deadline_miss_risk_scope: str | None = None
    last_admission_rejection: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError(
                "StageRuntimeSnapshot schema_version must be 1 or 2"
            )
        for field_name in (
            "stage_id",
            "control_epoch",
            "running_requests",
            "waiting_requests",
            "input_waiting_requests",
            "physical_max_num_seqs",
            "effective_max_num_seqs",
            "token_budget_total",
            "token_budget_available",
        ):
            _nonnegative_integer(getattr(self, field_name), field=field_name)
        captured = _finite_number(
            self.captured_monotonic_s,
            field="captured_monotonic_s",
        )
        if captured < 0:
            raise ValueError("captured_monotonic_s must be non-negative")
        object.__setattr__(self, "captured_monotonic_s", captured)
        if self.effective_max_num_seqs > self.physical_max_num_seqs:
            raise ValueError(
                "effective_max_num_seqs must not exceed physical_max_num_seqs"
            )
        if self.token_budget_available > self.token_budget_total:
            raise ValueError(
                "token_budget_available must not exceed token_budget_total"
            )

        for field_name in (
            "downstream_waiting_requests",
            "downstream_effective_cap",
            "connector_backlog",
            "deadline_miss_risk_population",
            "gpu_active_device_count",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _nonnegative_integer(value, field=field_name)
        for field_name in (
            "service_time_ewma_ms",
            "service_time_p95_ms",
            "gpu_active_sample_age_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonnegative_number(
                    getattr(self, field_name),
                    field=field_name,
                ),
            )
        gpu_active_fields = (
            self.gpu_active_ratio,
            self.gpu_active_sample_age_ms,
            self.gpu_active_source,
            self.gpu_active_device_count,
        )
        if any(value is not None for value in gpu_active_fields):
            if any(value is None for value in gpu_active_fields):
                raise ValueError(
                    "GPU active fields must be provided together"
                )
            if self.gpu_active_source != "nvml_mean_visible_devices":
                raise ValueError(
                    "gpu_active_source must be nvml_mean_visible_devices"
                )
            if self.gpu_active_device_count == 0:
                raise ValueError("gpu_active_device_count must be positive")
        for field_name in (
            "token_budget_utilization_ratio",
            "kv_cache_usage_ratio",
            "gpu_active_ratio",
            "deadline_miss_risk",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_ratio(getattr(self, field_name), field=field_name),
            )
        deadline_metadata = (
            self.deadline_miss_risk_population,
            self.deadline_miss_risk_scope,
        )
        if self.deadline_miss_risk is not None or any(
            value is not None for value in deadline_metadata
        ):
            if self.deadline_miss_risk is None or any(
                value is None for value in deadline_metadata
            ):
                raise ValueError(
                    "deadline risk fields must be provided together"
                )
            if self.deadline_miss_risk_scope != "runnable_running":
                raise ValueError(
                    "deadline_miss_risk_scope must be runnable_running"
                )
            if (
                self.deadline_miss_risk_population == 0
                and self.deadline_miss_risk != 0.0
            ):
                raise ValueError(
                    "empty deadline risk population must have zero risk"
                )
        if self.queue_growth_rate_rps is not None:
            object.__setattr__(
                self,
                "queue_growth_rate_rps",
                _finite_number(
                    self.queue_growth_rate_rps,
                    field="queue_growth_rate_rps",
                ),
            )
        if (
            self.last_admission_rejection is not None
            and not str(self.last_admission_rejection).strip()
        ):
            raise ValueError("last_admission_rejection must not be empty")

    def to_event_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "pressure_schema_version": self.schema_version,
            "control_epoch": self.control_epoch,
            "running_requests": self.running_requests,
            "waiting_requests": self.waiting_requests,
            "input_waiting_requests": self.input_waiting_requests,
            "physical_max_num_seqs": self.physical_max_num_seqs,
            "effective_max_num_seqs": self.effective_max_num_seqs,
            "token_budget_total": self.token_budget_total,
            "token_budget_available": self.token_budget_available,
            "captured_monotonic_s": self.captured_monotonic_s,
        }
        if self.schema_version == 2:
            fields["request_count_semantics"] = (
                "runnable_excludes_input_blocked"
            )
        for field_name in (
            "token_budget_utilization_ratio",
            "kv_cache_usage_ratio",
            "gpu_active_ratio",
            "gpu_active_sample_age_ms",
            "gpu_active_source",
            "gpu_active_device_count",
            "service_time_ewma_ms",
            "service_time_p95_ms",
            "queue_growth_rate_rps",
            "downstream_waiting_requests",
            "downstream_effective_cap",
            "connector_backlog",
            "deadline_miss_risk",
            "deadline_miss_risk_population",
            "deadline_miss_risk_scope",
            "last_admission_rejection",
        ):
            value = getattr(self, field_name)
            if value is not None:
                fields[field_name] = value
        return fields
