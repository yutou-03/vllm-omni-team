# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed multi-milestone Stage-DDL snapshot construction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vllm_omni.scheduling.contracts import (
    RequestPath,
    RequestSLOContract,
    SLOMilestone,
)


_STAGE_MILESTONE_SCOPE = {
    RequestPath.TEXT: {
        0: frozenset({SLOMilestone.TTFT, SLOMilestone.E2E}),
    },
    RequestPath.AUDIO: {
        0: frozenset(
            {SLOMilestone.TTFT, SLOMilestone.TTFP, SLOMilestone.E2E}
        ),
        1: frozenset({SLOMilestone.TTFP, SLOMilestone.E2E}),
        2: frozenset({SLOMilestone.TTFP, SLOMilestone.E2E}),
    },
}
_RECURRING_GAP_MILESTONES = frozenset(
    {SLOMilestone.TEXT_GAP, SLOMilestone.AUDIO_GAP}
)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _nonnegative_number(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _milestone(value: SLOMilestone | str, *, field: str) -> SLOMilestone:
    if isinstance(value, SLOMilestone):
        return value
    try:
        return SLOMilestone(str(value).lower())
    except ValueError as error:
        raise ValueError(f"{field} has unknown milestone {value!r}") from error


@dataclass(frozen=True, slots=True)
class MilestoneCostEstimate:
    """Frozen service budget from a stage decision point to one milestone.

    ``current_stage_service_ms`` has milestone-specific semantics.  For
    example, at Stage 0 it may mean time to first text for TTFT and time to
    the downstream handoff for TTFP.  Keeping it separate prevents a
    ready-to-done estimate from being silently reused for every milestone.
    """

    current_stage_service_ms: float
    downstream_service_ms: float = 0.0
    downstream_transfer_ms: float = 0.0
    reserve_ms: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "current_stage_service_ms",
            "downstream_service_ms",
            "downstream_transfer_ms",
            "reserve_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_number(
                    getattr(self, field_name),
                    field=field_name,
                ),
            )

    @property
    def downstream_reserve_ms(self) -> float:
        return (
            self.downstream_service_ms
            + self.downstream_transfer_ms
            + self.reserve_ms
        )

    def to_event_fields(self) -> dict[str, float]:
        return {
            "current_stage_service_ms": self.current_stage_service_ms,
            "downstream_service_ms": self.downstream_service_ms,
            "downstream_transfer_ms": self.downstream_transfer_ms,
            "reserve_ms": self.reserve_ms,
        }


@dataclass(frozen=True, slots=True)
class StageDeadlineSnapshot:
    """Deeply immutable Stage-DDL values used for one scheduler tick."""

    schema_version: int
    logical_request_id: str
    stage_id: int
    captured_monotonic_s: float
    deadline_version: int
    cost_model_version: str
    cost_source: str
    used_fallback: bool
    milestone_costs: Mapping[SLOMilestone, MilestoneCostEstimate]
    stage_event_deadlines_s: Mapping[SLOMilestone, float]
    milestone_laxities_s: Mapping[SLOMilestone, float]
    excluded_milestones: Mapping[SLOMilestone, str]
    controlling_milestone: SLOMilestone
    stage_event_deadline_s: float
    base_laxity_s: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("StageDeadlineSnapshot schema_version must be 1")
        if self.stage_id not in (0, 1, 2):
            raise ValueError("stage_id must be one of [0, 1, 2]")
        if (
            isinstance(self.deadline_version, bool)
            or not isinstance(self.deadline_version, int)
            or self.deadline_version <= 0
        ):
            raise ValueError("deadline_version must be a positive integer")
        if not str(self.logical_request_id).strip():
            raise ValueError("logical_request_id must not be empty")
        for field_name in ("cost_model_version", "cost_source"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isinstance(self.used_fallback, bool):
            raise ValueError("used_fallback must be a boolean")

        object.__setattr__(
            self,
            "captured_monotonic_s",
            _nonnegative_number(
                self.captured_monotonic_s,
                field="captured_monotonic_s",
            ),
        )
        for field_name in ("stage_event_deadline_s", "base_laxity_s"):
            object.__setattr__(
                self,
                field_name,
                _finite_number(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "milestone_costs",
            "stage_event_deadlines_s",
            "milestone_laxities_s",
            "excluded_milestones",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "stage_ddl_schema_version": self.schema_version,
            "logical_request_id": self.logical_request_id,
            "stage_id": self.stage_id,
            "captured_monotonic_s": self.captured_monotonic_s,
            "deadline_version": self.deadline_version,
            "cost_model_version": self.cost_model_version,
            "cost_source": self.cost_source,
            "used_fallback": self.used_fallback,
            "controlling_milestone": self.controlling_milestone.value,
            "stage_event_deadline_s": self.stage_event_deadline_s,
            "base_laxity_s": self.base_laxity_s,
            "milestone_costs": {
                milestone.value: cost.to_event_fields()
                for milestone, cost in self.milestone_costs.items()
            },
            "stage_event_deadlines_s": {
                milestone.value: value
                for milestone, value in self.stage_event_deadlines_s.items()
            },
            "milestone_laxities_s": {
                milestone.value: value
                for milestone, value in self.milestone_laxities_s.items()
            },
            "excluded_milestones": {
                milestone.value: reason
                for milestone, reason in self.excluded_milestones.items()
            },
        }


def _normalize_completed_milestones(
    values: Iterable[SLOMilestone | str],
    *,
    contract_milestones: set[SLOMilestone],
) -> set[SLOMilestone]:
    completed = {
        _milestone(value, field="completed_milestones") for value in values
    }
    unexpected = completed - contract_milestones
    if unexpected:
        names = sorted(milestone.value for milestone in unexpected)
        raise ValueError(
            f"completed_milestones are absent from the request contract: {names}"
        )
    return completed


def _normalize_costs(
    raw_costs: Mapping[SLOMilestone | str, MilestoneCostEstimate],
) -> dict[SLOMilestone, MilestoneCostEstimate]:
    if not isinstance(raw_costs, Mapping):
        raise ValueError("milestone_costs must be a mapping")
    normalized: dict[SLOMilestone, MilestoneCostEstimate] = {}
    for raw_milestone, cost in raw_costs.items():
        milestone = _milestone(raw_milestone, field="milestone_costs")
        if milestone in normalized:
            raise ValueError(
                f"milestone_costs repeats milestone {milestone.value}"
            )
        if not isinstance(cost, MilestoneCostEstimate):
            raise ValueError(
                f"milestone_costs.{milestone.value} must be a "
                "MilestoneCostEstimate"
            )
        normalized[milestone] = cost
    return normalized


def derive_stage_deadline_snapshot(
    contract: RequestSLOContract,
    *,
    stage_id: int,
    captured_monotonic_s: float,
    milestone_costs: Mapping[SLOMilestone | str, MilestoneCostEstimate],
    deadline_version: int,
    cost_model_version: str,
    cost_source: str,
    completed_milestones: Iterable[SLOMilestone | str] = (),
    used_fallback: bool = False,
) -> StageDeadlineSnapshot:
    """Reverse-decompose active milestones into one frozen priority snapshot.

    Recurring gap objectives are deliberately excluded.  They require
    produced/consumed stream state and are not equivalent to an absolute
    one-shot milestone deadline.
    """

    if not isinstance(contract, RequestSLOContract):
        raise ValueError("contract must be a RequestSLOContract")
    if stage_id not in (0, 1, 2):
        raise ValueError("stage_id must be one of [0, 1, 2]")
    path_scope = _STAGE_MILESTONE_SCOPE[contract.request_path]
    if stage_id not in path_scope:
        raise ValueError(
            f"{contract.request_path.value} path does not traverse stage {stage_id}"
        )

    contract_milestones = set(contract.milestone_deadlines_s)
    completed = _normalize_completed_milestones(
        completed_milestones,
        contract_milestones=contract_milestones,
    )
    stage_scope = path_scope[stage_id]
    expected = (contract_milestones & stage_scope) - completed
    if not expected:
        raise ValueError("no active one-shot milestone remains for this stage")

    costs = _normalize_costs(milestone_costs)
    unsupported = set(costs) - expected
    if unsupported:
        names = sorted(milestone.value for milestone in unsupported)
        raise ValueError(f"unsupported milestone costs for this stage: {names}")
    missing = expected - costs.keys()
    if missing:
        names = sorted(milestone.value for milestone in missing)
        raise ValueError(f"milestone_costs mismatch: missing milestones {names}")

    captured = _nonnegative_number(
        captured_monotonic_s,
        field="captured_monotonic_s",
    )
    stage_event_deadlines: dict[SLOMilestone, float] = {}
    laxities: dict[SLOMilestone, float] = {}
    for milestone in sorted(expected, key=lambda item: item.value):
        cost = costs[milestone]
        event_deadline = (
            contract.milestone_deadlines_s[milestone]
            - cost.downstream_reserve_ms / 1000.0
        )
        laxity = (
            event_deadline
            - captured
            - cost.current_stage_service_ms / 1000.0
        )
        stage_event_deadlines[milestone] = event_deadline
        laxities[milestone] = laxity

    controlling = min(
        expected,
        key=lambda milestone: (laxities[milestone], milestone.value),
    )
    excluded: dict[SLOMilestone, str] = {}
    for milestone in sorted(contract_milestones - expected, key=lambda item: item.value):
        if milestone in completed:
            reason = "completed"
        elif milestone in _RECURRING_GAP_MILESTONES:
            reason = "recurring_gap_deferred"
        else:
            reason = "not_affected_by_stage"
        excluded[milestone] = reason

    return StageDeadlineSnapshot(
        schema_version=1,
        logical_request_id=contract.logical_request_id,
        stage_id=stage_id,
        captured_monotonic_s=captured,
        deadline_version=deadline_version,
        cost_model_version=cost_model_version,
        cost_source=cost_source,
        used_fallback=used_fallback,
        milestone_costs=costs,
        stage_event_deadlines_s=stage_event_deadlines,
        milestone_laxities_s=laxities,
        excluded_milestones=excluded,
        controlling_milestone=controlling,
        stage_event_deadline_s=stage_event_deadlines[controlling],
        base_laxity_s=laxities[controlling],
    )
