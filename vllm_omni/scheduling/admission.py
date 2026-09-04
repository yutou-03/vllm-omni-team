# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure dynamic-admission controls for SLO-aware stage scheduling.

The controller deliberately does not mutate a scheduler.  It consumes one
immutable runtime snapshot and returns both an auditable decision and the next
immutable controller state.  Online integration can therefore use the same
function as offline trace replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from vllm_omni.scheduling.contracts import StageRuntimeSnapshot


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _ratio(value: object, *, field_name: str) -> float:
    number = _finite(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class StaticConcurrencyEnvelope:
    """Hash-bindable offline safe-cap envelope used by the online controller."""

    schema_version: int
    artifact_id: str
    safe_max_active_by_stage: Mapping[int, int]
    service_p95_limit_ms_by_stage: Mapping[int, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("StaticConcurrencyEnvelope schema_version must be 1")
        if not str(self.artifact_id).strip():
            raise ValueError("artifact_id must not be empty")
        safe_caps: dict[int, int] = {}
        for raw_stage, raw_cap in self.safe_max_active_by_stage.items():
            stage = _nonnegative_int(raw_stage, field_name="stage id")
            safe_caps[stage] = _positive_int(
                raw_cap, field_name=f"stage {stage} safe cap"
            )
        if not safe_caps:
            raise ValueError("safe_max_active_by_stage must not be empty")

        service_limits: dict[int, float] = {}
        for raw_stage, raw_limit in self.service_p95_limit_ms_by_stage.items():
            stage = _nonnegative_int(raw_stage, field_name="stage id")
            if stage not in safe_caps:
                raise ValueError(
                    "service p95 limit stage must exist in safe cap envelope"
                )
            limit = _finite(
                raw_limit,
                field_name=f"stage {stage} service p95 limit",
            )
            if limit <= 0:
                raise ValueError(
                    f"stage {stage} service p95 limit must be positive"
                )
            service_limits[stage] = limit
        object.__setattr__(
            self,
            "safe_max_active_by_stage",
            MappingProxyType(safe_caps),
        )
        object.__setattr__(
            self,
            "service_p95_limit_ms_by_stage",
            MappingProxyType(service_limits),
        )

    def safe_cap(self, stage_id: int) -> int:
        try:
            return self.safe_max_active_by_stage[stage_id]
        except KeyError as error:
            raise ValueError(
                f"stage {stage_id} is absent from concurrency envelope"
            ) from error


@dataclass(frozen=True, slots=True)
class AdmissionControllerConfig:
    """Frozen AIMD thresholds and hysteresis for one experiment version."""

    min_active: int = 1
    max_active: int = 1
    increase_hold: int = 3
    decrease_hold: int = 2
    cooldown_epochs: int = 1
    increase_step: int = 1
    decrease_factor: float = 0.5
    gpu_low_watermark: float = 0.6
    kv_low_watermark: float = 0.7
    kv_high_watermark: float = 0.9
    deadline_risk_low_watermark: float = 0.2
    deadline_risk_high_watermark: float = 0.6
    queue_growth_high_rps: float | None = None
    connector_backlog_high: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_active",
            "max_active",
            "increase_hold",
            "decrease_hold",
            "increase_step",
        ):
            _positive_int(getattr(self, name), field_name=name)
        _nonnegative_int(self.cooldown_epochs, field_name="cooldown_epochs")
        if self.min_active > self.max_active:
            raise ValueError("min_active must not exceed max_active")
        factor = _finite(self.decrease_factor, field_name="decrease_factor")
        if not 0.0 < factor < 1.0:
            raise ValueError("decrease_factor must be in (0, 1)")
        for name in (
            "gpu_low_watermark",
            "kv_low_watermark",
            "kv_high_watermark",
            "deadline_risk_low_watermark",
            "deadline_risk_high_watermark",
        ):
            object.__setattr__(
                self,
                name,
                _ratio(getattr(self, name), field_name=name),
            )
        if self.kv_low_watermark >= self.kv_high_watermark:
            raise ValueError("KV low watermark must be below high watermark")
        if (
            self.deadline_risk_low_watermark
            >= self.deadline_risk_high_watermark
        ):
            raise ValueError(
                "deadline-risk low watermark must be below high watermark"
            )
        if self.queue_growth_high_rps is not None:
            threshold = _finite(
                self.queue_growth_high_rps,
                field_name="queue_growth_high_rps",
            )
            if threshold <= 0:
                raise ValueError("queue_growth_high_rps must be positive")
            object.__setattr__(self, "queue_growth_high_rps", threshold)
        if self.connector_backlog_high is not None:
            _positive_int(
                self.connector_backlog_high,
                field_name="connector_backlog_high",
            )


@dataclass(frozen=True, slots=True)
class AdmissionControllerState:
    """Minimal replayable state; no wall-clock or scheduler references."""

    current_cap: int
    increase_streak: int = 0
    decrease_streak: int = 0
    last_change_epoch: int | None = None
    last_observed_epoch: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.current_cap, field_name="current_cap")
        _nonnegative_int(self.increase_streak, field_name="increase_streak")
        _nonnegative_int(self.decrease_streak, field_name="decrease_streak")
        for name in ("last_change_epoch", "last_observed_epoch"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int(value, field_name=name)

    @classmethod
    def initial(cls, cap: int) -> AdmissionControllerState:
        return cls(current_cap=cap)


class AdmissionAction(str, Enum):
    HOLD = "hold"
    INCREASE = "increase"
    DECREASE = "decrease"
    CLAMP = "clamp"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """One complete decision record suitable for conformance tracing."""

    stage_id: int
    control_epoch: int
    action: AdmissionAction
    old_cap: int
    new_cap: int
    safe_cap_ceiling: int
    trigger_reasons: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    drain_only: bool
    evicted_requests: int
    next_state: AdmissionControllerState

    def to_event_fields(self) -> dict[str, object]:
        return {
            "admission_schema_version": 1,
            "control_epoch": self.control_epoch,
            "admission_action": self.action.value,
            "old_effective_max_num_seqs": self.old_cap,
            "new_effective_max_num_seqs": self.new_cap,
            "safe_cap_ceiling": self.safe_cap_ceiling,
            "admission_trigger_reasons": list(self.trigger_reasons),
            "admission_blocked_reasons": list(self.blocked_reasons),
            "admission_drain_only": self.drain_only,
            "admission_evicted_requests": self.evicted_requests,
        }


def _pressure_reasons(
    config: AdmissionControllerConfig,
    envelope: StaticConcurrencyEnvelope,
    snapshot: StageRuntimeSnapshot,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        snapshot.connector_backlog is not None
        and config.connector_backlog_high is not None
        and snapshot.connector_backlog >= config.connector_backlog_high
    ):
        reasons.append("connector_backlog_high")
    if (
        snapshot.deadline_miss_risk is not None
        and snapshot.deadline_miss_risk
        >= config.deadline_risk_high_watermark
    ):
        reasons.append("deadline_risk_high")
    if (
        snapshot.downstream_waiting_requests is not None
        and snapshot.downstream_effective_cap is not None
        and snapshot.downstream_waiting_requests
        >= snapshot.downstream_effective_cap
    ):
        reasons.append("downstream_congested")
    if (
        snapshot.kv_cache_usage_ratio is not None
        and snapshot.kv_cache_usage_ratio >= config.kv_high_watermark
    ):
        reasons.append("kv_high_watermark")
    if (
        snapshot.queue_growth_rate_rps is not None
        and config.queue_growth_high_rps is not None
        and snapshot.queue_growth_rate_rps >= config.queue_growth_high_rps
    ):
        reasons.append("queue_growth_high")
    service_limit = envelope.service_p95_limit_ms_by_stage.get(
        snapshot.stage_id
    )
    if (
        snapshot.service_time_p95_ms is not None
        and service_limit is not None
        and snapshot.service_time_p95_ms > service_limit
    ):
        reasons.append("service_p95_above_envelope")
    return tuple(sorted(reasons))


def _increase_blockers(
    config: AdmissionControllerConfig,
    snapshot: StageRuntimeSnapshot,
    *,
    current_cap: int,
    safe_cap_ceiling: int,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if current_cap >= safe_cap_ceiling:
        blockers.append("safe_cap_ceiling_reached")
    if snapshot.waiting_requests == 0:
        blockers.append("no_runnable_waiting_demand")
    if snapshot.running_requests < current_cap:
        blockers.append("current_cap_not_occupied")

    required = {
        "deadline_miss_risk": snapshot.deadline_miss_risk,
        "gpu_active_ratio": snapshot.gpu_active_ratio,
        "kv_cache_usage_ratio": snapshot.kv_cache_usage_ratio,
    }
    blockers.extend(
        f"missing:{name}"
        for name, value in required.items()
        if value is None
    )
    if (
        snapshot.gpu_active_ratio is not None
        and snapshot.gpu_active_ratio > config.gpu_low_watermark
    ):
        blockers.append("gpu_not_underutilized")
    if (
        snapshot.kv_cache_usage_ratio is not None
        and snapshot.kv_cache_usage_ratio > config.kv_low_watermark
    ):
        blockers.append("kv_not_below_low_watermark")
    if (
        snapshot.deadline_miss_risk is not None
        and snapshot.deadline_miss_risk
        > config.deadline_risk_low_watermark
    ):
        blockers.append("deadline_risk_not_low")
    if (
        snapshot.downstream_waiting_requests is not None
        and snapshot.downstream_effective_cap is not None
        and snapshot.downstream_waiting_requests
        >= snapshot.downstream_effective_cap
    ):
        blockers.append("downstream_congested")
    if (
        snapshot.connector_backlog is not None
        and config.connector_backlog_high is not None
        and snapshot.connector_backlog >= config.connector_backlog_high
    ):
        blockers.append("connector_backlog_high")
    return tuple(sorted(blockers))


def decide_admission_cap(
    config: AdmissionControllerConfig,
    envelope: StaticConcurrencyEnvelope,
    snapshot: StageRuntimeSnapshot,
    state: AdmissionControllerState,
) -> AdmissionDecision:
    """Return one fail-closed AIMD cap decision without scheduler mutation."""

    if snapshot.schema_version != 2:
        raise ValueError(
            "dynamic admission requires pressure schema version 2"
        )
    if (
        state.last_observed_epoch is not None
        and snapshot.control_epoch <= state.last_observed_epoch
    ):
        raise ValueError("control_epoch must increase between decisions")
    if snapshot.physical_max_num_seqs < config.min_active:
        raise ValueError("physical cap is below configured min_active")

    ceiling = min(
        config.max_active,
        envelope.safe_cap(snapshot.stage_id),
        snapshot.physical_max_num_seqs,
    )
    if ceiling < config.min_active:
        raise ValueError("safe cap ceiling is below configured min_active")

    old_cap = state.current_cap
    pressure = _pressure_reasons(config, envelope, snapshot)
    blockers = _increase_blockers(
        config,
        snapshot,
        current_cap=old_cap,
        safe_cap_ceiling=ceiling,
    )
    action = AdmissionAction.HOLD
    new_cap = old_cap
    increase_streak = state.increase_streak
    decrease_streak = state.decrease_streak
    last_change_epoch = state.last_change_epoch
    trigger_reasons: tuple[str, ...] = ()

    if old_cap > ceiling or old_cap < config.min_active:
        action = AdmissionAction.CLAMP
        new_cap = min(max(old_cap, config.min_active), ceiling)
        increase_streak = 0
        decrease_streak = 0
        last_change_epoch = snapshot.control_epoch
        trigger_reasons = ("hard_bound_changed",)
        blockers = ()
    else:
        in_cooldown = (
            state.last_change_epoch is not None
            and snapshot.control_epoch - state.last_change_epoch
            < config.cooldown_epochs
        )
        if pressure:
            decrease_streak += 1
            increase_streak = 0
            trigger_reasons = pressure
        elif not blockers:
            increase_streak += 1
            decrease_streak = 0
            trigger_reasons = ("safe_headroom_with_demand",)
        else:
            increase_streak = 0
            decrease_streak = 0

        if in_cooldown:
            blockers = ("cooldown_active",)
        elif pressure and decrease_streak >= config.decrease_hold:
            action = AdmissionAction.DECREASE
            multiplicative = math.floor(old_cap * config.decrease_factor)
            new_cap = max(config.min_active, multiplicative)
            increase_streak = 0
            decrease_streak = 0
            if new_cap != old_cap:
                last_change_epoch = snapshot.control_epoch
            else:
                action = AdmissionAction.HOLD
        elif not pressure and not blockers and (
            increase_streak >= config.increase_hold
        ):
            action = AdmissionAction.INCREASE
            new_cap = min(ceiling, old_cap + config.increase_step)
            increase_streak = 0
            decrease_streak = 0
            if new_cap != old_cap:
                last_change_epoch = snapshot.control_epoch
            else:
                action = AdmissionAction.HOLD

    drain_only = new_cap < old_cap and new_cap <= snapshot.running_requests
    next_state = AdmissionControllerState(
        current_cap=new_cap,
        increase_streak=increase_streak,
        decrease_streak=decrease_streak,
        last_change_epoch=last_change_epoch,
        last_observed_epoch=snapshot.control_epoch,
    )
    return AdmissionDecision(
        stage_id=snapshot.stage_id,
        control_epoch=snapshot.control_epoch,
        action=action,
        old_cap=old_cap,
        new_cap=new_cap,
        safe_cap_ceiling=ceiling,
        trigger_reasons=trigger_reasons,
        blocked_reasons=blockers,
        drain_only=drain_only,
        evicted_requests=0,
        next_state=next_state,
    )


def replay_admission_trace(
    config: AdmissionControllerConfig,
    envelope: StaticConcurrencyEnvelope,
    snapshots: Sequence[StageRuntimeSnapshot],
    *,
    initial_cap: int,
) -> tuple[AdmissionDecision, ...]:
    """Replay snapshots with exactly the state transition used online."""

    state = AdmissionControllerState.initial(initial_cap)
    decisions: list[AdmissionDecision] = []
    for snapshot in snapshots:
        decision = decide_admission_cap(config, envelope, snapshot, state)
        decisions.append(decision)
        state = decision.next_state
    return tuple(decisions)
