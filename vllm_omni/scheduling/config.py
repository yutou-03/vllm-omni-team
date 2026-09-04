# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Typed, fail-closed configuration for SLO-aware scheduling experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from vllm_omni.recovery import PreemptionRecoveryMode
from vllm_omni.scheduling.admission import AdmissionControllerConfig


class SloSchedulingMode(str, Enum):
    """Versioned experiment modes exposed by the scheduling control plane."""

    NATIVE = "native"
    GLOBAL_EDF_FIXED = "global_edf_fixed"
    SRPF_FIXED = "srpf_fixed"
    PATH_RR_FIXED = "path_rr_fixed"
    STAGE_DDL_FIXED = "stage_ddl_fixed"
    CLOCKWORK_LATEST_START_FIXED = "clockwork_latest_start_fixed"
    STAGE_DDL_DYNAMIC_ADMISSION = "stage_ddl_dynamic_admission"
    STAGE_DDL_DYNAMIC_ADMISSION_FAIR = "stage_ddl_dynamic_admission_fair"
    PRIORITY_ONLY_PREEMPT = "priority_only_preempt"
    BUFFER_COST_PREEMPT = "buffer_cost_preempt"
    ADAPTIVE_RECOVERY = "adaptive_recovery"


_PROFILED_DEADLINE_MODES = frozenset(
    {
        SloSchedulingMode.STAGE_DDL_FIXED,
        SloSchedulingMode.CLOCKWORK_LATEST_START_FIXED,
        SloSchedulingMode.STAGE_DDL_DYNAMIC_ADMISSION,
        SloSchedulingMode.STAGE_DDL_DYNAMIC_ADMISSION_FAIR,
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _enum_value(enum_type: type[Enum], raw: Any, *, field_name: str) -> Enum:
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as error:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from error


def _reject_unknown_fields(
    raw: Mapping[str, Any],
    *,
    allowed: set[str],
    scope: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} fields: {unknown}")


def _nonnegative_finite(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class DynamicAdmissionConfig:
    """Hash-bound controller configuration for selected pipeline stages."""

    envelope_path: str | None = None
    envelope_sha256: str | None = None
    target_stage_ids: tuple[int, ...] = ()
    initial_active_by_stage: Mapping[int, int] = field(default_factory=dict)
    controller: AdmissionControllerConfig = AdmissionControllerConfig()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_active_by_stage",
            MappingProxyType(dict(self.initial_active_by_stage)),
        )

    @property
    def is_configured(self) -> bool:
        return self.envelope_path is not None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> DynamicAdmissionConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(
                "slo_scheduling.stage_ddl.dynamic_admission must be an object"
            )
        expected = {
            "envelope_path",
            "envelope_sha256",
            "target_stage_ids",
            "initial_active_by_stage",
            "controller",
        }
        _reject_unknown_fields(
            raw,
            allowed=expected,
            scope="slo_scheduling.stage_ddl.dynamic_admission",
        )
        missing = sorted(expected - raw.keys())
        if missing:
            raise ValueError(
                "slo_scheduling.stage_ddl.dynamic_admission missing fields: "
                f"{missing}"
            )
        path = str(raw["envelope_path"])
        if not path.strip():
            raise ValueError("dynamic admission envelope_path must not be empty")
        sha256 = str(raw["envelope_sha256"])
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError(
                "dynamic admission envelope_sha256 must be 64 lowercase hex "
                "characters"
            )
        raw_targets = raw["target_stage_ids"]
        if isinstance(raw_targets, (str, bytes)) or not isinstance(
            raw_targets, Sequence
        ):
            raise ValueError("dynamic admission target_stage_ids must be a list")
        targets: list[int] = []
        for value in raw_targets:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "dynamic admission target_stage_ids must contain "
                    "non-negative integers"
                )
            targets.append(value)
        if not targets or len(targets) != len(set(targets)):
            raise ValueError(
                "dynamic admission target_stage_ids must be non-empty and unique"
            )
        raw_initial = raw["initial_active_by_stage"]
        if not isinstance(raw_initial, Mapping):
            raise ValueError(
                "dynamic admission initial_active_by_stage must be an object"
            )
        initial: dict[int, int] = {}
        for raw_stage, raw_cap in raw_initial.items():
            try:
                stage = int(raw_stage)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "dynamic admission initial stage ids must be integers"
                ) from error
            if str(stage) != str(raw_stage) or stage < 0:
                raise ValueError(
                    "dynamic admission initial stage ids must be "
                    "non-negative integers"
                )
            if (
                isinstance(raw_cap, bool)
                or not isinstance(raw_cap, int)
                or raw_cap <= 0
            ):
                raise ValueError(
                    "dynamic admission initial caps must be positive integers"
                )
            initial[stage] = raw_cap
        if set(initial) != set(targets):
            raise ValueError(
                "dynamic admission initial_active_by_stage keys must exactly "
                "match target_stage_ids"
            )
        controller = AdmissionControllerConfig.from_mapping(raw["controller"])
        for stage, cap in initial.items():
            if not controller.min_active <= cap <= controller.max_active:
                raise ValueError(
                    f"dynamic admission initial cap for stage {stage} is "
                    "outside controller bounds"
                )
        return cls(
            envelope_path=path,
            envelope_sha256=sha256,
            target_stage_ids=tuple(targets),
            initial_active_by_stage=initial,
            controller=controller,
        )

    def to_dict(self) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        return {
            "envelope_path": self.envelope_path,
            "envelope_sha256": self.envelope_sha256,
            "target_stage_ids": list(self.target_stage_ids),
            "initial_active_by_stage": {
                str(stage): cap
                for stage, cap in self.initial_active_by_stage.items()
            },
            "controller": self.controller.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StageDDLConfig:
    """Hash-bound static milestone cost controls shared by every stage."""

    model_path: str | None = None
    model_sha256: str | None = None
    input_token_cutoff: int = 128
    milestone_guard_ms: Mapping[str, float] = field(
        default_factory=lambda: {"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0}
    )
    dynamic_admission: DynamicAdmissionConfig = DynamicAdmissionConfig()

    def __post_init__(self) -> None:
        guards = dict(self.milestone_guard_ms)
        object.__setattr__(self, "milestone_guard_ms", MappingProxyType(guards))

    @property
    def is_configured(self) -> bool:
        return self.model_path is not None and self.model_sha256 is not None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> StageDDLConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("slo_scheduling.stage_ddl must be an object")
        _reject_unknown_fields(
            raw,
            allowed={
                "model_path",
                "model_sha256",
                "input_token_cutoff",
                "milestone_guard_ms",
                "dynamic_admission",
            },
            scope="slo_scheduling.stage_ddl",
        )
        raw_path = raw.get("model_path")
        raw_sha256 = raw.get("model_sha256")
        if (raw_path is None) != (raw_sha256 is None):
            raise ValueError(
                "slo_scheduling.stage_ddl.model_path and model_sha256 "
                "must be provided together"
            )
        model_path = None
        model_sha256 = None
        if raw_path is not None:
            model_path = str(raw_path)
            if not model_path.strip():
                raise ValueError(
                    "slo_scheduling.stage_ddl.model_path must not be empty"
                )
            model_sha256 = str(raw_sha256)
            if not _SHA256_RE.fullmatch(model_sha256):
                raise ValueError(
                    "slo_scheduling.stage_ddl.model_sha256 must be 64 "
                    "lowercase hex characters"
                )
        cutoff = raw.get("input_token_cutoff", 128)
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
            raise ValueError(
                "slo_scheduling.stage_ddl.input_token_cutoff must be a "
                "positive integer"
            )
        raw_guards = raw.get(
            "milestone_guard_ms",
            {"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
        )
        if not isinstance(raw_guards, Mapping):
            raise ValueError(
                "slo_scheduling.stage_ddl.milestone_guard_ms must be an object"
            )
        expected = {"ttft", "ttfp", "e2e"}
        missing = sorted(expected - raw_guards.keys())
        extra = sorted(raw_guards.keys() - expected)
        if missing or extra:
            raise ValueError(
                "slo_scheduling.stage_ddl.milestone_guard_ms mismatch: "
                f"missing={missing}, extra={extra}"
            )
        guards = {
            milestone: _nonnegative_finite(
                raw_guards[milestone],
                field_name=(
                    "slo_scheduling.stage_ddl.milestone_guard_ms."
                    f"{milestone}"
                ),
            )
            for milestone in sorted(expected)
        }
        return cls(
            model_path=model_path,
            model_sha256=model_sha256,
            input_token_cutoff=cutoff,
            milestone_guard_ms=guards,
            dynamic_admission=DynamicAdmissionConfig.from_mapping(
                raw.get("dynamic_admission")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "input_token_cutoff": self.input_token_cutoff,
            "milestone_guard_ms": dict(self.milestone_guard_ms),
        }
        dynamic = self.dynamic_admission.to_dict()
        if dynamic is not None:
            result["dynamic_admission"] = dynamic
        return result


@dataclass(frozen=True, slots=True)
class SloRecoveryConfig:
    """Recovery mode selected for P0 and modes available to later planners."""

    mode: PreemptionRecoveryMode = PreemptionRecoveryMode.FULL_PIPELINE_RESTART
    allowed_modes: tuple[PreemptionRecoveryMode, ...] = (
        PreemptionRecoveryMode.FULL_PIPELINE_RESTART,
    )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> SloRecoveryConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("slo_scheduling.recovery must be an object")
        _reject_unknown_fields(
            raw,
            allowed={"mode", "allowed_modes"},
            scope="slo_scheduling.recovery",
        )
        mode = _enum_value(
            PreemptionRecoveryMode,
            raw.get("mode", PreemptionRecoveryMode.FULL_PIPELINE_RESTART.value),
            field_name="slo_scheduling.recovery.mode",
        )
        raw_allowed = raw.get(
            "allowed_modes",
            [PreemptionRecoveryMode.FULL_PIPELINE_RESTART.value],
        )
        if isinstance(raw_allowed, (str, bytes)) or not isinstance(
            raw_allowed, Sequence
        ):
            raise ValueError(
                "slo_scheduling.recovery.allowed_modes must be a list"
            )
        allowed_modes = tuple(
            _enum_value(
                PreemptionRecoveryMode,
                value,
                field_name="slo_scheduling.recovery.allowed_modes",
            )
            for value in raw_allowed
        )
        if not allowed_modes:
            raise ValueError(
                "slo_scheduling.recovery.allowed_modes must not be empty"
            )
        if len(allowed_modes) != len(set(allowed_modes)):
            raise ValueError(
                "slo_scheduling.recovery.allowed_modes must be unique"
            )
        if PreemptionRecoveryMode.FULL_PIPELINE_RESTART not in allowed_modes:
            raise ValueError(
                "slo_scheduling.recovery.allowed_modes must include "
                "full_pipeline_restart as the safe fallback"
            )
        if mode not in allowed_modes:
            raise ValueError(
                "slo_scheduling.recovery.mode must be included in allowed_modes"
            )
        return cls(
            mode=mode,  # type: ignore[arg-type]
            allowed_modes=allowed_modes,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "allowed_modes": [mode.value for mode in self.allowed_modes],
        }


@dataclass(frozen=True, slots=True)
class SloSchedulingConfig:
    """Master switch and named mode for reproducible scheduling runs."""

    enabled: bool = False
    mode: SloSchedulingMode = SloSchedulingMode.NATIVE
    recovery: SloRecoveryConfig = SloRecoveryConfig()
    stage_ddl: StageDDLConfig = StageDDLConfig()

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> SloSchedulingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("slo_scheduling must be an object")
        _reject_unknown_fields(
            raw,
            allowed={"enabled", "mode", "recovery", "stage_ddl"},
            scope="slo_scheduling",
        )
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("slo_scheduling.enabled must be a boolean")
        mode = _enum_value(
            SloSchedulingMode,
            raw.get("mode", SloSchedulingMode.NATIVE.value),
            field_name="slo_scheduling.mode",
        )
        stage_ddl = StageDDLConfig.from_mapping(raw.get("stage_ddl"))
        if (
            enabled
            and mode in _PROFILED_DEADLINE_MODES
            and not stage_ddl.is_configured
        ):
            raise ValueError(
                "slo_scheduling.stage_ddl.model_path and model_sha256 are "
                "required for an enabled profiled deadline mode"
            )
        dynamic_modes = {
            SloSchedulingMode.STAGE_DDL_DYNAMIC_ADMISSION,
            SloSchedulingMode.STAGE_DDL_DYNAMIC_ADMISSION_FAIR,
        }
        if enabled and mode in dynamic_modes and not (
            stage_ddl.dynamic_admission.is_configured
        ):
            raise ValueError(
                "slo_scheduling.stage_ddl.dynamic_admission is required "
                "for dynamic admission modes"
            )
        if mode not in dynamic_modes and (
            stage_ddl.dynamic_admission.is_configured
        ):
            raise ValueError(
                "slo_scheduling.stage_ddl.dynamic_admission is only valid "
                "for dynamic admission modes"
            )
        return cls(
            enabled=enabled,
            mode=mode,  # type: ignore[arg-type]
            recovery=SloRecoveryConfig.from_mapping(raw.get("recovery")),
            stage_ddl=stage_ddl,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "recovery": self.recovery.to_dict(),
            "stage_ddl": self.stage_ddl.to_dict(),
        }


def extract_slo_scheduling_config(runtime_config: Any) -> SloSchedulingConfig:
    """Read canonical SLO config from a config or vLLMConfig-like object."""

    if isinstance(runtime_config, SloSchedulingConfig):
        return runtime_config
    additional_config = getattr(runtime_config, "additional_config", None)
    if additional_config is None and isinstance(runtime_config, Mapping):
        additional_config = runtime_config.get("additional_config")
    if not isinstance(additional_config, Mapping):
        return SloSchedulingConfig()
    return SloSchedulingConfig.from_mapping(
        additional_config.get("slo_scheduling")
    )


def effective_preemption_recovery_mode(
    runtime_config: Any,
) -> PreemptionRecoveryMode:
    """Return baseline recovery while the master switch is disabled."""

    config = extract_slo_scheduling_config(runtime_config)
    if not config.enabled:
        return PreemptionRecoveryMode.FULL_PIPELINE_RESTART
    return config.recovery.mode
