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
class StageDDLConfig:
    """Hash-bound static milestone cost controls shared by every stage."""

    model_path: str | None = None
    model_sha256: str | None = None
    input_token_cutoff: int = 128
    milestone_guard_ms: Mapping[str, float] = field(
        default_factory=lambda: {"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0}
    )

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "input_token_cutoff": self.input_token_cutoff,
            "milestone_guard_ms": dict(self.milestone_guard_ms),
        }


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
