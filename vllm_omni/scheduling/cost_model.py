# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead, hash-bound milestone cost lookup for Stage-DDL."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from vllm_omni.scheduling.contracts import RequestPath, SLOMilestone
from vllm_omni.scheduling.deadline import MilestoneCostEstimate


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIZE_BUCKETS = ("short", "long")
_STAGE_SCOPE = {
    RequestPath.TEXT: {
        0: (SLOMilestone.TTFT, SLOMilestone.E2E),
    },
    RequestPath.AUDIO: {
        0: (SLOMilestone.TTFT, SLOMilestone.TTFP, SLOMilestone.E2E),
        1: (SLOMilestone.TTFP, SLOMilestone.E2E),
        2: (SLOMilestone.TTFP, SLOMilestone.E2E),
    },
}
_EXPECTED_KEYS = frozenset(
    (path, size, stage_id, milestone)
    for path, stages in _STAGE_SCOPE.items()
    for size in _SIZE_BUCKETS
    for stage_id, milestones in stages.items()
    for milestone in milestones
)
_GAP_MILESTONES = frozenset(
    {SLOMilestone.TEXT_GAP, SLOMilestone.AUDIO_GAP}
)


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _path(value: RequestPath | str) -> RequestPath:
    if isinstance(value, RequestPath):
        return value
    try:
        return RequestPath(str(value).lower())
    except ValueError as error:
        raise ValueError(f"unknown request path {value!r}") from error


def _milestone(value: SLOMilestone | str) -> SLOMilestone:
    if isinstance(value, SLOMilestone):
        return value
    try:
        return SLOMilestone(str(value).lower())
    except ValueError as error:
        raise ValueError(f"unknown SLO milestone {value!r}") from error


@dataclass(frozen=True, slots=True)
class _StaticCostCell:
    current_stage_service_ms: float
    downstream_residual_ms: float


@dataclass(frozen=True, slots=True)
class StaticMilestoneCostModel:
    """Pre-indexed static P95 model; prediction performs one dictionary lookup."""

    artifact_sha256: str
    deploy_profile_id: str
    quantile: float
    input_token_cutoff: int
    milestone_guard_ms: Mapping[SLOMilestone, float]
    cell_index: Mapping[
        tuple[RequestPath, str, int, SLOMilestone], _StaticCostCell
    ]
    use_conformal_reserve: bool = False
    cost_source: str = "static_component_p95_plus_first_output_guard"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "milestone_guard_ms",
            MappingProxyType(dict(self.milestone_guard_ms)),
        )
        object.__setattr__(
            self,
            "cell_index",
            MappingProxyType(dict(self.cell_index)),
        )

    @property
    def cost_model_version(self) -> str:
        return f"p3-static-milestone-p95:{self.artifact_sha256[:16]}"

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        artifact_sha256: str,
        input_token_cutoff: int,
        milestone_guard_ms: Mapping[SLOMilestone | str, float],
    ) -> StaticMilestoneCostModel:
        if not isinstance(artifact, Mapping):
            raise ValueError("cost model artifact must be an object")
        if artifact.get("schema_version") != 1:
            raise ValueError("cost model artifact schema_version must be 1")
        if artifact.get("artifact_type") != "p3_static_milestone_cost_model":
            raise ValueError("unexpected cost model artifact_type")
        if not _SHA256_RE.fullmatch(str(artifact_sha256)):
            raise ValueError("artifact_sha256 must be 64 lowercase hex characters")
        if (
            isinstance(input_token_cutoff, bool)
            or not isinstance(input_token_cutoff, int)
            or input_token_cutoff <= 0
        ):
            raise ValueError("input_token_cutoff must be a positive integer")
        deploy_profile_id = str(artifact.get("deploy_profile_id", ""))
        if not deploy_profile_id.strip():
            raise ValueError("deploy_profile_id must not be empty")
        quantile = _finite_nonnegative(
            artifact.get("quantile"), field="quantile"
        )
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")

        expected_guards = {
            SLOMilestone.TTFT,
            SLOMilestone.TTFP,
            SLOMilestone.E2E,
        }
        guards: dict[SLOMilestone, float] = {}
        for raw_milestone, value in milestone_guard_ms.items():
            milestone = _milestone(raw_milestone)
            if milestone in guards:
                raise ValueError(f"duplicate guard for {milestone.value}")
            guards[milestone] = _finite_nonnegative(
                value,
                field=f"milestone_guard_ms.{milestone.value}",
            )
        if set(guards) != expected_guards:
            missing = sorted(item.value for item in expected_guards - guards.keys())
            extra = sorted(item.value for item in guards.keys() - expected_guards)
            raise ValueError(
                f"milestone_guard_ms mismatch: missing={missing}, extra={extra}"
            )

        raw_cells = artifact.get("cells")
        if not isinstance(raw_cells, list):
            raise ValueError("cost model cells must be a list")
        cells: dict[
            tuple[RequestPath, str, int, SLOMilestone], _StaticCostCell
        ] = {}
        for index, raw_cell in enumerate(raw_cells):
            if not isinstance(raw_cell, Mapping):
                raise ValueError(f"cells[{index}] must be an object")
            raw_key = raw_cell.get("key")
            if not isinstance(raw_key, Mapping):
                raise ValueError(f"cells[{index}].key must be an object")
            path = _path(raw_key.get("request_path"))
            size = str(raw_key.get("size_bucket", ""))
            if size not in _SIZE_BUCKETS:
                raise ValueError(f"cells[{index}] has invalid size_bucket")
            stage_id = raw_key.get("stage_id")
            if (
                isinstance(stage_id, bool)
                or not isinstance(stage_id, int)
                or stage_id not in (0, 1, 2)
            ):
                raise ValueError(f"cells[{index}] has invalid stage_id")
            milestone = _milestone(raw_key.get("milestone"))
            key = (path, size, stage_id, milestone)
            if key in cells:
                raise ValueError(f"duplicate cell {key!r}")
            current = _finite_nonnegative(
                raw_cell.get("current_stage_service_p95_ms"),
                field=f"cells[{index}].current_stage_service_p95_ms",
            )
            downstream = _finite_nonnegative(
                raw_cell.get("downstream_residual_p95_ms"),
                field=f"cells[{index}].downstream_residual_p95_ms",
            )
            component = _finite_nonnegative(
                raw_cell.get("component_budget_ms"),
                field=f"cells[{index}].component_budget_ms",
            )
            if not math.isclose(
                current + downstream,
                component,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"cells[{index}] component budget does not add up")
            cells[key] = _StaticCostCell(current, downstream)
        missing = sorted(
            _EXPECTED_KEYS - cells.keys(),
            key=lambda key: tuple(str(item) for item in key),
        )
        extra = sorted(
            cells.keys() - _EXPECTED_KEYS,
            key=lambda key: tuple(str(item) for item in key),
        )
        if missing:
            raise ValueError(f"cost model is missing cells: {missing!r}")
        if extra:
            raise ValueError(f"cost model has unsupported cells: {extra!r}")

        return cls(
            artifact_sha256=str(artifact_sha256),
            deploy_profile_id=deploy_profile_id,
            quantile=quantile,
            input_token_cutoff=input_token_cutoff,
            milestone_guard_ms=guards,
            cell_index=cells,
        )

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        input_token_cutoff: int,
        milestone_guard_ms: Mapping[SLOMilestone | str, float],
    ) -> StaticMilestoneCostModel:
        model_path = Path(path)
        payload = model_path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "milestone cost model SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            artifact = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("milestone cost model is not valid UTF-8 JSON") from error
        return cls.from_artifact(
            artifact,
            artifact_sha256=actual_sha256,
            input_token_cutoff=input_token_cutoff,
            milestone_guard_ms=milestone_guard_ms,
        )

    def predict(
        self,
        *,
        request_path: RequestPath | str,
        stage_id: int,
        milestone: SLOMilestone | str,
        num_prompt_tokens: int,
    ) -> MilestoneCostEstimate:
        if (
            isinstance(num_prompt_tokens, bool)
            or not isinstance(num_prompt_tokens, int)
            or num_prompt_tokens <= 0
        ):
            raise ValueError("num_prompt_tokens must be a positive integer")
        normalized_path = _path(request_path)
        normalized_milestone = _milestone(milestone)
        if normalized_milestone in _GAP_MILESTONES:
            raise ValueError("recurring gap milestones are not one-shot cost cells")
        size = "short" if num_prompt_tokens <= self.input_token_cutoff else "long"
        key = (normalized_path, size, stage_id, normalized_milestone)
        cell = self.cell_index.get(key)
        if cell is None:
            raise ValueError(f"no milestone cost cell for {key!r}")
        return MilestoneCostEstimate(
            current_stage_service_ms=cell.current_stage_service_ms,
            downstream_service_ms=cell.downstream_residual_ms,
            downstream_transfer_ms=0.0,
            reserve_ms=self.milestone_guard_ms[normalized_milestone],
        )
