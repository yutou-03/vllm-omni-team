# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead, replayable inputs for stage admission control."""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


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


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def token_budget_utilization_ratio(
    *, total: int, available: int
) -> float | None:
    """Return the fraction of the scheduler token budget consumed this tick."""

    total = _nonnegative_int(total, field_name="total")
    available = _nonnegative_int(available, field_name="available")
    if available > total:
        raise ValueError("available must not exceed total")
    if total == 0:
        return None
    return (total - available) / total


def parse_visible_device_ids(raw: str) -> tuple[str | int, ...]:
    """Parse physical CUDA device indices or UUIDs from stage visibility."""

    result: list[str | int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GPUActiveSample:
    """One stage-scoped hardware utilization sample or failure record."""

    ratio: float | None
    source: str | None
    device_count: int
    sample_age_ms: float | None
    unavailable_reason: str | None = None


class NVMLGPUActiveSampler:
    """Rate-limited mean NVML utilization over this stage's visible GPUs."""

    def __init__(
        self,
        *,
        device_ids: Sequence[str | int],
        min_interval_s: float = 0.1,
        nvml_api: Any | None = None,
    ) -> None:
        interval = _finite(min_interval_s, field_name="min_interval_s")
        if interval <= 0:
            raise ValueError("min_interval_s must be positive")
        self._device_ids = tuple(device_ids)
        self._min_interval_s = interval
        self._nvml_api = nvml_api
        self._handles: tuple[Any, ...] | None = None
        self._last_ratio: float | None = None
        self._last_sample_s: float | None = None
        self._unavailable_reason: str | None = None

    @classmethod
    def from_environment(
        cls, *, min_interval_s: float = 0.1
    ) -> NVMLGPUActiveSampler:
        return cls(
            device_ids=parse_visible_device_ids(
                os.environ.get("CUDA_VISIBLE_DEVICES", "")
            ),
            min_interval_s=min_interval_s,
        )

    def _initialize(self) -> bool:
        if self._handles is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        if not self._device_ids:
            self._unavailable_reason = "no_visible_stage_devices"
            return False
        try:
            if self._nvml_api is None:
                from vllm.third_party import pynvml

                self._nvml_api = pynvml
            self._nvml_api.nvmlInit()
            handles = []
            for device_id in self._device_ids:
                if isinstance(device_id, int):
                    handle = self._nvml_api.nvmlDeviceGetHandleByIndex(
                        device_id
                    )
                else:
                    handle = self._nvml_api.nvmlDeviceGetHandleByUUID(
                        device_id
                    )
                handles.append(handle)
            self._handles = tuple(handles)
        except Exception as error:  # pragma: no cover - backend-specific types
            self._unavailable_reason = f"nvml_init_failed:{type(error).__name__}"
            return False
        return True

    def sample(self, *, captured_monotonic_s: float) -> GPUActiveSample:
        captured = _finite(
            captured_monotonic_s,
            field_name="captured_monotonic_s",
        )
        if captured < 0:
            raise ValueError("captured_monotonic_s must be non-negative")
        if (
            self._last_sample_s is not None
            and captured < self._last_sample_s
        ):
            raise ValueError("captured_monotonic_s must not regress")
        if not self._initialize():
            return GPUActiveSample(
                ratio=None,
                source=None,
                device_count=len(self._device_ids),
                sample_age_ms=None,
                unavailable_reason=self._unavailable_reason,
            )
        if (
            self._last_sample_s is not None
            and self._last_ratio is not None
            and captured - self._last_sample_s < self._min_interval_s
        ):
            return GPUActiveSample(
                ratio=self._last_ratio,
                source="nvml_mean_visible_devices",
                device_count=len(self._device_ids),
                sample_age_ms=(captured - self._last_sample_s) * 1000.0,
            )
        assert self._handles is not None
        try:
            percentages = [
                float(
                    self._nvml_api.nvmlDeviceGetUtilizationRates(handle).gpu
                )
                for handle in self._handles
            ]
            if any(value < 0 or value > 100 for value in percentages):
                raise ValueError("NVML GPU utilization is outside [0, 100]")
            ratio = sum(percentages) / len(percentages) / 100.0
        except Exception as error:  # pragma: no cover - backend-specific types
            return GPUActiveSample(
                ratio=None,
                source=None,
                device_count=len(self._device_ids),
                sample_age_ms=None,
                unavailable_reason=f"nvml_query_failed:{type(error).__name__}",
            )
        self._last_ratio = ratio
        self._last_sample_s = captured
        return GPUActiveSample(
            ratio=ratio,
            source="nvml_mean_visible_devices",
            device_count=len(self._device_ids),
            sample_age_ms=0.0,
        )


@dataclass(frozen=True, slots=True)
class DeadlineRiskEstimate:
    """Fraction of a named request population with non-positive laxity."""

    risk_ratio: float
    population: int
    scope: str = "runnable_running"


def estimate_deadline_miss_risk(
    running_laxities_s: Iterable[float],
) -> DeadlineRiskEstimate:
    """Estimate risk only for running requests, never blocked/waiting demand."""

    laxities = [
        _finite(value, field_name="running laxity")
        for value in running_laxities_s
    ]
    population = len(laxities)
    at_risk = sum(value <= 0.0 for value in laxities)
    return DeadlineRiskEstimate(
        risk_ratio=0.0 if population == 0 else at_risk / population,
        population=population,
    )


class QueueGrowthWindow:
    """Estimate runnable waiting-queue growth over a bounded time window."""

    def __init__(self, *, window_s: float = 1.0) -> None:
        self.window_s = _finite(window_s, field_name="window_s")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        self._samples: deque[tuple[float, int]] = deque()

    def observe(
        self,
        *,
        captured_monotonic_s: float,
        waiting_requests: int,
    ) -> float | None:
        captured = _finite(
            captured_monotonic_s,
            field_name="captured_monotonic_s",
        )
        if captured < 0:
            raise ValueError("captured_monotonic_s must be non-negative")
        waiting = _nonnegative_int(
            waiting_requests,
            field_name="waiting_requests",
        )
        if self._samples and captured < self._samples[-1][0]:
            raise ValueError("captured_monotonic_s must not regress")
        if self._samples and captured == self._samples[-1][0]:
            self._samples[-1] = (captured, waiting)
        else:
            self._samples.append((captured, waiting))

        cutoff = captured - self.window_s
        while len(self._samples) >= 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if len(self._samples) < 2:
            return None
        oldest_time, oldest_waiting = self._samples[0]
        elapsed = captured - oldest_time
        if elapsed <= 0:
            return None
        return (waiting - oldest_waiting) / elapsed
