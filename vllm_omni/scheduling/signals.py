# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead, replayable inputs for stage admission control."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable


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
