from __future__ import annotations

import pytest

from vllm_omni.scheduling.signals import (
    QueueGrowthWindow,
    estimate_deadline_miss_risk,
    token_budget_utilization_ratio,
)


def test_token_budget_utilization_is_exact_and_bounded() -> None:
    assert token_budget_utilization_ratio(total=4096, available=1024) == 0.75
    assert token_budget_utilization_ratio(total=0, available=0) is None
    with pytest.raises(ValueError, match="available must not exceed total"):
        token_budget_utilization_ratio(total=4, available=5)


def test_deadline_risk_counts_only_supplied_running_laxities() -> None:
    estimate = estimate_deadline_miss_risk([-0.1, 0.0, 0.2, 1.0])

    assert estimate.risk_ratio == 0.5
    assert estimate.population == 4
    assert estimate.scope == "runnable_running"


def test_empty_deadline_population_is_explicit_zero_not_missing() -> None:
    estimate = estimate_deadline_miss_risk([])

    assert estimate.risk_ratio == 0.0
    assert estimate.population == 0


def test_queue_growth_uses_bounded_window_and_allows_drain() -> None:
    window = QueueGrowthWindow(window_s=1.0)

    assert window.observe(captured_monotonic_s=10.0, waiting_requests=2) is None
    assert window.observe(captured_monotonic_s=10.5, waiting_requests=4) == pytest.approx(4.0)
    assert window.observe(captured_monotonic_s=11.0, waiting_requests=3) == pytest.approx(1.0)
    assert window.observe(captured_monotonic_s=11.5, waiting_requests=1) == pytest.approx(-3.0)


def test_queue_growth_rejects_clock_regression() -> None:
    window = QueueGrowthWindow(window_s=1.0)
    window.observe(captured_monotonic_s=10.0, waiting_requests=1)

    with pytest.raises(ValueError, match="must not regress"):
        window.observe(captured_monotonic_s=9.9, waiting_requests=2)
