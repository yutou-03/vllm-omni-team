from __future__ import annotations

import pytest

from vllm_omni.scheduling.signals import (
    NVMLGPUActiveSampler,
    QueueGrowthWindow,
    estimate_deadline_miss_risk,
    parse_visible_device_ids,
    token_budget_utilization_ratio,
)


class _Utilization:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _FakeNVML:
    def __init__(self, utilization: dict[object, int]) -> None:
        self.utilization = utilization
        self.init_calls = 0
        self.query_calls = 0

    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlDeviceGetHandleByIndex(self, device_id: int) -> object:
        return ("index", device_id)

    def nvmlDeviceGetHandleByUUID(self, device_id: str) -> object:
        return ("uuid", device_id)

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> _Utilization:
        self.query_calls += 1
        return _Utilization(self.utilization[handle])


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


def test_visible_device_parser_preserves_indices_and_uuids() -> None:
    assert parse_visible_device_ids("0, 4, GPU-deadbeef") == (
        0,
        4,
        "GPU-deadbeef",
    )
    assert parse_visible_device_ids("") == ()


def test_nvml_gpu_active_sampler_averages_stage_devices_and_caches() -> None:
    api = _FakeNVML({("index", 0): 40, ("uuid", "GPU-x"): 80})
    sampler = NVMLGPUActiveSampler(
        device_ids=(0, "GPU-x"),
        min_interval_s=0.1,
        nvml_api=api,
    )

    first = sampler.sample(captured_monotonic_s=10.0)
    cached = sampler.sample(captured_monotonic_s=10.05)
    refreshed = sampler.sample(captured_monotonic_s=10.2)

    assert first.ratio == pytest.approx(0.6)
    assert first.source == "nvml_mean_visible_devices"
    assert first.device_count == 2
    assert first.sample_age_ms == 0.0
    assert cached.ratio == first.ratio
    assert cached.sample_age_ms == pytest.approx(50.0)
    assert refreshed.sample_age_ms == 0.0
    assert api.init_calls == 1
    assert api.query_calls == 4


def test_nvml_gpu_active_sampler_fails_closed_without_stage_devices() -> None:
    sampler = NVMLGPUActiveSampler(
        device_ids=(),
        min_interval_s=0.1,
        nvml_api=_FakeNVML({}),
    )

    sample = sampler.sample(captured_monotonic_s=10.0)

    assert sample.ratio is None
    assert sample.unavailable_reason == "no_visible_stage_devices"
