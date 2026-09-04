from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.scheduling.contracts import StageRuntimeSnapshot


def _snapshot(**overrides):
    values = {
        "schema_version": 1,
        "stage_id": 1,
        "captured_monotonic_s": 100.0,
        "control_epoch": 7,
        "running_requests": 2,
        "waiting_requests": 3,
        "input_waiting_requests": 1,
        "physical_max_num_seqs": 8,
        "effective_max_num_seqs": 4,
        "token_budget_total": 4096,
        "token_budget_available": 1024,
        "kv_cache_usage_ratio": 0.75,
        "gpu_active_ratio": 0.9,
        "service_time_ewma_ms": 12.5,
        "service_time_p95_ms": 18.0,
        "queue_growth_rate_rps": 1.25,
        "downstream_waiting_requests": 5,
        "downstream_effective_cap": 3,
        "connector_backlog": 2,
        "deadline_miss_risk": 0.2,
        "last_admission_rejection": "effective_cap_reached",
    }
    values.update(overrides)
    return StageRuntimeSnapshot(**values)


def test_stage_runtime_snapshot_is_immutable_and_serializable():
    snapshot = _snapshot()

    assert snapshot.to_event_fields() == {
        "pressure_schema_version": 1,
        "control_epoch": 7,
        "running_requests": 2,
        "waiting_requests": 3,
        "input_waiting_requests": 1,
        "physical_max_num_seqs": 8,
        "effective_max_num_seqs": 4,
        "token_budget_total": 4096,
        "token_budget_available": 1024,
        "kv_cache_usage_ratio": 0.75,
        "gpu_active_ratio": 0.9,
        "service_time_ewma_ms": 12.5,
        "service_time_p95_ms": 18.0,
        "queue_growth_rate_rps": 1.25,
        "downstream_waiting_requests": 5,
        "downstream_effective_cap": 3,
        "connector_backlog": 2,
        "deadline_miss_risk": 0.2,
        "last_admission_rejection": "effective_cap_reached",
        "captured_monotonic_s": 100.0,
    }
    with pytest.raises(FrozenInstanceError):
        snapshot.control_epoch = 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage_id": -1}, "stage_id must be a non-negative integer"),
        (
            {"effective_max_num_seqs": 9},
            "effective_max_num_seqs must not exceed physical_max_num_seqs",
        ),
        (
            {"token_budget_available": 5000},
            "token_budget_available must not exceed token_budget_total",
        ),
        ({"kv_cache_usage_ratio": 1.1}, "kv_cache_usage_ratio must be in"),
        ({"deadline_miss_risk": -0.1}, "deadline_miss_risk must be in"),
        ({"service_time_p95_ms": -1}, "service_time_p95_ms must be non-negative"),
    ],
)
def test_stage_runtime_snapshot_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _snapshot(**overrides)


def test_stage_runtime_snapshot_omits_unavailable_optional_signals():
    snapshot = _snapshot(
        kv_cache_usage_ratio=None,
        gpu_active_ratio=None,
        service_time_ewma_ms=None,
        service_time_p95_ms=None,
        queue_growth_rate_rps=None,
        downstream_waiting_requests=None,
        downstream_effective_cap=None,
        connector_backlog=None,
        deadline_miss_risk=None,
        last_admission_rejection=None,
    )

    fields = snapshot.to_event_fields()
    assert "kv_cache_usage_ratio" not in fields
    assert "downstream_waiting_requests" not in fields
    assert "last_admission_rejection" not in fields
