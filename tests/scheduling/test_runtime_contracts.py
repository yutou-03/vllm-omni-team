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
        "token_budget_utilization_ratio": 0.75,
        "kv_cache_usage_ratio": 0.75,
        "gpu_active_ratio": 0.9,
        "gpu_active_sample_age_ms": 12.0,
        "gpu_active_source": "nvml_mean_visible_devices",
        "gpu_active_device_count": 1,
        "service_time_ewma_ms": 12.5,
        "service_time_p95_ms": 18.0,
        "queue_growth_rate_rps": 1.25,
        "downstream_waiting_requests": 5,
        "downstream_effective_cap": 3,
        "downstream_stage_id": 2,
        "downstream_control_epoch": 11,
        "downstream_sample_age_ms": 25.0,
        "downstream_source": "engine_core_utility",
        "connector_backlog": 2,
        "connector_backlog_scope": "sender_pending_save_and_inflight",
        "deadline_miss_risk": 0.2,
        "deadline_miss_risk_population": 5,
        "deadline_miss_risk_scope": "runnable_running",
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
        "token_budget_utilization_ratio": 0.75,
        "kv_cache_usage_ratio": 0.75,
        "gpu_active_ratio": 0.9,
        "gpu_active_sample_age_ms": 12.0,
        "gpu_active_source": "nvml_mean_visible_devices",
        "gpu_active_device_count": 1,
        "service_time_ewma_ms": 12.5,
        "service_time_p95_ms": 18.0,
        "queue_growth_rate_rps": 1.25,
        "downstream_waiting_requests": 5,
        "downstream_effective_cap": 3,
        "downstream_stage_id": 2,
        "downstream_control_epoch": 11,
        "downstream_sample_age_ms": 25.0,
        "downstream_source": "engine_core_utility",
        "connector_backlog": 2,
        "connector_backlog_scope": "sender_pending_save_and_inflight",
        "deadline_miss_risk": 0.2,
        "deadline_miss_risk_population": 5,
        "deadline_miss_risk_scope": "runnable_running",
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
        (
            {"deadline_miss_risk_scope": "all_queued"},
            "deadline_miss_risk_scope must be runnable_running",
        ),
        (
            {"token_budget_utilization_ratio": 1.1},
            "token_budget_utilization_ratio must be in",
        ),
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
        gpu_active_sample_age_ms=None,
        gpu_active_source=None,
        gpu_active_device_count=None,
        service_time_ewma_ms=None,
        service_time_p95_ms=None,
        queue_growth_rate_rps=None,
        downstream_waiting_requests=None,
        downstream_effective_cap=None,
        downstream_stage_id=None,
        downstream_control_epoch=None,
        downstream_sample_age_ms=None,
        downstream_source=None,
        connector_backlog=None,
        connector_backlog_scope=None,
        deadline_miss_risk=None,
        deadline_miss_risk_population=None,
        deadline_miss_risk_scope=None,
        last_admission_rejection=None,
    )

    fields = snapshot.to_event_fields()
    assert "kv_cache_usage_ratio" not in fields
    assert "downstream_waiting_requests" not in fields
    assert "last_admission_rejection" not in fields


def test_deadline_risk_metadata_must_be_all_present_or_all_absent():
    with pytest.raises(ValueError, match="deadline risk fields must be provided together"):
        _snapshot(deadline_miss_risk_population=None)


def test_gpu_active_metadata_must_be_all_present_or_all_absent():
    with pytest.raises(ValueError, match="GPU active fields must be provided together"):
        _snapshot(gpu_active_sample_age_ms=None)


def test_stage_runtime_snapshot_v2_marks_runnable_count_semantics():
    fields = _snapshot(schema_version=2).to_event_fields()

    assert fields["pressure_schema_version"] == 2
    assert fields["request_count_semantics"] == "runnable_excludes_input_blocked"


def test_v2_downstream_pressure_requires_complete_provenance():
    with pytest.raises(
        ValueError,
        match="downstream pressure fields must be provided together",
    ):
        _snapshot(schema_version=2, downstream_sample_age_ms=None)

    with pytest.raises(
        ValueError,
        match="downstream_source must be engine_core_utility",
    ):
        _snapshot(schema_version=2, downstream_source="local_queue_guess")


def test_v2_connector_backlog_requires_sender_scope():
    with pytest.raises(
        ValueError,
        match="connector backlog fields must be provided together",
    ):
        _snapshot(schema_version=2, connector_backlog_scope=None)

    with pytest.raises(
        ValueError,
        match="connector_backlog_scope must be sender_pending_save_and_inflight",
    ):
        _snapshot(schema_version=2, connector_backlog_scope="receiver_waiting")
