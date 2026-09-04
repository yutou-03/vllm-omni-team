from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_omni.scheduling.admission import (
    AdmissionAction,
    AdmissionControllerConfig,
    AdmissionControllerState,
    StaticConcurrencyEnvelope,
    decide_admission_cap,
    replay_admission_trace,
)
from vllm_omni.scheduling.contracts import StageRuntimeSnapshot


def _snapshot(**overrides) -> StageRuntimeSnapshot:
    values = {
        "schema_version": 2,
        "stage_id": 0,
        "captured_monotonic_s": 100.0,
        "control_epoch": 1,
        "running_requests": 2,
        "waiting_requests": 2,
        "input_waiting_requests": 7,
        "physical_max_num_seqs": 8,
        "effective_max_num_seqs": 2,
        "token_budget_total": 4096,
        "token_budget_available": 1024,
        "kv_cache_usage_ratio": 0.4,
        "gpu_active_ratio": 0.5,
        "gpu_active_sample_age_ms": 0.0,
        "gpu_active_source": "nvml_mean_visible_devices",
        "gpu_active_device_count": 1,
        "service_time_p95_ms": 10.0,
        "deadline_miss_risk": 0.1,
        "deadline_miss_risk_population": 2,
        "deadline_miss_risk_scope": "runnable_running",
    }
    values.update(overrides)
    return StageRuntimeSnapshot(**values)


def _config(**overrides) -> AdmissionControllerConfig:
    values = {
        "min_active": 1,
        "max_active": 8,
        "increase_hold": 2,
        "decrease_hold": 2,
        "cooldown_epochs": 1,
        "increase_step": 1,
        "decrease_factor": 0.5,
        "gpu_low_watermark": 0.6,
        "kv_low_watermark": 0.6,
        "kv_high_watermark": 0.85,
        "deadline_risk_low_watermark": 0.2,
        "deadline_risk_high_watermark": 0.6,
        "queue_growth_high_rps": 5.0,
        "connector_backlog_high": 2,
        "require_downstream_pressure": False,
    }
    values.update(overrides)
    return AdmissionControllerConfig(**values)


def _envelope(**overrides) -> StaticConcurrencyEnvelope:
    values = {
        "schema_version": 1,
        "artifact_id": "p4-test-envelope",
        "safe_max_active_by_stage": {0: 6, 1: 4, 2: 3},
        "service_p95_limit_ms_by_stage": {0: 20.0, 1: 30.0, 2: 40.0},
    }
    values.update(overrides)
    return StaticConcurrencyEnvelope(**values)


def test_envelope_and_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="safe cap.*positive integer"):
        _envelope(safe_max_active_by_stage={0: 0})
    with pytest.raises(ValueError, match="low watermark.*high watermark"):
        _config(kv_low_watermark=0.9, kv_high_watermark=0.8)
    with pytest.raises(ValueError, match="decrease_factor must be in"):
        _config(decrease_factor=1.0)


def test_controller_rejects_legacy_raw_queue_pressure_semantics() -> None:
    with pytest.raises(ValueError, match="pressure schema version 2"):
        decide_admission_cap(
            _config(),
            _envelope(),
            _snapshot(schema_version=1),
            AdmissionControllerState.initial(2),
        )


def test_safe_low_pressure_increases_only_after_hold() -> None:
    config = _config()
    envelope = _envelope()
    state = AdmissionControllerState.initial(2)

    first = decide_admission_cap(config, envelope, _snapshot(), state)
    assert first.action is AdmissionAction.HOLD
    assert first.new_cap == 2
    assert first.next_state.increase_streak == 1

    second = decide_admission_cap(
        config,
        envelope,
        _snapshot(control_epoch=2, captured_monotonic_s=100.1),
        first.next_state,
    )
    assert second.action is AdmissionAction.INCREASE
    assert second.new_cap == 3
    assert second.trigger_reasons == ("safe_headroom_with_demand",)


def test_missing_required_signal_blocks_increase_without_implying_pressure() -> None:
    decision = decide_admission_cap(
        _config(increase_hold=1),
        _envelope(),
        _snapshot(
            gpu_active_ratio=None,
            gpu_active_sample_age_ms=None,
            gpu_active_source=None,
            gpu_active_device_count=None,
        ),
        AdmissionControllerState.initial(2),
    )

    assert decision.action is AdmissionAction.HOLD
    assert decision.new_cap == 2
    assert decision.trigger_reasons == ()
    assert decision.blocked_reasons == ("missing:gpu_active_ratio",)
    assert decision.next_state.decrease_streak == 0


def test_required_downstream_pressure_blocks_increase_when_missing() -> None:
    decision = decide_admission_cap(
        _config(
            increase_hold=1,
            require_downstream_pressure=True,
            connector_backlog_high=None,
        ),
        _envelope(),
        _snapshot(),
        AdmissionControllerState.initial(2),
    )

    assert decision.action is AdmissionAction.HOLD
    assert decision.trigger_reasons == ()
    assert decision.blocked_reasons == (
        "missing:downstream_effective_cap",
        "missing:downstream_waiting_requests",
    )


def test_input_waiting_requests_are_not_runnable_demand() -> None:
    decision = decide_admission_cap(
        _config(increase_hold=1),
        _envelope(),
        _snapshot(waiting_requests=0, input_waiting_requests=99),
        AdmissionControllerState.initial(2),
    )

    assert decision.action is AdmissionAction.HOLD
    assert "no_runnable_waiting_demand" in decision.blocked_reasons


def test_high_pressure_uses_aimd_decrease_and_never_evicts_running() -> None:
    config = _config(decrease_hold=2)
    pressure = _snapshot(
        running_requests=5,
        effective_max_num_seqs=6,
        kv_cache_usage_ratio=0.9,
        deadline_miss_risk=0.8,
    )
    state = AdmissionControllerState.initial(6)

    first = decide_admission_cap(config, _envelope(), pressure, state)
    second = decide_admission_cap(
        config,
        _envelope(),
        replace(pressure, control_epoch=2, captured_monotonic_s=100.1),
        first.next_state,
    )

    assert second.action is AdmissionAction.DECREASE
    assert second.new_cap == 3
    assert second.new_cap < pressure.running_requests
    assert second.drain_only is True
    assert second.evicted_requests == 0
    assert second.trigger_reasons == (
        "deadline_risk_high",
        "kv_high_watermark",
    )


def test_cap_is_clamped_to_all_hard_upper_bounds() -> None:
    decision = decide_admission_cap(
        _config(max_active=7),
        _envelope(
            safe_max_active_by_stage={0: 5},
            service_p95_limit_ms_by_stage={0: 20.0},
        ),
        _snapshot(
            physical_max_num_seqs=4,
            effective_max_num_seqs=4,
            running_requests=4,
        ),
        AdmissionControllerState.initial(7),
    )

    assert decision.action is AdmissionAction.CLAMP
    assert decision.new_cap == 4
    assert decision.safe_cap_ceiling == 4
    assert decision.drain_only is True


def test_cooldown_prevents_high_frequency_reversal() -> None:
    config = _config(increase_hold=1, decrease_hold=1, cooldown_epochs=2)
    envelope = _envelope()
    increased = decide_admission_cap(
        config,
        envelope,
        _snapshot(),
        AdmissionControllerState.initial(2),
    )
    assert increased.new_cap == 3

    pressure = _snapshot(
        control_epoch=2,
        captured_monotonic_s=100.1,
        running_requests=3,
        effective_max_num_seqs=3,
        kv_cache_usage_ratio=0.95,
    )
    held = decide_admission_cap(
        config, envelope, pressure, increased.next_state
    )
    assert held.action is AdmissionAction.HOLD
    assert held.new_cap == 3
    assert held.blocked_reasons == ("cooldown_active",)

    decreased = decide_admission_cap(
        config,
        envelope,
        replace(pressure, control_epoch=3, captured_monotonic_s=100.2),
        held.next_state,
    )
    assert decreased.action is AdmissionAction.DECREASE
    assert decreased.new_cap == 1


def test_replay_is_deterministic_and_rejects_stale_epochs() -> None:
    config = _config(increase_hold=1)
    envelope = _envelope()
    snapshots = [
        _snapshot(control_epoch=1),
        _snapshot(
            control_epoch=2,
            captured_monotonic_s=100.1,
            running_requests=3,
            effective_max_num_seqs=3,
        ),
    ]

    first = replay_admission_trace(
        config, envelope, snapshots, initial_cap=2
    )
    second = replay_admission_trace(
        config, envelope, snapshots, initial_cap=2
    )
    assert first == second
    assert [decision.new_cap for decision in first] == [3, 4]

    with pytest.raises(ValueError, match="control_epoch must increase"):
        decide_admission_cap(
            config,
            envelope,
            snapshots[0],
            first[0].next_state,
        )


def test_explicit_pressure_signals_are_recorded_in_stable_order() -> None:
    decision = decide_admission_cap(
        _config(decrease_hold=1),
        _envelope(),
        _snapshot(
            service_time_p95_ms=25.0,
            queue_growth_rate_rps=7.0,
            downstream_waiting_requests=3,
            downstream_effective_cap=3,
            downstream_stage_id=1,
            downstream_control_epoch=7,
            downstream_sample_age_ms=25.0,
            downstream_source="engine_core_utility",
            connector_backlog=2,
            connector_backlog_scope="sender_pending_save_and_inflight",
        ),
        AdmissionControllerState.initial(4),
    )

    assert decision.action is AdmissionAction.DECREASE
    assert decision.trigger_reasons == (
        "connector_backlog_high",
        "downstream_congested",
        "queue_growth_high",
        "service_p95_above_envelope",
    )
