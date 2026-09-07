from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
)
from vllm_omni.recovery import PreemptionRecoveryMode
from vllm_omni.scheduling.config import (
    BoundedWaitFairnessConfig,
    DynamicAdmissionConfig,
    StageDDLConfig,
    SloSchedulingConfig,
    SloSchedulingMode,
    effective_preemption_recovery_mode,
)


def _dynamic_admission() -> dict:
    return {
        "envelope_path": "/models/p4-envelope.json",
        "envelope_sha256": "b" * 64,
        "target_stage_ids": [1],
        "initial_active_by_stage": {"1": 2},
        "controller": {
            "min_active": 1,
            "max_active": 2,
            "increase_hold": 50,
            "decrease_hold": 20,
            "cooldown_epochs": 100,
            "increase_step": 1,
            "decrease_factor": 0.5,
            "gpu_low_watermark": 0.6,
            "kv_low_watermark": 0.01,
            "kv_high_watermark": 0.9,
            "queue_growth_high_rps": None,
            "connector_backlog_high": 2,
            "require_downstream_pressure": True,
            "reset_to_safe_cap_when_fully_idle": False,
        },
    }


def test_slo_config_defaults_are_disabled_and_native():
    config = SloSchedulingConfig.from_mapping(None)

    assert config.enabled is False
    assert config.mode is SloSchedulingMode.NATIVE
    assert config.recovery.mode is PreemptionRecoveryMode.FULL_PIPELINE_RESTART
    assert config.recovery.allowed_modes == (
        PreemptionRecoveryMode.FULL_PIPELINE_RESTART,
    )
    assert config.stage_ddl == StageDDLConfig()


def test_stage_ddl_config_parses_hash_bound_static_model():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "stage_ddl_fixed",
            "stage_ddl": {
                "model_path": "/models/p3.json",
                "model_sha256": "a" * 64,
                "input_token_cutoff": 128,
                "milestone_guard_ms": {
                    "ttft": 5.0,
                    "ttfp": 5.0,
                    "e2e": 0.0,
                },
            },
        }
    )

    assert config.stage_ddl.model_path == "/models/p3.json"
    assert config.stage_ddl.model_sha256 == "a" * 64
    assert config.stage_ddl.input_token_cutoff == 128
    assert config.stage_ddl.milestone_guard_ms == {
        "ttft": 5.0,
        "ttfp": 5.0,
        "e2e": 0.0,
    }
    assert config.to_dict()["stage_ddl"] == {
        "model_path": "/models/p3.json",
        "model_sha256": "a" * 64,
        "input_token_cutoff": 128,
        "milestone_guard_ms": {"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
    }


def test_clockwork_latest_start_reuses_hash_bound_cost_model():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "clockwork_latest_start_fixed",
            "stage_ddl": {
                "model_path": "/models/p3.json",
                "model_sha256": "a" * 64,
            },
        }
    )

    assert config.mode is SloSchedulingMode.CLOCKWORK_LATEST_START_FIXED
    assert config.stage_ddl.is_configured


def test_dynamic_admission_config_is_hash_bound_and_typed():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "stage_ddl_dynamic_admission",
            "stage_ddl": {
                "model_path": "/models/p3.json",
                "model_sha256": "a" * 64,
                "dynamic_admission": _dynamic_admission(),
            },
        }
    )

    dynamic = config.stage_ddl.dynamic_admission
    assert isinstance(dynamic, DynamicAdmissionConfig)
    assert dynamic.envelope_path == "/models/p4-envelope.json"
    assert dynamic.envelope_sha256 == "b" * 64
    assert dynamic.target_stage_ids == (1,)
    assert dynamic.initial_active_by_stage == {1: 2}
    assert dynamic.controller.max_active == 2
    assert dynamic.controller.require_downstream_pressure is True
    assert config.to_dict()["stage_ddl"]["dynamic_admission"] == (
        _dynamic_admission()
    )


def test_bounded_wait_fair_fixed_config_is_typed_and_stage_scoped():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "stage_ddl_fair_fixed",
            "stage_ddl": {
                "model_path": "/models/p3.json",
                "model_sha256": "a" * 64,
                "fairness": {
                    "bounded_wait_ms_by_stage": {"0": 2000.0},
                },
            },
        }
    )

    assert config.mode is SloSchedulingMode.STAGE_DDL_FAIR_FIXED
    assert isinstance(config.stage_ddl.fairness, BoundedWaitFairnessConfig)
    assert config.stage_ddl.fairness.bounded_wait_ms_by_stage == {0: 2000.0}
    assert config.stage_ddl.fairness.threshold_ms(0) == 2000.0
    assert config.stage_ddl.fairness.threshold_ms(1) is None
    assert config.to_dict()["stage_ddl"]["fairness"] == {
        "bounded_wait_ms_by_stage": {"0": 2000.0}
    }


@pytest.mark.parametrize(
    "raw, match",
    [
        (
            {
                "enabled": True,
                "mode": "stage_ddl_fair_fixed",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                },
            },
            "fairness.*required",
        ),
        (
            {
                "enabled": True,
                "mode": "stage_ddl_fixed",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                    "fairness": {
                        "bounded_wait_ms_by_stage": {"0": 2000.0},
                    },
                },
            },
            "fairness.*only valid",
        ),
        (
            {
                "enabled": True,
                "mode": "stage_ddl_fair_fixed",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                    "fairness": {"bounded_wait_ms_by_stage": {"0": 0.0}},
                },
            },
            "positive",
        ),
        (
            {
                "enabled": True,
                "mode": "stage_ddl_fair_fixed",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                    "fairness": {
                        "bounded_wait_ms_by_stage": {"stage-zero": 2000.0}
                    },
                },
            },
            "stage ids",
        ),
    ],
)
def test_invalid_bounded_wait_fairness_config_fails_closed(raw, match):
    with pytest.raises(ValueError, match=match):
        SloSchedulingConfig.from_mapping(raw)


def test_enabled_stage_local_mode_must_be_explicitly_allowed():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "adaptive_recovery",
            "recovery": {
                "mode": "stage_local_recompute",
                "allowed_modes": [
                    "full_pipeline_restart",
                    "stage_local_recompute",
                ],
            },
        }
    )

    assert config.enabled is True
    assert config.mode is SloSchedulingMode.ADAPTIVE_RECOVERY
    assert (
        effective_preemption_recovery_mode(config)
        is PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
    )


def test_disabled_master_switch_forces_baseline_recovery():
    config = SloSchedulingConfig.from_mapping(
        {
            "enabled": False,
            "mode": "adaptive_recovery",
            "recovery": {
                "mode": "stage_local_recompute",
                "allowed_modes": [
                    "full_pipeline_restart",
                    "stage_local_recompute",
                ],
            },
        }
    )

    assert (
        effective_preemption_recovery_mode(config)
        is PreemptionRecoveryMode.FULL_PIPELINE_RESTART
    )


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"unknown": True}, "unknown slo_scheduling fields"),
        (
            {
                "recovery": {
                    "mode": "stage_local_recompute",
                    "allowed_modes": ["full_pipeline_restart"],
                }
            },
            "must be included in allowed_modes",
        ),
        (
            {"recovery": {"allowed_modes": []}},
            "allowed_modes must not be empty",
        ),
        (
            {"recovery": {"allowed_modes": ["stage_local_recompute"]}},
            "must include full_pipeline_restart",
        ),
        (
            {"enabled": True, "mode": "stage_ddl_fixed"},
            "stage_ddl.model_path.*required",
        ),
        (
            {"enabled": True, "mode": "clockwork_latest_start_fixed"},
            "stage_ddl.model_path.*required",
        ),
        (
            {
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "bad",
                }
            },
            "model_sha256",
        ),
        (
            {"stage_ddl": {"milestone_guard_ms": {"ttft": 5.0}}},
            "milestone_guard_ms.*missing",
        ),
        (
            {
                "enabled": True,
                "mode": "stage_ddl_dynamic_admission",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                },
            },
            "dynamic_admission.*required",
        ),
        (
            {
                "enabled": True,
                "mode": "stage_ddl_fixed",
                "stage_ddl": {
                    "model_path": "/models/p3.json",
                    "model_sha256": "a" * 64,
                    "dynamic_admission": _dynamic_admission(),
                },
            },
            "dynamic_admission.*only valid",
        ),
    ],
)
def test_invalid_slo_config_fails_closed(raw, match):
    with pytest.raises(ValueError, match=match):
        SloSchedulingConfig.from_mapping(raw)


def test_deploy_config_propagates_canonical_switch_to_every_stage(tmp_path):
    deploy_path = tmp_path / "qwen3_omni.yaml"
    deploy_path.write_text(
        """
async_chunk: true
slo_scheduling:
  enabled: true
  mode: adaptive_recovery
  recovery:
    mode: stage_local_recompute
    allowed_modes: [full_pipeline_restart, stage_local_recompute]
stages:
  - stage_id: 0
  - stage_id: 1
  - stage_id: 2
""".strip(),
        encoding="utf-8",
    )

    deploy = load_deploy_config(deploy_path)
    stages = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, deploy)

    assert deploy.slo_scheduling.enabled is True
    canonical = deploy.slo_scheduling.to_dict()
    for stage in stages:
        assert stage.yaml_engine_args["additional_config"]["slo_scheduling"] == (
            canonical
        )


def test_cli_default_additional_config_is_a_deploy_override_field():
    """The parser's default {} must be nullified instead of erasing YAML."""
    from vllm_omni.config.stage_config import deploy_override_field_names

    assert "additional_config" in deploy_override_field_names()


def test_vllm_config_resolves_typed_recovery_mode():
    config = SimpleNamespace(
        additional_config={
            "slo_scheduling": {
                "enabled": True,
                "mode": "adaptive_recovery",
                "recovery": {
                    "mode": "stage_local_recompute",
                    "allowed_modes": [
                        "full_pipeline_restart",
                        "stage_local_recompute",
                    ],
                },
            }
        }
    )

    assert (
        effective_preemption_recovery_mode(config)
        is PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
    )
