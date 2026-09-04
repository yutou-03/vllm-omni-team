from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
)
from vllm_omni.recovery import PreemptionRecoveryMode
from vllm_omni.scheduling.config import (
    StageDDLConfig,
    SloSchedulingConfig,
    SloSchedulingMode,
    effective_preemption_recovery_mode,
)


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
