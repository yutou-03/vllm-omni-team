import hashlib
import json
from types import MappingProxyType

import pytest

from vllm_omni.scheduling.contracts import RequestPath, SLOMilestone
from vllm_omni.scheduling.cost_model import StaticMilestoneCostModel


SCOPES = {
    "text": {0: ("ttft", "e2e")},
    "audio": {
        0: ("ttft", "ttfp", "e2e"),
        1: ("ttfp", "e2e"),
        2: ("ttfp", "e2e"),
    },
}


def _artifact():
    cells = []
    for path, stages in SCOPES.items():
        for size in ("short", "long"):
            for stage, milestones in stages.items():
                for milestone in milestones:
                    cells.append(
                        {
                            "key": {
                                "request_path": path,
                                "size_bucket": size,
                                "stage_id": stage,
                                "milestone": milestone,
                            },
                            "sample_count": 36,
                            "current_stage_service_p95_ms": 10.0 + stage,
                            "downstream_residual_p95_ms": 20.0 + stage,
                            "component_budget_ms": 30.0 + 2 * stage,
                            "joint_remaining_p95_ms": 29.0,
                            "conformal_reserve_ms": 999.0,
                            "conformal_sample_count": 12,
                            "conformal_rank": 12,
                        }
                    )
    return {
        "schema_version": 1,
        "artifact_type": "p3_static_milestone_cost_model",
        "deploy_profile_id": "deploy-v1",
        "quantile": 0.95,
        "cells": cells,
    }


def _model():
    return StaticMilestoneCostModel.from_artifact(
        _artifact(),
        artifact_sha256="a" * 64,
        input_token_cutoff=128,
        milestone_guard_ms={"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
    )


def test_static_model_uses_o1_frozen_cell_index_and_ignores_conformal():
    model = _model()

    estimate = model.predict(
        request_path=RequestPath.AUDIO,
        stage_id=1,
        milestone=SLOMilestone.TTFP,
        num_prompt_tokens=36,
    )

    assert estimate.current_stage_service_ms == 11.0
    assert estimate.downstream_service_ms == 21.0
    assert estimate.downstream_transfer_ms == 0.0
    assert estimate.reserve_ms == 5.0
    assert estimate.downstream_reserve_ms == 26.0
    assert isinstance(model.cell_index, MappingProxyType)
    assert model.use_conformal_reserve is False
    assert model.cost_source == "static_component_p95_plus_first_output_guard"


def test_input_cutoff_selects_short_and_long_cells():
    artifact = _artifact()
    for cell in artifact["cells"]:
        if cell["key"]["size_bucket"] == "long":
            cell["current_stage_service_p95_ms"] += 100.0
            cell["component_budget_ms"] += 100.0
    model = StaticMilestoneCostModel.from_artifact(
        artifact,
        artifact_sha256="b" * 64,
        input_token_cutoff=128,
        milestone_guard_ms={"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
    )

    short = model.predict(
        request_path="text",
        stage_id=0,
        milestone="ttft",
        num_prompt_tokens=128,
    )
    long = model.predict(
        request_path="text",
        stage_id=0,
        milestone="ttft",
        num_prompt_tokens=129,
    )
    assert long.current_stage_service_ms - short.current_stage_service_ms == 100.0


def test_json_loader_verifies_exact_artifact_hash(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_artifact(), sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    model = StaticMilestoneCostModel.from_json_file(
        path,
        expected_sha256=digest,
        input_token_cutoff=128,
        milestone_guard_ms={"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
    )
    assert model.artifact_sha256 == digest

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        StaticMilestoneCostModel.from_json_file(
            path,
            expected_sha256="0" * 64,
            input_token_cutoff=128,
            milestone_guard_ms={"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["cells"].pop(), "missing cells"),
        (lambda value: value["cells"].append(value["cells"][0]), "duplicate cell"),
        (
            lambda value: value["cells"][0].__setitem__(
                "current_stage_service_p95_ms", -1
            ),
            "must be non-negative",
        ),
        (
            lambda value: value.__setitem__("artifact_type", "wrong"),
            "artifact_type",
        ),
    ],
)
def test_artifact_validation_fails_closed(mutation, message):
    artifact = _artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        StaticMilestoneCostModel.from_artifact(
            artifact,
            artifact_sha256="a" * 64,
            input_token_cutoff=128,
            milestone_guard_ms={"ttft": 5.0, "ttfp": 5.0, "e2e": 0.0},
        )


def test_prediction_rejects_path_stage_milestone_and_token_mismatches():
    model = _model()
    with pytest.raises(ValueError, match="no milestone cost cell"):
        model.predict(
            request_path="text",
            stage_id=1,
            milestone="e2e",
            num_prompt_tokens=36,
        )
    with pytest.raises(ValueError, match="recurring gap"):
        model.predict(
            request_path="audio",
            stage_id=1,
            milestone="audio_gap",
            num_prompt_tokens=36,
        )
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(
            request_path="audio",
            stage_id=1,
            milestone="ttfp",
            num_prompt_tokens=0,
        )
