from types import MappingProxyType

import pytest

from vllm_omni.scheduling.contracts import RequestSLOContract, SLOMilestone
from vllm_omni.scheduling.deadline import (
    MilestoneCostEstimate,
    derive_stage_deadline_snapshot,
)


def _contract(path: str = "audio", *, include_gap: bool = False):
    deadlines = {
        "ttft": 101.0,
        "e2e": 108.0,
    }
    if path == "audio":
        deadlines["ttfp"] = 103.0
        if include_gap:
            deadlines["audio_gap"] = 102.0
    elif include_gap:
        deadlines["text_gap"] = 102.0
    return RequestSLOContract(
        schema_version=2,
        logical_request_id="logical-1",
        ingress_monotonic_s=100.0,
        request_path=path,
        service_class="interactive",
        milestone_deadlines_s=deadlines,
        text_consume_rate_tps=20.0,
        audio_consume_rate=1.0 if path == "audio" else None,
        max_preemptions=1,
    )


def _cost(
    current_ms: float,
    downstream_ms: float = 0.0,
    transfer_ms: float = 0.0,
    reserve_ms: float = 0.0,
):
    return MilestoneCostEstimate(
        current_stage_service_ms=current_ms,
        downstream_service_ms=downstream_ms,
        downstream_transfer_ms=transfer_ms,
        reserve_ms=reserve_ms,
    )


def test_stage_deadline_uses_tightest_multi_milestone_laxity():
    snapshot = derive_stage_deadline_snapshot(
        _contract(),
        stage_id=0,
        captured_monotonic_s=100.5,
        milestone_costs={
            "ttft": _cost(600.0),
            "ttfp": _cost(1000.0, 1200.0, 100.0, 100.0),
            "e2e": _cost(2000.0, 3000.0, 200.0, 300.0),
        },
        deadline_version=1,
        cost_model_version="b0-static-p95-v1",
        cost_source="static_p95",
        used_fallback=True,
    )

    assert snapshot.stage_event_deadlines_s == {
        SLOMilestone.TTFT: pytest.approx(101.0),
        SLOMilestone.TTFP: pytest.approx(101.6),
        SLOMilestone.E2E: pytest.approx(104.5),
    }
    assert snapshot.milestone_laxities_s == {
        SLOMilestone.TTFT: pytest.approx(-0.1),
        SLOMilestone.TTFP: pytest.approx(0.1),
        SLOMilestone.E2E: pytest.approx(2.0),
    }
    assert snapshot.controlling_milestone is SLOMilestone.TTFT
    assert snapshot.base_laxity_s == pytest.approx(-0.1)
    assert snapshot.stage_event_deadline_s == pytest.approx(101.0)
    assert snapshot.used_fallback is True


def test_negative_laxity_is_retained_as_an_urgency_signal():
    snapshot = derive_stage_deadline_snapshot(
        _contract("text"),
        stage_id=0,
        captured_monotonic_s=101.5,
        milestone_costs={
            "ttft": _cost(200.0),
            "e2e": _cost(1000.0),
        },
        deadline_version=2,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )

    assert snapshot.controlling_milestone is SLOMilestone.TTFT
    assert snapshot.base_laxity_s == pytest.approx(-0.7)


def test_completed_milestone_is_removed_from_the_frozen_snapshot():
    snapshot = derive_stage_deadline_snapshot(
        _contract(),
        stage_id=1,
        captured_monotonic_s=101.0,
        milestone_costs={
            "ttfp": _cost(300.0, 500.0),
            "e2e": _cost(1000.0, 2000.0),
        },
        completed_milestones={"ttft"},
        deadline_version=1,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )

    assert set(snapshot.milestone_laxities_s) == {
        SLOMilestone.TTFP,
        SLOMilestone.E2E,
    }


def test_gap_contract_is_not_silently_treated_as_one_shot_deadline():
    snapshot = derive_stage_deadline_snapshot(
        _contract(include_gap=True),
        stage_id=2,
        captured_monotonic_s=101.0,
        milestone_costs={
            "ttfp": _cost(500.0),
            "e2e": _cost(2000.0),
        },
        deadline_version=1,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )
    assert snapshot.excluded_milestones == {
        SLOMilestone.AUDIO_GAP: "recurring_gap_deferred",
        SLOMilestone.TTFT: "not_affected_by_stage",
    }

    with pytest.raises(ValueError, match="unsupported milestone.*audio_gap"):
        derive_stage_deadline_snapshot(
            _contract(include_gap=True),
            stage_id=2,
            captured_monotonic_s=101.0,
            milestone_costs={
                "ttfp": _cost(500.0),
                "audio_gap": _cost(100.0),
                "e2e": _cost(2000.0),
            },
            deadline_version=1,
            cost_model_version="b0-v1",
            cost_source="static_p95",
        )


def test_snapshot_and_nested_mappings_are_immutable():
    snapshot = derive_stage_deadline_snapshot(
        _contract("text"),
        stage_id=0,
        captured_monotonic_s=100.0,
        milestone_costs={"ttft": _cost(100.0), "e2e": _cost(500.0)},
        deadline_version=1,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )

    assert isinstance(snapshot.milestone_costs, MappingProxyType)
    assert isinstance(snapshot.milestone_laxities_s, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.milestone_laxities_s[SLOMilestone.TTFT] = 99.0
    with pytest.raises(AttributeError):
        snapshot.base_laxity_s = 99.0


@pytest.mark.parametrize(
    ("stage_id", "message"),
    [(1, "text path does not traverse stage 1"), (3, "stage_id must be one of")],
)
def test_stage_path_scope_fails_closed(stage_id, message):
    with pytest.raises(ValueError, match=message):
        derive_stage_deadline_snapshot(
            _contract("text"),
            stage_id=stage_id,
            captured_monotonic_s=100.0,
            milestone_costs={"ttft": _cost(100.0), "e2e": _cost(500.0)},
            deadline_version=1,
            cost_model_version="b0-v1",
            cost_source="static_p95",
        )


def test_missing_extra_and_invalid_costs_fail_closed():
    common = dict(
        contract=_contract(),
        stage_id=1,
        captured_monotonic_s=101.0,
        deadline_version=1,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )
    with pytest.raises(ValueError, match="milestone_costs mismatch.*missing.*e2e"):
        derive_stage_deadline_snapshot(
            **common,
            milestone_costs={"ttfp": _cost(100.0)},
        )
    with pytest.raises(ValueError, match="unsupported milestone.*ttft"):
        derive_stage_deadline_snapshot(
            **common,
            milestone_costs={
                "ttft": _cost(100.0),
                "ttfp": _cost(100.0),
                "e2e": _cost(100.0),
            },
        )
    with pytest.raises(ValueError, match="downstream_service_ms must be non-negative"):
        _cost(100.0, -1.0)


def test_equal_laxity_uses_deterministic_milestone_tie_breaker():
    snapshot = derive_stage_deadline_snapshot(
        _contract(),
        stage_id=0,
        captured_monotonic_s=100.0,
        milestone_costs={
            "ttft": _cost(1000.0),
            "ttfp": _cost(3000.0),
            "e2e": _cost(8000.0),
        },
        deadline_version=1,
        cost_model_version="b0-v1",
        cost_source="static_p95",
    )

    assert snapshot.base_laxity_s == pytest.approx(0.0)
    assert snapshot.controlling_milestone is SLOMilestone.E2E
