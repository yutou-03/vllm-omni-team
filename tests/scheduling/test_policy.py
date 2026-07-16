from __future__ import annotations

import pytest

from vllm_omni.scheduling.policy import (
    BASELINE_POLICY_ENV,
    BaselineSchedulingPolicy,
    SchedulingMetadataError,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_key,
    predict_remaining_ms,
)


class FakeRequest:
    def __init__(
        self,
        request_id: str,
        scheduling: dict,
        *,
        num_computed_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.additional_information = {"meta": scheduling}
        self.num_computed_tokens = num_computed_tokens


def _scheduling(
    source_request_id: str,
    *,
    ingress_order: int,
    deadline: float,
) -> dict:
    return {
        "sched_schema_version": 1,
        "sched_source_request_id": source_request_id,
        "sched_ingress_order": ingress_order,
        "sched_deadline_monotonic_s": deadline,
        "sched_predicted_stage_ms": [100.0, 200.0, 300.0],
        "sched_predicted_stage_work_units": [10.0, 20.0, 30.0],
    }


def test_policy_environment_defaults_to_native_fcfs():
    assert get_baseline_scheduling_policy({}) is BaselineSchedulingPolicy.NATIVE_FCFS


def test_policy_environment_normalizes_and_rejects_unknown_values():
    assert get_baseline_scheduling_policy(
        {BASELINE_POLICY_ENV: " Final_Deadline_EDF_NP "}
    ) is BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP

    with pytest.raises(ValueError, match=BASELINE_POLICY_ENV):
        get_baseline_scheduling_policy({BASELINE_POLICY_ENV: "silent-fallback"})


def test_stage2_only_edf_has_explicit_activation_scope():
    policy = BaselineSchedulingPolicy.S2_ONLY_EDF_NP

    assert not policy_applies_to_stage(policy, 0)
    assert not policy_applies_to_stage(policy, 1)
    assert policy_applies_to_stage(policy, 2)


def test_edf_key_uses_deadline_then_stable_tie_breakers():
    earlier_deadline = FakeRequest(
        "engine-b",
        _scheduling("source-b", ingress_order=9, deadline=10.0),
    )
    earlier_ingress = FakeRequest(
        "engine-a",
        _scheduling("source-a", ingress_order=3, deadline=20.0),
    )
    later_ingress = FakeRequest(
        "engine-c",
        _scheduling("source-c", ingress_order=4, deadline=20.0),
    )

    ordered = sorted(
        [later_ingress, earlier_ingress, earlier_deadline],
        key=lambda request: policy_key(
            request,
            policy=BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
            stage_id=0,
            data_ready_time=1.0,
            now=2.0,
        ),
    )

    assert [request.request_id for request in ordered] == [
        "engine-b",
        "engine-a",
        "engine-c",
    ]


def test_srpf_remaining_prediction_is_monotonic_and_stage_local():
    request = FakeRequest(
        "engine-a",
        _scheduling("source-a", ingress_order=0, deadline=20.0),
        num_computed_tokens=5,
    )
    first_key = policy_key(
        request,
        policy=BaselineSchedulingPolicy.SRPF_LOCAL_NP,
        stage_id=0,
        data_ready_time=1.0,
        now=2.0,
    )
    request.num_computed_tokens = 8
    second_key = policy_key(
        request,
        policy=BaselineSchedulingPolicy.SRPF_LOCAL_NP,
        stage_id=0,
        data_ready_time=1.0,
        now=3.0,
    )

    assert first_key == (50.0, 1.0, "source-a", "engine-a")
    assert second_key == (20.0, 1.0, "source-a", "engine-a")
    assert second_key < first_key
    assert predict_remaining_ms(
        total_predicted_ms=100,
        total_work_units=10,
        completed_work_units=100,
    ) == 0.0

    with pytest.raises(ValueError, match="finite"):
        predict_remaining_ms(
            total_predicted_ms=100,
            total_work_units=10,
            completed_work_units=float("nan"),
        )


def test_enabled_policy_fails_fast_on_missing_metadata():
    request = FakeRequest("missing", {})

    with pytest.raises(SchedulingMetadataError, match="no scheduling metadata"):
        policy_key(
            request,
            policy=BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
            stage_id=0,
            data_ready_time=1.0,
            now=2.0,
        )


def test_native_policy_cannot_accidentally_enter_custom_sorting():
    request = FakeRequest(
        "native",
        _scheduling("native", ingress_order=0, deadline=20.0),
    )

    with pytest.raises(ValueError, match="does not reorder"):
        policy_key(
            request,
            policy=BaselineSchedulingPolicy.NATIVE_FCFS,
            stage_id=0,
            data_ready_time=1.0,
            now=2.0,
        )
