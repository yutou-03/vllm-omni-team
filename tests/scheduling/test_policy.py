from __future__ import annotations

import pytest

from vllm_omni.scheduling.contracts import (
    RequestSLOContract,
    SLOMilestone,
)
from vllm_omni.scheduling.deadline import (
    MilestoneCostEstimate,
    derive_stage_deadline_snapshot,
)
from vllm_omni.scheduling.policy import (
    BASELINE_POLICY_ENV,
    BaselineSchedulingPolicy,
    SchedulingMetadataError,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_key,
    remaining_prefill_tokens,
)


class FakeRequest:
    def __init__(
        self,
        request_id: str,
        scheduling: dict,
        *,
        num_prompt_tokens: int = 10,
        num_computed_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.additional_information = {"meta": scheduling}
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens


def _scheduling(
    source_request_id: str,
    *,
    ingress_order: int,
    deadline: float,
    stage_deadlines: list[float | None] | None = None,
) -> dict:
    return {
        "sched_schema_version": 1,
        "sched_source_request_id": source_request_id,
        "sched_ingress_order": ingress_order,
        "sched_deadline_monotonic_s": deadline,
        "sched_stage_deadline_monotonic_s": stage_deadlines
        or [deadline - 2.0, deadline - 1.0, deadline],
        "sched_predicted_stage_ms": [100.0, 200.0, 300.0],
        "sched_predicted_stage_work_units": [10.0, 20.0, 30.0],
    }


def _stage_ddl_request(request_id: str, ingress_order: int, ttft: float, e2e: float):
    scheduling = {
        "sched_schema_version": 2,
        "sched_source_request_id": f"source-{request_id}",
        "sched_ingress_order": ingress_order,
        "sched_logical_request_id": request_id,
        "sched_ingress_monotonic_s": 100.0,
        "sched_request_path": "text",
        "sched_service_class": "interactive",
        "sched_milestone_deadlines_monotonic_s": {
            "ttft": ttft,
            "e2e": e2e,
        },
        "sched_max_preemptions": 1,
    }
    return FakeRequest(request_id, scheduling)


def _stage_ddl_snapshot(request, *, ttft_cost_ms, e2e_cost_ms):
    contract = RequestSLOContract.from_server_metadata(
        request.additional_information["meta"]
    )
    return derive_stage_deadline_snapshot(
        contract,
        stage_id=0,
        captured_monotonic_s=100.5,
        milestone_costs={
            "ttft": MilestoneCostEstimate(ttft_cost_ms),
            "e2e": MilestoneCostEstimate(e2e_cost_ms),
        },
        deadline_version=1,
        cost_model_version="static-v1",
        cost_source="static_p95",
    )


def test_policy_environment_defaults_to_native_fcfs():
    assert get_baseline_scheduling_policy({}) is BaselineSchedulingPolicy.NATIVE_FCFS


def test_policy_environment_normalizes_and_rejects_unknown_values():
    assert get_baseline_scheduling_policy(
        {BASELINE_POLICY_ENV: " Final_Deadline_EDF_NP "}
    ) is BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP

    with pytest.raises(ValueError, match=BASELINE_POLICY_ENV):
        get_baseline_scheduling_policy({BASELINE_POLICY_ENV: "silent-fallback"})


def test_stage_deadline_edf_applies_to_all_stages():
    policy = BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP

    assert policy_applies_to_stage(policy, 0)
    assert policy_applies_to_stage(policy, 1)
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


def test_stage_deadline_edf_can_reverse_final_deadline_order_upstream():
    earlier_stage = FakeRequest(
        "stage-first",
        _scheduling(
            "source-stage",
            ingress_order=0,
            deadline=100.0,
            stage_deadlines=[80.0, 90.0, 100.0],
        ),
    )
    earlier_final = FakeRequest(
        "final-first",
        _scheduling(
            "source-final",
            ingress_order=1,
            deadline=95.0,
            stage_deadlines=[92.0, 94.0, 95.0],
        ),
    )

    final_order = sorted(
        [earlier_stage, earlier_final],
        key=lambda request: policy_key(
            request,
            policy=BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
            stage_id=0,
            data_ready_time=0.0,
            now=0.0,
        ),
    )
    stage_order = sorted(
        [earlier_stage, earlier_final],
        key=lambda request: policy_key(
            request,
            policy=BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP,
            stage_id=0,
            data_ready_time=0.0,
            now=0.0,
        ),
    )

    assert final_order == [earlier_final, earlier_stage]
    assert stage_order == [earlier_stage, earlier_final]


def test_multi_milestone_stage_ddl_orders_by_base_laxity_not_final_deadline():
    earlier_final = _stage_ddl_request("final-first", 0, 102.0, 104.0)
    tighter_ttft = _stage_ddl_request("ttft-first", 1, 101.0, 110.0)
    snapshots = {
        earlier_final.request_id: _stage_ddl_snapshot(
            earlier_final, ttft_cost_ms=100.0, e2e_cost_ms=500.0
        ),
        tighter_ttft.request_id: _stage_ddl_snapshot(
            tighter_ttft, ttft_cost_ms=700.0, e2e_cost_ms=500.0
        ),
    }

    ordered = sorted(
        [earlier_final, tighter_ttft],
        key=lambda request: policy_key(
            request,
            policy=BaselineSchedulingPolicy.STAGE_DDL_FIXED,
            stage_id=0,
            data_ready_time=100.0,
            now=100.5,
            stage_deadline_snapshot=snapshots[request.request_id],
        ),
    )

    assert ordered == [tighter_ttft, earlier_final]


def test_clockwork_latest_start_uses_profiled_e2e_remaining_time_only():
    earlier_e2e = _stage_ddl_request("e2e-first", 0, 102.0, 104.0)
    tighter_ttft = _stage_ddl_request("ttft-first", 1, 101.0, 110.0)
    snapshots = {
        earlier_e2e.request_id: _stage_ddl_snapshot(
            earlier_e2e, ttft_cost_ms=100.0, e2e_cost_ms=500.0
        ),
        tighter_ttft.request_id: _stage_ddl_snapshot(
            tighter_ttft, ttft_cost_ms=700.0, e2e_cost_ms=500.0
        ),
    }

    ordered = sorted(
        [tighter_ttft, earlier_e2e],
        key=lambda request: policy_key(
            request,
            policy=BaselineSchedulingPolicy.CLOCKWORK_LATEST_START_NP,
            stage_id=0,
            data_ready_time=100.0,
            now=100.5,
            stage_deadline_snapshot=snapshots[request.request_id],
        ),
    )

    assert ordered == [earlier_e2e, tighter_ttft]
    assert policy_key(
        earlier_e2e,
        policy=BaselineSchedulingPolicy.CLOCKWORK_LATEST_START_NP,
        stage_id=0,
        data_ready_time=100.0,
        now=100.5,
        stage_deadline_snapshot=snapshots[earlier_e2e.request_id],
    )[0] == snapshots[earlier_e2e.request_id].milestone_laxities_s[
        SLOMilestone.E2E
    ]


def test_stage_ddl_key_requires_matching_frozen_snapshot():
    request = _stage_ddl_request("request-a", 0, 101.0, 105.0)
    with pytest.raises(SchedulingMetadataError, match="requires a frozen snapshot"):
        policy_key(
            request,
            policy=BaselineSchedulingPolicy.STAGE_DDL_FIXED,
            stage_id=0,
            data_ready_time=100.0,
            now=100.5,
        )

    other = _stage_ddl_request("other", 1, 101.0, 105.0)
    with pytest.raises(SchedulingMetadataError, match="snapshot identity"):
        policy_key(
            request,
            policy=BaselineSchedulingPolicy.STAGE_DDL_FIXED,
            stage_id=0,
            data_ready_time=100.0,
            now=100.5,
            stage_deadline_snapshot=_stage_ddl_snapshot(
                other, ttft_cost_ms=100.0, e2e_cost_ms=500.0
            ),
        )


def test_srpf_uses_sarathi_style_local_remaining_prefill_tokens():
    request = FakeRequest(
        "engine-a",
        _scheduling("source-a", ingress_order=0, deadline=20.0),
        num_prompt_tokens=10,
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

    assert first_key == (5, 0, "source-a", "engine-a")
    assert second_key == (2, 0, "source-a", "engine-a")
    assert second_key < first_key
    request.num_computed_tokens = 100
    assert remaining_prefill_tokens(request) == 0


def test_srpf_does_not_require_stage_predictions_or_include_output_length():
    scheduling = {
        "sched_source_request_id": "source-a",
        "sched_ingress_order": 3,
    }
    short_prefill = FakeRequest(
        "short",
        scheduling,
        num_prompt_tokens=8,
        num_computed_tokens=4,
    )
    short_prefill.max_tokens = 1000
    long_prefill = FakeRequest(
        "long",
        scheduling,
        num_prompt_tokens=20,
        num_computed_tokens=10,
    )
    long_prefill.max_tokens = 1

    short_key = policy_key(
        short_prefill,
        policy=BaselineSchedulingPolicy.SRPF_LOCAL_NP,
        stage_id=2,
        data_ready_time=float("inf"),
        now=2.0,
    )
    long_key = policy_key(
        long_prefill,
        policy=BaselineSchedulingPolicy.SRPF_LOCAL_NP,
        stage_id=2,
        data_ready_time=float("inf"),
        now=2.0,
    )

    assert short_key < long_key
    assert short_key[0] == 4
    assert long_key[0] == 10


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


def test_stage_deadline_edf_fails_fast_when_stage_value_is_missing():
    request = FakeRequest(
        "text-stage-1",
        _scheduling(
            "text-source",
            ingress_order=0,
            deadline=20.0,
            stage_deadlines=[20.0, None, None],
        ),
    )

    with pytest.raises(SchedulingMetadataError, match="no value for stage 1"):
        policy_key(
            request,
            policy=BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP,
            stage_id=1,
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
