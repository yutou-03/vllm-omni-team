"""Unit tests for cross-stage scheduling metadata propagation."""

import pytest

from vllm.sampling_params import SamplingParams

from vllm_omni.core.sched.omni_scheduler_mixin import (
    extract_request_scheduling_trace_fields,
    preserve_streaming_scheduling_metadata,
)
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.orchestrator import (
    OrchestratorRequestState,
    build_engine_core_request_from_tokens,
    extract_scheduling_metadata_from_request_input,
    merge_scheduling_metadata_into_request_input,
)
from vllm_omni.scheduling.contracts import RequestPath, SLOMilestone
from vllm_omni.scheduling.metadata import build_server_scheduling_metadata


def test_rebuilt_stage_request_preserves_identical_scheduling_metadata():
    scheduling = {
        "sched_schema_version": 1,
        "sched_source_request_id": "request-6",
        "sched_ingress_order": 6,
        "sched_ingress_monotonic_s": 20.0,
        "sched_deadline_monotonic_s": 21.5,
        "sched_stage_deadline_monotonic_s": [20.8, 21.1, 21.5],
        "sched_slo_ms": 1500.0,
    }
    stage0_prompt = {
        "prompt_token_ids": [1, 2],
        "additional_information": {
            "meta": scheduling,
        },
    }
    stage0_request = build_engine_core_request_from_tokens(
        "request-6",
        stage0_prompt,
        SamplingParams(max_tokens=2),
    )

    rebuilt_stage1 = merge_scheduling_metadata_into_request_input(
        build_engine_core_request_from_tokens(
            "request-6",
            {
                "prompt_token_ids": [3, 4],
                "additional_information": {
                    "meta": {"codec_streaming": True},
                },
            },
            SamplingParams(max_tokens=2),
        ),
        extract_scheduling_metadata_from_request_input(stage0_request),
    )
    rebuilt_stage2 = merge_scheduling_metadata_into_request_input(
        build_engine_core_request_from_tokens(
            "request-6",
            {"prompt_token_ids": [5, 6]},
            SamplingParams(max_tokens=2),
        ),
        extract_scheduling_metadata_from_request_input(rebuilt_stage1),
    )

    assert extract_scheduling_metadata_from_request_input(stage0_request) == scheduling
    assert extract_scheduling_metadata_from_request_input(rebuilt_stage1) == scheduling
    assert extract_scheduling_metadata_from_request_input(rebuilt_stage2) == scheduling


def test_streaming_replacement_preserves_deadline_and_new_chunk_fields():
    existing = serialize_additional_information(
        {
            "meta": {
                "sched_schema_version": 1,
                "sched_source_request_id": "stream-1",
                "sched_deadline_monotonic_s": 42.0,
            }
        }
    )
    incoming = serialize_additional_information(
        {"meta": {"codec_streaming": True}}
    )

    merged = preserve_streaming_scheduling_metadata(existing, incoming)
    decoded = deserialize_additional_information(merged)

    assert decoded["meta"] == {
        "codec_streaming": True,
        "sched_schema_version": 1,
        "sched_source_request_id": "stream-1",
        "sched_deadline_monotonic_s": 42.0,
    }


def test_stage_admission_trace_decodes_scheduling_fields():
    scheduling = {
        "sched_schema_version": 1,
        "sched_source_request_id": "trace-1",
        "sched_ingress_order": 1,
        "sched_deadline_monotonic_s": 101.0,
    }
    request = type(
        "FakeRequest",
        (),
        {
            "additional_information": serialize_additional_information(
                {"meta": scheduling}
            )
        },
    )()

    assert extract_request_scheduling_trace_fields(request) == scheduling


def test_orchestrator_state_materializes_v2_contract():
    scheduling = build_server_scheduling_metadata(
        {
            "schema_version": 2,
            "source_request_id": "source-1",
            "ingress_order": 1,
            "request_path": "text",
            "service_class": "interactive",
            "milestone_slo_ms": {"ttft": 500, "e2e": 2000},
            "max_preemptions": 1,
        },
        ingress_monotonic_s=100.0,
        ingress_wall_s=1700000000.0,
        logical_request_id="logical-1",
    )

    state = OrchestratorRequestState(
        request_id="logical-1",
        final_stage_id=0,
        scheduling_metadata=scheduling,
    )

    assert state.slo_contract is not None
    assert state.slo_contract.request_path is RequestPath.TEXT
    assert state.slo_contract.milestone_deadlines_s[SLOMilestone.E2E] == 102.0


def test_orchestrator_state_rejects_mismatched_v2_logical_id():
    scheduling = build_server_scheduling_metadata(
        {
            "schema_version": 2,
            "source_request_id": "source-1",
            "ingress_order": 1,
            "request_path": "text",
            "service_class": "interactive",
            "milestone_slo_ms": {"ttft": 500, "e2e": 2000},
            "max_preemptions": 1,
        },
        ingress_monotonic_s=100.0,
        ingress_wall_s=1700000000.0,
        logical_request_id="other-logical-id",
    )

    with pytest.raises(ValueError, match="logical request id mismatch"):
        OrchestratorRequestState(
            request_id="logical-1",
            final_stage_id=0,
            scheduling_metadata=scheduling,
        )
