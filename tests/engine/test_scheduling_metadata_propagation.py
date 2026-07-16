"""Unit tests for cross-stage scheduling metadata propagation."""

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
    build_engine_core_request_from_tokens,
    extract_scheduling_metadata_from_request_input,
    merge_scheduling_metadata_into_request_input,
)


def test_rebuilt_stage_request_preserves_identical_scheduling_metadata():
    scheduling = {
        "sched_schema_version": 1,
        "sched_source_request_id": "request-6",
        "sched_ingress_order": 6,
        "sched_ingress_monotonic_s": 20.0,
        "sched_deadline_monotonic_s": 21.5,
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
