import pytest

from vllm_omni.scheduling.metadata import (
    build_client_scheduling_metadata,
    build_server_scheduling_metadata,
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
    normalize_client_scheduling_metadata,
    preserve_scheduling_metadata,
)


def test_build_client_scheduling_metadata_normalizes_stage_mapping():
    metadata = build_client_scheduling_metadata(
        source_request_id="request-4",
        ingress_order=4,
        slo_ms=1500,
        request_path="AUDIO",
        predicted_stage_ms={"0": 12, "2": 90},
        predicted_stage_work_units={0: 8, 2: 120},
    )

    assert metadata == {
        "schema_version": 1,
        "source_request_id": "request-4",
        "ingress_order": 4,
        "slo_ms": 1500.0,
        "request_path": "audio",
        "predicted_stage_ms": [12.0, None, 90.0],
        "predicted_stage_work_units": [8.0, None, 120.0],
    }


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {"schema_version": 2, "source_request_id": "r", "ingress_order": 0},
            "unsupported schema_version",
        ),
        (
            {"schema_version": 1, "source_request_id": "r", "ingress_order": -1},
            "ingress_order must be a non-negative integer",
        ),
        (
            {
                "schema_version": 1,
                "source_request_id": "r",
                "ingress_order": 0,
                "predicted_stage_ms": [1, 2],
            },
            "must contain exactly 3 stage entries",
        ),
        (
            {
                "schema_version": 1,
                "source_request_id": "r",
                "ingress_order": 0,
                "predicted_stage_ms": [1, None, 3],
                "predicted_stage_work_units": [1, 2, 3],
            },
            "must cover the same stages",
        ),
        (
            {
                "schema_version": 1,
                "source_request_id": "r",
                "ingress_order": 0,
                "typo_deadline": 1,
            },
            "unknown fields",
        ),
    ],
)
def test_invalid_client_scheduling_metadata_fails_early(metadata, message):
    with pytest.raises(ValueError, match=message):
        normalize_client_scheduling_metadata(metadata)


def test_server_metadata_owns_ingress_and_absolute_deadline():
    metadata = build_server_scheduling_metadata(
        {
            "schema_version": 1,
            "source_request_id": "request-9",
            "ingress_order": 9,
            "slo_ms": 2500,
            "request_path": "text",
        },
        ingress_monotonic_s=100.25,
        ingress_wall_s=1700000000.5,
    )

    assert metadata == {
        "sched_schema_version": 1,
        "sched_source_request_id": "request-9",
        "sched_ingress_order": 9,
        "sched_ingress_monotonic_s": 100.25,
        "sched_ingress_wall_s": 1700000000.5,
        "sched_slo_ms": 2500.0,
        "sched_request_path": "text",
        "sched_deadline_monotonic_s": 102.75,
    }


def test_scheduling_merge_preserves_existing_payload_fields():
    scheduling = {
        "sched_schema_version": 1,
        "sched_source_request_id": "request-2",
    }
    merged = merge_scheduling_metadata_into_additional_information(
        {
            "speaker": ["default"],
            "meta": {"codec_streaming": True},
        },
        scheduling,
    )

    assert merged["speaker"] == ["default"]
    assert merged["meta"]["codec_streaming"] is True
    assert extract_scheduling_metadata(merged) == scheduling


def test_runtime_payload_replacement_preserves_server_scheduling_fields():
    existing = {
        "meta": {
            "sched_source_request_id": "source-1",
            "sched_deadline_monotonic_s": 123.0,
        }
    }
    incoming = {
        "codes": {"audio": [1]},
        "meta": {
            "finished": True,
            "sched_deadline_monotonic_s": -1.0,
        },
    }

    merged = preserve_scheduling_metadata(existing, incoming)

    assert merged["codes"] == {"audio": [1]}
    assert merged["meta"]["finished"] is True
    assert merged["meta"]["sched_source_request_id"] == "source-1"
    assert merged["meta"]["sched_deadline_monotonic_s"] == 123.0
