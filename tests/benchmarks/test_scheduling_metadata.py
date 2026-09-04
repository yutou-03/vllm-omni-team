import pytest

from vllm_omni.scheduling.metadata import (
    CURRENT_SCHEDULING_SCHEMA_VERSION,
    build_client_scheduling_metadata,
    build_server_scheduling_metadata,
    derive_stage_deadlines,
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
    normalize_client_scheduling_metadata,
    preserve_scheduling_metadata,
)


def _valid_v2_metadata(**overrides):
    metadata = {
        "schema_version": 2,
        "source_request_id": "request-v2",
        "ingress_order": 7,
        "request_path": "audio",
        "service_class": "realtime",
        "milestone_slo_ms": {
            "ttft": 500,
            "ttfp": 900,
            "audio_gap": 120,
            "e2e": 3000,
        },
        "text_consume_rate_tps": 25,
        "audio_consume_rate": 1,
        "max_preemptions": 2,
    }
    metadata.update(overrides)
    return metadata


def test_current_schema_is_v2_while_v1_remains_accepted():
    assert CURRENT_SCHEDULING_SCHEMA_VERSION == 2
    assert normalize_client_scheduling_metadata(
        {"schema_version": 1, "source_request_id": "legacy", "ingress_order": 0}
    ) == {"schema_version": 1, "source_request_id": "legacy", "ingress_order": 0}


@pytest.mark.parametrize(
    "missing",
    [
        "source_request_id",
        "ingress_order",
        "request_path",
        "service_class",
        "milestone_slo_ms",
        "max_preemptions",
    ],
)
def test_v2_missing_required_field_fails_closed(missing):
    metadata = _valid_v2_metadata()
    del metadata[missing]

    with pytest.raises(ValueError, match=f"missing {missing}"):
        normalize_client_scheduling_metadata(metadata)


def test_server_owns_v2_logical_id_and_absolute_milestone_deadlines():
    metadata = build_server_scheduling_metadata(
        _valid_v2_metadata(),
        ingress_monotonic_s=100.0,
        ingress_wall_s=1700000000.0,
        logical_request_id="chatcmpl-server-id",
    )

    assert metadata["sched_schema_version"] == 2
    assert metadata["sched_logical_request_id"] == "chatcmpl-server-id"
    assert metadata["sched_milestone_deadlines_monotonic_s"] == {
        "ttft": 100.5,
        "ttfp": 100.9,
        "audio_gap": 100.12,
        "e2e": 103.0,
    }
    assert metadata["sched_deadline_monotonic_s"] == 103.0
    assert metadata["sched_service_class"] == "realtime"
    assert metadata["sched_max_preemptions"] == 2


def test_v2_rejects_path_inapplicable_milestones():
    with pytest.raises(ValueError, match="milestones not applicable to text.*ttfp"):
        normalize_client_scheduling_metadata(
            _valid_v2_metadata(
                request_path="text",
                milestone_slo_ms={"ttft": 500, "ttfp": 900, "e2e": 3000},
                audio_consume_rate=None,
            )
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
            {"schema_version": 3, "source_request_id": "r", "ingress_order": 0},
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


def test_server_derives_audio_stage_deadlines_from_final_slo():
    metadata = build_server_scheduling_metadata(
        {
            "schema_version": 1,
            "source_request_id": "audio-1",
            "ingress_order": 1,
            "slo_ms": 2000,
            "request_path": "audio",
            "predicted_stage_ms": [500, 300, 700],
        },
        ingress_monotonic_s=100.0,
        ingress_wall_s=1700000000.0,
    )

    assert metadata["sched_deadline_monotonic_s"] == 102.0
    assert metadata["sched_stage_deadline_monotonic_s"] == [101.0, 101.3, 102.0]


def test_text_stage_deadline_uses_final_deadline_and_rejects_path_mismatch():
    assert derive_stage_deadlines(
        deadline_monotonic_s=12.0,
        request_path="text",
        predicted_stage_ms=[250.0, None, None],
    ) == [12.0, None, None]

    with pytest.raises(ValueError, match="must match request_path"):
        derive_stage_deadlines(
            deadline_monotonic_s=12.0,
            request_path="audio",
            predicted_stage_ms=[250.0, None, 500.0],
        )


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
