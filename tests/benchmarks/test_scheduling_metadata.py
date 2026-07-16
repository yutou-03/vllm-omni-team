import pytest

from vllm_omni.scheduling.metadata import (
    build_client_scheduling_metadata,
    normalize_client_scheduling_metadata,
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
