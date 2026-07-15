import json

import pytest

from vllm_omni.benchmarks.data_modules.servegen_schema import (
    load_servegen_jsonl,
    mm_item_config,
    normalize_servegen_record,
)


def test_normalize_record_preserves_experiment_metadata():
    record = normalize_servegen_record(
        {
            "request_id": 17,
            "timestamp": 1.25,
            "input_tokens": 32,
            "output_tokens": 64,
            "output_modalities": ["text", "audio"],
            "slo_ms": 2500,
            "request_path": "audio",
            "predicted_stage_ms": {"0": 120, "1": 80, "2": 400},
        },
        line_number=1,
    )

    assert record["request_id"] == "17"
    assert record["mm_items"] == []
    assert record["output_modalities"] == ["text", "audio"]
    assert record["slo_ms"] == 2500.0
    assert record["predicted_stage_ms"] == {
        "0": 120.0,
        "1": 80.0,
        "2": 400.0,
    }


def test_video_frame_count_is_not_multiplied_by_fps():
    assert mm_item_config(
        {
            "modality": "video",
            "h": 256,
            "w": 512,
            "t": 60,
            "fps": 30,
        },
        request_id="video-1",
    ) == (256, 512, 60)


def test_jsonl_rejects_duplicate_ids(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    records = [
        {"request_id": "r1", "timestamp": 1.0, "output_tokens": 8},
        {"request_id": "r1", "timestamp": 0.5, "output_tokens": 8},
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in records))

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_servegen_jsonl(trace_path)


def test_jsonl_rejects_unsorted_timestamps(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    records = [
        {"request_id": "r1", "timestamp": 1.0, "output_tokens": 8},
        {"request_id": "r2", "timestamp": 0.5, "output_tokens": 8},
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in records))

    with pytest.raises(ValueError, match="timestamps must be nondecreasing"):
        load_servegen_jsonl(trace_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", -0.1, "timestamp must be non-negative"),
        ("output_tokens", 0, "output_tokens must be a positive integer"),
        ("slo_ms", 0, "slo_ms must be positive"),
        ("output_modalities", ["image"], "unsupported output modalities"),
    ],
)
def test_invalid_contract_fields_fail_early(field, value, message):
    record = {"request_id": "r1", "timestamp": 0, "output_tokens": 8}
    record[field] = value

    with pytest.raises(ValueError, match=message):
        normalize_servegen_record(record, line_number=3)
