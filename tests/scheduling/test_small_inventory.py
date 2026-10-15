from collections import defaultdict
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import assert_payload
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import talker2code2wav_async_chunk
from vllm_omni.profiling import small_inventory

pytestmark = pytest.mark.cpu


def _payloads(monkeypatch, enabled):
    monkeypatch.setenv("VLLM_OMNI_SMALL_HOLD_PROBE", "1" if enabled else "0")
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 2, "codec_left_context_frames": 2}}),
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
    )
    request = SimpleNamespace(
        request_id="engine-1", output_connector_req_id="wire-1",
        additional_information={"meta": {"sched_logical_request_id": "logical-1"}},
    )
    payloads = []
    for index in range(5):
        payload = talker2code2wav_async_chunk(
            manager, {"codes": {"audio": torch.full((1, 16), index + 1)}},
            request, is_finished=index == 4,
        )
        if payload is not None:
            payloads.append(payload)
            manager.put_req_chunk["wire-1"] += 1
    return payloads


def test_metadata_does_not_change_codec_content_and_excludes_replayed_context(monkeypatch):
    tagged = _payloads(monkeypatch, True)
    plain = _payloads(monkeypatch, False)
    assert len(tagged) == len(plain) == 3
    for new, old in zip(tagged, plain):
        assert_payload(new)
        assert new["codes"] == old["codes"]
        assert new["meta"]["finished"] == old["meta"]["finished"]
        assert new["meta"]["left_context_size"] == old["meta"]["left_context_size"]
        assert not small_inventory.inventory_fields(old)
    fields = [small_inventory.inventory_fields(p) for p in tagged]
    assert [f["chunk_id"] for f in fields] == [0, 1, 2]
    assert [f["codec_total_frames"] for f in fields] == [2, 4, 5]
    assert [f["codec_payload_frames"] for f in fields] == [2, 4, 3]
    assert [f["new_frames"] for f in fields] == [2, 2, 1]
    assert [f["left_context_frames"] for f in fields] == [0, 2, 2]
    assert [f["stream_finished"] for f in fields] == [False, False, True]


def test_receive_is_recorded_only_after_publication_not_cancelled_get(monkeypatch):
    payload = _payloads(monkeypatch, True)[0]
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.connector = SimpleNamespace(stage_id=2, get=lambda *args: (payload, 128))
    adapter.model_mode = "generation"
    adapter.get_req_chunk = defaultdict(int)
    adapter.request_ids_mapping = {"engine-1": "wire-1"}
    adapter.finished_requests = set()
    adapter._finished_load_reqs = set()
    adapter._drain_state_lock = Lock()
    adapter._cancelled_load_reqs = {"engine-1"}
    request = SimpleNamespace(request_id="engine-1", additional_information={}, prompt_token_ids=[])
    events = []
    monkeypatch.setattr(small_inventory, "emit_stage_event", lambda event, **kwargs: events.append((event, kwargs)))
    assert adapter._poll_single_request(request)
    assert not events
    assert not adapter.get_req_chunk
    assert adapter._poll_single_request(request)
    assert "engine-1" in adapter._finished_load_reqs
    assert request.prompt_token_ids == payload["codes"]["audio"]
    assert events[0][0] == "small_hold_chunk_received"
    assert events[0][1]["received_chunk_id"] == events[0][1]["chunk_id"] == 0
    assert events[0][1]["logical_request_id"] == "logical-1"


def test_chunk_identity_survives_engine_payload_serialization(monkeypatch):
    payload = _payloads(monkeypatch, True)[-1]
    decoded = deserialize_additional_information(serialize_additional_information(payload))
    assert small_inventory.inventory_fields(decoded) == small_inventory.inventory_fields(payload)


def test_execution_snapshot_excludes_unscheduled_chunks_and_respects_guard(monkeypatch):
    payload = _payloads(monkeypatch, True)[0]
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"new": 32, "cached": 32},
        scheduled_new_reqs=[SimpleNamespace(req_id="new", additional_information=payload)],
        scheduled_cached_reqs=SimpleNamespace(additional_information={"cached": payload, "unscheduled": payload}),
    )
    assert set(small_inventory.scheduled_inventory_fields(scheduler)) == {"new", "cached"}
    monkeypatch.delenv("VLLM_OMNI_SMALL_HOLD_PROBE")
    assert small_inventory.scheduled_inventory_fields(scheduler) == {}


def test_completed_conversion_counts_actual_trimmed_audio_and_not_codec_tokens(monkeypatch):
    from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

    fields = small_inventory.inventory_fields(_payloads(monkeypatch, True)[-1])
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model_config = SimpleNamespace(stage_id=2)
    runner.execute_model_state = (
        None, None, None, None, None, None, None, None, None,
        {"audio": [torch.zeros(1, 1920)]}, None,
    )
    runner.kv_connector_output = None
    runner.input_batch = SimpleNamespace(req_ids=["engine-1"], req_id_to_index={"engine-1": 0}, num_reqs=1)
    runner.speculative_config = None
    runner.supports_mm_inputs = False
    runner.use_async_scheduling = False
    runner._small_hold_inventory_steps = {"engine-1": fields}
    events = []
    monkeypatch.setattr(small_inventory, "emit_stage_event", lambda event, **kwargs: events.append((event, kwargs)))
    output = runner.sample_tokens()
    assert output.pooler_output[0]["audio"].shape == (1, 1920)
    assert events[0][0] == "small_hold_chunk_conversion_completed"
    assert events[0][1]["audio_samples"] == 1920
    assert events[0][1]["new_frames"] == 1
    assert events[0][1]["client_delivery_confirmed"] is False
    assert runner._small_hold_inventory_steps == {}
