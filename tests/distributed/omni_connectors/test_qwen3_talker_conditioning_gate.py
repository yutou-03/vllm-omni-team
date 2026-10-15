# SPDX-License-Identifier: Apache-2.0
"""CPU regression for Thinker embedding-lag readiness under chunked prefill."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.talker_conditioning import (
    qwen3_talker_conditioning_ready,
)
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
    TalkerConditioningUnavailable,
)


def payload(total: int):
    # The first sampled token has no forward row yet. Its row arrives when
    # the second output ID is sampled.
    forward_rows = 10 + max(total - 1, 0)
    return {"ids": {"all": list(range(10 + total)), "prompt": list(range(10))},
            "embed": {"prefill": torch.zeros((forward_rows, 1))}}


def adapter_with_request(total: int, output_tokens: int = 0):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._qwen3_talker_conditioning_gate = True
    adapter._qwen3_bootstrap_published = set()
    adapter._qwen3_prompt_rows = {"r": 10}
    adapter._qwen3_received_entity_rows = {"r": 10 + max(total - 1, 0)}
    adapter._qwen3_entity_layout_invalid = set()
    adapter.request_ids_mapping = {}
    adapter.request_payload = {"r": payload(total)}
    adapter.finished_requests = set()
    adapter.requests_with_ready_chunks = set()
    adapter.requests_origin_status = {}
    loaded = []
    adapter.load_async = lambda request: loaded.append(request.request_id)
    request = SimpleNamespace(
        request_id="r", status=RequestStatus.WAITING,
        num_output_tokens=output_tokens,
    )
    return adapter, request, loaded


def test_embedding_lag_and_terminal_marker():
    assert not qwen3_talker_conditioning_ready(payload(1), 0)
    assert qwen3_talker_conditioning_ready(payload(2), 0)
    assert not qwen3_talker_conditioning_ready(payload(2), 1)
    assert qwen3_talker_conditioning_ready(payload(3), 1)
    assert qwen3_talker_conditioning_ready(payload(1), 1, upstream_finished=True)


def test_initial_chunk_waits_until_first_embedding():
    adapter, request, loaded = adapter_with_request(total=1)
    queue = [request]
    waiting = deque()
    adapter._process_chunk_queue(queue, waiting, RequestStatus.WAITING, set())
    assert loaded == ["r"]
    assert queue == [] and list(waiting) == [request]
    assert request.status == RequestStatus.WAITING_FOR_CHUNK

    adapter.request_payload["r"] = payload(2)
    adapter._qwen3_received_entity_rows["r"] = 11
    queue[:] = waiting
    waiting.clear()
    adapter._process_chunk_queue(queue, waiting, RequestStatus.WAITING, {"r"})
    assert queue == [request] and list(waiting) == []
    assert request.status == RequestStatus.WAITING
    assert adapter.requests_with_ready_chunks == {"r"}


def test_chunked_prefill_reuses_first_embedding_then_decode_waits():
    adapter, request, loaded = adapter_with_request(total=2)
    request.status = RequestStatus.RUNNING
    queue = [request]
    waiting = deque()
    adapter._process_chunk_queue(queue, waiting, RequestStatus.RUNNING, set())
    assert queue == [request] and not loaded
    assert adapter.requests_with_ready_chunks == set()

    # A partial prompt batch has no codec output; its next prompt batch may
    # proceed using the same first text embedding.
    adapter.requests_with_ready_chunks.clear()
    adapter._process_chunk_queue(queue, waiting, RequestStatus.RUNNING, set())
    assert queue == [request] and not loaded

    # The first actual codec token requires embedding index 1, which the
    # current two-token Thinker payload does not yet provide.
    adapter.requests_with_ready_chunks.clear()
    request.num_output_tokens = 1
    adapter._process_chunk_queue(queue, waiting, RequestStatus.RUNNING, set())
    assert loaded == ["r"]
    assert request.status == RequestStatus.WAITING_FOR_CHUNK
    assert list(waiting) == [request]

    adapter.request_payload["r"] = payload(3)
    adapter._qwen3_received_entity_rows["r"] = 12
    queue[:] = waiting
    waiting.clear()
    adapter._process_chunk_queue(queue, waiting, RequestStatus.RUNNING, {"r"})
    assert request.status == RequestStatus.RUNNING
    assert adapter.requests_with_ready_chunks == {"r"}



def test_cached_conditioning_does_not_reinject_old_delta():
    adapter, request, _ = adapter_with_request(total=3, output_tokens=1)
    request.additional_information = {"embed": {"decode": "fresh-delta"}}
    merged = []
    fake_worker = SimpleNamespace(
        _update_intermediate_buffer=lambda rid, value: merged.append((rid, value))
    )

    def scheduler_output():
        return SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], additional_information={}),
        )

    # Executable from already cached conditioning, but no new connector
    # chunk. The real worker update path must receive no delta.
    output = scheduler_output()
    adapter.attach_cached_additional_information(output, {"r": request})
    OmniGPUModelRunner._update_additional_information(fake_worker, output)
    assert merged == []

    # One actual newly received chunk carries its delta exactly once.
    adapter.requests_with_ready_chunks.add("r")
    output = scheduler_output()
    adapter.attach_cached_additional_information(output, {"r": request})
    OmniGPUModelRunner._update_additional_information(fake_worker, output)
    assert merged == [("r", request.additional_information)]
    adapter._clear_chunk_ready(output)

    output = scheduler_output()
    adapter.attach_cached_additional_information(output, {"r": request})
    OmniGPUModelRunner._update_additional_information(fake_worker, output)
    assert len(merged) == 1

def test_nonterminal_model_decode_fails_closed():
    model = SimpleNamespace()
    method = Qwen3OmniMoeForConditionalGeneration._thinker_decode_to_talker_decode
    with pytest.raises(TalkerConditioningUnavailable):
        method(
            model,
            {"ids": {"output": [1, 2]}, "embed": {},
             "meta": {"num_processed_tokens": 1, "finished": False}},
            device="cpu",
            update_dict={},
        )


def test_real_omni_model_configs_select_only_async_talker(monkeypatch):
    """Exercise the same OmniEngineArgs -> OmniModelConfig path as StageCore."""
    from pathlib import Path
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.distributed.omni_connectors.transfer_adapter.talker_conditioning import (
        is_qwen3_omni_async_talker,
    )
    from vllm_omni.distributed.omni_connectors.transfer_adapter.base import (
        OmniTransferAdapterBase,
    )

    model = "/data/models/Qwen3-Omni-30B-A3B-Instruct"
    if not Path(model).exists():
        pytest.skip("local Qwen3-Omni config unavailable")
    prefix = "vllm_omni.model_executor.stage_input_processors.qwen3_omni."
    configs = []
    for stage_id, model_stage, next_processor in (
        (0, "thinker", prefix + "thinker2talker_async_chunk"),
        (1, "talker", prefix + "talker2code2wav_async_chunk"),
        (2, "code2wav", None),
    ):
        config = OmniEngineArgs(
            model=model, stage_id=stage_id,
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            model_stage=model_stage, async_chunk=True,
            custom_process_next_stage_input_func=next_processor,
        ).create_model_config()
        configs.append(config)
    assert [hasattr(c, "custom_process_input_func") for c in configs] == [False] * 3
    assert [is_qwen3_omni_async_talker(c) for c in configs] == [False, True, False]

    connector = SimpleNamespace(stage_id=1)
    monkeypatch.setattr(OmniTransferAdapterBase, "__init__", lambda self, config: None)
    monkeypatch.setattr(OmniChunkTransferAdapter, "create_connector",
                        classmethod(lambda cls, config: connector))
    adapter = OmniChunkTransferAdapter(SimpleNamespace(
        model_config=configs[1], scheduler_config=SimpleNamespace(max_num_seqs=64)))
    assert adapter._qwen3_talker_conditioning_gate is True


def test_real_poll_schedule_worker_chain_waits_for_embedding_and_preserves_prefill():
    """Reproduce A01: unready first chunk, long prefill, then first decode."""
    import threading
    from collections import defaultdict
    import torch
    from vllm.v1.core.sched.output import CachedRequestData
    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    class WaitingQueue(list):
        def add_request(self, request):
            self.append(request)
        def prepend_requests(self, requests):
            self[:0] = list(requests)

    initial = {
        "embed": {"prefill": torch.tensor([[1.0], [2.0], [3.0]]),
                  "tts_bos": torch.tensor([[2.0]]),
                  "tts_eos": torch.tensor([[3.0]]),
                  "tts_pad": torch.tensor([[4.0]])},
        "hidden_states": {"output": torch.tensor([[5.0], [6.0], [7.0]])},
        "ids": {"all": [100, 101, 102, 10], "prompt": [100, 101, 102]},
        "meta": {"finished": False},
    }
    def next_chunk(output_ids, value):
        return {
            "embed": {"decode": torch.tensor([[float(value)]])},
            "ids": {"output": list(output_ids)},
            "meta": {"finished": False,
                     "override_keys": [["embed", "decode"], ["ids", "output"]]},
        }
    chunks = deque([initial, next_chunk([10, 11], 11),
                    next_chunk([10, 11, 12], 12)])
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.connector = SimpleNamespace(stage_id=1, get=lambda *args: (chunks.popleft(), 8))
    adapter.model_mode = "ar"
    adapter.scheduler_max_num_seqs = 64
    adapter._qwen3_talker_conditioning_gate = True
    adapter._qwen3_bootstrap_published = set()
    adapter._qwen3_received_entity_rows = {}
    adapter._qwen3_prompt_rows = {}
    adapter._qwen3_entity_layout_invalid = set()
    adapter.request_payload = {}
    adapter.get_req_chunk = defaultdict(int)
    adapter.request_ids_mapping = {}
    adapter.finished_requests = set()
    adapter._finished_load_reqs = set()
    adapter._cancelled_load_reqs = set()
    adapter._drain_state_lock = threading.Lock()
    adapter.requests_with_ready_chunks = set()
    adapter.requests_origin_status = {}
    adapter.waiting_for_chunk_waiting_requests = deque()
    adapter.waiting_for_chunk_running_requests = deque()
    load_calls = []
    adapter.load_async = lambda request: load_calls.append(request.request_id)
    request = SimpleNamespace(
        request_id="r", status=RequestStatus.WAITING,
        num_output_tokens=0, num_computed_tokens=0,
        additional_information=None,
    )
    waiting, running = WaitingQueue([request]), []

    adapter.process_pending_chunks(waiting, running)
    assert request.status == RequestStatus.WAITING_FOR_CHUNK
    assert adapter._poll_single_request(request)
    adapter.restore_queues(waiting, running)
    adapter.process_pending_chunks(waiting, running)
    assert request.status == RequestStatus.WAITING_FOR_CHUNK
    assert not waiting and not adapter.requests_with_ready_chunks
    assert len(load_calls) == 2

    # A second Thinker chunk provides the first ready embedding. The first
    # prefill scheduler output must still contain the original 151 MB-class
    # prefill fields, although this CPU fixture uses tiny tensors.
    assert adapter._poll_single_request(request)
    adapter.restore_queues(waiting, running)
    adapter.process_pending_chunks(waiting, running)
    assert request.status == RequestStatus.WAITING
    assert adapter.requests_with_ready_chunks == {"r"}
    cumulative = request.additional_information
    assert cumulative["ids"]["output"] == [10, 11]
    assert cumulative["ids"]["prompt"] == [100, 101, 102]
    assert torch.equal(cumulative["embed"]["prefill"],
                       torch.tensor([[1.0], [2.0], [3.0], [11.0]]))
    assert cumulative["meta"]["_qwen3_bootstrap_promoted_decode_rows"] == 1
    assert cumulative["meta"]["_qwen3_bootstrap_decode_normalized"] is True
    assert adapter._qwen3_received_entity_rows["r"] == 4
    assert torch.equal(cumulative["hidden_states"]["output"], initial["hidden_states"]["output"])
    assert torch.equal(cumulative["embed"]["decode"], torch.tensor([[11.0]]))

    # Execute the actual assistant-prefill composition. Before the delayed
    # Thinker forward arrives it writes a zero at first-text position 8;
    # after promotion it uses the real embedding and does not consume it
    # again during the first Talker decode (cursor starts at index 1).
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            talker_config=SimpleNamespace(
                codec_nothink_id=1, codec_think_bos_id=2,
                codec_think_eos_id=3, codec_pad_id=5, codec_bos_id=6,
                text_config=SimpleNamespace(hidden_size=1),
            ),
            tts_pad_token_id=0,
        ),
        talker=SimpleNamespace(
            text_projection=lambda x: x,
            embed_input_ids=lambda ids: torch.zeros((ids.numel(), 1), dtype=torch.bfloat16),
        ),
    )
    compose = Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts
    zero = torch.zeros((1, 1), dtype=torch.bfloat16)
    early, _, _, early_detail = compose(fake_model, 0, 4, 4,
                          initial["embed"]["prefill"].to(torch.bfloat16),
                          zero, zero, zero)
    actual, _, _, actual_detail = compose(fake_model, 0, 4, 4,
                           cumulative["embed"]["prefill"].to(torch.bfloat16),
                           zero, zero, zero)
    assert early[8, 0].item() == 0.0
    assert early_detail["first_text_source"] == "zero_fallback"
    assert actual[8, 0].item() == 11.0
    assert actual_detail["first_text_source"] == "real_prefill"
    assert actual_detail["first_text_prefill_index"] == 3

    worker = SimpleNamespace(
        requests={"r": SimpleNamespace()}, model_intermediate_buffer={},
        model=SimpleNamespace(gpu_resident_buffer_keys=set()),
    )
    worker._store_value = OmniGPUModelRunner._store_value.__get__(worker)
    worker._update_intermediate_buffer = OmniGPUModelRunner._update_intermediate_buffer.__get__(worker)
    def cached_data(output_tokens):
        return CachedRequestData(
            req_ids=["r"], resumed_req_ids=set(), new_token_ids=[],
            all_token_ids={}, new_block_ids=[None], num_computed_tokens=[0],
            num_output_tokens=[output_tokens],
        )
    new = OmniNewRequestData(
        req_id="r", prompt_token_ids=[0, 1], mm_features=[],
        sampling_params=None, pooling_params=None, block_ids=([],),
        num_computed_tokens=0, lora_request=None,
        additional_information=cumulative,
    )
    output = SimpleNamespace(scheduled_new_reqs=[new],
                             scheduled_cached_reqs=cached_data(0))
    adapter.postprocess_scheduler_output(output, {"r": request})
    OmniGPUModelRunner._update_additional_information(worker, output)
    assert adapter._qwen3_bootstrap_published == {"r"}
    assert adapter._qwen3_received_entity_rows["r"] == 4
    assert not adapter.requests_with_ready_chunks
    assert torch.equal(worker.model_intermediate_buffer["r"]["embed"]["prefill"],
                       torch.tensor([[1.0], [2.0], [3.0], [11.0]]))

    # A long prompt can need another prefill batch with the same cached
    # conditioning. No connector delta should be injected a second time.
    waiting.remove(request)
    request.status = RequestStatus.RUNNING
    running.append(request)
    adapter.process_pending_chunks(waiting, running)
    partial = SimpleNamespace(scheduled_new_reqs=[],
                              scheduled_cached_reqs=cached_data(0))
    adapter.postprocess_scheduler_output(partial, {"r": request})
    assert partial.scheduled_cached_reqs.additional_information == {}
    assert running == [request]
    assert len(load_calls) == 2

    # Last prefill batch samples one codec token. The worker handoff cursor is
    # one, so ready_exclusive=1 is insufficient for first decode at index 1.
    request.num_output_tokens = 1
    worker.model_intermediate_buffer["r"].setdefault("meta", {})["num_processed_tokens"] = 1
    adapter.process_pending_chunks(waiting, running)
    assert request.status == RequestStatus.WAITING_FOR_CHUNK
    assert not running and len(load_calls) == 3
    assert adapter._qwen3_received_entity_rows["r"] == 4

    assert adapter._poll_single_request(request)
    adapter.restore_queues(waiting, running)
    adapter.process_pending_chunks(waiting, running)
    assert request.status == RequestStatus.RUNNING
    assert adapter.requests_with_ready_chunks == {"r"}
    assert adapter._qwen3_received_entity_rows["r"] == 5
    assert request.num_output_tokens == worker.model_intermediate_buffer["r"]["meta"]["num_processed_tokens"]
    resumed = SimpleNamespace(scheduled_new_reqs=[],
                              scheduled_cached_reqs=cached_data(1))
    adapter.postprocess_scheduler_output(resumed, {"r": request})
    assert resumed.scheduled_cached_reqs.additional_information["r"] is request.additional_information
    OmniGPUModelRunner._update_additional_information(worker, resumed)
    state = worker.model_intermediate_buffer["r"]
    assert state["ids"]["output"] == [10, 11, 12]
    assert torch.equal(state["embed"]["decode"], torch.tensor([[12.0]]))
    state["embed"]["cached_decode"] = torch.tensor([[11.0]])
    model = SimpleNamespace(talker=SimpleNamespace(text_projection=lambda x: x))
    step = Qwen3OmniMoeForConditionalGeneration._thinker_decode_to_talker_decode(
        model, state, torch.device("cpu"), {})
    assert torch.equal(step, torch.tensor([[12.0]]))

    repeated = SimpleNamespace(scheduled_new_reqs=[],
                               scheduled_cached_reqs=cached_data(1))
    adapter.postprocess_scheduler_output(repeated, {"r": request})
    assert repeated.scheduled_cached_reqs.additional_information == {}


def test_bootstrap_keeps_all_unsent_decode_deltas_then_returns_to_incremental_mode():
    import torch
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.request_payload = {}
    first = {
        "embed": {"prefill": torch.tensor([[7.0]])},
        "hidden_states": {"output": torch.tensor([[8.0]])},
        "ids": {"all": [100], "prompt": [100]},
        "meta": {"finished": False},
    }
    adapter._update_request_payload("r", first, preserve_bootstrap=True)
    def delta(ids, value):
        return {
            "embed": {"decode": torch.tensor([[float(value)]])},
            "ids": {"output": list(ids)},
            "meta": {"finished": False,
                     "override_keys": [["embed", "decode"], ["ids", "output"]]},
        }
    adapter._update_request_payload("r", delta([10], 10), preserve_bootstrap=True)
    pending = adapter._update_request_payload("r", delta([10, 11], 11),
                                              preserve_bootstrap=True)
    assert pending["ids"]["output"] == [10, 11]  # cumulative IDs replaced
    assert pending["ids"]["prompt"] == [100]
    assert torch.equal(pending["embed"]["prefill"], torch.tensor([[7.0]]))
    assert torch.equal(pending["embed"]["decode"], torch.tensor([[10.0], [11.0]]))
    assert torch.equal(pending["hidden_states"]["output"], torch.tensor([[8.0]]))

    # Once the first scheduled new request has delivered bootstrap to the
    # worker, each later connector chunk is a single fresh delta again.
    later = adapter._update_request_payload("r", delta([10, 11, 12], 12),
                                            preserve_bootstrap=False)
    assert later["ids"]["output"] == [10, 11, 12]
    assert torch.equal(later["embed"]["decode"], torch.tensor([[12.0]]))
    assert "prefill" not in later["embed"]


def test_prefill_generated_row_is_cached_at_absolute_output_index_zero():
    """A first packet with a generated forward row must seed decode index 0."""
    first = {
        "ids": {"prompt": [1, 2, 3], "all": [1, 2, 3, 10]},
        "embed": {"prefill": torch.tensor([[1.0], [2.0], [3.0], [10.0]])},
        "meta": {"finished": False},
    }
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.request_payload = {}
    prepared = adapter._update_request_payload("r", first, preserve_bootstrap=True)
    assert prepared is not first
    assert not adapter._prepare_qwen3_talker_bootstrap(prepared)
    assert "decode" not in first["embed"]  # raw packet remains unchanged
    assert torch.equal(prepared["embed"]["decode"], torch.tensor([[10.0]]))
    prefill_update = {}
    Qwen3OmniMoeForConditionalGeneration._talker_cache_thinker_decode_embeds(
        SimpleNamespace(), prepared["embed"], prefill_update
    )
    assert torch.equal(prefill_update["embed"]["cached_decode"],
                       torch.tensor([[10.0]]))
    second = {
        "ids": {"output": [10, 11]},
        "embed": {"decode": torch.tensor([[11.0]])},
        "meta": {"finished": False, "override_keys": [["embed", "decode"], ["ids", "output"]]},
    }
    cumulative = adapter._update_request_payload("r", second, preserve_bootstrap=True)
    assert not adapter._prepare_qwen3_talker_bootstrap(cumulative)
    assert torch.equal(cumulative["embed"]["decode"],
                       torch.tensor([[10.0], [11.0]]))
    assert torch.equal(cumulative["embed"]["prefill"],
                       torch.tensor([[1.0], [2.0], [3.0], [10.0]]))


def test_missing_first_word_fails_closed_in_async_model():
    model = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(async_chunk=True)),
        config=SimpleNamespace(
            talker_config=SimpleNamespace(
                codec_nothink_id=1, codec_think_bos_id=2,
                codec_think_eos_id=3, codec_pad_id=5, codec_bos_id=6,
                text_config=SimpleNamespace(hidden_size=1),
            ),
            tts_pad_token_id=0,
        ),
        talker=SimpleNamespace(
            text_projection=lambda x: x,
            embed_input_ids=lambda ids: torch.zeros((ids.numel(), 1), dtype=torch.bfloat16),
        ),
    )
    with pytest.raises(TalkerConditioningUnavailable, match="first text embedding"):
        Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts(
            model, 0, 3, 4, torch.ones((3, 1), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            conditioning_payload={"meta": {"finished": False}},
        )


def test_id_only_readiness_cannot_override_missing_physical_forward_rows():
    adapter, request, _ = adapter_with_request(total=4, output_tokens=0)
    assert adapter._talker_has_next_conditioning(request)
    adapter._qwen3_received_entity_rows["r"] = 10
    assert not adapter._talker_has_next_conditioning(request)
    adapter._qwen3_received_entity_rows["r"] = 11
    assert adapter._talker_has_next_conditioning(request)
    adapter._qwen3_entity_layout_invalid.add("r")
    assert not adapter._talker_has_next_conditioning(request)
