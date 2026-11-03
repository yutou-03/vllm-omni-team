"""Receiver ownership and release gates, with actual adapter publication paths."""
import time
import torch
import pytest
from vllm.v1.request import RequestStatus
from vllm_omni.distributed.omni_connectors.transfer_adapter.receiver_buffer import ReceiverBuffer
from tests.distributed.omni_connectors.test_chunk_transfer_adapter import build_adapter, _req


def setup(build_adapter, monkeypatch, **config):
    monkeypatch.delenv('VLLM_OMNI_RECEIVER_BUFFER_CONFIG', raising=False)
    adapter, connector = build_adapter(stage_id=2, model_mode='generation')
    adapter._receiver_buffer = ReceiverBuffer(adapter, config)
    request = _req('r', RequestStatus.WAITING, external_req_id='attempt-0')
    request.additional_information = {'meta': {'sched_slo_ms': 120000,
        'sched_deadline_monotonic_s': time.monotonic()+120}}
    return adapter, connector, request


def packet(cid, finished=False):
    return ({'codes': {'audio': [cid+1]*16}, 'meta': {'finished': finished,
        'small_hold_chunk_id': cid, 'small_hold_new_frames': 1}}, 128)


def test_prefetch_does_not_publish_and_preserves_order(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch, max_chunks=3)
    connector.get.side_effect = [packet(0), packet(1), packet(2, True)]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._receiver_buffer.streams['r'].received == 3
    assert request.prompt_token_ids == []
    assert adapter.get_req_chunk.get('r', 0) == 0
    for cid in range(3):
        assert adapter._poll_single_request(request)
        assert request.prompt_token_ids == [cid+1]*16
        assert adapter.get_req_chunk['r'] == cid+1
    assert adapter._receiver_buffer.total_bytes == 0
    assert 'r' in adapter.finished_requests


def test_full_queue_stops_get_and_resumes_after_release(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch, max_chunks=2)
    connector.get.side_effect = [packet(0), packet(1), packet(2, True)]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    adapter._receiver_buffer.prefetch_tick()
    assert connector.get.call_count == 2
    adapter._poll_single_request(request)
    adapter._receiver_buffer.prefetch_tick()
    assert connector.get.call_count == 3


def test_cancel_during_get_does_not_resurrect_state(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch)
    adapter.load_async(request)
    def get(*args):
        adapter.cleanup_receiver('r')
        return packet(0)
    connector.get.side_effect = get
    adapter._receiver_buffer.prefetch_tick()
    assert not adapter._receiver_buffer.streams
    assert adapter._receiver_buffer.total_bytes == 0
    assert not adapter.get_req_chunk


def test_reused_id_requires_cleanup(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch)
    adapter.load_async(request)
    replacement = _req('r', RequestStatus.WAITING, external_req_id='attempt-1')
    with pytest.raises(RuntimeError, match='identity reused'):
        adapter._receiver_buffer.register(replacement)
    adapter.cleanup_receiver('r')
    adapter.load_async(replacement)
    assert adapter._receiver_buffer.streams['r'].received == 0


def test_build_gate_skips_first_chunk_and_releases_on_deadline(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch,
        build_after_chunks=1, build_delay_s=.4)
    connector.get.side_effect = [packet(0), packet(1), None]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._poll_single_request(request)
    assert not adapter._poll_single_request(request)
    stream = adapter._receiver_buffer.streams['r']
    assert stream.bytes > 0 and stream.units == 1
    stream.gate_until = time.monotonic()-1
    assert adapter._poll_single_request(request)
    assert stream.gate_done


def test_eos_drains_without_waiting_for_build_timer(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch,
        build_after_chunks=1, build_delay_s=1)
    connector.get.side_effect = [packet(0), packet(1, True)]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._poll_single_request(request)
    assert adapter._poll_single_request(request)
    adapter.cleanup_receiver('r')
    assert adapter._receiver_buffer.total_bytes == 0


def test_release_rate_bounds_even_final_chunk(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch,
        release_units_per_s=16)
    connector.get.side_effect = [packet(0), packet(1, True)]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._poll_single_request(request)
    assert not adapter._poll_single_request(request)
    stream = adapter._receiver_buffer.streams['r']
    assert stream.release_rate == 16
    assert stream.next_release > time.monotonic()
    stream.next_release = time.monotonic()-1
    assert adapter._poll_single_request(request)


def test_deadline_drain_increases_rate_without_losing_order(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch,
        release_units_per_s=1, deadline_drain_guard_s=5)
    request.additional_information['meta']['sched_deadline_monotonic_s'] = time.monotonic()+5.1
    connector.get.side_effect = [packet(0), packet(1, True)]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._poll_single_request(request)
    stream = adapter._receiver_buffer.streams['r']
    assert 0 < stream.next_release-time.monotonic() < .15
    assert stream.received == 2 and stream.released == 1
    assert not adapter._poll_single_request(request)
    stream.next_release = time.monotonic()-1
    assert adapter._poll_single_request(request)
    assert request.prompt_token_ids == [2]*16


def test_sufficient_receiver_stock_skips_build_delay(build_adapter, monkeypatch):
    adapter, connector, request = setup(build_adapter, monkeypatch,
        build_after_chunks=0, build_delay_s=10, build_target_units=2)
    connector.get.side_effect = [packet(0), packet(1), None]
    adapter.load_async(request)
    adapter._receiver_buffer.prefetch_tick()
    assert adapter._poll_single_request(request)
    stream = adapter._receiver_buffer.streams['r']
    assert stream.gate_done and stream.gate_until is None
    assert adapter._poll_single_request(request)
