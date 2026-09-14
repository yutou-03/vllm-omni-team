import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats

from vllm_omni.engine import OmniEngineCoreOutputs
from vllm_omni.engine.snapshot_output_queue import SnapshotOutputQueue, snapshot_key


def snapshot(t=1., engine=0, **kwargs):
    return OmniEngineCoreOutputs(timestamp=t, engine_index=engine,
                                 scheduler_stats=SchedulerStats(**kwargs))


def test_fifty_thousand_snapshots_keep_latest_before_audio():
    q = SnapshotOutputQueue()
    for i in range(50000):
        latest = snapshot(float(i+1), num_running_reqs=i)
        q.put_nowait(latest)
    audio = snapshot(50001.)
    audio.outputs = [SimpleNamespace(request_id="audio")]
    q.put_nowait(audio)
    assert q.qsize() == 2
    assert q.coalesced_snapshots == 49999
    assert q.get_nowait() is latest
    assert q.get_nowait() is audio


@pytest.mark.parametrize("kind", ["data", "finished", "preemption", "wave_zero", "exception", "delta", "reset", "unknown"])
def test_never_coalesce_across_barriers(kind):
    barrier = snapshot(3.)
    if kind == "data":
        barrier.outputs = [SimpleNamespace(request_id="audio")]
    elif kind == "finished":
        barrier.finished_requests = {"r"}
    elif kind == "preemption":
        barrier.stage_preempted_request_ids = ["r"]
    elif kind == "wave_zero":
        barrier.start_wave = 0
    elif kind == "exception":
        barrier = RuntimeError("engine dead")
    elif kind == "delta":
        barrier.scheduler_stats.prefix_cache_stats.hits = 8
    elif kind == "reset":
        barrier.scheduler_stats.prefix_cache_stats.reset = True
    elif kind == "unknown":
        barrier = SimpleNamespace(outputs=[], scheduler_stats=SchedulerStats())
    before, after = snapshot(2.), snapshot(5.)
    q = SnapshotOutputQueue()
    for item in [snapshot(1.), before, barrier, snapshot(4.), after]:
        q.put_nowait(item)
    assert [q.get_nowait() for _ in range(q.qsize())] == [before, barrier, after]


@pytest.mark.parametrize("field,value", [
    ("step_counter", 1), ("current_wave", 1), ("kv_cache_eviction_events", ["event"]),
    ("kv_connector_stats", {"bytes": 1}), ("spec_decoding_stats", object()),
    ("perf_stats", object()), ("cudagraph_stats", object()),
    ("running_lora_adapters", {"adapter": 1}), ("waiting_lora_adapters", {"adapter": 1}),
])
def test_non_gauge_statistics_are_preserved(field, value):
    item = snapshot()
    setattr(item.scheduler_stats, field, value)
    assert snapshot_key(item) is None


def test_unknown_schema_and_extra_fields_fail_closed():
    class Extended(EngineCoreOutputs):
        future_control: int | None = None
    assert snapshot_key(Extended(scheduler_stats=SchedulerStats())) is None
    @dataclass
    class ExtraStats(SchedulerStats):
        new_counter: int = 0
    item = snapshot()
    item.scheduler_stats = ExtraStats()
    assert snapshot_key(item) is None
    item.scheduler_stats = SchedulerStats()
    item.scheduler_stats.future_counter = 0
    assert snapshot_key(item) is None


def test_different_engines_and_backwards_timestamps_are_barriers():
    items = [snapshot(3., 0), snapshot(4., 1), snapshot(2., 1)]
    q = SnapshotOutputQueue()
    for item in items:
        q.put_nowait(item)
    assert [q.get_nowait() for _ in items] == items


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), "invalid", None])
def test_invalid_timestamps_fail_closed(timestamp):
    item = snapshot()
    item.timestamp = timestamp
    assert snapshot_key(item) is None


@pytest.mark.asyncio
async def test_async_waiter_task_done_and_join():
    q = SnapshotOutputQueue()
    first, latest = snapshot(1.), snapshot(2.)
    consumer = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    await q.put(first)
    await q.put(latest)
    assert await asyncio.wait_for(consumer, 1) is latest
    q.task_done()
    await asyncio.wait_for(q.join(), 1)
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
    # Once consumed, the next snapshot is a new queued task, not a replacement.
    await q.put(snapshot(3.))
    assert q.qsize() == 1
    await q.get()
    q.task_done()
    await asyncio.wait_for(q.join(), 1)


def test_default_stage2_installs_production_queue(monkeypatch):
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
    monkeypatch.delenv("VLLM_OMNI_OUTPUT_QUEUE_DIAGNOSTIC", raising=False)
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 2
    client.resources = SimpleNamespace(output_queue_task=None)
    client.outputs_queue = asyncio.Queue()
    captured = []
    def start(self):
        if self.resources.output_queue_task is None:
            captured.append(self.outputs_queue)
            self.resources.output_queue_task = "started"
    monkeypatch.setattr(AsyncMPClient, "_ensure_output_queue_task", start)
    client._ensure_output_queue_task()
    client._ensure_output_queue_task()
    assert len(captured) == 1
    assert isinstance(captured[0], SnapshotOutputQueue)
