import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats

from vllm_omni.profiling.output_queue_diagnostic import DiagnosticOutputQueue, is_pure_statistics


def message(**kwargs):
    return EngineCoreOutputs(scheduler_stats=SchedulerStats(), **kwargs)


def test_pure_statistics_preserves_control_and_unknown_fields():
    assert is_pure_statistics(message())
    assert not is_pure_statistics(message(finished_requests={"done"}))
    assert not is_pure_statistics(message(start_wave=0))
    assert not is_pure_statistics(message(wave_complete=0))
    assert not is_pure_statistics(RuntimeError("dead"))
    assert not is_pure_statistics(SimpleNamespace(outputs=[], scheduler_stats=1))
    class Extended:
        __struct_fields__ = ("outputs", "scheduler_stats", "future_control")
        outputs = []
        scheduler_stats = 1
        future_control = False
    assert not is_pure_statistics(Extended())


@pytest.mark.parametrize("mode", ["observe", "drop_stats"])
def test_queue_retains_data_control_exception_and_records_wait(mode):
    events = []
    clock = [10.]
    q = DiagnosticOutputQueue(stage_id=2, mode=mode,
                              emit=lambda event, **fields: events.append((event, fields)),
                              clock=lambda: clock[0])
    stats = message()
    audio = message(outputs=[SimpleNamespace(request_id="audio-1")])
    control = message(finished_requests={"audio-1"})
    error = RuntimeError("dead")
    for item in (stats, audio, control, error):
        q.put_nowait(item)
    clock[0] = 13.
    expected = [stats, audio, control, error] if mode == "observe" else [audio, control, error]
    assert [q.get_nowait() for _ in expected] == expected
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
    enq = next(fields for name, fields in events if name == "output_queue_enqueued")
    deq = next(fields for name, fields in events if name == "output_queue_dequeued")
    assert enq["depth_before"] == (1 if mode == "observe" else 0)
    assert deq["queue_wait_s"] == 3.
    assert enq["output_message_seq"] == deq["output_message_seq"]
    assert not q.metadata
    assert not any(q.depths.values())
    assert q.counts["received_stats"] == 1
    assert q.counts["dropped_stats"] == (1 if mode == "drop_stats" else 0)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        DiagnosticOutputQueue(stage_id=2, mode="typo", emit=lambda *a, **kw: None)


@pytest.mark.parametrize("stage,mode,wrapped", [(2, "observe", True), (2, "off", False), (0, "drop_stats", False)])
def test_receiver_captures_wrapped_queue_only_when_requested(monkeypatch, stage, mode, wrapped):
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = stage
    client.resources = SimpleNamespace(output_queue_task=None)
    client.outputs_queue = asyncio.Queue()
    captured = []
    def start(self):
        if self.resources.output_queue_task is None:
            captured.append(self.outputs_queue)
            self.resources.output_queue_task = "receiver"
    monkeypatch.setattr(AsyncMPClient, "_ensure_output_queue_task", start)
    monkeypatch.setenv("VLLM_OMNI_OUTPUT_QUEUE_DIAGNOSTIC", mode)
    client._ensure_output_queue_task()
    client._ensure_output_queue_task()
    assert len(captured) == 1
    assert captured[0] is client.outputs_queue
    assert isinstance(client.outputs_queue, DiagnosticOutputQueue) is wrapped
