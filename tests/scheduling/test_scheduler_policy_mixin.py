from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import FCFSRequestQueue

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.profiling import stage_queue_trace
from vllm_omni.scheduling.policy import BASELINE_POLICY_ENV
from vllm_omni.scheduling.request_queue import PolicyOrderedRequestQueue


class FakeRequest:
    def __init__(
        self,
        request_id: str,
        *,
        deadline: float | None,
        ingress_order: int = 0,
        stage_deadlines: list[float | None] | None = None,
        num_prompt_tokens: int = 10,
        num_computed_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.additional_information = None
        if deadline is not None:
            self.additional_information = {
                "meta": {
                    "sched_source_request_id": request_id,
                    "sched_ingress_order": ingress_order,
                    "sched_deadline_monotonic_s": deadline,
                    "sched_stage_deadline_monotonic_s": stage_deadlines
                    or [deadline - 2.0, deadline - 1.0, deadline],
                }
            }


def _write_concurrent_trace_records(
    trace_dir: str,
    worker_id: int,
    records: int,
) -> None:
    os.environ.pop("STAGE_QUEUE_TRACE_DISABLE", None)
    stage_queue_trace._TRACE_DIR = trace_dir
    for record_id in range(records):
        stage_queue_trace._write_jsonl(
            "iteration_events.jsonl",
            {
                "worker_id": worker_id,
                "record_id": record_id,
                "large_payload": "x" * 16_384,
            },
        )


class FakeBaseScheduler:
    def __init__(self, stage_id: int, *, chunk_adapter=None) -> None:
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(stage_id=stage_id)
        )
        self.waiting = FCFSRequestQueue()
        self.skipped_waiting = FCFSRequestQueue()
        self.running = []
        self.chunk_transfer_adapter = chunk_adapter
        self.native_waiting = self.waiting
        self.native_skipped_waiting = self.skipped_waiting

    def add_request(self, request, *args, **kwargs) -> None:
        self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self):
        return self.skipped_waiting or self.waiting or None

    def _free_request(self, request, *args, **kwargs):
        return request.request_id


class SchedulerHarness(OmniSchedulerMixin, FakeBaseScheduler):
    def __init__(self, stage_id: int, *, chunk_adapter=None) -> None:
        super().__init__(stage_id, chunk_adapter=chunk_adapter)
        self._initialize_baseline_scheduling()


@pytest.fixture(autouse=True)
def disable_stage_trace(monkeypatch):
    monkeypatch.setenv("STAGE_QUEUE_TRACE_DISABLE", "1")


def test_native_fcfs_keeps_original_scheduler_queues(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    scheduler = SchedulerHarness(0)

    assert scheduler.waiting is scheduler.native_waiting
    assert scheduler.skipped_waiting is scheduler.native_skipped_waiting


@pytest.mark.parametrize("stage_id", [0, 1, 2])
def test_stage_deadline_edf_activation(monkeypatch, stage_id):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "stage_deadline_edf_np")
    scheduler = SchedulerHarness(stage_id)

    assert isinstance(scheduler.waiting, PolicyOrderedRequestQueue)


def test_stage_deadline_edf_uses_current_stage_subdeadline(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "stage_deadline_edf_np")
    scheduler = SchedulerHarness(0)
    earlier_stage = FakeRequest(
        "stage-first",
        deadline=100.0,
        ingress_order=0,
        stage_deadlines=[80.0, 90.0, 100.0],
    )
    earlier_final = FakeRequest(
        "final-first",
        deadline=95.0,
        ingress_order=1,
        stage_deadlines=[92.0, 94.0, 95.0],
    )
    scheduler.add_request(earlier_final)
    scheduler.add_request(earlier_stage)

    assert scheduler.waiting.peek_request() is earlier_stage


def test_final_edf_orders_waiting_and_running_without_eviction(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    scheduler = SchedulerHarness(0)
    later = FakeRequest("later", deadline=20.0, ingress_order=0)
    earlier = FakeRequest("earlier", deadline=10.0, ingress_order=1)

    scheduler.add_request(later)
    scheduler.add_request(earlier)
    assert scheduler.waiting.peek_request() is earlier

    scheduler.running = [later, earlier]
    original_members = set(scheduler.running)
    scheduler._baseline_prepare_schedule()
    assert scheduler.running == [earlier, later]
    assert set(scheduler.running) == original_members


def test_final_edf_compares_waiting_and_skipped_queue_heads(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    scheduler = SchedulerHarness(1)
    later = FakeRequest("later", deadline=20.0)
    earlier = FakeRequest("earlier", deadline=10.0)
    scheduler.waiting.add_request(later)
    scheduler.skipped_waiting.add_request(earlier)

    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.skipped_waiting


def test_enabled_policy_rejects_missing_metadata_at_admission(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    scheduler = SchedulerHarness(0)

    with pytest.raises(ValueError, match="no scheduling metadata"):
        scheduler.add_request(FakeRequest("missing", deadline=None))
    assert not scheduler.waiting


@pytest.mark.parametrize("stage_id", [0, 1, 2])
def test_srpf_is_enabled_at_every_stage_without_predictor_metadata(
    monkeypatch,
    stage_id,
):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "srpf_local_np")
    scheduler = SchedulerHarness(stage_id)
    long_request = FakeRequest(
        "long",
        deadline=20.0,
        ingress_order=0,
        num_prompt_tokens=20,
        num_computed_tokens=5,
    )
    short_request = FakeRequest(
        "short",
        deadline=20.0,
        ingress_order=1,
        num_prompt_tokens=10,
        num_computed_tokens=5,
    )
    scheduler.add_request(long_request)
    scheduler.add_request(short_request)

    assert isinstance(scheduler.waiting, PolicyOrderedRequestQueue)
    assert scheduler.waiting.peek_request() is short_request

    long_request.num_computed_tokens = 19
    assert scheduler.waiting.peek_request() is long_request


def test_async_chunk_ready_time_is_recorded_only_after_real_chunk(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    adapter = SimpleNamespace(requests_with_ready_chunks=set())
    scheduler = SchedulerHarness(1, chunk_adapter=adapter)
    request = FakeRequest("chunked", deadline=10.0)
    scheduler.add_request(request)

    assert request.request_id not in scheduler._baseline_data_ready_time
    adapter.requests_with_ready_chunks.add(request.request_id)
    scheduler._baseline_prepare_schedule()
    assert scheduler._baseline_data_ready_time[request.request_id] > 0


def test_request_cleanup_removes_data_ready_state(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    scheduler = SchedulerHarness(0)
    request = FakeRequest("cleanup", deadline=10.0)
    scheduler.add_request(request)
    assert request.request_id in scheduler._baseline_data_ready_time

    scheduler._free_request(request)
    assert request.request_id not in scheduler._baseline_data_ready_time


def test_conformance_trace_records_replayable_edf_decision(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    monkeypatch.setenv("VLLM_OMNI_CONFORMANCE_TRACE", "1")
    scheduler = SchedulerHarness(0)
    scheduler.max_num_running_reqs = 1
    scheduler.max_num_scheduled_tokens = 1
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.25)
    later = FakeRequest("later", deadline=20.0, ingress_order=0)
    earlier = FakeRequest("earlier", deadline=10.0, ingress_order=1)
    scheduler.add_request(later)
    scheduler.add_request(earlier)

    scheduler._baseline_prepare_schedule(token_budget_before=1)
    captured = {}
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_iteration_event",
        lambda **fields: captured.update(fields),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"earlier": 1},
        total_num_scheduled_tokens=1,
        preempted_req_ids=set(),
    )
    scheduler._trace_scheduler_output(
        scheduler_output,
        iteration_id=3,
        timestamp_start=1.0,
        timestamp_end=1.1,
        num_running_before=0,
        num_waiting_before=2,
    )

    assert captured["policy"] == "final_deadline_edf_np"
    assert captured["runnable_req_ids"] == ["earlier", "later"]
    assert captured["policy_keys"]["earlier"][:2] == [10.0, 1]
    assert captured["selected_req_ids"] == ["earlier"]
    assert captured["num_scheduled_tokens"] == {"earlier": 1}
    assert captured["token_budget_before"] == 1
    assert captured["token_budget_after"] == 0
    assert captured["sequence_slots_before"] == 1
    assert captured["ineligible_reasons"] == {}
    assert captured["policy_activation"] is True


def test_conformance_trace_default_path_is_safe_temp_directory():
    assert stage_queue_trace._DEFAULT_TRACE_DIR.startswith(
        tempfile.gettempdir() + os.sep
    )
    assert "motivation/stage_queue" not in stage_queue_trace._DEFAULT_TRACE_DIR


def test_request_metrics_trace_mode_keeps_only_calibration_join_events(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("STAGE_QUEUE_TRACE_DISABLE", raising=False)
    monkeypatch.setenv("STAGE_QUEUE_TRACE_MODE", "request_metrics")
    monkeypatch.setattr(stage_queue_trace, "_TRACE_DIR", str(tmp_path))

    stage_queue_trace._write_jsonl(
        "stage_events.jsonl",
        {"event": "request_ingress", "request_id": "server-id"},
    )
    stage_queue_trace._write_jsonl(
        "stage_events.jsonl",
        {"event": "stage_done_metrics", "request_id": "server-id"},
    )
    stage_queue_trace._write_jsonl(
        "stage_events.jsonl",
        {"event": "stage_forward", "request_id": None},
    )
    stage_queue_trace._write_jsonl(
        "iteration_events.jsonl",
        {"event": "iteration", "iteration_id": 1},
    )
    stage_queue_trace._write_jsonl(
        "connector_events.jsonl",
        {"event": "transfer_rx", "request_id": "server-id"},
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "stage_events.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "request_ingress",
        "stage_done_metrics",
    ]
    assert not (tmp_path / "iteration_events.jsonl").exists()
    assert not (tmp_path / "connector_events.jsonl").exists()


def test_unknown_trace_mode_fails_open_to_full_evidence(monkeypatch):
    monkeypatch.setenv("STAGE_QUEUE_TRACE_MODE", "misspelled")
    assert stage_queue_trace._record_enabled(
        "iteration_events.jsonl",
        {"event": "iteration"},
    )


def test_trace_jsonl_records_do_not_interleave_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    worker_count = 6
    records_per_worker = 20
    processes = [
        context.Process(
            target=_write_concurrent_trace_records,
            args=(str(tmp_path), worker_id, records_per_worker),
        )
        for worker_id in range(worker_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("concurrent trace writer did not exit")
        assert process.exitcode == 0

    lines = (tmp_path / "iteration_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == worker_count * records_per_worker
    assert all(len(record["large_payload"]) == 16_384 for record in records)
    assert {
        (record["worker_id"], record["record_id"])
        for record in records
    } == {
        (worker_id, record_id)
        for worker_id in range(worker_count)
        for record_id in range(records_per_worker)
    }
