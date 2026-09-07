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
from vllm_omni.scheduling.deadline import MilestoneCostEstimate
from vllm_omni.scheduling.admission import StaticConcurrencyEnvelope
from vllm_omni.scheduling.policy import BaselineSchedulingPolicy
from vllm_omni.scheduling.request_queue import PolicyOrderedRequestQueue
from vllm_omni.scheduling.signals import GPUActiveSample


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
        max_tokens: int | None = 8,
        num_output_placeholders: int = 0,
    ) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.max_tokens = max_tokens
        self.num_output_placeholders = num_output_placeholders
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
    def __init__(
        self,
        stage_id: int,
        *,
        chunk_adapter=None,
        additional_config=None,
        max_num_running_reqs=None,
    ) -> None:
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(stage_id=stage_id),
            additional_config=additional_config or {},
        )
        self.waiting = FCFSRequestQueue()
        self.skipped_waiting = FCFSRequestQueue()
        self.running = []
        self.chunk_transfer_adapter = chunk_adapter
        if max_num_running_reqs is not None:
            self.max_num_running_reqs = max_num_running_reqs
        self.native_waiting = self.waiting
        self.native_skipped_waiting = self.skipped_waiting

    def add_request(self, request, *args, **kwargs) -> None:
        self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self):
        return self.skipped_waiting or self.waiting or None

    def _free_request(self, request, *args, **kwargs):
        return request.request_id


class SchedulerHarness(OmniSchedulerMixin, FakeBaseScheduler):
    def __init__(
        self,
        stage_id: int,
        *,
        chunk_adapter=None,
        additional_config=None,
        max_num_running_reqs=None,
    ) -> None:
        super().__init__(
            stage_id,
            chunk_adapter=chunk_adapter,
            additional_config=additional_config,
            max_num_running_reqs=max_num_running_reqs,
        )
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


def _stage_ddl_config(mode="stage_ddl_fixed"):
    return {
        "slo_scheduling": {
            "enabled": True,
            "mode": mode,
            "stage_ddl": {
                "model_path": "/models/p3.json",
                "model_sha256": "a" * 64,
                "input_token_cutoff": 128,
                "milestone_guard_ms": {
                    "ttft": 5.0,
                    "ttfp": 5.0,
                    "e2e": 0.0,
                },
            },
        }
    }


def _fair_stage_ddl_config():
    config = _stage_ddl_config("stage_ddl_fair_fixed")
    config["slo_scheduling"]["stage_ddl"]["fairness"] = {
        "bounded_wait_ms_by_stage": {"0": 2000.0}
    }
    return config


def _dynamic_stage_ddl_config(*, decrease_hold=20):
    config = _stage_ddl_config("stage_ddl_dynamic_admission")
    config["slo_scheduling"]["stage_ddl"]["dynamic_admission"] = {
        "envelope_path": "/models/p4-envelope.json",
        "envelope_sha256": "b" * 64,
        "target_stage_ids": [1],
        "initial_active_by_stage": {"1": 2},
        "controller": {
            "min_active": 1,
            "max_active": 2,
            "increase_hold": 50,
            "decrease_hold": decrease_hold,
            "cooldown_epochs": 100,
            "increase_step": 1,
            "decrease_factor": 0.5,
            "gpu_low_watermark": 0.6,
            "kv_low_watermark": 0.01,
            "kv_high_watermark": 0.9,
            "queue_growth_high_rps": None,
            "connector_backlog_high": 2,
            "require_downstream_pressure": True,
            "reset_to_safe_cap_when_fully_idle": True,
        },
    }
    return config


class FakeMilestoneModel:
    cost_model_version = "p3-static:test"
    cost_source = "static_component_p95_plus_first_output_guard"

    def predict(self, *, request_path, stage_id, milestone, num_prompt_tokens):
        del request_path, stage_id
        name = getattr(milestone, "value", milestone)
        if name == "ttft":
            current = 900.0 if num_prompt_tokens > 128 else 100.0
        elif name == "ttfp":
            current = 200.0
        else:
            current = 500.0
        return MilestoneCostEstimate(current_stage_service_ms=current)


def _set_stage_ddl_metadata(request, *, ingress_order, ttft, e2e):
    request.additional_information = {
        "meta": {
            "sched_schema_version": 2,
            "sched_source_request_id": request.request_id,
            "sched_ingress_order": ingress_order,
            "sched_ingress_monotonic_s": 100.0,
            "sched_logical_request_id": request.request_id,
            "sched_request_path": "text",
            "sched_service_class": "interactive",
            "sched_milestone_deadlines_monotonic_s": {
                "ttft": ttft,
                "e2e": e2e,
            },
            "sched_max_preemptions": 1,
        }
    }
    return request


def test_stage_ddl_mode_loads_model_and_orders_only_waiting(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    loaded = []

    def load_model(path, **kwargs):
        loaded.append((path, kwargs))
        return FakeMilestoneModel()

    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        load_model,
    )
    scheduler = SchedulerHarness(0, additional_config=_stage_ddl_config())
    short = _set_stage_ddl_metadata(
        FakeRequest("short", deadline=None, num_prompt_tokens=36),
        ingress_order=0,
        ttft=102.0,
        e2e=110.0,
    )
    long = _set_stage_ddl_metadata(
        FakeRequest("long", deadline=None, num_prompt_tokens=216),
        ingress_order=1,
        ttft=102.0,
        e2e=110.0,
    )
    scheduler.add_request(short)
    scheduler.add_request(long)

    assert scheduler._baseline_policy is BaselineSchedulingPolicy.STAGE_DDL_FIXED
    assert isinstance(scheduler.waiting, PolicyOrderedRequestQueue)
    assert scheduler.waiting.peek_request() is long
    assert loaded == [
        (
            "/models/p3.json",
            {
                "expected_sha256": "a" * 64,
                "input_token_cutoff": 128,
                "milestone_guard_ms": {
                    "ttft": 5.0,
                    "ttfp": 5.0,
                    "e2e": 0.0,
                },
            },
        )
    ]

    scheduler.running = [short, long]
    scheduler._baseline_prepare_schedule()
    assert scheduler.running == [short, long]


def test_fair_fixed_mode_promotes_waiter_without_initializing_dynamic_cap(
    monkeypatch,
):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.time.monotonic",
        lambda: 102.0,
    )
    scheduler = SchedulerHarness(
        0,
        additional_config=_fair_stage_ddl_config(),
        max_num_running_reqs=1,
    )
    urgent = _set_stage_ddl_metadata(
        FakeRequest("urgent", deadline=None, num_prompt_tokens=216),
        ingress_order=1,
        ttft=103.0,
        e2e=110.0,
    )
    loose = _set_stage_ddl_metadata(
        FakeRequest("loose", deadline=None, num_prompt_tokens=36),
        ingress_order=0,
        ttft=112.0,
        e2e=120.0,
    )
    scheduler.add_request(urgent)
    scheduler.add_request(loose)
    scheduler._baseline_data_ready_time["urgent"] = 101.9
    scheduler._baseline_data_ready_time["loose"] = 100.0

    assert scheduler._baseline_policy is (
        BaselineSchedulingPolicy.STAGE_DDL_FAIR_FIXED
    )
    assert isinstance(scheduler.waiting, PolicyOrderedRequestQueue)
    assert scheduler._slo_admission_state is None
    assert scheduler._slo_physical_max_num_seqs is None
    assert scheduler.max_num_running_reqs == 1
    assert scheduler.waiting.peek_request() is loose


def test_fair_fixed_conformance_trace_marks_protected_candidates(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_CONFORMANCE_TRACE", "1")
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.time.monotonic",
        lambda: 102.0,
    )
    scheduler = SchedulerHarness(
        0,
        additional_config=_fair_stage_ddl_config(),
        max_num_running_reqs=1,
    )
    protected = _set_stage_ddl_metadata(
        FakeRequest("protected", deadline=None),
        ingress_order=0,
        ttft=112.0,
        e2e=120.0,
    )
    unprotected = _set_stage_ddl_metadata(
        FakeRequest("unprotected", deadline=None),
        ingress_order=1,
        ttft=103.0,
        e2e=110.0,
    )
    scheduler.add_request(protected)
    scheduler.add_request(unprotected)
    scheduler._baseline_data_ready_time["protected"] = 100.0
    scheduler._baseline_data_ready_time["unprotected"] = 101.9

    scheduler._baseline_prepare_schedule(token_budget_before=1)

    snapshot = scheduler._baseline_conformance_snapshot
    assert snapshot is not None
    assert snapshot["bounded_wait_fairness"] == {
        "threshold_ms": 2000.0,
        "protected_by_request": {
            "protected": True,
            "unprotected": False,
        },
    }


def test_profiled_control_epoch_advances_once_per_scheduler_tick(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    scheduler = SchedulerHarness(1, additional_config=_stage_ddl_config())

    assert scheduler._slo_control_epoch == 0
    scheduler._baseline_prepare_schedule()
    assert scheduler._slo_control_epoch == 1
    scheduler._baseline_prepare_schedule()
    assert scheduler._slo_control_epoch == 2


def test_dynamic_stage1_loads_envelope_and_applies_initial_cap(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    loaded = []

    def load_envelope(path, **kwargs):
        loaded.append((path, kwargs))
        return StaticConcurrencyEnvelope(
            schema_version=1,
            artifact_id="p4-stage1-cap-v3",
            safe_max_active_by_stage={0: 1, 1: 2, 2: 1},
        )

    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticConcurrencyEnvelope.from_json_file",
        load_envelope,
    )
    scheduler = SchedulerHarness(
        1,
        additional_config=_dynamic_stage_ddl_config(),
        max_num_running_reqs=2,
    )

    assert scheduler._baseline_policy is (
        BaselineSchedulingPolicy.STAGE_DDL_FIXED
    )
    assert scheduler._slo_physical_max_num_seqs == 2
    assert scheduler.effective_max_num_seqs == 2
    assert scheduler.max_num_running_reqs == 2
    assert scheduler._slo_admission_state.current_cap == 2
    assert loaded == [
        (
            "/models/p4-envelope.json",
            {"expected_sha256": "b" * 64},
        )
    ]


def test_dynamic_stage1_applies_resource_driven_drain_only_cap(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticConcurrencyEnvelope.from_json_file",
        lambda *_args, **_kwargs: StaticConcurrencyEnvelope(
            schema_version=1,
            artifact_id="p4-stage1-cap-v3",
            safe_max_active_by_stage={0: 1, 1: 2, 2: 1},
        ),
    )
    decisions = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **fields: decisions.append((event, fields)),
    )
    scheduler = SchedulerHarness(
        1,
        additional_config=_dynamic_stage_ddl_config(decrease_hold=1),
        max_num_running_reqs=2,
    )
    scheduler.max_num_scheduled_tokens = 100
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.95)
    scheduler._stage_gpu_active_sampler = SimpleNamespace(
        sample=lambda **_kwargs: GPUActiveSample(
            ratio=0.7,
            source="nvml_mean_visible_devices",
            device_count=1,
            sample_age_ms=5.0,
        )
    )
    now = 101.0
    at_risk = _set_stage_ddl_metadata(
        FakeRequest("at-risk", deadline=None),
        ingress_order=0,
        ttft=100.1,
        e2e=101.1,
    )
    at_risk.additional_information["meta"]["sched_request_path"] = "audio"
    at_risk.additional_information["meta"][
        "sched_milestone_deadlines_monotonic_s"
    ]["ttfp"] = 100.1
    at_risk.status = SimpleNamespace(name="RUNNING")
    safe = _set_stage_ddl_metadata(
        FakeRequest("safe", deadline=None),
        ingress_order=1,
        ttft=100.1,
        e2e=101.1,
    )
    safe.additional_information["meta"]["sched_request_path"] = "audio"
    safe.additional_information["meta"][
        "sched_milestone_deadlines_monotonic_s"
    ]["ttfp"] = 100.1
    safe.status = SimpleNamespace(name="RUNNING")
    scheduler.running = [at_risk, safe]
    scheduler._update_slo_downstream_pressure(
        {
            "pressure_feedback_schema_version": 1,
            "stage_id": 2,
            "control_epoch": 9,
            "captured_monotonic_s": now,
            "waiting_requests": 0,
            "effective_max_num_seqs": 1,
        }
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.time.monotonic",
        lambda: now,
    )

    scheduler._baseline_prepare_schedule(token_budget_before=100)

    assert scheduler.effective_max_num_seqs == 1
    # vLLM asserts running <= max_num_running_reqs before it can drain.  The
    # enforcement ceiling therefore follows the existing population down,
    # while the logical effective cap immediately blocks new admissions.
    assert scheduler.max_num_running_reqs == 2
    assert len(scheduler.running) == 2
    decision = scheduler._slo_last_admission_decision
    assert decision.action.value == "decrease"
    assert decision.drain_only is True
    assert decision.evicted_requests == 0
    assert decisions[-1][0] == "stage_admission_decision"
    assert decisions[-1][1]["new_effective_max_num_seqs"] == 1
    assert decisions[-1][1]["scheduler_enforcement_max_num_seqs"] == 2

    scheduler.running = [at_risk]
    scheduler._baseline_prepare_schedule(token_budget_before=100)
    assert scheduler.effective_max_num_seqs == 1
    assert scheduler.max_num_running_reqs == 1

    scheduler.running = []
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        requests_with_ready_chunks=set(),
        waiting_for_chunk_running_requests=[],
        waiting_for_chunk_waiting_requests=[],
        get_control_signal_snapshot=lambda: {
            "sender_pending_save_requests": 0,
            "sender_save_inflight": 0,
        },
    )
    scheduler._baseline_prepare_schedule(token_budget_before=100)
    assert scheduler._slo_last_admission_decision.action.value == "reset"
    assert scheduler.effective_max_num_seqs == 2
    assert scheduler.max_num_running_reqs == 2

    pressure = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=now,
    )
    assert pressure["physical_max_num_seqs"] == 2
    assert pressure["effective_max_num_seqs"] == 2


def test_clockwork_mode_loads_same_model_but_orders_by_e2e_laxity(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *args, **kwargs: FakeMilestoneModel(),
    )
    scheduler = SchedulerHarness(
        0, additional_config=_stage_ddl_config("clockwork_latest_start_fixed")
    )
    earlier_e2e = _set_stage_ddl_metadata(
        FakeRequest("e2e-first", deadline=None, num_prompt_tokens=36),
        ingress_order=0,
        ttft=102.0,
        e2e=104.0,
    )
    tighter_ttft = _set_stage_ddl_metadata(
        FakeRequest("ttft-first", deadline=None, num_prompt_tokens=216),
        ingress_order=1,
        ttft=101.0,
        e2e=110.0,
    )
    scheduler.add_request(tighter_ttft)
    scheduler.add_request(earlier_e2e)

    assert (
        scheduler._baseline_policy
        is BaselineSchedulingPolicy.CLOCKWORK_LATEST_START_NP
    )
    assert scheduler.waiting.peek_request() is earlier_e2e


def test_stage_ddl_freezes_one_snapshot_time_per_scheduler_tick(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *args, **kwargs: FakeMilestoneModel(),
    )
    scheduler = SchedulerHarness(0, additional_config=_stage_ddl_config())
    first = _set_stage_ddl_metadata(
        FakeRequest("first", deadline=None),
        ingress_order=0,
        ttft=999999.0,
        e2e=1000000.0,
    )
    second = _set_stage_ddl_metadata(
        FakeRequest("second", deadline=None),
        ingress_order=1,
        ttft=999999.0,
        e2e=1000000.0,
    )
    scheduler.add_request(first)
    scheduler.add_request(second)
    list(scheduler.waiting)
    initial = dict(scheduler._stage_ddl_snapshot_by_request)
    assert len(initial) == 2
    assert len({item.captured_monotonic_s for item in initial.values()}) == 1

    scheduler._baseline_prepare_schedule()
    list(scheduler.waiting)
    refreshed = scheduler._stage_ddl_snapshot_by_request
    assert len({item.captured_monotonic_s for item in refreshed.values()}) == 1
    assert {item.deadline_version for item in refreshed.values()} == {
        next(iter(initial.values())).deadline_version + 1
    }


def test_stage_ddl_rejects_legacy_metadata_and_policy_conflicts(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *args, **kwargs: FakeMilestoneModel(),
    )
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    scheduler = SchedulerHarness(0, additional_config=_stage_ddl_config())
    with pytest.raises(ValueError, match="schema_version"):
        scheduler.add_request(FakeRequest("legacy", deadline=200.0))

    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    with pytest.raises(ValueError, match="conflicts with enabled SLO mode"):
        SchedulerHarness(0, additional_config=_stage_ddl_config())


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
    assert scheduler._baseline_waiting_queue_choices == [
        {
            "waiting_head_request_id": "later",
            "waiting_head_policy_key": [20.0, 0, "later", "later"],
            "skipped_waiting_head_request_id": "earlier",
            "skipped_waiting_head_policy_key": [
                10.0,
                0,
                "earlier",
                "earlier",
            ],
            "chosen_queue": "skipped_waiting",
            "differs_from_native_fcfs": False,
        }
    ]


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
    assert scheduler._baseline_data_ready_order[request.request_id] == 0


def test_request_cleanup_removes_data_ready_state(monkeypatch):
    monkeypatch.setenv(BASELINE_POLICY_ENV, "final_deadline_edf_np")
    scheduler = SchedulerHarness(0)
    request = FakeRequest("cleanup", deadline=10.0)
    scheduler.add_request(request)
    assert request.request_id in scheduler._baseline_data_ready_time

    scheduler._free_request(request)
    assert request.request_id not in scheduler._baseline_data_ready_time
    assert request.request_id not in scheduler._baseline_data_ready_order


def test_stage_data_ready_is_traced_once_for_native_fcfs(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    captured = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **fields: captured.append((event, fields)),
    )
    scheduler = SchedulerHarness(0)
    request = FakeRequest("ready", deadline=None)

    scheduler.add_request(request)
    scheduler._baseline_mark_data_ready(
        request.request_id,
        ready_time=999.0,
    )

    assert [event for event, _ in captured] == [
        "stage_data_ready",
        "server_receive",
        "stage_enqueue",
    ]
    ready_fields = captured[0][1]
    assert ready_fields["request_id"] == "ready"
    assert ready_fields["stage_ready_order"] == 0


def test_stage_data_ready_reuses_admission_identity(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    captured = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **fields: captured.append((event, fields)),
    )
    scheduler = SchedulerHarness(2)
    request = FakeRequest("logical__pipeline0__stage2__attempt1", deadline=None)
    request.additional_information = {
        "meta": {
            "sched_logical_request_id": "logical",
            "omni_logical_request_id": "logical",
            "omni_execution_id": "logical",
            "omni_pipeline_attempt_id": 0,
            "omni_stage_attempt_id": 1,
            "omni_recovery_target_stage_id": 1,
            "omni_recovery_target_num_computed_tokens": 37,
            "omni_recovery_preemption_cause": "controlled_restart",
        }
    }

    scheduler.add_request(request)

    ready_fields = next(
        fields for event, fields in captured if event == "stage_data_ready"
    )
    assert ready_fields["logical_request_id"] == "logical"
    assert ready_fields["execution_id"] == "logical"
    assert ready_fields["pipeline_attempt_id"] == 0
    assert ready_fields["stage_attempt_id"] == 1


def test_stage_data_ready_captures_prediction_features_and_pressure(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.stage_trace_enabled",
        lambda: True,
    )
    captured = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **fields: captured.append((event, fields)),
    )
    scheduler = SchedulerHarness(0)
    scheduler.max_num_running_reqs = 4
    scheduler.effective_max_num_seqs = 3
    scheduler.max_num_scheduled_tokens = 1024
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.25)
    scheduler.num_waiting_for_streaming_input = 2
    request = FakeRequest(
        "profile-me",
        deadline=None,
        num_prompt_tokens=17,
        num_computed_tokens=3,
        max_tokens=29,
        num_output_placeholders=5,
    )

    scheduler.add_request(request)

    ready_fields = next(
        fields for event, fields in captured if event == "stage_data_ready"
    )
    assert ready_fields["stage_cost_features"] == {
        "feature_schema_version": 1,
        "num_prompt_tokens": 17,
        "max_output_tokens": 29,
        "num_computed_tokens": 3,
        "num_output_placeholders": 5,
        "candidate_sequence_demand": 1,
    }
    assert ready_fields["pressure_snapshot"] == {
        "pressure_schema_version": 2,
        "request_count_semantics": "runnable_excludes_input_blocked",
        "captured_monotonic_s": pytest.approx(
            ready_fields["data_ready_monotonic_s"]
        ),
        "control_epoch": 0,
        "running_requests": 0,
        "waiting_requests": 0,
        "input_waiting_requests": 2,
        "physical_max_num_seqs": 4,
        "effective_max_num_seqs": 3,
        "token_budget_total": 1024,
        "token_budget_available": 1024,
        "kv_cache_usage_ratio": 0.25,
    }


def test_disabled_trace_skips_data_ready_feature_snapshot(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.stage_trace_enabled",
        lambda: False,
    )
    scheduler = SchedulerHarness(0)
    monkeypatch.setattr(
        scheduler,
        "_capture_stage_runtime_snapshot",
        lambda **_kwargs: pytest.fail("disabled trace built pressure snapshot"),
    )
    request = FakeRequest("no-trace-work", deadline=None)

    scheduler.add_request(request)


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
    assert captured["stage_ready_order"] == {"later": 0, "earlier": 1}
    assert captured["queue_orders_before_policy"]["waiting"] == [
        "later",
        "earlier",
    ]
    assert captured["queue_orders_after_policy"]["waiting"] == [
        "earlier",
        "later",
    ]
    assert captured["policy_reordered"] is True
    assert captured["policy_reordered_domains"] == ["waiting"]
    assert captured["unselected_runnable_req_ids"] == ["later"]
    assert captured["selection_differs_from_pre_policy_prefix"] is True
    assert captured["selection_changed_domains"] == ["waiting"]
    shadow = captured["shadow_policy_decision"]
    assert shadow["comparison"] == "fixed_location_admission_count"
    assert shadow["nonpreemptive_running_req_ids"] == []
    assert shadow["policies"]["native_fcfs"]["waiting"] == {
        "candidate_order": ["later", "earlier"],
        "actual_selected_req_ids": ["earlier"],
        "selected_count": 1,
        "fcfs_prefix_req_ids": ["later"],
        "shadow_prefix_req_ids": ["later"],
        "order_differs_from_fcfs": False,
        "fixed_count_substitution_from_fcfs": False,
        "fixed_count_substitution_from_actual": True,
    }
    assert shadow["policies"]["final_deadline_edf_np"]["waiting"] == {
        "candidate_order": ["earlier", "later"],
        "actual_selected_req_ids": ["earlier"],
        "selected_count": 1,
        "fcfs_prefix_req_ids": ["later"],
        "shadow_prefix_req_ids": ["earlier"],
        "order_differs_from_fcfs": True,
        "fixed_count_substitution_from_fcfs": True,
        "fixed_count_substitution_from_actual": False,
    }


def test_iteration_trace_carries_immutable_stage_pressure_snapshot(monkeypatch):
    scheduler = SchedulerHarness(1)
    scheduler.max_num_running_reqs = 8
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.75)
    scheduler.running = [FakeRequest("running", deadline=None)]
    scheduler.waiting.add_request(FakeRequest("waiting", deadline=None))
    captured = {}
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.stage_trace_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_iteration_event",
        lambda **fields: captured.update(fields),
    )

    scheduler._trace_scheduler_output(
        SimpleNamespace(
            num_scheduled_tokens={"running": 128},
            total_num_scheduled_tokens=128,
            preempted_req_ids=set(),
        ),
        iteration_id=3,
        timestamp_start=1.0,
        timestamp_end=1.1,
        num_running_before=1,
        num_waiting_before=1,
    )

    pressure = dict(captured["pressure_snapshot"])
    captured_monotonic_s = pressure.pop("captured_monotonic_s")
    assert captured_monotonic_s >= 0
    assert pressure == {
        "pressure_schema_version": 2,
        "request_count_semantics": "runnable_excludes_input_blocked",
        "control_epoch": 0,
        "running_requests": 1,
        "waiting_requests": 1,
        "input_waiting_requests": 1,
        "physical_max_num_seqs": 8,
        "effective_max_num_seqs": 8,
        "token_budget_total": 4096,
        "token_budget_available": 3968,
        "token_budget_utilization_ratio": 0.03125,
        "kv_cache_usage_ratio": 0.75,
    }


def test_profiled_pressure_tracks_queue_growth_and_running_deadline_risk(
    monkeypatch,
):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin."
        "StaticMilestoneCostModel.from_json_file",
        lambda *_args, **_kwargs: FakeMilestoneModel(),
    )
    scheduler = SchedulerHarness(0, additional_config=_stage_ddl_config())
    scheduler.max_num_running_reqs = 2
    scheduler.effective_max_num_seqs = 2
    scheduler.max_num_scheduled_tokens = 100
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.25)
    scheduler._stage_gpu_active_sampler = SimpleNamespace(
        sample=lambda **_kwargs: GPUActiveSample(
            ratio=0.42,
            source="nvml_mean_visible_devices",
            device_count=1,
            sample_age_ms=7.0,
        )
    )
    at_risk = _set_stage_ddl_metadata(
        FakeRequest("at-risk", deadline=None),
        ingress_order=0,
        ttft=100.05,
        e2e=100.4,
    )
    safe = _set_stage_ddl_metadata(
        FakeRequest("safe", deadline=None),
        ingress_order=1,
        ttft=102.0,
        e2e=103.0,
    )
    at_risk.status = SimpleNamespace(name="RUNNING")
    safe.status = SimpleNamespace(name="RUNNING")
    scheduler.running = [at_risk, safe]
    waiting_1 = _set_stage_ddl_metadata(
        FakeRequest("waiting-1", deadline=None),
        ingress_order=2,
        ttft=102.0,
        e2e=103.0,
    )
    scheduler.waiting.add_request(waiting_1)
    scheduler._stage_ddl_tick_monotonic_s = 100.0
    scheduler._stage_ddl_snapshot_by_request.clear()

    first = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=100.0,
        token_budget_available=75,
        update_control_signals=True,
    )
    waiting_2 = _set_stage_ddl_metadata(
        FakeRequest("waiting-2", deadline=None),
        ingress_order=3,
        ttft=102.0,
        e2e=103.0,
    )
    scheduler.waiting.add_request(waiting_2)
    second = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=100.5,
        token_budget_available=50,
        update_control_signals=True,
    )

    assert first["token_budget_utilization_ratio"] == 0.25
    assert "queue_growth_rate_rps" not in first
    assert first["deadline_miss_risk"] == 0.5
    assert first["deadline_miss_risk_population"] == 2
    assert first["deadline_miss_risk_scope"] == "runnable_running"
    assert first["gpu_active_ratio"] == 0.42
    assert first["gpu_active_source"] == "nvml_mean_visible_devices"
    assert first["gpu_active_device_count"] == 1
    assert first["gpu_active_sample_age_ms"] == 7.0
    assert second["token_budget_utilization_ratio"] == 0.5
    assert second["queue_growth_rate_rps"] == pytest.approx(2.0)


def test_pressure_snapshot_excludes_input_blocked_requests_from_active_counts():
    scheduler = SchedulerHarness(2)
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_scheduled_tokens = 1024
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.25)
    ready_running = FakeRequest("ready-running", deadline=None)
    ready_running.status = SimpleNamespace(name="RUNNING")
    chunk_blocked_running = FakeRequest("chunk-running", deadline=None)
    chunk_blocked_running.status = SimpleNamespace(name="WAITING_FOR_CHUNK")
    ready_waiting = FakeRequest("ready-waiting", deadline=None)
    ready_waiting.status = SimpleNamespace(name="WAITING")
    input_blocked_waiting = FakeRequest("input-waiting", deadline=None)
    input_blocked_waiting.status = SimpleNamespace(name="WAITING_FOR_INPUT")
    adapter_only = FakeRequest("adapter-only", deadline=None)
    adapter_only.status = SimpleNamespace(name="WAITING_FOR_CHUNK")

    scheduler.running = [ready_running, chunk_blocked_running]
    scheduler.waiting.add_request(ready_waiting)
    scheduler.waiting.add_request(input_blocked_waiting)
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        waiting_for_chunk_running_requests=[chunk_blocked_running],
        waiting_for_chunk_waiting_requests=[adapter_only],
    )

    pressure = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=100.0,
        token_budget_available=512,
    )

    assert pressure["running_requests"] == 1
    assert pressure["waiting_requests"] == 1
    assert pressure["input_waiting_requests"] == 3
    assert pressure["pressure_schema_version"] == 2
    assert pressure["request_count_semantics"] == (
        "runnable_excludes_input_blocked"
    )


def test_scheduler_exports_compact_runnable_pressure_for_control_plane():
    scheduler = SchedulerHarness(2)
    scheduler.max_num_running_reqs = 4
    scheduler.effective_max_num_seqs = 3
    scheduler._slo_control_epoch = 17
    ready_waiting = FakeRequest("ready", deadline=None)
    ready_waiting.status = SimpleNamespace(name="WAITING")
    blocked = FakeRequest("blocked", deadline=None)
    blocked.status = SimpleNamespace(name="WAITING_FOR_CHUNK")
    scheduler.waiting.add_request(ready_waiting)
    scheduler.waiting.add_request(blocked)

    exported = scheduler._export_slo_runtime_pressure(
        captured_monotonic_s=100.0,
    )

    assert exported == {
        "pressure_feedback_schema_version": 1,
        "stage_id": 2,
        "control_epoch": 17,
        "captured_monotonic_s": 100.0,
        "waiting_requests": 1,
        "effective_max_num_seqs": 3,
    }


def test_scheduler_joins_fresh_downstream_pressure_and_sender_backlog():
    scheduler = SchedulerHarness(1)
    scheduler.max_num_running_reqs = 2
    scheduler.effective_max_num_seqs = 2
    scheduler.max_num_scheduled_tokens = 100
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        waiting_for_chunk_running_requests=[],
        waiting_for_chunk_waiting_requests=[],
        get_control_signal_snapshot=lambda: {
            "sender_pending_save_requests": 2,
            "sender_save_inflight": 1,
        },
    )
    scheduler._update_slo_downstream_pressure(
        {
            "pressure_feedback_schema_version": 1,
            "stage_id": 2,
            "control_epoch": 19,
            "captured_monotonic_s": 100.0,
            "waiting_requests": 3,
            "effective_max_num_seqs": 1,
        }
    )

    pressure = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=100.2,
        token_budget_available=50,
        update_control_signals=True,
    )

    assert pressure["downstream_waiting_requests"] == 3
    assert pressure["downstream_effective_cap"] == 1
    assert pressure["downstream_stage_id"] == 2
    assert pressure["downstream_control_epoch"] == 19
    assert pressure["downstream_sample_age_ms"] == pytest.approx(200.0)
    assert pressure["downstream_source"] == "engine_core_utility"
    assert pressure["connector_backlog"] == 3
    assert pressure["connector_backlog_scope"] == (
        "sender_pending_save_and_inflight"
    )


def test_scheduler_omits_stale_downstream_pressure_and_rejects_bad_updates():
    scheduler = SchedulerHarness(1)
    scheduler.max_num_running_reqs = 2
    scheduler.max_num_scheduled_tokens = 100
    sample = {
        "pressure_feedback_schema_version": 1,
        "stage_id": 2,
        "control_epoch": 19,
        "captured_monotonic_s": 100.0,
        "waiting_requests": 3,
        "effective_max_num_seqs": 1,
    }
    scheduler._update_slo_downstream_pressure(sample)

    stale = scheduler._capture_stage_runtime_snapshot(
        captured_monotonic_s=100.501,
        update_control_signals=True,
    )
    assert "downstream_waiting_requests" not in stale
    assert "downstream_sample_age_ms" not in stale

    with pytest.raises(ValueError, match="immediate downstream stage"):
        scheduler._update_slo_downstream_pressure({**sample, "stage_id": 3})
    with pytest.raises(ValueError, match="control_epoch must increase"):
        scheduler._update_slo_downstream_pressure(sample)

    refreshed = {**sample, "captured_monotonic_s": 100.1}
    scheduler._update_slo_downstream_pressure(refreshed)
    assert scheduler._slo_downstream_pressure == refreshed


def test_conformance_trace_records_kv_preemption_requeue_and_resume(monkeypatch):
    monkeypatch.delenv(BASELINE_POLICY_ENV, raising=False)
    monkeypatch.setenv("VLLM_OMNI_CONFORMANCE_TRACE", "1")
    scheduler = SchedulerHarness(1)
    scheduler.max_num_running_reqs = 64
    scheduler.max_num_scheduled_tokens = 32_768
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.99)
    victim = FakeRequest(
        "victim",
        deadline=None,
        num_computed_tokens=321,
    )
    scheduler.running = [victim]
    scheduler._baseline_prepare_schedule(token_budget_before=32_768)

    # Mirror vLLM's post-preemption state: the victim has been reset and
    # prepended to waiting before SchedulerOutput is returned.
    victim.num_computed_tokens = 0
    scheduler.running = []
    scheduler.waiting.prepend_request(victim)
    captured = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_iteration_event",
        lambda **fields: captured.append(fields),
    )
    scheduler._trace_scheduler_output(
        SimpleNamespace(
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            preempted_req_ids={"victim"},
        ),
        iteration_id=1,
        timestamp_start=1.0,
        timestamp_end=1.1,
        num_running_before=1,
        num_waiting_before=0,
    )

    first = captured[0]
    assert first["mechanism_trace_version"] == 2
    assert first["kv_cache_usage_before"] == 0.99
    assert first["kv_allocation_failed"] is True
    assert first["kv_allocation_failure_victim_req_ids"] == ["victim"]
    assert first["preempted_num_computed_tokens_before"] == {"victim": 321}
    assert first["preempted_requeue_order"] == ["victim"]
    assert first["queue_orders_after_schedule"]["waiting"] == ["victim"]
    assert first["resumed_preempted_req_ids"] == []

    scheduler.waiting = FCFSRequestQueue()
    scheduler.running = [victim]
    scheduler.kv_cache_manager.usage = 0.01
    scheduler._baseline_prepare_schedule(token_budget_before=32_768)
    scheduler._trace_scheduler_output(
        SimpleNamespace(
            num_scheduled_tokens={"victim": 1},
            total_num_scheduled_tokens=1,
            preempted_req_ids=set(),
        ),
        iteration_id=2,
        timestamp_start=2.0,
        timestamp_end=2.1,
        num_running_before=0,
        num_waiting_before=1,
    )

    second = captured[1]
    assert second["kv_allocation_failed"] is False
    assert second["resumed_preempted_req_ids"] == ["victim"]
    assert second["queue_orders_after_schedule"]["running"] == ["victim"]


@pytest.mark.parametrize(
    (
        "num_computed_tokens",
        "max_tokens",
        "num_output_placeholders",
        "expected_runnable",
    ),
    [
        pytest.param(100, 4, 0, True, id="no-placeholder"),
        pytest.param(12, 4, 1, True, id="one-placeholder-one-token-short"),
        pytest.param(13, 4, 1, False, id="one-placeholder-at-limit"),
        pytest.param(14, 4, 3, True, id="draft-placeholders-one-token-short"),
        pytest.param(15, 4, 3, False, id="draft-placeholders-at-limit"),
        pytest.param(100, None, 1, True, id="unset-max-tokens"),
    ],
)
def test_conformance_snapshot_matches_async_final_placeholder_boundary(
    monkeypatch,
    num_computed_tokens,
    max_tokens,
    num_output_placeholders,
    expected_runnable,
):
    monkeypatch.setenv("VLLM_OMNI_CONFORMANCE_TRACE", "1")
    scheduler = SchedulerHarness(0)
    request = FakeRequest(
        "async-decode",
        deadline=None,
        num_prompt_tokens=10,
        num_computed_tokens=num_computed_tokens,
        max_tokens=max_tokens,
        num_output_placeholders=num_output_placeholders,
    )
    scheduler.running = [request]
    # Being selected in the previous async step is not sufficient by itself
    # to make a request ineligible; normal continuous decode must remain in
    # the replayable runnable set.
    scheduler.prev_step_scheduled_req_ids = {request.request_id}

    scheduler._baseline_prepare_schedule(token_budget_before=1)

    snapshot = scheduler._baseline_conformance_snapshot
    assert snapshot is not None
    if expected_runnable:
        assert snapshot["runnable_req_ids"] == [request.request_id]
        assert snapshot["ineligible_reasons"] == {}
    else:
        assert snapshot["runnable_req_ids"] == []
        assert snapshot["ineligible_reasons"] == {
            request.request_id: "async_output_pending_at_token_limit"
        }


def test_conformance_snapshot_does_not_apply_final_placeholder_skip_to_waiting(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_OMNI_CONFORMANCE_TRACE", "1")
    scheduler = SchedulerHarness(0)
    scheduler.max_num_running_reqs = 1
    request = FakeRequest(
        "waiting-placeholder",
        deadline=None,
        num_prompt_tokens=10,
        num_computed_tokens=13,
        max_tokens=4,
        num_output_placeholders=1,
    )
    scheduler.waiting.add_request(request)

    scheduler._baseline_prepare_schedule(token_budget_before=1)

    snapshot = scheduler._baseline_conformance_snapshot
    assert snapshot is not None
    assert snapshot["runnable_req_ids"] == [request.request_id]
    assert snapshot["ineligible_reasons"] == {}


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
        {"event": "stage_data_ready", "request_id": "server-id"},
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
        "stage_data_ready",
        "stage_done_metrics",
    ]
    assert not (tmp_path / "iteration_events.jsonl").exists()
    assert not (tmp_path / "connector_events.jsonl").exists()


def test_scheduler_resource_trace_mode_keeps_only_iterations(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("STAGE_QUEUE_TRACE_DISABLE", raising=False)
    monkeypatch.setenv("STAGE_QUEUE_TRACE_MODE", "scheduler_resource")
    monkeypatch.setattr(stage_queue_trace, "_TRACE_DIR", str(tmp_path))

    stage_queue_trace._write_jsonl(
        "iteration_events.jsonl",
        {"event": "iteration", "iteration_id": 1},
    )
    stage_queue_trace._write_jsonl(
        "stage_events.jsonl",
        {"event": "stage_schedule_done", "request_id": None},
    )
    stage_queue_trace._write_jsonl(
        "stage_events.jsonl",
        {"event": "request_ingress", "request_id": "server-id"},
    )
    stage_queue_trace._write_jsonl(
        "connector_events.jsonl",
        {"event": "transfer_rx", "request_id": "server-id"},
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "iteration_events.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["iteration"]
    assert not (tmp_path / "stage_events.jsonl").exists()
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
