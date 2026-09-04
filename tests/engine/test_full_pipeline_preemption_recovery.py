import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARAsyncScheduler,
    _load_controlled_preemption_rules,
    _load_slo_active_preemption_config,
)
from vllm_omni.core.sched.omni_generation_scheduler import (
    OmniGenerationScheduler,
)
from vllm_omni.core.sched.omni_scheduler_mixin import (
    extract_request_recovery_trace_fields,
)
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pool:
    def __init__(self, *, final_output=False):
        self.final_output = final_output
        self.aborted = []
        self.released = []
        self.submitted = []

    async def abort_requests(self, request_ids):
        self.aborted.extend(request_ids)

    def release_binding(self, request_id):
        self.released.append(request_id)

    def release_bindings(self, request_ids):
        self.released.extend(request_ids)

    async def submit_initial(
        self, request_id, req_state, request, *, prompt_text=None, **_kwargs
    ):
        self.submitted.append((request_id, req_state, request, prompt_text))


class _Waiting:
    def __init__(self):
        self.requests = []

    def prepend_request(self, request):
        self.requests.insert(0, request)

    def add_request(self, request):
        self.requests.append(request)

    def __iter__(self):
        return iter(self.requests)

    def remove_requests(self, requests):
        for request in requests:
            if request in self.requests:
                self.requests.remove(request)


def _orchestrator_with_request(*, emitted_audio_outputs=7):
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [_Pool(), _Pool(), _Pool(final_output=True)]
    orchestrator.async_chunk = True
    orchestrator._drain_state_epoch = 0
    template = SimpleNamespace(
        request_id="logical",
        external_req_id="logical",
        connector_req_id="logical",
        output_connector_req_id="logical",
        prompt_token_ids=[1, 2],
        additional_information=serialize_additional_information(
            {"global_request_id": ["logical"]}
        ),
    )
    state = OrchestratorRequestState(
        request_id="logical",
        final_stage_id=2,
        stage0_request_template=template,
        stage0_prompt_text="fixed prompt",
        stage_request_ids={0: "logical", 1: "logical", 2: "logical"},
        current_execution_id="logical",
    )
    state.stage_submit_ts = {0: 1.0, 1: 2.0, 2: 3.0}
    for _ in range(emitted_audio_outputs):
        state.classify_and_commit_output(2)
    state.streaming.bridge_states = {"qwen3_omni": {"dirty": True}}
    orchestrator.request_states = {"logical": state}
    orchestrator._physical_to_logical_request_ids = {"logical": "logical"}
    return orchestrator, state, template


def test_request_state_initializes_explicit_execution_ledger():
    _, state, _ = _orchestrator_with_request()

    assert state.execution_ledger is not None
    assert state.execution_ledger.pipeline_attempt_id == 0
    assert {
        stage_id: ref.physical_request_id
        for stage_id, ref in state.execution_ledger.current_stages.items()
    } == {0: "logical", 1: "logical", 2: "logical"}


def test_suppressed_replay_terminal_still_reaches_frontend(monkeypatch):
    orchestrator, state, _ = _orchestrator_with_request(
        emitted_audio_outputs=5
    )
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator._pd_pair = None
    orchestrator._omni_nvtx_s2_route_seen = set()
    state.execution_attempt = 1
    state.current_execution_id = "logical__execution1"
    state.begin_output_replay([2])
    cleanup_calls = []
    events = []

    async def _cleanup(request_ids):
        cleanup_calls.append(request_ids)

    monkeypatch.setattr(orchestrator, "_cleanup_request_ids", _cleanup)
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    duplicate_terminal = OmniRequestOutput(
        request_id="logical",
        finished=True,
        stage_id=2,
        final_output_type="audio",
        request_output=SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[1])],
            multimodal_output={"audio": b"duplicate"},
        ),
    )
    asyncio.run(
        orchestrator._route_output(
            2,
            0,
            duplicate_terminal,
            state,
            stage_metrics=None,
        )
    )

    message = orchestrator.output_async_queue.get_nowait()
    assert message["finished"] is True
    assert message["engine_outputs"].finished is True
    assert message["engine_outputs"].request_id == "logical"
    assert message["engine_outputs"].outputs == []
    assert message["engine_outputs"].multimodal_output == {}
    assert state.replay_outputs_to_drop == {2: 4}
    assert cleanup_calls == [["logical"]]
    assert [event for event, _ in events] == [
        "stage_replay_output_suppressed",
        "stage_replay_terminal_forwarded",
        "full_pipeline_recovery_completed",
    ]


def test_scheduler_does_not_requeue_retired_physical_request(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler.waiting = _Waiting()
    scheduler.chunk_transfer_adapter = None
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler._omni_preemption_causes_pending = {}
    scheduler._omni_preemption_progress_pending = {}
    scheduler._omni_preemption_rule_ids_pending = {}
    request = SimpleNamespace(
        request_id="logical",
        _output_token_ids=[4, 5],
        num_computed_tokens=37,
        output_connector_req_id="logical",
        connector_req_id="logical",
        external_req_id="logical",
        _omni_execution_attempt=0,
    )

    def _base_preempt(self, victim, _timestamp):
        victim.num_computed_tokens = 0
        self.waiting.prepend_request(victim)

    events = []
    monkeypatch.setattr(Scheduler, "_preempt_request", _base_preempt)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_ar_scheduler.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    scheduler._preempt_request(request, 1.0)

    assert scheduler.waiting.requests == []
    assert request._omni_execution_attempt == 1
    assert events == [
        (
            "full_pipeline_restart_requested",
            {
                "stage_id": 1,
                "request_id": "logical",
                "discarded_output_tokens": 2,
                "discarded_computed_tokens": 37,
                "connector_req_id": "logical",
                "preemption_cause": "allocation_failure",
                "controlled_rule_id": None,
            },
        )
    ]


def test_controlled_preemption_uses_same_recovery_entrypoint(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler.waiting = _Waiting()
    scheduler.chunk_transfer_adapter = None
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler._controlled_preemption_rules = [
        {
            "rule_id": "s1-half",
            "victim_request_id": "logical",
            "target_stage": 1,
            "target_num_computed_tokens": 30,
            "occurrence_index": 1,
        }
    ]
    scheduler._controlled_preemption_fired_rule_ids = set()
    scheduler._omni_preemption_causes_pending = {}
    scheduler._omni_preemption_progress_pending = {}
    scheduler._omni_preemption_rule_ids_pending = {}
    request = SimpleNamespace(
        request_id="logical__execution1",
        _output_token_ids=[4, 5],
        num_computed_tokens=37,
        num_output_placeholders=0,
        output_connector_req_id="logical__execution1",
        connector_req_id="logical__execution1",
        external_req_id="logical",
        _omni_execution_attempt=0,
    )
    scheduler.running = [request]

    def _base_preempt(self, victim, _timestamp):
        victim.num_computed_tokens = 0
        self.waiting.prepend_request(victim)

    events = []
    monkeypatch.setattr(Scheduler, "_preempt_request", _base_preempt)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_ar_scheduler.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    injected = scheduler._inject_controlled_preemptions(1.0)

    assert injected == {"logical__execution1"}
    assert scheduler.running == []
    assert scheduler.waiting.requests == []
    assert scheduler._controlled_preemption_fired_rule_ids == {"s1-half"}
    assert scheduler._omni_preemption_causes_pending == {
        "logical__execution1": "controlled_restart"
    }
    assert scheduler._omni_preemption_progress_pending == {
        "logical__execution1": 37
    }
    assert scheduler._omni_preemption_rule_ids_pending == {
        "logical__execution1": "s1-half"
    }
    assert [event for event, _ in events] == [
        "full_pipeline_restart_requested",
        "controlled_preemption_injected",
    ]


def test_controlled_preemption_file_contract(tmp_path, monkeypatch):
    path = tmp_path / "events.json"
    path.write_text(
        '{"schema_version":1,"events":[{'
        '"rule_id":"r1","victim_request_id":"logical",'
        '"target_stage":1,"target_num_computed_tokens":42,'
        '"occurrence_index":0}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_OMNI_CONTROLLED_PREEMPTION_FILE", str(path))

    assert _load_controlled_preemption_rules() == [
        {
            "rule_id": "r1",
            "victim_request_id": "logical",
            "target_stage": 1,
            "target_num_computed_tokens": 42,
            "occurrence_index": 0,
        }
    ]


def test_generation_stage_controlled_preemption_uses_cumulative_progress(
    monkeypatch,
):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler.waiting = _Waiting()
    scheduler.chunk_transfer_adapter = None
    scheduler._omni_stage_id_for_trace = lambda: 2
    scheduler._controlled_preemption_rules = [
        {
            "rule_id": "s2-half",
            "victim_request_id": "logical",
            "target_stage": 2,
            "target_num_computed_tokens": 8,
            "occurrence_index": 0,
        }
    ]
    scheduler._controlled_preemption_fired_rule_ids = set()
    scheduler._omni_preemption_causes_pending = {}
    scheduler._omni_preemption_progress_pending = {}
    scheduler._omni_preemption_rule_ids_pending = {}
    scheduler._omni_cumulative_confirmed_tokens = {"logical": 10}
    request = SimpleNamespace(
        request_id="logical",
        _output_token_ids=[],
        num_computed_tokens=0,
        _omni_cumulative_confirmed_tokens=10,
        output_connector_req_id="logical",
        connector_req_id="logical",
        external_req_id="logical",
        _omni_execution_attempt=0,
    )
    scheduler.running = [request]

    def _base_preempt(self, victim, _timestamp):
        victim.num_computed_tokens = 0
        self.waiting.prepend_request(victim)

    events = []
    monkeypatch.setattr(Scheduler, "_preempt_request", _base_preempt)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_generation_scheduler.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    injected = scheduler._inject_controlled_preemption_after_output(
        request, 1.0
    )

    assert injected is True
    assert scheduler.running == []
    assert scheduler.waiting.requests == []
    assert scheduler._omni_preemption_progress_pending == {"logical": 10}
    assert scheduler._omni_preemption_rule_ids_pending == {
        "logical": "s2-half"
    }
    assert [event for event, _ in events] == [
        "full_pipeline_restart_requested",
        "controlled_preemption_injected",
    ]
    assert events[0][1]["progress_metric"] == (
        "completed_streaming_chunks"
    )


def test_slo_active_preemption_config_contract(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_SLO_ACTIVE_PREEMPTION", "true")
    monkeypatch.setenv("VLLM_OMNI_SLO_ACTIVE_STAGES", "0,1")
    monkeypatch.setenv("VLLM_OMNI_SLO_ACTIVE_GUARD_MS", "25.5")
    monkeypatch.setenv("VLLM_OMNI_SLO_PRIORITY_TRACE", "1")

    assert _load_slo_active_preemption_config() == (True, {0, 1}, 25.5, True)


def test_slo_active_hook_is_noop_by_default_without_metadata():
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler._slo_active_preemption_enabled = False
    scheduler._slo_priority_trace_enabled = False
    scheduler._slo_active_preemption_stages = {1}
    scheduler.running = [SimpleNamespace(request_id="ordinary-running")]
    scheduler.waiting = _Waiting()
    scheduler.waiting.add_request(SimpleNamespace(request_id="ordinary-waiting"))
    scheduler.max_num_running_reqs = 1

    assert scheduler._maybe_preempt_for_slo(1.0) == set()


def test_slo_active_switch_replaces_later_deadline_running_request(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler.waiting = _Waiting()
    scheduler.chunk_transfer_adapter = None
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler.max_num_running_reqs = 1
    scheduler._slo_active_preemption_enabled = True
    scheduler._slo_active_preemption_stages = {1}
    scheduler._slo_active_guard_ms = 0.0
    scheduler._slo_priority_trace_enabled = True
    scheduler._slo_priority_witness_seen = set()
    scheduler._omni_preemption_causes_pending = {}
    scheduler._omni_preemption_progress_pending = {}
    scheduler._omni_preemption_rule_ids_pending = {}

    def _request(request_id, *, deadline, predicted_ms, progress):
        return SimpleNamespace(
            request_id=request_id,
            _output_token_ids=[],
            num_computed_tokens=progress,
            num_output_placeholders=0,
            output_connector_req_id=request_id,
            connector_req_id=request_id,
            external_req_id=request_id,
            additional_information=serialize_additional_information(
                {
                    "meta": {
                        "sched_source_request_id": request_id,
                        "sched_ingress_order": 0,
                        "sched_deadline_monotonic_s": deadline + 1.0,
                        "sched_stage_deadline_monotonic_s": [
                            deadline - 1.0,
                            deadline,
                            deadline + 1.0,
                        ],
                        "sched_predicted_stage_ms": [100.0, predicted_ms, 100.0],
                    }
                }
            ),
        )

    victim = _request("low", deadline=20.0, predicted_ms=500.0, progress=73)
    aggressor = _request("high", deadline=10.0, predicted_ms=1200.0, progress=0)
    scheduler.running = [victim]
    scheduler.waiting.add_request(aggressor)

    def _base_preempt(self, request, _timestamp):
        request.num_computed_tokens = 0
        self.waiting.prepend_request(request)

    events = []
    monkeypatch.setattr(Scheduler, "_preempt_request", _base_preempt)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_ar_scheduler.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    preempted = scheduler._maybe_preempt_for_slo(9.0)

    assert preempted == {"low"}
    assert scheduler.running == []
    assert list(scheduler.waiting) == [aggressor]
    assert scheduler._omni_preemption_causes_pending == {
        "low": "slo_active_switch"
    }
    assert scheduler._omni_preemption_progress_pending == {"low": 73}
    assert [event for event, _ in events] == [
        "strict_priority_blocking_witness",
        "full_pipeline_restart_requested",
        "slo_active_preemption",
    ]


def test_preemption_restarts_entire_pipeline_with_new_execution(monkeypatch):
    orchestrator, state, template = _orchestrator_with_request()
    prewarm_calls = []

    async def _prewarm(request_id, stage0_request, req_state, **kwargs):
        prewarm_calls.append((request_id, stage0_request, req_state, kwargs))

    monkeypatch.setattr(orchestrator, "_prewarm_async_chunk_stages", _prewarm)
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    asyncio.run(
        orchestrator._handle_stage_preemptions(
            1,
            ["logical"],
            preemption_causes={"logical": "controlled_restart"},
            preemption_progress={"logical": 37},
            preemption_rule_ids={"logical": "s1-half"},
        )
    )

    execution_id = "logical__execution1"
    assert all(pool.aborted == ["logical"] for pool in orchestrator.stage_pools)
    assert all(pool.released == ["logical"] for pool in orchestrator.stage_pools)
    assert state.current_execution_id == execution_id
    assert state.execution_attempt == 1
    assert state.stage_request_ids == {0: execution_id, 1: execution_id, 2: execution_id}
    assert state.execution_ledger is not None
    assert state.execution_ledger.pipeline_attempt_id == 1
    assert {
        stage_id: ref.physical_request_id
        for stage_id, ref in state.execution_ledger.current_stages.items()
    } == {0: execution_id, 1: execution_id, 2: execution_id}
    assert state.stage_submit_ts.keys() == {0}
    assert state.streaming.bridge_states == {}
    assert state.replay_outputs_to_drop == {2: 7}
    assert set(state.discard_stage_outputs_before) == {0, 1, 2}
    assert orchestrator._physical_to_logical_request_ids[execution_id] == "logical"

    submitted_id, _, restarted_request, prompt_text = orchestrator.stage_pools[0].submitted[0]
    assert submitted_id == execution_id
    assert restarted_request.request_id == execution_id
    assert restarted_request.external_req_id == "logical"
    assert restarted_request.connector_req_id == execution_id
    assert restarted_request.output_connector_req_id == execution_id
    assert deserialize_additional_information(
        restarted_request.additional_information
    )["global_request_id"] == [execution_id]
    assert extract_request_recovery_trace_fields(restarted_request) == {
        "logical_request_id": "logical",
        "execution_id": execution_id,
        "pipeline_attempt_id": 1,
        "stage_attempt_id": 0,
        "recovery_target_stage_id": 1,
        "recovery_target_num_computed_tokens": 37,
        "preemption_cause": "controlled_restart",
        "controlled_rule_id": "s1-half",
    }
    assert restarted_request.prompt_token_ids == [1, 2]
    assert prompt_text == "fixed prompt"
    assert restarted_request is not template
    assert template.request_id == "logical"

    # Downstream replacement requests are not submitted alongside Stage 0:
    # each one waits for output from the replacement upstream stage.
    assert prewarm_calls == []
    assert state.current_stage0_request is restarted_request
    assert state.pending_restart_stages == {1, 2}

    asyncio.run(
        orchestrator._submit_next_restart_stage_after_upstream_output(
            0, [SimpleNamespace(request_id=execution_id)]
        )
    )
    assert prewarm_calls[0][0] == "logical"
    assert prewarm_calls[0][1] is restarted_request
    assert prewarm_calls[0][3]["start_stage_id"] == 1
    assert prewarm_calls[0][3]["end_stage_id"] == 1
    assert prewarm_calls[0][3]["recovery"] is True
    assert prewarm_calls[0][3]["connector_req_id"] == execution_id


def test_local_recompute_trace_identity_is_preserved_at_pipeline_attempt_zero():
    request = SimpleNamespace(
        additional_information={
            "meta": {
                "omni_logical_request_id": "logical",
                "omni_execution_id": "logical",
                "omni_pipeline_attempt_id": 0,
                "omni_stage_attempt_id": 1,
                "omni_recovery_target_stage_id": 1,
                "omni_recovery_target_num_computed_tokens": 37,
                "omni_recovery_preemption_cause": "controlled_restart",
                "omni_controlled_rule_id": "s1-half",
            }
        }
    )

    assert extract_request_recovery_trace_fields(request) == {
        "logical_request_id": "logical",
        "execution_id": "logical",
        "pipeline_attempt_id": 0,
        "stage_attempt_id": 1,
        "recovery_target_stage_id": 1,
        "recovery_target_num_computed_tokens": 37,
        "preemption_cause": "controlled_restart",
        "controlled_rule_id": "s1-half",
    }
    assert state.pending_restart_stages == {2}

    asyncio.run(
        orchestrator._submit_next_restart_stage_after_upstream_output(
            1, [SimpleNamespace(request_id=execution_id)]
        )
    )
    assert prewarm_calls[1][3]["start_stage_id"] == 2
    assert prewarm_calls[1][3]["end_stage_id"] == 2
    assert state.pending_restart_stages == set()


def test_recovery_progress_events_are_confirmed_once(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler._omni_recovery_trace_by_request = {
        "logical__execution1": {
            "logical_request_id": "logical",
            "pipeline_attempt_id": 1,
            "recovery_target_stage_id": 1,
            "recovery_target_num_computed_tokens": 37,
            "preemption_cause": "controlled_restart",
            "controlled_rule_id": "s1-half",
        }
    }
    scheduler._omni_first_recovery_progress_seen = set()
    scheduler._omni_recovery_catchup_seen = set()
    request = SimpleNamespace(
        request_id="logical__execution1",
        num_computed_tokens=40,
    )
    events = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    scheduler._trace_recovery_progress_after_output(
        request,
        num_tokens_scheduled=20,
    )
    scheduler._trace_recovery_progress_after_output(
        request,
        num_tokens_scheduled=1,
    )

    assert [event for event, _ in events] == [
        "first_recomputed_progress",
        "full_pipeline_progress_caught_up",
    ]
    assert events[1][1]["target_num_computed_tokens"] == 37
    assert events[1][1]["confirmed_num_computed_tokens"] == 40
    assert events[1][1]["overshoot_num_computed_tokens"] == 3


def test_generation_recovery_catchup_uses_cumulative_progress(monkeypatch):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._omni_stage_id_for_trace = lambda: 2
    scheduler._omni_recovery_trace_by_request = {
        "logical__execution1": {
            "logical_request_id": "logical",
            "pipeline_attempt_id": 1,
            "recovery_target_stage_id": 2,
            "recovery_target_num_computed_tokens": 10,
            "preemption_cause": "controlled_restart",
            "controlled_rule_id": "s2-half",
        }
    }
    scheduler._omni_first_recovery_progress_seen = set()
    scheduler._omni_recovery_catchup_seen = set()
    request = SimpleNamespace(
        request_id="logical__execution1",
        num_computed_tokens=0,
        _omni_cumulative_confirmed_tokens=10,
    )
    events = []
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    scheduler._trace_recovery_progress_after_output(
        request,
        num_tokens_scheduled=160,
    )

    assert [event for event, _ in events] == [
        "first_recomputed_progress",
        "full_pipeline_progress_caught_up",
    ]
    assert events[1][1]["confirmed_num_computed_tokens"] == 10


def test_delayed_preemption_from_retired_execution_is_ignored(monkeypatch):
    orchestrator, state, _ = _orchestrator_with_request()
    assert state.execution_ledger is not None
    state.execution_ledger.begin_full_restart(
        physical_request_ids={
            0: "logical__execution1",
            1: "logical__execution1",
            2: "logical__execution1",
        },
        started_monotonic_s=state.execution_ledger.current_stages[
            0
        ].started_monotonic_s,
    )
    state.stage_request_ids = {
        stage_id: "logical__execution1" for stage_id in range(3)
    }
    state.current_execution_id = "logical__execution1"
    state.execution_attempt = 1
    orchestrator._physical_to_logical_request_ids["logical__execution1"] = "logical"
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    asyncio.run(orchestrator._handle_stage_preemptions(1, ["logical"]))

    assert state.execution_attempt == 1
    assert state.execution_ledger.pipeline_attempt_id == 1
    assert all(pool.aborted == [] for pool in orchestrator.stage_pools)
    assert orchestrator.stage_pools[0].submitted == []


def test_logical_completion_aborts_and_releases_all_executions():
    orchestrator, _, _ = _orchestrator_with_request()
    orchestrator._physical_to_logical_request_ids.update(
        {
            "logical__execution1": "logical",
            "logical__execution2": "logical",
        }
    )
    orchestrator._pd_kv_params = {}
    orchestrator._omni_nvtx_s2_raw_output_seen = set()
    orchestrator._omni_nvtx_s2_route_seen = set()
    orchestrator._running_counter = None

    asyncio.run(orchestrator._cleanup_request_ids(["logical"]))

    expected = {"logical", "logical__execution1", "logical__execution2"}
    assert all(set(pool.released) == expected for pool in orchestrator.stage_pools)
    assert all(set(pool.aborted) == expected for pool in orchestrator.stage_pools)
    assert orchestrator.request_states == {}
    assert orchestrator._physical_to_logical_request_ids == {}


def test_cleanup_uses_ledger_history_when_registry_only_has_current_execution():
    orchestrator, state, _ = _orchestrator_with_request()
    assert state.execution_ledger is not None
    start = state.execution_ledger.current_stages[0].started_monotonic_s
    state.execution_ledger.begin_full_restart(
        physical_request_ids={
            0: "logical__execution1",
            1: "logical__execution1",
            2: "logical__execution1",
        },
        started_monotonic_s=start,
    )
    state.execution_ledger.begin_full_restart(
        physical_request_ids={
            0: "logical__execution2",
            1: "logical__execution2",
            2: "logical__execution2",
        },
        started_monotonic_s=start,
    )
    orchestrator._physical_to_logical_request_ids = {
        "logical__execution2": "logical"
    }
    orchestrator._pd_kv_params = {}
    orchestrator._omni_nvtx_s2_raw_output_seen = set()
    orchestrator._omni_nvtx_s2_route_seen = set()
    orchestrator._running_counter = None

    asyncio.run(orchestrator._cleanup_request_ids(["logical__execution2"]))

    expected = {"logical", "logical__execution1", "logical__execution2"}
    assert all(set(pool.released) == expected for pool in orchestrator.stage_pools)
    assert all(set(pool.aborted) == expected for pool in orchestrator.stage_pools)
    assert orchestrator.request_states == {}
    assert orchestrator._physical_to_logical_request_ids == {}
