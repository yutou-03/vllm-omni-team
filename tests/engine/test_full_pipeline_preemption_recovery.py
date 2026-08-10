import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)

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

    def remove_requests(self, requests):
        for request in requests:
            if request in self.requests:
                self.requests.remove(request)


def _orchestrator_with_request():
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
    state.final_outputs_emitted = {2: 7}
    state.streaming.bridge_states = {"qwen3_omni": {"dirty": True}}
    orchestrator.request_states = {"logical": state}
    orchestrator._physical_to_logical_request_ids = {"logical": "logical"}
    return orchestrator, state, template


def test_scheduler_does_not_requeue_retired_physical_request(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler.waiting = _Waiting()
    scheduler.chunk_transfer_adapter = None
    scheduler._omni_stage_id_for_trace = lambda: 1
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
            },
        )
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

    asyncio.run(orchestrator._handle_stage_preemptions(1, ["logical"]))

    execution_id = "logical__execution1"
    assert all(pool.aborted == ["logical"] for pool in orchestrator.stage_pools)
    assert all(pool.released == ["logical"] for pool in orchestrator.stage_pools)
    assert state.current_execution_id == execution_id
    assert state.execution_attempt == 1
    assert state.stage_request_ids == {0: execution_id, 1: execution_id, 2: execution_id}
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
    assert state.pending_restart_stages == {2}

    asyncio.run(
        orchestrator._submit_next_restart_stage_after_upstream_output(
            1, [SimpleNamespace(request_id=execution_id)]
        )
    )
    assert prewarm_calls[1][3]["start_stage_id"] == 2
    assert prewarm_calls[1][3]["end_stage_id"] == 2
    assert state.pending_restart_stages == set()


def test_delayed_preemption_from_retired_execution_is_ignored(monkeypatch):
    orchestrator, state, _ = _orchestrator_with_request()
    state.stage_request_ids[1] = "logical__execution1"
    state.current_execution_id = "logical__execution1"
    orchestrator._physical_to_logical_request_ids["logical__execution1"] = "logical"
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    asyncio.run(orchestrator._handle_stage_preemptions(1, ["logical"]))

    assert state.execution_attempt == 0
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
