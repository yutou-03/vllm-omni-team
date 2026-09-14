import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARAsyncScheduler,
    _load_preemption_recovery_mode,
)
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.recovery import PreemptionRecoveryMode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pool:
    def __init__(self, *, final_output=False):
        self.final_output = final_output
        self.aborted = []
        self.released = []
        self.submitted = []
        self.discarded_output_states = []

    def discard_terminal_output_states(self, request_ids):
        self.discarded_output_states.extend(request_ids)

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
    for _ in range(7):
        state.classify_and_commit_output(2)
    orchestrator.request_states = {"logical": state}
    orchestrator._physical_to_logical_request_ids = {"logical": "logical"}
    return orchestrator, state, template


def test_recovery_mode_defaults_to_full_pipeline_restart(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_PREEMPTION_RECOVERY_MODE", raising=False)

    assert (
        _load_preemption_recovery_mode()
        is PreemptionRecoveryMode.FULL_PIPELINE_RESTART
    )


def test_recovery_mode_accepts_stage_local_recompute(monkeypatch):
    monkeypatch.setenv(
        "VLLM_OMNI_PREEMPTION_RECOVERY_MODE",
        "stage_local_recompute",
    )

    assert (
        _load_preemption_recovery_mode()
        is PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
    )


def test_recovery_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_PREEMPTION_RECOVERY_MODE", "unsafe_resume")

    with pytest.raises(ValueError, match="VLLM_OMNI_PREEMPTION_RECOVERY_MODE"):
        _load_preemption_recovery_mode()


def test_talker_stage_local_preemption_requeues_only_immutable_input(monkeypatch):
    scheduler = object.__new__(OmniARAsyncScheduler)
    scheduler.waiting = SimpleNamespace(requests=[])
    scheduler.waiting.prepend_request = scheduler.waiting.requests.insert
    scheduler.waiting.remove_requests = lambda requests: None
    scheduler.chunk_transfer_adapter = None
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage="talker")
    )
    scheduler._preemption_recovery_mode = (
        PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
    )
    scheduler._omni_stage_id_for_trace = lambda: 1
    scheduler._omni_preemption_causes_pending = {}
    scheduler._omni_preemption_progress_pending = {}
    scheduler._omni_preemption_rule_ids_pending = {}
    scheduler._omni_preemption_recovery_modes_pending = {}
    scheduler._omni_recovery_trace_by_request = {}
    scheduler._omni_request_trace_fields_by_request = {
        "logical": {
            "logical_request_id": "logical",
            "execution_id": "logical",
            "pipeline_attempt_id": 0,
            "stage_attempt_id": 0,
        }
    }
    request = SimpleNamespace(
        request_id="logical",
        external_req_id="logical",
        connector_req_id="logical",
        output_connector_req_id="logical",
        _all_token_ids=[1, 2, 40, 41],
        _output_token_ids=[40, 41],
        num_prompt_tokens=2,
        num_computed_tokens=4,
        num_output_placeholders=1,
        discard_latest_async_tokens=True,
        _omni_execution_attempt=0,
        num_preemptions=0,
        stop_reason="length",
    )

    def _base_preempt(self, victim, _timestamp):
        victim.num_computed_tokens = 0
        victim.num_preemptions += 1
        self.waiting.requests.insert(0, victim)

    events = []
    monkeypatch.setattr(Scheduler, "_preempt_request", _base_preempt)
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_ar_scheduler.emit_stage_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    scheduler._preempt_request(request, 1.0)

    assert scheduler.waiting.requests == [request]
    assert request._all_token_ids == [1, 2]
    assert request._output_token_ids == []
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is False
    assert request.stop_reason is None
    assert request._omni_execution_attempt == 1
    assert request.output_connector_req_id == "logical__stage1__attempt1"
    assert scheduler._omni_preemption_recovery_modes_pending == {
        "logical": "stage_local_recompute"
    }
    assert scheduler._omni_recovery_trace_by_request["logical"] == {
        "logical_request_id": "logical",
        "execution_id": "logical",
        "pipeline_attempt_id": 0,
        "stage_attempt_id": 1,
        "recovery_target_stage_id": 1,
        "recovery_target_num_computed_tokens": 4,
        "preemption_cause": "allocation_failure",
    }
    assert (
        scheduler._omni_request_trace_fields_by_request["logical"]
        == scheduler._omni_recovery_trace_by_request["logical"]
    )
    assert [event for event, _ in events] == [
        "stage_local_dynamic_state_discarded"
    ]


def test_stage_local_talker_recompute_replaces_only_downstream(monkeypatch):
    orchestrator, state, _ = _orchestrator_with_request()
    state.processed_terminal_stages.update({0, 1, 2})
    state.streaming.bridge_states = {
        "qwen3_omni": {
            "logical": {"dirty": True},
            "unrelated": {"keep": True},
        }
    }
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
            connector_req_ids={
                "logical": "logical__stage1__attempt1",
            },
            preemption_causes={"logical": "controlled_restart"},
            preemption_progress={"logical": 37},
            preemption_rule_ids={"logical": "s1-half"},
            recovery_modes={"logical": "stage_local_recompute"},
        )
    )

    stage2_id = "logical__pipeline0__stage2__attempt1"
    assert orchestrator.stage_pools[0].aborted == []
    assert orchestrator.stage_pools[1].aborted == []
    assert orchestrator.stage_pools[2].aborted == ["logical"]
    assert orchestrator.stage_pools[2].released == ["logical"]
    assert state.current_execution_id == "logical"
    assert state.execution_attempt == 0
    assert state.stage_request_ids == {0: "logical", 1: "logical", 2: stage2_id}
    assert state.processed_terminal_stages == {0}
    assert state.execution_ledger is not None
    assert state.execution_ledger.pipeline_attempt_id == 0
    assert {
        stage_id: ref.stage_attempt_id
        for stage_id, ref in state.execution_ledger.current_stages.items()
    } == {0: 0, 1: 1, 2: 1}
    assert state.streaming.bridge_states == {
        "qwen3_omni": {"unrelated": {"keep": True}}
    }
    assert state.replay_outputs_to_drop == {2: 7}
    assert set(state.discard_stage_outputs_before) == {1, 2}
    assert state.stage_submit_ts == {0: 1.0, 1: 2.0}
    assert orchestrator._physical_to_logical_request_ids[stage2_id] == "logical"
    assert len(prewarm_calls) == 1
    assert prewarm_calls[0][0] == "logical"
    assert prewarm_calls[0][3] == {
        "start_stage_id": 2,
        "recovery": True,
        "connector_req_id": "logical__stage1__attempt1",
    }


def test_stage2_local_mode_falls_back_to_full_pipeline_restart(monkeypatch):
    orchestrator, state, _ = _orchestrator_with_request()
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    asyncio.run(
        orchestrator._handle_stage_preemptions(
            2,
            ["logical"],
            recovery_modes={"logical": "stage_local_recompute"},
        )
    )

    assert state.execution_attempt == 1
    assert state.current_execution_id == "logical__execution1"
    assert all(pool.aborted == ["logical"] for pool in orchestrator.stage_pools)
    assert orchestrator.stage_pools[0].submitted[0][0] == "logical__execution1"
