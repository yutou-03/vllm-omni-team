import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.recovery import OutputDecision

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pool:
    def __init__(self, *, final_output=False, final_output_type=None):
        self.final_output = final_output
        self.final_output_type = final_output_type
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


def _runtime():
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [
        _Pool(final_output=True, final_output_type="text"),
        _Pool(),
        _Pool(final_output=True, final_output_type="audio"),
    ]
    orchestrator.async_chunk = True
    orchestrator._drain_state_epoch = 0
    orchestrator._pd_pair = None
    orchestrator._omni_nvtx_s2_route_seen = set()
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    template = SimpleNamespace(
        request_id="req",
        external_req_id="req",
        connector_req_id="req",
        output_connector_req_id="req",
        prompt_token_ids=[1, 2],
        additional_information=None,
    )
    state = OrchestratorRequestState(
        request_id="req",
        final_stage_id=2,
        stage0_request_template=template,
        current_stage0_request=template,
        stage_request_ids={0: "req", 1: "req", 2: "req"},
        current_execution_id="req",
    )
    state.stage_submit_ts = {0: 1.0, 1: 2.0, 2: 3.0}
    orchestrator.request_states = {"req": state}
    orchestrator._physical_to_logical_request_ids = {"req": "req"}
    orchestrator.output_async_queue = asyncio.Queue()
    return orchestrator, state


def test_request_state_initializes_stage_scoped_output_ledger():
    _, state = _runtime()

    assert state.output_commit_ledger is not None
    assert set(state.output_commit_ledger.states) == {
        "stage:0",
        "stage:1",
        "stage:2",
    }
    assert state.output_attempt_sequences == {0: 0, 1: 0, 2: 0}


def test_full_restart_snapshots_text_and_audio_paths(monkeypatch):
    orchestrator, state = _runtime()
    assert state.output_commit_ledger is not None
    assert (
        state.classify_and_commit_output(0, finished=True)
        is OutputDecision.COMMIT
    )
    assert state.classify_and_commit_output(2) is OutputDecision.COMMIT
    assert state.classify_and_commit_output(2) is OutputDecision.COMMIT
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    asyncio.run(orchestrator._handle_stage_preemptions(1, ["req"]))

    text_state = state.output_commit_ledger.states["stage:0"]
    audio_state = state.output_commit_ledger.states["stage:2"]
    assert text_state.terminal_committed is True
    assert text_state.replay_watermark == -1
    assert audio_state.replay_watermark == 1
    assert state.output_attempt_sequences == {0: 0, 1: 0, 2: 0}
    assert state.replay_outputs_to_drop == {0: 1, 2: 2}


def test_replay_dedup_is_independent_for_text_and_audio(monkeypatch):
    orchestrator, state = _runtime()
    assert state.classify_and_commit_output(0, finished=True) is OutputDecision.COMMIT
    assert state.classify_and_commit_output(2) is OutputDecision.COMMIT
    state.begin_output_replay([0, 2])
    monkeypatch.setattr(
        "vllm_omni.engine.orchestrator.emit_stage_event",
        lambda *args, **kwargs: None,
    )

    replayed_text = OmniRequestOutput(
        request_id="req",
        finished=True,
        stage_id=0,
        final_output_type="text",
        request_output=SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1])]),
    )
    replayed_audio = OmniRequestOutput(
        request_id="req",
        finished=False,
        stage_id=2,
        final_output_type="audio",
        request_output=SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[2])],
            multimodal_output={"audio": b"old"},
        ),
    )
    fresh_audio = OmniRequestOutput(
        request_id="req",
        finished=False,
        stage_id=2,
        final_output_type="audio",
        request_output=SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[3])],
            multimodal_output={"audio": b"new"},
        ),
    )

    asyncio.run(orchestrator._route_output(0, 0, replayed_text, state, None))
    asyncio.run(orchestrator._route_output(2, 0, replayed_audio, state, None))
    asyncio.run(orchestrator._route_output(2, 0, fresh_audio, state, None))

    queued = orchestrator.output_async_queue.get_nowait()
    assert orchestrator.output_async_queue.empty()
    assert queued["engine_outputs"] is fresh_audio
    assert state.final_outputs_emitted == {0: 1, 2: 2}
