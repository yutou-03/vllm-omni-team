"""Late upstream data must precede request completion, in every drain order."""
import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setup_request(final_stage=2):
    orch = object.__new__(Orchestrator)
    state = OrchestratorRequestState(request_id="r", final_stage_id=final_stage,
                                    stage_submit_ts={s: 1. for s in range(final_stage + 1)})
    orch.request_states = {"r": state}
    orch.stage_pools = [SimpleNamespace(final_output=s != 1) for s in range(final_stage + 1)]
    orch._cfg_tracker = SimpleNamespace(is_companion=lambda _: False, cleanup_parent=lambda _: [])
    orch._pd_pair = None
    orch.async_chunk = True
    orch._omni_nvtx_s2_route_seen = set()
    orch.output_async_queue = asyncio.Queue()
    async def cleanup(ids, **kwargs):
        orch.request_states.pop("r", None)
    orch._cleanup_request_ids = AsyncMock(side_effect=cleanup)
    return orch, state


async def route(orch, state, stage, finished=True):
    output = SimpleNamespace(request_id="r", finished=finished, content=f"stage-{stage}")
    await orch._route_output(stage, 0, output, state, None)
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
async def test_all_stage_terminal_orders(order):
    orch, state = setup_request()
    sent = []
    for i, stage in enumerate(order):
        await route(orch, state, stage)
        while not orch.output_async_queue.empty():
            sent.append(orch.output_async_queue.get_nowait())
        assert ("r" in orch.request_states) == (i != 2)
        assert sum(bool(m.get("finished")) for m in sent) == (i == 2)
    assert [m["stage_id"] for m in sent if m["type"] == "output"] == [s for s in order if s != 1]
    assert sent[-1]["finished"]
    assert sum(m["type"] == "request_done" for m in sent) == (order[-1] != 2)
    orch._cleanup_request_ids.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_not_held_and_late_text_not_dropped():
    orch, state = setup_request()
    audio = await route(orch, state, 2)
    message = orch.output_async_queue.get_nowait()
    assert message["engine_outputs"] is audio and not message["finished"]
    assert "r" in orch.request_states
    text = await route(orch, state, 0, finished=False)
    assert orch.output_async_queue.get_nowait()["engine_outputs"] is text
    await route(orch, state, 0)
    assert not orch.output_async_queue.get_nowait()["finished"]
    assert "r" in orch.request_states  # intermediate terminal still outstanding
    await route(orch, state, 1)
    assert orch.output_async_queue.get_nowait()["type"] == "request_done"
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_text_only_retains_original_terminal_envelope():
    orch, state = setup_request(0)
    await route(orch, state, 0)
    message = orch.output_async_queue.get_nowait()
    assert message["type"] == "output" and message["finished"]
    assert orch.output_async_queue.empty() and "r" not in orch.request_states


@pytest.mark.asyncio
async def test_error_bypasses_pending_terminal_barrier():
    orch, state = setup_request()
    await route(orch, state, 2)
    await orch._handle_stage_error(0, SimpleNamespace(request_id="r", error="failure"))
    assert "r" not in orch.request_states
    orch._cleanup_request_ids.assert_awaited_once_with(["r"], abort=True)


def test_frontend_control_is_not_an_output_or_metric():
    base = object.__new__(OmniBase)
    state = SimpleNamespace(stage_id=0)
    base.request_states = {"r": state}
    message = {"type":"request_done", "request_id":"r", "stage_id":2, "finished":True}
    assert base._handle_output_message(message) == (False, "r", 2, state)
    assert base._process_single_result(message, 2, Mock(), {}, 0., 2) is None


def test_pending_restart_prevents_completion():
    _, state = setup_request()
    state.processed_terminal_stages = {0, 1, 2}
    state.pending_restart_stages = {1}
    assert not state.outputs_drained()


@pytest.mark.asyncio
async def test_cancel_bypasses_pending_terminal_barrier():
    orch, state = setup_request()
    orch._cfg_tracker.abort_parents = lambda ids: ids
    await route(orch, state, 2)
    await orch._handle_abort({"request_ids": ["r"]})
    orch._cleanup_request_ids.assert_awaited_once_with(["r"], abort=True)
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_async_consumer_yields_late_text_before_done_without_fake_chunk():
    orch, state = setup_request()
    await route(orch, state, 2)
    await route(orch, state, 0)
    await route(orch, state, 1)
    frontend = object.__new__(AsyncOmni)
    frontend.request_states = {"r": SimpleNamespace(queue=orch.output_async_queue)}
    def process(result, stage, metrics, starts, wall, final):
        if result["type"] == "request_done":
            return OmniBase._process_single_result(frontend, result, stage, metrics, starts, wall, final)
        return result["engine_outputs"]
    frontend._process_single_result = process
    async def collect():
        return [o async for o in frontend._process_orchestrator_results("r", Mock(), 2, {}, 0.)]
    outputs = await asyncio.wait_for(collect(), timeout=1.)
    assert [o.content for o in outputs] == ["stage-2", "stage-0"]
    assert orch.output_async_queue.empty()
