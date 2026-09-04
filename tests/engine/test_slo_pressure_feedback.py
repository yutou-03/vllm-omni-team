from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sample(stage_id: int, epoch: int) -> dict[str, Any]:
    return {
        "pressure_feedback_schema_version": 1,
        "stage_id": stage_id,
        "control_epoch": epoch,
        "captured_monotonic_s": 100.0 + stage_id,
        "waiting_requests": stage_id + 1,
        "effective_max_num_seqs": 2,
    }


def test_stage_core_pressure_utilities_delegate_to_scheduler_owner():
    updates = []
    scheduler = SimpleNamespace(
        _export_slo_runtime_pressure=lambda: _sample(2, 7),
        _update_slo_downstream_pressure=updates.append,
    )
    core = SimpleNamespace(scheduler=scheduler)

    assert StageEngineCoreProc.get_slo_runtime_pressure(core) == _sample(2, 7)
    assert (
        StageEngineCoreProc.update_slo_downstream_pressure(
            core,
            _sample(2, 7),
        )
        is None
    )
    assert updates == [_sample(2, 7)]


@pytest.mark.asyncio
async def test_stage_client_pressure_wrappers_use_utility_channel():
    calls = []

    async def call_utility_async(method, *args):
        calls.append((method, args))
        return _sample(2, 9) if method == "get_slo_runtime_pressure" else None

    client = SimpleNamespace(call_utility_async=call_utility_async)

    exported = await StageEngineCoreClientBase.get_slo_runtime_pressure_async(
        client,
        timeout=0.25,
    )
    await StageEngineCoreClientBase.update_slo_downstream_pressure_async(
        client,
        _sample(2, 9),
        timeout=0.25,
    )

    assert exported == _sample(2, 9)
    assert calls == [
        ("get_slo_runtime_pressure", ()),
        ("update_slo_downstream_pressure", (_sample(2, 9),)),
    ]


class _FeedbackClient:
    def __init__(self, stage_id: int, epoch: int) -> None:
        self.stage_id = stage_id
        self.sample = _sample(stage_id, epoch)
        self.get_calls = 0
        self.updates = []

    async def get_slo_runtime_pressure_async(self, *, timeout: float):
        assert timeout == 0.25
        self.get_calls += 1
        return dict(self.sample)

    async def update_slo_downstream_pressure_async(
        self,
        sample: dict[str, Any],
        *,
        timeout: float,
    ) -> None:
        assert timeout == 0.25
        self.updates.append(dict(sample))


def _feedback_orchestrator(clients: list[_FeedbackClient]) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.stage_pools = [
        SimpleNamespace(num_replicas=1, clients=[client])
        for client in clients
    ]
    orchestrator._slo_feedback_last_epoch_by_edge = {}
    return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_relays_adjacent_pressure_downstream_to_upstream():
    stage0 = _FeedbackClient(0, 3)
    stage1 = _FeedbackClient(1, 5)
    stage2 = _FeedbackClient(2, 7)
    orchestrator = _feedback_orchestrator([stage0, stage1, stage2])

    await orchestrator._relay_slo_downstream_pressure_once(timeout=0.25)

    assert stage0.updates == [_sample(1, 5)]
    assert stage1.updates == [_sample(2, 7)]
    assert stage2.updates == []

    # Repeated source epochs are not resent; idle cores may retain an epoch.
    await orchestrator._relay_slo_downstream_pressure_once(timeout=0.25)
    assert stage0.updates == [_sample(1, 5)]
    assert stage1.updates == [_sample(2, 7)]

    stage2.sample = _sample(2, 8)
    await orchestrator._relay_slo_downstream_pressure_once(timeout=0.25)
    assert stage1.updates[-1] == _sample(2, 8)


@pytest.mark.asyncio
async def test_orchestrator_feedback_fails_closed_for_multi_replica_stage():
    stage0 = _FeedbackClient(0, 3)
    stage1a = _FeedbackClient(1, 5)
    stage1b = _FeedbackClient(1, 6)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.stage_pools = [
        SimpleNamespace(num_replicas=1, clients=[stage0]),
        SimpleNamespace(num_replicas=2, clients=[stage1a, stage1b]),
    ]
    orchestrator._slo_feedback_last_epoch_by_edge = {}

    await orchestrator._relay_slo_downstream_pressure_once(timeout=0.25)

    assert stage0.updates == []
    assert stage1a.get_calls == 0
    assert stage1b.get_calls == 0
