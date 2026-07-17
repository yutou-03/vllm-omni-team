from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.engine.core import EngineShutdownState

from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import _drain_status_response

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_proxy() -> AsyncOmniEngine:
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=queue.Queue())
    engine.rpc_output_queue = SimpleNamespace(sync_q=queue.Queue())
    engine._rpc_lock = threading.Lock()
    engine.orchestrator_thread = SimpleNamespace(is_alive=lambda: True)
    return engine


def test_engine_proxy_drain_rpc_uses_dedicated_control_queue() -> None:
    engine = _make_engine_proxy()
    received: list[dict[str, Any]] = []

    def _respond() -> None:
        message = engine.request_queue.sync_q.get(timeout=1)
        received.append(message)
        engine.rpc_output_queue.sync_q.put_nowait(
            {
                "type": "drain_status_result",
                "rpc_id": message["rpc_id"],
                "schema_version": 1,
                "healthy": True,
                "drained": True,
            }
        )

    responder = threading.Thread(target=_respond)
    responder.start()
    status = engine.get_drain_status(timeout=1)
    responder.join(timeout=1)

    assert status["drained"] is True
    assert received[0]["type"] == "drain_status"
    assert 0 < received[0]["timeout"] < 1


def test_engine_proxy_drain_rpc_lock_contention_is_bounded() -> None:
    engine = _make_engine_proxy()
    engine._rpc_lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="control-RPC lock"):
            engine.get_drain_status(timeout=0.01)
    finally:
        engine._rpc_lock.release()


class _FakeScheduler:
    def __init__(self) -> None:
        self.unfinished = 0
        self.requests: dict[str, object] = {}
        self.running: list[object] = []
        self.waiting: list[object] = []
        self.skipped_waiting: list[object] = []
        self.finished_req_ids: set[str] = set()
        self.finished_req_ids_dict: dict[int, set[str]] = {}
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()
        self.chunk_transfer_adapter = None

    def get_num_unfinished_requests(self) -> int:
        return self.unfinished

    def has_requests(self) -> bool:
        return bool(self.unfinished or self.requests)


def _make_core() -> StageEngineCoreProc:
    core = object.__new__(StageEngineCoreProc)
    core.scheduler = _FakeScheduler()
    core.batch_queue = deque()
    core.input_queue = queue.Queue()
    core.output_queue = queue.Queue()
    core.aborts_queue = queue.Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    core.engine_index = 0
    return core


@pytest.mark.parametrize(
    "layer",
    [
        "unfinished",
        "scheduler_requests",
        "finished_pending",
        "batch_queue",
        "core_output_queue",
    ],
)
def test_engine_core_any_nonempty_layer_is_not_drained(layer: str) -> None:
    core = _make_core()
    if layer == "unfinished":
        core.scheduler.unfinished = 1
    elif layer == "scheduler_requests":
        core.scheduler.requests["request-1"] = object()
    elif layer == "finished_pending":
        core.scheduler.finished_req_ids.add("request-1")
    elif layer == "batch_queue":
        core.batch_queue.append(object())
    else:
        core.output_queue.put_nowait(object())

    status = core.get_drain_status()

    assert status["healthy"] is True
    assert status["drained"] is False


def test_engine_core_all_empty_is_drained() -> None:
    status = _make_core().get_drain_status()

    assert status["healthy"] is True
    assert status["drained"] is True
    assert all(value == 0 for value in status["counters"].values())


def test_engine_core_snapshot_error_fails_closed() -> None:
    core = _make_core()
    core.scheduler = None

    status = core.get_drain_status()

    assert status["healthy"] is False
    assert status["drained"] is False
    assert "error" in status


def _make_chunk_adapter() -> OmniChunkTransferAdapter:
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._drain_state_lock = threading.Lock()
    adapter._pending_load_reqs = deque()
    adapter._pending_save_reqs = deque()
    adapter._recv_inflight = 0
    adapter._save_inflight = 0
    adapter._background_fault = None
    adapter._finished_load_reqs = set()
    adapter._finished_save_reqs = set()
    adapter.finished_requests = set()
    adapter.requests_with_ready_chunks = set()
    adapter.waiting_for_chunk_waiting_requests = deque()
    adapter.waiting_for_chunk_running_requests = deque()
    adapter.put_req_chunk = defaultdict(int)
    adapter.get_req_chunk = defaultdict(int)
    adapter.request_payload = {}
    adapter.code_prompt_token_ids = defaultdict(list)
    adapter.request_ids_mapping = {}
    adapter.requests_origin_status = {}
    adapter.stop_event = threading.Event()
    adapter.recv_thread = SimpleNamespace(is_alive=lambda: True)
    adapter.save_thread = SimpleNamespace(is_alive=lambda: True)
    return adapter


def test_chunk_transfer_detail_is_part_of_strict_drain() -> None:
    adapter = _make_chunk_adapter()
    assert adapter.get_drain_status()["drained"] is True

    adapter.requests_with_ready_chunks.add("request-1")
    status = adapter.get_drain_status()

    assert status["healthy"] is True
    assert status["stable_detail"] is True
    assert status["drained"] is False


def test_chunk_transfer_stopped_adapter_fails_closed() -> None:
    adapter = _make_chunk_adapter()
    adapter.stop_event.set()

    status = adapter.get_drain_status()

    assert status["healthy"] is False
    assert status["drained"] is False


@pytest.mark.parametrize("worker", ["recv_thread", "save_thread"])
def test_chunk_transfer_dead_worker_fails_closed(worker: str) -> None:
    adapter = _make_chunk_adapter()
    setattr(adapter, worker, SimpleNamespace(is_alive=lambda: False))

    status = adapter.get_drain_status()

    assert status["healthy"] is False
    assert status["drained"] is False


def test_chunk_transfer_persistent_save_fault_fails_closed() -> None:
    adapter = _make_chunk_adapter()
    adapter._save_cond = threading.Condition()
    adapter._pending_save_reqs.append({"request_id": "request-1"})

    def _fail_save(_task: object) -> None:
        # Stop after this iteration so save_loop can be exercised synchronously.
        adapter.stop_event.set()
        raise RuntimeError("connector put failed")

    adapter._send_single_request = _fail_save
    adapter.save_loop()
    adapter.stop_event.clear()

    status = adapter.get_drain_status()

    assert status["healthy"] is False
    assert status["drained"] is False
    assert status["error"] == "save: RuntimeError: connector put failed"


def _run_one_chunk_save(adapter: OmniChunkTransferAdapter) -> dict[str, Any]:
    adapter._save_cond = threading.Condition()
    adapter._pending_save_reqs.append(
        {
            "pooling_output": None,
            "request": SimpleNamespace(
                request_id="request-1",
                external_req_id="request-1",
            ),
            "is_finished": False,
        }
    )
    adapter.save_loop()
    adapter.stop_event.clear()
    return adapter.get_drain_status()


def test_chunk_transfer_connector_false_persists_save_fault() -> None:
    adapter = _make_chunk_adapter()

    class _Connector:
        stage_id = 0

        def put(self, **_kwargs: Any) -> tuple[bool, int, dict[str, str]]:
            adapter.stop_event.set()
            return False, 0, {"reason": "synthetic failure"}

    adapter.connector = _Connector()
    adapter.custom_process_next_stage_input_func = (
        lambda **_kwargs: {"meta": {"finished": False}}
    )

    status = _run_one_chunk_save(adapter)

    assert status["healthy"] is False
    assert status["drained"] is False
    assert "connector put returned unsuccessful" in status["error"]


def test_chunk_transfer_payload_builder_error_persists_save_fault() -> None:
    adapter = _make_chunk_adapter()
    adapter.connector = SimpleNamespace(stage_id=0)

    def _raise_builder(**_kwargs: Any) -> dict[str, Any]:
        adapter.stop_event.set()
        raise ValueError("synthetic payload failure")

    adapter.custom_process_next_stage_input_func = _raise_builder

    status = _run_one_chunk_save(adapter)

    assert status["healthy"] is False
    assert status["drained"] is False
    assert "custom next-stage payload builder failed" in status["error"]


def test_chunk_transfer_recv_program_error_persists_fault() -> None:
    adapter = _make_chunk_adapter()
    adapter._recv_cond = threading.Condition()

    class _BadRequest:
        @property
        def request_id(self) -> str:
            adapter.stop_event.set()
            raise ValueError("synthetic request decode failure")

    adapter._pending_load_reqs.append(_BadRequest())
    adapter.recv_loop()
    adapter.stop_event.clear()

    status = adapter.get_drain_status()

    assert status["healthy"] is False
    assert status["drained"] is False
    assert "recv: ValueError: synthetic request decode failure" in status["error"]


class _FakeDrainClient:
    def __init__(self, core_status: dict[str, Any]) -> None:
        self.core_status = core_status
        self.outputs_queue: queue.Queue[Any] = queue.Queue()
        self.pending_messages: list[tuple[Any, Any]] = []
        self.utility_results: dict[str, object] = {}

    async def get_drain_status_async(self, timeout: float) -> dict[str, Any]:
        assert timeout > 0
        return self.core_status


@pytest.mark.asyncio
async def test_stage_pool_output_queue_is_part_of_strict_drain() -> None:
    client = _FakeDrainClient(
        {"schema_version": 1, "healthy": True, "drained": True}
    )
    pool = StagePool(
        0,
        client,
        output_processor=SimpleNamespace(request_states={}),
    )
    assert (await pool.get_replica_drain_status(0, timeout=1))["drained"]

    client.outputs_queue.put_nowait(object())
    status = await pool.get_replica_drain_status(0, timeout=1)

    assert status["healthy"] is True
    assert status["drained"] is False


class _RaisingDrainClient(_FakeDrainClient):
    async def get_drain_status_async(self, timeout: float) -> dict[str, Any]:
        raise RuntimeError("core RPC failed")


@pytest.mark.asyncio
async def test_stage_pool_rpc_error_fails_closed() -> None:
    pool = StagePool(
        0,
        _RaisingDrainClient({}),
        output_processor=SimpleNamespace(request_states={}),
    )

    status = await pool.get_replica_drain_status(0, timeout=1)

    assert status["healthy"] is False
    assert status["drained"] is False
    assert status["error"] == "RuntimeError: core RPC failed"


@pytest.mark.parametrize(
    "core_status",
    [
        {"healthy": True, "drained": True},
        {"schema_version": 2, "healthy": True, "drained": True},
        {"schema_version": 1, "healthy": 1, "drained": True},
        {"schema_version": 1, "healthy": True, "drained": "yes"},
    ],
)
@pytest.mark.asyncio
async def test_stage_pool_malformed_core_status_fails_closed(
    core_status: dict[str, Any],
) -> None:
    pool = StagePool(
        0,
        _FakeDrainClient(core_status),
        output_processor=SimpleNamespace(request_states={}),
    )

    status = await pool.get_replica_drain_status(0, timeout=1)

    assert status["healthy"] is False
    assert status["drained"] is False
    assert status["error"] == "EngineCore returned an invalid drain-status schema"


class _FakeDrainPool:
    stage_id = 0
    num_replicas = 1

    def __init__(
        self,
        *,
        local_busy: bool = False,
        replica_status: dict[str, Any] | None = None,
        rpc_error: Exception | None = None,
        on_rpc: Any = None,
    ) -> None:
        self.local_busy = local_busy
        self.replica_status = replica_status or {
            "schema_version": 1,
            "stage_id": 0,
            "replica_id": 0,
            "healthy": True,
            "drained": True,
        }
        self.rpc_error = rpc_error
        self.on_rpc = on_rpc

    def get_local_drain_status(self) -> dict[str, Any]:
        count = int(self.local_busy)
        return {
            "stage_id": self.stage_id,
            "counters": {"request_bindings": count},
            "drained": not self.local_busy,
        }

    async def get_replica_drain_status(
        self,
        replica_id: int,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        assert replica_id == 0
        assert timeout > 0
        if self.on_rpc is not None:
            self.on_rpc()
        if self.rpc_error is not None:
            raise self.rpc_error
        return self.replica_status


def _make_orchestrator(pool: _FakeDrainPool) -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [pool]
    orchestrator._cfg_tracker = CfgCompanionTracker()
    orchestrator.request_async_queue = asyncio.Queue()
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator.rpc_async_queue = asyncio.Queue()
    orchestrator.request_states = {}
    orchestrator._pd_kv_params = {}
    orchestrator._drain_output_inflight = 0
    orchestrator._drain_state_epoch = 0
    orchestrator._fatal_error = None
    orchestrator._stages_shutdown = False
    orchestrator._shutdown_event = asyncio.Event()
    return orchestrator


async def _orchestrator_status(
    orchestrator: Orchestrator,
) -> dict[str, Any]:
    await orchestrator._handle_drain_status(
        {"rpc_id": "drain-test", "timeout": 1.0}
    )
    return await orchestrator.rpc_async_queue.get()


@pytest.mark.asyncio
async def test_orchestrator_all_empty_is_drained() -> None:
    status = await _orchestrator_status(_make_orchestrator(_FakeDrainPool()))

    assert status["healthy"] is True
    assert status["stable"] is True
    assert status["drained"] is True


@pytest.mark.parametrize(
    "layer",
    [
        "request_states",
        "admission_queue",
        "output_queue",
        "cfg_pending",
        "stage_binding",
        "stage_replica",
    ],
)
@pytest.mark.asyncio
async def test_orchestrator_any_nonempty_layer_is_not_drained(
    layer: str,
) -> None:
    replica_status = None
    if layer == "stage_replica":
        replica_status = {
            "schema_version": 1,
            "stage_id": 0,
            "replica_id": 0,
            "healthy": True,
            "drained": False,
        }
    orchestrator = _make_orchestrator(
        _FakeDrainPool(
            local_busy=layer == "stage_binding",
            replica_status=replica_status,
        )
    )
    if layer == "request_states":
        orchestrator.request_states["request-1"] = object()
    elif layer == "admission_queue":
        await orchestrator.request_async_queue.put({"type": "add_request"})
    elif layer == "output_queue":
        await orchestrator.output_async_queue.put({"type": "output"})
    elif layer == "cfg_pending":
        orchestrator._cfg_tracker.register_parent("request-1")

    status = await _orchestrator_status(orchestrator)

    assert status["healthy"] is True
    assert status["drained"] is False


@pytest.mark.asyncio
async def test_snapshot_churn_is_busy_not_unhealthy() -> None:
    orchestrator = _make_orchestrator(_FakeDrainPool())
    orchestrator.stage_pools[0].on_rpc = lambda: setattr(
        orchestrator,
        "_drain_state_epoch",
        orchestrator._drain_state_epoch + 1,
    )

    status = await _orchestrator_status(orchestrator)

    assert status["stable"] is False
    assert status["healthy"] is True
    assert status["drained"] is False


def _frontend_snapshot(*, epoch: int = 0, busy: bool = False) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "healthy": True,
        "drained": not busy,
        "counters": {"request_states": int(busy)},
    }


def _orchestrator_snapshot(*, drained: bool = True) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "healthy": True,
        "drained": drained,
    }
    stage = {
        "schema_version": 1,
        "stage_id": 0,
        "replica_id": 0,
        "healthy": True,
        "drained": drained,
        "core": core,
    }
    return {
        "schema_version": 1,
        "healthy": True,
        "drained": drained,
        "stable": True,
        "replicas": [stage],
        "invalid_reasons": [],
        "errors": [],
    }


def test_public_drain_schema_all_empty_and_frontend_busy() -> None:
    empty = _frontend_snapshot()
    backend = _orchestrator_snapshot()
    drained = AsyncOmni._combine_drain_snapshots(empty, backend, empty)

    assert set(drained) == {
        "schema_version",
        "healthy",
        "drained",
        "snapshot_epoch",
        "frontend",
        "orchestrator",
        "stages",
        "invalid_reasons",
        "errors",
    }
    assert drained["healthy"] is True
    assert drained["drained"] is True

    busy = _frontend_snapshot(busy=True)
    not_drained = AsyncOmni._combine_drain_snapshots(
        busy,
        backend,
        busy,
    )
    assert not_drained["healthy"] is True
    assert not_drained["drained"] is False


def test_frontend_snapshot_churn_is_busy_not_unhealthy() -> None:
    status = AsyncOmni._combine_drain_snapshots(
        _frontend_snapshot(epoch=4),
        _orchestrator_snapshot(),
        _frontend_snapshot(epoch=5),
    )

    assert status["healthy"] is True
    assert status["drained"] is False
    assert "frontend_snapshot_changed" in status["invalid_reasons"]


def test_contradictory_orchestrator_error_fails_closed() -> None:
    backend = _orchestrator_snapshot()
    backend["errors"] = ["synthetic hidden layer fault"]
    snapshot = _frontend_snapshot()

    status = AsyncOmni._combine_drain_snapshots(
        snapshot,
        backend,
        snapshot,
    )

    assert status["healthy"] is False
    assert status["drained"] is False


@pytest.mark.parametrize(
    "replicas",
    [
        [],
        ["not-an-object"],
        [{"schema_version": 1, "healthy": True, "drained": True}],
        [
            {
                "schema_version": 1,
                "stage_id": 0,
                "replica_id": 0,
                "healthy": True,
                "drained": True,
                "core": None,
            }
        ],
    ],
)
def test_empty_or_malformed_stage_replicas_fail_closed(
    replicas: list[Any],
) -> None:
    backend = _orchestrator_snapshot()
    backend["replicas"] = replicas
    snapshot = _frontend_snapshot()

    status = AsyncOmni._combine_drain_snapshots(
        snapshot,
        backend,
        snapshot,
    )

    assert status["healthy"] is False
    assert status["drained"] is False


def test_orchestrator_cannot_override_busy_stage() -> None:
    backend = _orchestrator_snapshot()
    backend["replicas"][0]["drained"] = False
    backend["replicas"][0]["core"]["drained"] = False
    snapshot = _frontend_snapshot()

    status = AsyncOmni._combine_drain_snapshots(
        snapshot,
        backend,
        snapshot,
    )

    assert status["healthy"] is True
    assert status["drained"] is False


class _RaisingDrainEngine:
    async def get_drain_status(self, timeout: float) -> dict[str, Any]:
        raise RuntimeError("synthetic RPC failure")


@pytest.mark.asyncio
async def test_rpc_error_returns_503_and_false() -> None:
    response = await _drain_status_response(_RaisingDrainEngine(), 1.0)
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["drained"] is False
    assert payload["errors"] == ["RuntimeError: synthetic RPC failure"]


class _BusyDrainEngine:
    async def get_drain_status(self, timeout: float) -> dict[str, Any]:
        return AsyncOmni._combine_drain_snapshots(
            _frontend_snapshot(busy=True),
            _orchestrator_snapshot(drained=False),
            _frontend_snapshot(busy=True),
        )


@pytest.mark.asyncio
async def test_healthy_busy_returns_200_and_false() -> None:
    response = await _drain_status_response(_BusyDrainEngine(), 1.0)
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["healthy"] is True
    assert payload["drained"] is False


class _DrainedEngine:
    async def get_drain_status(self, timeout: float) -> dict[str, Any]:
        snapshot = _frontend_snapshot()
        return AsyncOmni._combine_drain_snapshots(
            snapshot,
            _orchestrator_snapshot(),
            snapshot,
        )


@pytest.mark.asyncio
async def test_fully_drained_returns_200_and_true() -> None:
    response = await _drain_status_response(_DrainedEngine(), 1.0)
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["healthy"] is True
    assert payload["drained"] is True


@pytest.mark.asyncio
async def test_unsupported_engine_returns_503_and_false() -> None:
    response = await _drain_status_response(object(), 1.0)
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["drained"] is False
    assert payload["invalid_reasons"] == ["drain_status_unsupported"]


class _InvalidSchemaEngine:
    async def get_drain_status(self, timeout: float) -> dict[str, Any]:
        return {"schema_version": 2, "healthy": True, "drained": True}


@pytest.mark.asyncio
async def test_invalid_schema_returns_503_and_false() -> None:
    response = await _drain_status_response(_InvalidSchemaEngine(), 1.0)
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["drained"] is False
    assert payload["invalid_reasons"] == ["drain_status_schema_invalid"]
