"""CPU checks of the hold primitive against the real scheduler/runner methods."""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

# Omni must patch Request before this test binds the vLLM request classes.
# isort: off
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.scheduling.small_hold_probe import (
    CONTROLLER_ENV,
    CONTROL_FILE_ENV,
    HOLD_ENV,
    SmallHoldProbe,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(request_id="victim"):
    request = Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
        arrival_time=0.0,
    )
    request.append_output_token_ids([4, 5])
    request.num_computed_tokens = 4
    request.status = RequestStatus.RUNNING
    return request


def _scheduler(*requests, stage=1, cap=64):
    # Avoid creating an engine/model/device. Native schedule() below is real;
    # KV allocation is a CPU spy because this test checks ownership, not kernels.
    scheduler = object.__new__(OmniARScheduler)
    scheduler._omni_stage_id_for_trace = lambda: stage
    scheduler.scheduler_config = SimpleNamespace(
        async_scheduling=False, long_prefill_token_threshold=0
    )
    scheduler.max_num_running_reqs = cap
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 128
    scheduler.num_spec_tokens = 0
    scheduler.num_lookahead_tokens = 0
    scheduler.use_pp = False
    scheduler.use_v2_model_runner = False
    scheduler.need_mamba_block_aligned_split = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.lora_config = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[None])
    scheduler.kv_cache_manager = SimpleNamespace(
        new_step_starts=Mock(),
        allocate_slots=Mock(return_value=KVCacheBlocks(([],))),
        get_num_common_prefix_blocks=Mock(return_value=[0]),
        get_block_ids=Mock(return_value=([31, 32],)),
        free=Mock(),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.running = list(requests)
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.finished_req_ids = set()
    scheduler.prev_step_scheduled_req_ids = set(scheduler.requests)
    scheduler.chunk_transfer_adapter = None
    return scheduler


def _probe(scheduler, **environment):
    events = []
    clock = SimpleNamespace(now=100.0)
    probe = SmallHoldProbe(
        scheduler,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
        clock=lambda: clock.now,
        environ={HOLD_ENV: "1", **environment},
    )
    scheduler._small_hold_probe = probe
    return probe, clock, events


def _native_step(scheduler, probe):
    with probe.scheduling_context():
        output = Scheduler.schedule(scheduler)
    probe.record_scheduled(output)
    return output


def test_real_scheduler_retains_kv_progress_and_reschedules_with_all_tokens():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(victim, peer)
    probe, clock, events = _probe(scheduler)
    original_tokens = list(victim.all_token_ids)
    assert scheduler.request_hold("victim", 0.025, "correctness", {"sample": 1})
    assert not scheduler.request_hold("victim", 0.025, "duplicate", {})
    output = _native_step(scheduler, probe)
    assert output.num_scheduled_tokens == {"peer": 1}
    assert victim in scheduler.running
    assert victim.status == RequestStatus.RUNNING
    assert victim.num_computed_tokens == 4
    assert list(victim.all_token_ids) == original_tokens
    assert scheduler.prev_step_scheduled_req_ids == {"peer"}
    assert not output.preempted_req_ids
    assert scheduler.max_num_running_reqs == 64
    scheduler.kv_cache_manager.free.assert_not_called()
    allocations = scheduler.kv_cache_manager.allocate_slots.call_args_list
    assert [call.args[0].request_id for call in allocations] == ["peer"]

    clock.now = probe.holds["victim"].deadline_monotonic_s
    output = _native_step(scheduler, probe)
    assert output.num_scheduled_tokens == {"victim": 1}
    assert output.scheduled_cached_reqs.all_token_ids["victim"] == original_tokens
    assert not output.scheduled_cached_reqs.resumed_req_ids
    assert not output.scheduled_new_reqs
    assert not output.preempted_req_ids
    assert victim.num_computed_tokens == 5
    assert [event["event"] for event in events] == [
        "small_hold_requested", "small_hold_released", "small_hold_first_rescheduled"
    ]
    assert events[0]["kv_block_ids"] == events[-1]["kv_block_ids"] == ([31, 32],)
    assert events[-1]["first_rescheduled_monotonic_s"] == clock.now
    assert events[-1]["held_schedule_steps"] == 1
    assert not probe.holds


def test_full_resident_cap_does_not_admit_waiting_request():
    requests = [_request(str(index)) for index in range(64)]
    scheduler = _scheduler(*requests)
    waiting = _request("waiting")
    waiting.status = RequestStatus.WAITING
    scheduler.waiting.add_request(waiting)
    probe, _, _ = _probe(scheduler)
    assert probe.request_hold("0", 0.05, "cap", {})
    output = _native_step(scheduler, probe)
    assert len(output.num_scheduled_tokens) == 63
    assert len(scheduler.running) == 64
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting.peek_request() is waiting


def test_native_allocation_failure_cannot_preempt_held_request():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(peer, victim)
    scheduler.log_stats = False
    scheduler.encoder_cache_manager.free = Mock()
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler._preempt_request = MethodType(Scheduler._preempt_request, scheduler)
    probe, _, _ = _probe(scheduler)
    assert probe.request_hold("victim", 0.05, "kv", {})
    output = _native_step(scheduler, probe)
    assert output.preempted_req_ids == {"peer"}
    assert scheduler.running == [victim]
    assert victim.num_computed_tokens == 4
    assert victim.status == RequestStatus.RUNNING
    assert victim.num_preemptions == 0
    scheduler.kv_cache_manager.free.assert_called_once_with(peer)


def test_hidden_chunk_residents_are_counted_and_reserve_slots():
    victim, peer, blocked = _request(), _request("peer"), _request("blocked")
    scheduler = _scheduler(victim, peer, cap=3)
    scheduler.chunk_transfer_adapter = SimpleNamespace(waiting_for_chunk_running_requests=[blocked])
    probe, _, events = _probe(scheduler)
    assert probe.request_hold("victim", 0.01, "resident", {})
    with probe.scheduling_context():
        assert scheduler.max_num_running_reqs == 1
        assert scheduler.running == [peer]
    assert scheduler.max_num_running_reqs == 3
    assert scheduler.running == [victim, peer]
    assert events[0]["resident_count"] == 3
    assert events[0]["input_blocked_resident_count"] == 1


def test_context_restores_on_error_but_does_not_revive_cancelled_request():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(victim, peer)
    running_list = scheduler.running
    probe, _, _ = _probe(scheduler)
    assert probe.request_hold("victim", 0.01, "exception", {})
    with pytest.raises(RuntimeError, match="injected"):
        with probe.scheduling_context():
            raise RuntimeError("injected")
    assert scheduler.running is running_list
    assert scheduler.running == [victim, peer]
    assert scheduler.max_num_running_reqs == 64
    with probe.scheduling_context():
        del scheduler.requests["victim"]
        victim.status = RequestStatus.FINISHED_ABORTED
    assert scheduler.running == [peer]
    assert scheduler.max_num_running_reqs == 64
    with probe.scheduling_context():
        pass
    assert not probe.holds


def test_context_preserves_other_preemptions_and_admissions():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(peer, victim)
    probe, _, _ = _probe(scheduler)
    assert probe.request_hold("victim", 0.01, "restore", {})
    newcomer = _request("new")
    with probe.scheduling_context():
        scheduler.running.remove(peer)
        peer.status = RequestStatus.PREEMPTED
        scheduler.running.append(newcomer)
        scheduler.requests["new"] = newcomer
    assert scheduler.running == [victim, newcomer]


@pytest.mark.parametrize("duration", [0, -1, 0.100001, float("inf"), float("nan"), True])
def test_invalid_duration_is_rejected(duration):
    probe, _, _ = _probe(_scheduler(_request()))
    with pytest.raises(ValueError, match="duration_s"):
        probe.request_hold("victim", duration, "invalid", {})


@pytest.mark.parametrize("mutation", ["prefill", "placeholder", "spec", "not_running", "no_output", "no_next_token"])
def test_only_confirmed_decode_boundary_can_be_held(mutation):
    request = _request()
    if mutation == "prefill":
        request.num_computed_tokens = 1
    elif mutation == "placeholder":
        request.num_output_placeholders = 1
    elif mutation == "spec":
        request.spec_token_ids = [6]
    elif mutation == "not_running":
        request.status = RequestStatus.WAITING
    elif mutation == "no_output":
        request._output_token_ids.clear()
    else:
        request.num_computed_tokens = request.num_tokens
    probe, _, events = _probe(_scheduler(request))
    assert not probe.request_hold("victim", 0.01, "boundary", {})
    assert not events


@pytest.mark.parametrize("stage,enabled", [(0, "1"), (2, "1"), (1, "0")])
def test_disabled_or_other_stage_is_inert(stage, enabled):
    scheduler = _scheduler(_request(), stage=stage)
    probe, _, events = _probe(scheduler, **{HOLD_ENV: enabled, CONTROL_FILE_ENV: "/not/read"})
    assert not probe.request_hold("victim", 0.01, "disabled", {})
    probe.maybe_force_hold()
    with probe.scheduling_context():
        assert scheduler.running
        assert scheduler.max_num_running_reqs == 64
    assert not events


@pytest.mark.parametrize("environment", [
    {"VLLM_OMNI_SLO_ACTIVE_PREEMPTION": "1"},
    {"VLLM_OMNI_CONTROLLED_PREEMPTION_FILE": "/preempt.json"},
])
def test_preemption_cannot_be_combined_with_hold(environment):
    with pytest.raises(ValueError, match="coexist"):
        _probe(_scheduler(_request()), **environment)


def test_async_scheduler_or_async_flag_or_large_cap_is_rejected():
    scheduler = _scheduler(_request())
    scheduler.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="synchronous"):
        _probe(scheduler)
    scheduler.scheduler_config.async_scheduling = False
    scheduler.__class__ = OmniARAsyncScheduler
    with pytest.raises(ValueError, match="synchronous"):
        _probe(scheduler)
    with pytest.raises(ValueError, match="cap"):
        _probe(_scheduler(_request(), cap=65))


def test_forced_control_is_loaded_once_and_fires_once_at_threshold(tmp_path):
    control = tmp_path / "hold.json"
    control.write_text(json.dumps({"schema_version": 1, "forced_hold": {
        "request_id": "*", "duration_s": 0.01, "min_output_tokens": 3
    }}))
    request = _request()
    scheduler = _scheduler(request)
    probe, _, events = _probe(scheduler, **{CONTROL_FILE_ENV: str(control)})
    control.write_text("invalid replacement is intentionally ignored")
    probe.maybe_force_hold()
    assert not events
    request.append_output_token_ids([6])
    request.num_computed_tokens += 1
    probe.maybe_force_hold()
    probe.maybe_force_hold()
    assert len(events) == 1
    assert probe.forced_hold_fired


def test_controller_hook_uses_same_primitive_and_does_not_load_when_disabled(monkeypatch):
    import vllm_omni.scheduling.small_hold_probe as module

    calls = []

    def controller(scheduler, now):
        calls.append(now)
        scheduler.request_hold("victim", 0.01, "controller", {"now": now})

    loader = Mock(return_value=SimpleNamespace(before_schedule=controller))
    monkeypatch.setattr(module.importlib, "import_module", loader)
    scheduler = _scheduler(_request())
    probe, _, events = _probe(scheduler, **{CONTROLLER_ENV: "policy:before_schedule"})
    probe.before_schedule()
    assert calls == [100.0]
    assert events[0]["reason"] == "controller"
    loader.assert_called_once_with("policy")
    loader.reset_mock()
    probe, _, _ = _probe(scheduler, **{HOLD_ENV: "0", CONTROLLER_ENV: "missing:callback"})
    probe.before_schedule()
    loader.assert_not_called()


@pytest.mark.parametrize("split", [False, True])
def test_real_runner_unschedule_and_readd_preserves_talker_state(monkeypatch, split):
    from vllm_omni.worker import gpu_model_runner as runner_module

    request = _request()
    scheduler = _scheduler(request)
    probe, clock, _ = _split_probe(scheduler, cap=1) if split else _probe(scheduler)
    assert probe.request_hold("victim", 0.01, "runner", {})
    held_output = _native_step(scheduler, probe)

    runner = object.__new__(runner_module.OmniGPUModelRunner)
    state = SimpleNamespace(
        req_id="victim", prev_num_draft_len=0, output_token_ids=[4, 5],
        block_ids=([31, 32],), num_computed_tokens=4,
    )
    batch = SimpleNamespace(req_id_to_index={"victim": 0})
    batch.remove_request = lambda request_id: batch.req_id_to_index.pop(request_id, None)
    batch.add_request = lambda req: batch.req_id_to_index.update({req.req_id: 0})
    batch.condense = lambda: None
    batch.refresh_metadata = lambda: None
    batch.update_req_spec_token_ids = lambda *_args: None
    runner.input_batch = batch
    runner.requests = {"victim": state}
    intermediate = {"codec": object()}
    rng = object()
    runner.model_intermediate_buffer = {"victim": intermediate}
    runner._talker_mtp_generators = {"victim": rng}
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.omni_prefix_cache = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner._may_reorder_batch = lambda _output: None
    monkeypatch.setattr(runner_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    runner._update_states(held_output)
    assert not batch.req_id_to_index
    assert runner.requests["victim"] is state
    assert runner.model_intermediate_buffer["victim"] is intermediate
    assert runner._talker_mtp_generators["victim"] is rng
    assert state.block_ids == ([31, 32],)
    assert state.output_token_ids == [4, 5]
    clock.now += 0.02
    resumed_output = _native_step(scheduler, probe)
    runner._update_states(resumed_output)
    assert batch.req_id_to_index == {"victim": 0}
    assert runner.requests["victim"] is state
    assert runner.model_intermediate_buffer["victim"] is intermediate
    assert runner._talker_mtp_generators["victim"] is rng
    assert state.num_computed_tokens == 4  # input position, before the next forward
    assert state.output_token_ids == [4, 5]


# These opt-in tests use the real native scheduling loop with CPU KV spies.
def _split_probe(scheduler, cap=2, enabled=True, stages="1"):
    from vllm_omni.scheduling.small_hold_probe import EXECUTION_CAP_ENV, STAGES_ENV
    return _probe(scheduler, **{EXECUTION_CAP_ENV: str(cap), STAGES_ENV: stages,
                               HOLD_ENV: "1" if enabled else "0"})


def _enable_native_admission(scheduler):
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.use_eagle = False
    scheduler.log_stats = False
    empty = KVCacheBlocks(([],))
    scheduler.kv_cache_manager.get_computed_blocks = Mock(return_value=(empty, 0))
    scheduler.kv_cache_manager.empty_kv_cache_blocks = empty
    scheduler.kv_cache_manager.get_blocks = Mock(return_value=empty)


def _enqueue(scheduler, name):
    request = Request(request_id=name, prompt_token_ids=[10, 11],
                      sampling_params=SamplingParams(max_tokens=32),
                      pooling_params=None, arrival_time=1.0)
    scheduler.requests[name] = request
    scheduler.waiting.add_request(request)
    return request


def _accept_cpu_outputs(scheduler, output):
    for rid in output.num_scheduled_tokens:
        scheduler.requests[rid].append_output_token_ids([99])


def test_split_cap_hold_admits_newcomer_then_waits_for_resume_slot():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(victim, peer, cap=3)
    _enable_native_admission(scheduler)
    newcomer = _enqueue(scheduler, "newcomer")
    probe, clock, events = _split_probe(scheduler)
    tokens = list(victim.all_token_ids)
    assert probe.request_hold("victim", .05, "split", {})
    first = _native_step(scheduler, probe)
    assert set(first.num_scheduled_tokens) == {"peer", "newcomer"}
    assert scheduler.running == [victim, peer, newcomer]
    assert probe._resident_fields()["resident_count"] == 3
    assert probe._resident_fields()["execution_cap"] == 2
    assert list(victim.all_token_ids) == tokens and victim.num_computed_tokens == 4
    scheduler.kv_cache_manager.free.assert_not_called()
    _accept_cpu_outputs(scheduler, first)
    clock.now += .06
    pending = _native_step(scheduler, probe)
    assert set(pending.num_scheduled_tokens) == {"peer", "newcomer"}
    assert probe.holds["victim"].released_monotonic_s is not None
    assert probe._resident_fields()["resume_pending_count"] == 1
    assert not any(e["event"] == "small_hold_first_rescheduled" for e in events)
    _accept_cpu_outputs(scheduler, pending)
    scheduler.running.remove(peer)
    scheduler.requests.pop("peer")
    peer.status = RequestStatus.FINISHED_STOPPED
    _enqueue(scheduler, "later")
    resumed = _native_step(scheduler, probe)
    assert set(resumed.num_scheduled_tokens) == {"victim", "newcomer"}
    assert scheduler.requests["victim"] is victim and list(victim.all_token_ids) == tokens
    assert victim.num_computed_tokens == 5 and "victim" not in probe.holds
    assert not resumed.preempted_req_ids and not resumed.scheduled_cached_reqs.resumed_req_ids
    assert scheduler.waiting.peek_request().request_id == "later"
    assert scheduler.max_num_running_reqs == 3


def test_split_cap_without_holds_limits_fcfs_continuous_batch():
    scheduler = _scheduler(cap=3)
    _enable_native_admission(scheduler)
    for rid in ("one", "two", "three"):
        _enqueue(scheduler, rid)
    probe, _, _ = _split_probe(scheduler, enabled=False)
    output = _native_step(scheduler, probe)
    assert set(output.num_scheduled_tokens) == {"one", "two"}
    assert len(scheduler.waiting) == 1 and scheduler.max_num_running_reqs == 3
    assert not probe.request_hold("one", .01, "disabled", {})


def test_split_cap_counts_input_wait_and_defers_returning_resident():
    victim, peer, blocked = _request(), _request("peer"), _request("blocked")
    scheduler = _scheduler(victim, peer, cap=3)
    scheduler.requests["blocked"] = blocked
    scheduler.chunk_transfer_adapter = SimpleNamespace(waiting_for_chunk_running_requests=[blocked])
    probe, clock, _ = _split_probe(scheduler)
    assert probe.request_hold("victim", .05, "input", {})
    with probe.scheduling_context():
        assert scheduler.running == [peer] and scheduler.max_num_running_reqs == 1
    scheduler.chunk_transfer_adapter.waiting_for_chunk_running_requests.clear()
    scheduler.running.append(blocked)
    with probe.scheduling_context():
        assert scheduler.running == [peer, blocked] and scheduler.max_num_running_reqs == 2
    clock.now += .06
    with probe.scheduling_context():
        assert scheduler.running == [peer, blocked]
    assert len(scheduler.running) == 3
    assert probe.holds["victim"].released_monotonic_s is not None


def test_split_cap_ready_input_waits_if_newcomer_owns_freed_execution_slot():
    peer, returning, newcomer = _request("peer"), _request("returning"), _request("newcomer")
    scheduler = _scheduler(peer, cap=3)
    scheduler.requests["returning"] = returning
    scheduler.chunk_transfer_adapter = SimpleNamespace(waiting_for_chunk_running_requests=[returning])
    probe, _, _ = _split_probe(scheduler, enabled=False)
    with probe.scheduling_context():
        assert scheduler.max_num_running_reqs == 2
        scheduler.requests["newcomer"] = newcomer
        scheduler.running.append(newcomer)
    scheduler.chunk_transfer_adapter.waiting_for_chunk_running_requests.clear()
    scheduler.running.append(returning)
    with probe.scheduling_context():
        assert scheduler.running == [peer, newcomer] and scheduler.max_num_running_reqs == 2
    assert scheduler.running == [peer, newcomer, returning]


def test_split_cap_reactive_preemption_cannot_select_hidden_hold():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(peer, victim, cap=3)
    scheduler.log_stats = False
    scheduler.encoder_cache_manager.free = Mock()
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler._preempt_request = MethodType(Scheduler._preempt_request, scheduler)
    probe, _, _ = _split_probe(scheduler)
    assert probe.request_hold("victim", .05, "kv", {})
    output = _native_step(scheduler, probe)
    assert output.preempted_req_ids == {"peer"} and scheduler.running == [victim]
    assert victim.num_computed_tokens == 4
    scheduler.kv_cache_manager.free.assert_called_once_with(peer)


def test_split_cap_failed_new_allocation_stays_waiting():
    victim = _request()
    scheduler = _scheduler(victim, cap=2)
    _enable_native_admission(scheduler)
    newcomer = _enqueue(scheduler, "newcomer")
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    probe, _, _ = _split_probe(scheduler, cap=1)
    assert probe.request_hold("victim", .05, "kv", {})
    output = _native_step(scheduler, probe)
    assert not output.num_scheduled_tokens
    assert scheduler.waiting.peek_request() is newcomer and scheduler.running == [victim]
    scheduler.kv_cache_manager.free.assert_not_called()


def test_split_cap_exception_cancellation_and_protected_transfer_accounting():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(victim, peer, cap=3)
    scheduler.kv_cache_manager.coordinator = SimpleNamespace(single_type_managers=(
        SimpleNamespace(req_to_blocks={"transfer": [object()], "empty": []}),))
    probe, _, _ = _split_probe(scheduler)
    assert probe.request_hold("victim", .05, "cancel", {})
    with pytest.raises(RuntimeError, match="injected"):
        with probe.scheduling_context():
            assert scheduler.max_num_running_reqs == 1
            del scheduler.requests["victim"]
            victim.status = RequestStatus.FINISHED_ABORTED
            raise RuntimeError("injected")
    assert scheduler.running == [peer] and scheduler.max_num_running_reqs == 3
    assert not probe.holds and probe._resident_fields()["resident_count"] == 2


@pytest.mark.parametrize("cap", ["0", "-1", "2.5", "true", "4", ""])
def test_split_cap_invalid_bounds(cap):
    from vllm_omni.scheduling.small_hold_probe import EXECUTION_CAP_ENV
    with pytest.raises(ValueError):
        _probe(_scheduler(cap=3), **{EXECUTION_CAP_ENV: cap})


@pytest.mark.parametrize("stages", ["", "2", "-1", "0,2", "0,"])
def test_split_cap_invalid_stage_selection(stages):
    from vllm_omni.scheduling.small_hold_probe import STAGES_ENV
    with pytest.raises(ValueError, match="Stage"):
        _probe(_scheduler(), **{STAGES_ENV: stages})


def test_split_cap_stage_selection_and_resident_overflow_defense():
    scheduler = _scheduler(_request(), stage=1, cap=3)
    probe, _, _ = _split_probe(scheduler, cap=1, stages="0")
    assert not probe.enabled and probe.execution_cap is None
    with probe.scheduling_context():
        assert scheduler.max_num_running_reqs == 3
    scheduler = _scheduler(*[_request(str(i)) for i in range(4)], cap=3)
    probe, _, _ = _split_probe(scheduler)
    with pytest.raises(RuntimeError, match="resident count"):
        with probe.scheduling_context():
            pass
    assert len(scheduler.running) == 4 and scheduler.max_num_running_reqs == 3


def test_split_cap_rejects_native_remote_kv_connector():
    scheduler = _scheduler(cap=3)
    scheduler.connector = object()
    with pytest.raises(ValueError, match="remote KV"):
        _split_probe(scheduler, enabled=False)


def test_split_stage_zero_capacity_event_and_equal_caps_keep_resident_limit():
    victim, peer = _request(), _request("peer")
    scheduler = _scheduler(victim, peer, cap=2, stage=0)
    _enable_native_admission(scheduler)
    newcomer = _enqueue(scheduler, "newcomer")
    probe, _, events = _split_probe(scheduler, cap=2, stages="0")
    assert events[0]["event"] == "small_hold_capacity_configured"
    assert events[0]["execution_cap"] == events[0]["resident_cap"] == 2
    assert probe.request_hold("victim", .05, "equal_caps", {})
    output = _native_step(scheduler, probe)
    assert set(output.num_scheduled_tokens) == {"peer"}
    assert scheduler.waiting.peek_request() is newcomer
