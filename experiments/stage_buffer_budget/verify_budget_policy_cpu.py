"""Standalone CPU verification for the isolated successor buffer policy."""
from collections import defaultdict, deque
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'vllm_omni/scheduling/small_hold_probe.py'

trace = ModuleType('vllm_omni.profiling.stage_queue_trace')
trace.emit_stage_event = lambda *_args, **_kwargs: None
sys.modules.setdefault('vllm_omni', ModuleType('vllm_omni'))
sys.modules.setdefault('vllm_omni.profiling', ModuleType('vllm_omni.profiling'))
sys.modules['vllm_omni.profiling.stage_queue_trace'] = trace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hold_module = load('budget_hold_probe_cpu', SOURCE)
policy_module = load('budget_controller_cpu', ROOT.parents[1]/'stage_buffer_controller_budget.py')
sys.modules['stage_buffer_controller_budget'] = policy_module


class OmniARScheduler:
    def __init__(self, requests, stage=0):
        self._omni_stage_id_for_trace = lambda: stage
        self.scheduler_config = SimpleNamespace(async_scheduling=False)
        self.max_num_running_reqs = 64
        self.max_num_scheduled_tokens = 128
        self.num_spec_tokens = 0
        self.use_pp = False
        self.running = list(requests)
        self.requests = {request.request_id: request for request in requests}
        self.waiting = []
        self.chunk_transfer_adapter = None
        self.kv_cache_manager = SimpleNamespace(get_block_ids=lambda _rid: ([31, 32],))


def request(rid, *, decode=True):
    return SimpleNamespace(
        request_id=rid, status=SimpleNamespace(name='RUNNING'),
        num_computed_tokens=4 if decode else 1, num_prompt_tokens=3,
        num_output_tokens=2, num_output_placeholders=0, spec_token_ids=[],
        num_tokens=5,
    )


def probe(requests):
    scheduler = OmniARScheduler(requests)
    clock = SimpleNamespace(now=100.0)
    events = []
    p = hold_module.SmallHoldProbe(
        scheduler, emit=lambda name, **kw: events.append(dict(event=name, **kw)),
        clock=lambda: clock.now,
        environ={hold_module.HOLD_ENV: '1', 'VLLM_OMNI_STAGE_BUFFER_POLICY': '1'},
    )
    scheduler._small_hold_probe = p
    return scheduler, p, clock, events


def controller(stage=0):
    c = object.__new__(policy_module.Controller)
    c.stage = stage
    c.config = json.loads((ROOT/'controller_budget_config.json').read_text())
    c.c = c.config['stages'][str(stage)]
    c.state = defaultdict(policy_module.Inventory)
    c.last_call = None
    c.last_had_running = False
    c.ticks = deque(maxlen=64)
    c.last_end = {}
    c.last_log = {}
    c.reasons = defaultdict(int)
    c.last_heartbeat = 100.0
    c.watermark = 0.0
    c.offset = 0
    c.refresh = lambda: None
    c.record = lambda *_args, **_kwargs: None
    return c


class BudgetPolicyTests(unittest.TestCase):
    def test_held_slice_blocks_new_prefill_and_caps_work(self):
        scheduler, p, clock, events = probe([request('victim'), request('peer')])
        scheduler.waiting.append(request('large_prefill', decode=False))
        self.assertTrue(p.request_hold('victim', .05, 'budget', {}))
        with p.scheduling_context():
            self.assertEqual([r.request_id for r in scheduler.running], ['peer'])
            self.assertEqual(scheduler.max_num_running_reqs, 1)
            self.assertEqual(scheduler.max_num_scheduled_tokens, 64)
            self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(scheduler.max_num_running_reqs, 64)
        self.assertEqual(scheduler.max_num_scheduled_tokens, 128)
        self.assertEqual([r.request_id for r in scheduler.running], ['victim', 'peer'])
        clock.now = 100.05
        with p.scheduling_context():
            self.assertEqual(len(scheduler.running), 2)
        self.assertFalse(p.buffer_policy_suspended)
        self.assertEqual(events[-1]['deadline_lateness_s'], 0.0)

    def test_prefill_peer_cannot_be_held_behind(self):
        scheduler, p, _clock, _events = probe([request('victim'), request('prefill', decode=False)])
        self.assertFalse(p.request_hold('victim', .05, 'budget', {}))
        self.assertEqual(len(scheduler.running), 2)

    def test_resident_peer_leaving_decode_releases_hold_before_schedule(self):
        scheduler, p, _clock, events = probe([request('victim'), request('peer')])
        self.assertTrue(p.request_hold('victim', .05, 'budget', {}))
        scheduler.requests['peer'].num_tokens = 20
        with p.scheduling_context():
            self.assertEqual([r.request_id for r in scheduler.running],
                             ['victim', 'peer'])
            self.assertEqual(scheduler.max_num_scheduled_tokens, 128)
        self.assertEqual([e['event'] for e in events], [
            'small_hold_requested', 'small_hold_resume_requested',
            'small_hold_released'])
        self.assertEqual(events[1]['resume_reason'], 'resident_left_decode')

    def test_last_peer_finishing_releases_hold_before_idle_step(self):
        scheduler, p, _clock, events = probe([request('victim'), request('peer')])
        self.assertTrue(p.request_hold('victim', .05, 'budget', {}))
        scheduler.running.pop()
        with p.scheduling_context():
            self.assertEqual([r.request_id for r in scheduler.running], ['victim'])
        self.assertEqual(events[1]['resume_reason'], 'resident_left_decode')

    def test_deadline_overrun_fails_closed(self):
        scheduler, p, clock, events = probe([request('victim'), request('peer')])
        self.assertTrue(p.request_hold('victim', .05, 'budget', {}))
        clock.now = 100.101  # 51 ms beyond the 50 ms test hold.
        with p.scheduling_context():
            self.assertEqual(len(scheduler.running), 2)
        self.assertTrue(p.buffer_policy_suspended)
        self.assertFalse(p.request_hold('victim', .01, 'again', {}))
        self.assertEqual([e['event'] for e in events], [
            'small_hold_requested', 'small_hold_released', 'stage_buffer_policy_suspended'])
        self.assertAlmostEqual(events[1]['deadline_lateness_s'], .051)

    def test_relaxed_hold_cap_accepts_100ms_but_not_more(self):
        scheduler, p, _clock, _events = probe([request('victim'), request('peer')])
        self.assertTrue(p.request_hold('victim', .10, 'budget', {}))
        with self.assertRaises(ValueError):
            p.request_hold('peer', .101, 'too_long', {})

    def test_controller_uses_repository_config_without_machine_path(self):
        fake_scheduler = SimpleNamespace(_omni_stage_id_for_trace=lambda: 0)
        with patch.dict(policy_module.os.environ, {
            'STAGE_QUEUE_TRACE_DIR': '/tmp/budget_test/traces',
            'VLLM_OMNI_STAGE_BUFFER_CONFIG': '',
        }):
            c = policy_module.Controller(fake_scheduler)
        self.assertEqual(c.config['max_hold_s'], .10)

    def test_receiver_ready_inventory_and_variable_duration(self):
        c = controller()
        inv = c.state['victim']
        sent = dict(event='stage_study_text_payload_sent', monotonic=1., chunk_id=0,
                    raw_entity_embedding_layout_valid=True,
                    raw_entity_embedding_rows_total=220, raw_ids_prompt_count=215)
        inv.update(sent)
        self.assertEqual(inv.q(0), 5)
        self.assertIsNone(inv.q_ready(0))
        inv.update(dict(sent, event='stage_study_text_payload_received'))
        inv.update(dict(sent, event='stage_study_text_payload_received'))
        self.assertEqual(inv.q_ready(0), 5)
        fields = c.fields('victim', 100., .05)
        self.assertGreater(fields['requested_hold_s'], .01)
        self.assertLess(fields['requested_hold_s'], .02)
        calls = []
        fake_probe = SimpleNamespace(holds={}, buffer_policy_suspended=False,
                                     _at_decode_boundary=lambda _request: True)
        fake_scheduler = SimpleNamespace(
            _small_hold_probe=fake_probe,
            running=[SimpleNamespace(request_id='victim')],
            request_hold=lambda rid, duration, reason, evidence:
                calls.append(duration) or True,
        )
        with patch.object(policy_module.time, 'monotonic', return_value=100.):
            c(fake_scheduler, 100.)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0], fields['requested_hold_s'])
        inv.rows += 15
        inv.received_rows += 15
        self.assertEqual(c.fields('victim', 100., .05)['requested_hold_s'], .10)
        inv.received_rows = 0
        self.assertIsNone(c.fields('victim', 100., .05)['coverage_s'])

    def test_stage1_chunk_inventory_has_no_safe_budget_at_25_frames(self):
        c = controller(1)
        inv = c.state['victim']
        inv.codec_sent[0] = 25
        inv.codec_received[0] = 25
        fields = c.fields('victim', 100., .02)
        self.assertLess(fields['coverage_s'], fields['high_s'])
        self.assertEqual(fields['requested_hold_s'], 0.)


def urgent_setup(*, protected_extra=False):
    victim = request('victim')
    urgent = request('urgent', decode=False)
    urgent.status = SimpleNamespace(name='WAITING')
    scheduler = OmniARScheduler([victim])
    scheduler.max_num_running_reqs = 2
    scheduler.requests[urgent.request_id] = urgent
    scheduler.waiting.append(urgent)
    scheduler._baseline_policy = SimpleNamespace(value='stage_deadline_edf_np')
    scheduler._slo_stage_fields = lambda req: (
        (100.1, 200.) if req is urgent else (101., 200.))
    blocks = {'victim': [31, 32]}
    if protected_extra:
        blocks['other_protected'] = [41]
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda rid: ([blocks.get(rid, [])]),
        coordinator=SimpleNamespace(single_type_managers=[
            SimpleNamespace(req_to_blocks=blocks)]))
    clock = SimpleNamespace(now=100.)
    events = []
    probe = hold_module.SmallHoldProbe(
        scheduler, emit=lambda name, **kw: events.append(dict(event=name, **kw)),
        clock=lambda: clock.now,
        environ={hold_module.HOLD_ENV: '1',
                 'VLLM_OMNI_STAGE_BUFFER_POLICY': '1',
                 hold_module.EXECUTION_CAP_ENV: '1',
                 hold_module.STAGES_ENV: '0',
                 hold_module.URGENT_KEEP_ENV: '1',
                 hold_module.CONTROLLER_ENV: 'stage_buffer_controller_budget:decide'})
    scheduler._small_hold_probe = probe
    scheduler.request_hold = probe.request_hold
    c = controller()
    c.sequence = 0
    c.record = policy_module.Controller.record.__get__(c)
    c.fields = lambda rid, now, guard: dict(
        coverage_s=.3, high_s=.17, requested_hold_s=.05,
        low_s=.15)
    return scheduler, probe, c, clock, events, blocks, victim, urgent


class UrgentKeepTests(unittest.TestCase):
    def test_urgent_newcomer_enters_freed_execution_slot_and_victim_resumes(self):
        scheduler, probe, c, clock, events, blocks, victim, urgent = urgent_setup()
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertIn('victim', probe.holds)
        self.assertEqual(blocks['victim'], [31, 32])
        with probe.scheduling_context():
            self.assertEqual(scheduler.running, [])
            self.assertEqual(scheduler.max_num_running_reqs, 1)
            self.assertEqual(scheduler.max_num_scheduled_tokens, 64)
            scheduler.waiting.remove(urgent)
            urgent.status = SimpleNamespace(name='RUNNING')
            scheduler.running.append(urgent)
            blocks['urgent'] = [51]
        probe.record_scheduled(SimpleNamespace(num_scheduled_tokens={'urgent': 1}))
        self.assertEqual([e['event'] for e in events].count('urgent_keep_admitted'), 1)
        self.assertIn('victim', probe.holds)
        clock.now = 100.06
        with probe.scheduling_context():
            self.assertEqual(scheduler.running, [urgent])
            self.assertTrue(probe.holds['victim'].released_monotonic_s)
        self.assertIn('victim', probe.holds)
        scheduler.running.remove(urgent)
        scheduler.requests.pop('urgent')
        blocks.pop('urgent')
        with probe.scheduling_context():
            self.assertEqual(scheduler.running, [victim])
        probe.record_scheduled(SimpleNamespace(num_scheduled_tokens={'victim': 1}))
        self.assertNotIn('victim', probe.holds)
        self.assertEqual(blocks['victim'], [31, 32])

    def test_pending_admission_is_recorded_once_and_not_counted_as_success(self):
        scheduler, probe, c, clock, events, _blocks, _victim, _urgent = urgent_setup()
        c._try_urgent_keep(scheduler, clock.now, .05)
        with probe.scheduling_context():
            self.assertEqual(scheduler.running, [])
        empty = SimpleNamespace(num_scheduled_tokens={})
        probe.record_scheduled(empty)
        probe.record_scheduled(empty)
        names = [event['event'] for event in events]
        self.assertEqual(names.count('urgent_keep_admission_pending'), 1)
        self.assertNotIn('urgent_keep_admitted', names)

    def test_full_resident_capacity_rejects_urgent_hold(self):
        scheduler, probe, c, clock, _events, _blocks, _victim, _urgent = (
            urgent_setup(protected_extra=True))
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertFalse(probe.holds)
        self.assertEqual(c.reasons['resident_cap_full'], 1)

    def test_deadline_inversion_and_inventory_are_required(self):
        scheduler, probe, c, clock, _events, _blocks, victim, urgent = urgent_setup()
        scheduler._slo_stage_fields = lambda req: (
            (102., 200.) if req is urgent else (101., 200.))
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertEqual(c.reasons['no_deadline_inversion'], 1)
        scheduler._slo_stage_fields = lambda req: (
            (100.1, 200.) if req is urgent else (101., 200.))
        c.fields = lambda rid, now, guard: dict(
            coverage_s=.01, high_s=.17, requested_hold_s=0., low_s=.15)
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertEqual(c.reasons['no_eligible_victim'], 1)
        self.assertFalse(probe.holds)

    def test_large_deadline_miss_stops_further_urgent_holds(self):
        scheduler, probe, c, clock, events, _blocks, _victim, _urgent = urgent_setup()
        c._try_urgent_keep(scheduler, clock.now, .05)
        clock.now = 100.101
        with probe.scheduling_context():
            self.assertEqual(len(scheduler.running), 1)
        self.assertTrue(probe.buffer_policy_suspended)
        self.assertEqual([e['event'] for e in events].count(
            'stage_buffer_policy_suspended'), 1)

    def test_ordinary_admission_works_before_any_hold(self):
        scheduler, probe, _c, _clock, _events, _blocks, victim, _urgent = urgent_setup()
        scheduler.running.clear()
        scheduler.requests.pop('victim')
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks.clear()
        with probe.scheduling_context():
            self.assertEqual(scheduler.max_num_running_reqs, 1)
            self.assertEqual(scheduler.max_num_scheduled_tokens, 128)

    def test_cancelled_urgent_request_releases_victim_and_clears_link(self):
        scheduler, probe, c, clock, _events, _blocks, _victim, urgent = urgent_setup()
        c._try_urgent_keep(scheduler, clock.now, .05)
        scheduler.requests.pop('urgent')
        with probe.scheduling_context():
            self.assertFalse(probe._urgent_pending)
            self.assertEqual(len(scheduler.running), 1)
        self.assertTrue(probe.holds['victim'].released_monotonic_s)

    def test_wrong_queue_policy_fails_at_initialization(self):
        scheduler, _probe, _c, _clock, _events, _blocks, _victim, _urgent = urgent_setup()
        scheduler._baseline_policy = SimpleNamespace(value='native_fcfs')
        with self.assertRaisesRegex(ValueError, 'stage_deadline_edf_np'):
            hold_module.SmallHoldProbe(scheduler, emit=lambda *a, **k: None,
                environ={hold_module.HOLD_ENV: '1',
                         'VLLM_OMNI_STAGE_BUFFER_POLICY': '1',
                         hold_module.EXECUTION_CAP_ENV: '1',
                         hold_module.STAGES_ENV: '0',
                         hold_module.URGENT_KEEP_ENV: '1',
                         hold_module.CONTROLLER_ENV: 'stage_buffer_controller_budget:decide'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
