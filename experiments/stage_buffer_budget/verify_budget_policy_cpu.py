"""Standalone CPU verification for the isolated successor buffer policy."""
from collections import defaultdict, deque
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
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
        self.max_num_scheduled_tokens = 32768
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
    def test_paced_receiver_reserves_one_inflight_chunk(self):
        c = controller(stage=1)
        inv = c.state['r']
        inv.codec_sent = {0:25,1:25,2:25,3:25}
        inv.codec_received = dict(inv.codec_sent)
        inv.receiver_release_rate = 16.
        inv.receiver_burst_units = 25
        before = c.fields('r',100.,.02)
        self.assertLess(before['coverage_s'],.06)
        c.config['allow_paced_receiver_budget'] = True
        after = c.fields('r',100.,.02)
        self.assertAlmostEqual(after['coverage_s'],75/16)
        self.assertEqual(after['receiver_pacing_burst_reserved_units'],25)
        self.assertEqual(after['raw_consumption_rate_upper_units_s'],before['consumption_rate_units_s'])

    def test_variable_deadline_pacer_cannot_fund_stage1_hold(self):
        c = controller(stage=1)
        c.config['allow_paced_receiver_budget'] = True
        inv = c.state['r']
        inv.codec_sent = {0:25,1:25,2:25,3:25}
        inv.codec_received = dict(inv.codec_sent)
        inv.receiver_release_rate = 16.
        inv.receiver_burst_units = 25
        self.assertGreater(c.fields('r',100.,.02)['coverage_s'],4.)
        inv.update(dict(event='stage_study_receiver_buffer_released',stage_id=2,
            monotonic=100.,fixed_release_bound=False,release_units_per_s=16.))
        self.assertLess(c.fields('r',100.,.02)['coverage_s'],.06)
        self.assertFalse(inv.invalid) # Does not invalidate Stage0 conditioning inventory.

    def test_held_slice_blocks_new_prefill_and_preserves_token_budget(self):
        scheduler, p, clock, events = probe([request('victim'), request('peer')])
        scheduler.waiting.append(request('large_prefill', decode=False))
        self.assertTrue(p.request_hold('victim', .05, 'budget', {}))
        with p.scheduling_context():
            self.assertEqual([r.request_id for r in scheduler.running], ['peer'])
            self.assertEqual(scheduler.max_num_running_reqs, 1)
            self.assertEqual(scheduler.max_num_scheduled_tokens, 32768)
            self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(scheduler.max_num_running_reqs, 64)
        self.assertEqual(scheduler.max_num_scheduled_tokens, 32768)
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
            self.assertEqual(scheduler.max_num_scheduled_tokens, 32768)
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

    def test_restart_inventory_uses_physical_execution_and_attempt(self):
        # The old execution may publish more chunks than its receiver loads.
        # A new execution starts chunk IDs at zero and must not inherit them.
        c = controller()
        c.refresh = policy_module.Controller.refresh.__get__(c)
        with tempfile.TemporaryDirectory() as directory:
            c.root = Path(directory)
            trace_path = c.root / 'traces/stage_events.jsonl'
            trace_path.parent.mkdir()
            events = []
            for physical, attempt, rows in (('job', 0, 220), ('job__execution1', 1, 230)):
                for event in ('stage_study_text_payload_sent', 'stage_study_text_payload_received'):
                    events.append(dict(event=event, logical_request_id='job',
                        physical_request_id=physical, pipeline_attempt_id=attempt,
                        stage_attempt_id=0, monotonic=100.+attempt,
                        chunk_id=0, raw_entity_embedding_layout_valid=True,
                        raw_entity_embedding_rows_total=rows, raw_ids_prompt_count=215))
            trace_path.write_text(''.join(json.dumps(event)+'\n' for event in events))
            c.refresh()
        self.assertEqual(c.fields('job', 102., .05)['receiver_ready_inventory_units'], 5)
        self.assertEqual(c.fields('job__execution1', 102., .05)['receiver_ready_inventory_units'], 15)
        conflict = dict(events[-1], pipeline_attempt_id=2, chunk_id=1)
        c.state['job__execution1'].update(conflict)
        self.assertTrue(c.fields('job__execution1', 102., .05)['invalid_inventory'])
        self.assertIsNone(c.fields('job__execution1', 102., .05)['inventory_units'])

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
    def test_repeat_pause_uses_new_cycle_after_restore(self):
        scheduler,p,c,clock,events,blocks,victim,urgent=urgent_setup()
        c.config['max_holds_per_request']=0
        c._try_urgent_keep(scheduler,clock.now,.05)
        self.assertEqual(c.hold_counts['victim'],1)
        p.holds.clear();p._urgent_pending.clear()
        clock.now=101.
        scheduler._slo_stage_fields=lambda r:(101.1,200.) if r is urgent else (110.,200.)
        c._try_urgent_keep(scheduler,clock.now,.05)
        self.assertEqual(c.hold_counts['victim'],2)
        self.assertEqual(p.holds['victim'].evidence['pause_cycle'],2)

    def test_inventory_lease_preserves_pending_urgent_slot_until_admission(self):
        scheduler,p,c,clock,events,blocks,victim,urgent=urgent_setup()
        c.config.update(inventory_deadline_lease=True,guard_full_urgent_service=True)
        c.fields=lambda *args:dict(coverage_s=2.,high_s=.17,requested_hold_s=.1,low_s=.15,budget_s=1.84)
        c._try_urgent_keep(scheduler,clock.now,.05)
        clock.now=100.2
        # The nominal 100 ms slice elapsed, but the FIFO urgent has not yet
        # entered a synchronous batch; live stock still covers admission.
        with p.scheduling_context():
            self.assertEqual(scheduler.running, [])
        with patch.object(policy_module.time,'monotonic',return_value=clock.now):
            c(scheduler,clock.now)
        self.assertIsNone(p.holds['victim'].released_monotonic_s)
        self.assertFalse(p.buffer_policy_suspended)
        self.assertEqual([e['event'] for e in events].count('small_hold_released'),0)
        with p.scheduling_context():
            scheduler.waiting.remove(urgent)
            urgent.status=SimpleNamespace(name='RUNNING')
            scheduler.running.append(urgent)
            blocks['urgent']=[51]
        p.record_scheduled(SimpleNamespace(num_scheduled_tokens={'urgent':1}))
        self.assertEqual([e['event'] for e in events].count('urgent_keep_admitted'),1)
        clock.now=100.3
        with patch.object(policy_module.time,'monotonic',return_value=clock.now):
            c(scheduler,clock.now)
        release=[e for e in events if e['event']=='small_hold_released']
        self.assertFalse(release) # release is applied at the next scheduler boundary
        with p.scheduling_context():
            self.assertEqual(scheduler.running,[urgent])
        release=[e for e in events if e['event']=='small_hold_released'][0]
        self.assertGreater(release['deadline_lateness_s'],.05)
        self.assertFalse(p.buffer_policy_suspended)

    def test_pending_urgent_lease_ends_when_live_inventory_cannot_cover_service(self):
        scheduler,p,c,clock,events,_blocks,_victim,_urgent=urgent_setup()
        c.config.update(inventory_deadline_lease=True,guard_full_urgent_service=True)
        c.fields=lambda *args:dict(coverage_s=2.,high_s=.17,requested_hold_s=.1,low_s=.15,budget_s=1.84)
        c._try_urgent_keep(scheduler,clock.now,.05)
        clock.now=100.04
        c.fields=lambda *args:dict(coverage_s=.3,high_s=.17,requested_hold_s=.1,low_s=.15,budget_s=.14)
        with patch.object(policy_module.time,'monotonic',return_value=clock.now):
            c(scheduler,clock.now)
        self.assertEqual(c.reasons['resume_requested'],1)
        self.assertEqual([e['event'] for e in events].count('urgent_keep_admitted'),0)
        with p.scheduling_context():
            self.assertEqual(scheduler.running,[scheduler.requests['victim']])
        self.assertFalse(p.buffer_policy_suspended)

    def test_pause_count_tracks_waiters_candidates_and_resident_headroom(self):
        # (execution slots, urgent waiters, safe candidates, resident spare, expected)
        for k,waiters,safe,spare,expected in [(4,3,4,4,3),(4,4,4,4,4),
                (4,4,2,4,2),(4,4,4,1,1),(4,1,4,4,1)]:
            with self.subTest(k=k,waiters=waiters,safe=safe,spare=spare):
                scheduler,p,c,clock,events,blocks,victim,urgent=urgent_setup()
                victims=[victim]+[request('v'+str(i)) for i in range(1,k)]
                urgents=[urgent]+[request('u'+str(i),decode=False) for i in range(1,waiters)]
                for r in urgents:r.status=SimpleNamespace(name='WAITING')
                scheduler.running[:]=victims;scheduler.waiting[:]=urgents
                scheduler.requests={r.request_id:r for r in victims+urgents}
                blocks.clear();blocks.update({r.request_id:[30+i] for i,r in enumerate(victims)})
                scheduler.max_num_running_reqs=p.resident_cap=k+spare;p.execution_cap=k
                scheduler._baseline_policy=SimpleNamespace(value='native_fcfs')
                scheduler._slo_stage_fields=lambda r:(100.1,200.) if r in urgents else (110.,200.)
                eligible={r.request_id for r in victims[:safe]}
                c.fields=lambda rid,*args:dict(coverage_s=.5 if rid in eligible else 0.,
                    high_s=.17,requested_hold_s=.05,low_s=.15)
                c.config['max_concurrent_holds']=k
                c._try_urgent_keep(scheduler,clock.now,.05)
                self.assertEqual(len(p.holds),expected)
                self.assertEqual(set(p._urgent_pending.values()),{r.request_id for r in urgents[:expected]})
                self.assertEqual(scheduler.waiting,urgents)
                with p.scheduling_context():
                    self.assertEqual(len(scheduler.running),k-expected)
                    self.assertEqual(scheduler.max_num_running_reqs,k)

    def test_multiple_victims_admit_fifo_waiters_and_resume_without_kv_loss(self):
        scheduler, p, c, clock, events, blocks, victim, urgent = urgent_setup()
        victim2=request('victim2'); urgent2=request('urgent2', decode=False)
        urgent2.status=SimpleNamespace(name='WAITING')
        scheduler.running.append(victim2); scheduler.waiting.append(urgent2)
        scheduler.requests.update(victim2=victim2,urgent2=urgent2)
        blocks['victim2']=[61,62]
        scheduler.max_num_running_reqs=p.resident_cap=4; p.execution_cap=2
        scheduler._baseline_policy=SimpleNamespace(value='native_fcfs')
        scheduler._slo_stage_fields=lambda r: (100.1,200.) if r in (urgent,urgent2) else (110.,200.)
        c.config['max_concurrent_holds']=2
        c._try_urgent_keep(scheduler,clock.now,.05)
        self.assertEqual(len(p.holds),2)
        self.assertEqual(set(p._urgent_pending.values()),{'urgent','urgent2'})
        self.assertEqual(scheduler.waiting,[urgent,urgent2])
        with p.scheduling_context():
            self.assertEqual(scheduler.max_num_running_reqs,2)
            self.assertEqual(scheduler.running,[])
            for r in (urgent,urgent2):
                scheduler.waiting.remove(r);r.status=SimpleNamespace(name='RUNNING')
                scheduler.running.append(r);blocks[r.request_id]=[71]
        p.record_scheduled(SimpleNamespace(num_scheduled_tokens={'urgent':1,'urgent2':1}))
        self.assertEqual(sum(e['event']=='urgent_keep_admitted' for e in events),2)
        clock.now=100.06
        for r in (urgent,urgent2):
            scheduler.running.remove(r);scheduler.requests.pop(r.request_id);blocks.pop(r.request_id)
        with p.scheduling_context():
            self.assertEqual({r.request_id for r in scheduler.running},{'victim','victim2'})
        p.record_scheduled(SimpleNamespace(num_scheduled_tokens={'victim':1,'victim2':1}))
        self.assertFalse(p.holds)
        self.assertEqual(blocks['victim'],[31,32]);self.assertEqual(blocks['victim2'],[61,62])

    def test_multi_hold_reserves_resident_capacity_before_allocation(self):
        scheduler,p,c,clock,events,blocks,victim,urgent=urgent_setup()
        victim2=request('victim2');urgent2=request('urgent2',decode=False)
        urgent2.status=SimpleNamespace(name='WAITING')
        scheduler.running.append(victim2);scheduler.waiting.append(urgent2)
        scheduler.requests.update(victim2=victim2,urgent2=urgent2);blocks['victim2']=[61]
        scheduler.max_num_running_reqs=p.resident_cap=3;p.execution_cap=2
        scheduler._baseline_policy=SimpleNamespace(value='native_fcfs')
        scheduler._slo_stage_fields=lambda r:(100.1,200.) if r in (urgent,urgent2) else (110.,200.)
        c.config['max_concurrent_holds']=2
        c._try_urgent_keep(scheduler,clock.now,.05)
        self.assertEqual(len(p.holds),1)
        self.assertEqual(c.reasons['resident_cap_full'],1)

    def test_full_urgent_residency_is_part_of_coverage_budget(self):
        scheduler, probe, c, clock, events, blocks, victim, urgent = urgent_setup()
        c.config['guard_full_urgent_service'] = True
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertFalse(probe.holds)  # .3 s stock cannot cover .2 + .15 + margin.
        c.fields = lambda *args: dict(coverage_s=.5, high_s=.17,
                                     requested_hold_s=.1, low_s=.15)
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertIn('victim', probe.holds)

    def test_native_order_remains_native_with_urgent_hold(self):
        scheduler, probe, c, clock, events, blocks, victim, urgent = urgent_setup()
        scheduler._baseline_policy = SimpleNamespace(value='native_fcfs')
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertIn('victim', probe.holds)
        self.assertEqual(scheduler._baseline_policy.value, 'native_fcfs')

    def test_urgent_newcomer_enters_freed_execution_slot_and_victim_resumes(self):
        scheduler, probe, c, clock, events, blocks, victim, urgent = urgent_setup()
        c._try_urgent_keep(scheduler, clock.now, .05)
        self.assertIn('victim', probe.holds)
        self.assertEqual(blocks['victim'], [31, 32])
        with probe.scheduling_context():
            self.assertEqual(scheduler.running, [])
            self.assertEqual(scheduler.max_num_running_reqs, 1)
            self.assertEqual(scheduler.max_num_scheduled_tokens, 32768)
            scheduler.waiting.remove(urgent)
            urgent.status = SimpleNamespace(name='RUNNING')
            scheduler.running.append(urgent)
            blocks['urgent'] = [51]
        probe.record_scheduled(SimpleNamespace(num_scheduled_tokens={'urgent': 1}))
        admitted = [e for e in events if e['event'] == 'urgent_keep_admitted']
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['victim_kv_block_ids'], [[31, 32]])
        self.assertEqual(admitted[0]['urgent_kv_block_ids'], [[51]])
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

    def test_no_eligible_victim_reports_guard_reason(self):
        scheduler, _probe, c, clock, _events, _blocks, _victim, _urgent = urgent_setup()
        records = []
        c.record = lambda rid, now, reason, fields, **extra: records.append((reason, extra))
        c.fields = lambda rid, now, guard: dict(
            coverage_s=0., high_s=.17, requested_hold_s=0., low_s=.15,
            inventory_units=12, receiver_ready_inventory_units=0)
        c._try_urgent_keep(scheduler, clock.now, .05)
        reason, detail = records[-1]
        self.assertEqual(reason, 'no_eligible_victim')
        self.assertEqual(detail['rejection_counts'],
                         {'insufficient_downstream_coverage': 1})
        self.assertEqual(detail['rejection_example']['receiver_ready_inventory_units'], 0)

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
            self.assertEqual(scheduler.max_num_scheduled_tokens, 32768)

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
        scheduler._baseline_policy = SimpleNamespace(value='srpf_local_np')
        with self.assertRaisesRegex(ValueError, 'explicit native or stage EDF'):
            hold_module.SmallHoldProbe(scheduler, emit=lambda *a, **k: None,
                environ={hold_module.HOLD_ENV: '1',
                         'VLLM_OMNI_STAGE_BUFFER_POLICY': '1',
                         hold_module.EXECUTION_CAP_ENV: '1',
                         hold_module.STAGES_ENV: '0',
                         hold_module.URGENT_KEEP_ENV: '1',
                         hold_module.CONTROLLER_ENV: 'stage_buffer_controller_budget:decide'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
