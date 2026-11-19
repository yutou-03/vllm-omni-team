"""Policy ticks coalesce prefill; native execution and emergency returns continue."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from .pause import PausedRequest

from .scheduler import ClientBufferScheduler
from .test_core import request
from .stream import atomic_json


class PolicyCadenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'snapshot.json'
        self.now = 10.
        self.s = NS(running=[],waiting=[],skipped_waiting=[],requests={},
                    max_num_running_reqs=4,kv_cache_manager=NS(),_omni_stage_id_for_trace=lambda:0)
        self.events=[]
        self.config=dict(snapshot_path=str(self.path),resume_to_client_s=.5,session='s',
                         modality='text',policy_interval_s=.5)
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        self.members={};self.paused={};self.feedback={};self.acks={};self.quota=False

    def add(self,rid,*,running=False,path='text',coverage=None,executed=False):
        r=request(rid,len(self.s.requests));r.client_buffer_test_metadata['sched_request_path']=path
        r.status.name='RUNNING' if running else 'WAITING'
        self.s.requests[rid]=r
        (self.s.running if running else self.s.waiting).append(r)
        if path=='text':self.members[rid]=dict(rate=1.)
        if coverage is not None:self.credit(rid,coverage)
        if executed:self.adapter.executed_requests[rid]=r
        return r

    def credit(self,rid,coverage):
        self.feedback[rid]=dict(request_id=rid,session='s',modality='text',rate=1.,
            received=coverage,consumed=0.,observed_at=self.now,sequence=int(self.now*1000),
            started=True,complete=False)

    def write(self,**updates):
        state=dict(schema_version=1,session='s',modality='text',written_at=self.now,
            mode='buffer',demand=float(len(self.members)),gamma=20.,capacity_valid_until=100.,
            members=self.members,paused_members=self.paused,feedback=self.feedback,
            engine_queue_admission=True,registered_ids=list(set(self.members)|set(self.paused)))
        if self.quota:state.update(pause_admission_protocol=1,pause_acknowledgements=self.acks)
        state.update(updates);atomic_json(self.path, state)

    def pause(self,rid,coverage,*,quota=False):
        r=self.s.requests[rid];self.credit(rid,coverage)
        self.adapter.pauses[rid] = PausedRequest(9., float("inf"), .8)
        self.adapter.pauses[rid].resume_by=self.now+coverage-.8
        self.adapter.pauses[rid].resume_guard_s=.8
        self.adapter.executed_requests[rid]=r
        if quota:
            self.quota=True;self.paused[rid]=self.members.pop(rid)
            self.adapter.pauses[rid].releases_quota = True;self.acks[rid]=dict(epoch=9.,state='paused')

    def step(self,*,write=True):
        if write:self.write()
        with self.adapter.scheduling_context():
            visible_waiters=[r.request_id for r in self.s.skipped_waiting+self.s.waiting]
            while (self.s.skipped_waiting or self.s.waiting) and len(self.s.running)<self.s.max_num_running_reqs:
                q=self.s.skipped_waiting or self.s.waiting
                r=q.pop(0);r.status.name='RUNNING';self.s.running.append(r)
            selected={r.request_id:1 for r in self.s.running}
        self.adapter.record_scheduled(NS(num_scheduled_tokens=selected))
        return set(selected),visible_waiters

    def test_initial_cohort_immediate_and_later_new_prefills_share_tick(self):
        self.add('a');self.add('b')
        self.assertEqual(self.step()[0],{'a','b'})
        self.now=10.1;self.add('c');self.assertEqual(self.step()[0],{'a','b'})
        self.now=10.3;self.add('d');self.assertEqual(self.step()[0],{'a','b'})
        self.now=10.5;self.assertEqual(self.step()[0],{'a','b','c','d'})
        self.assertEqual(self.adapter.decision_counts['periodic_policy_decisions'],2)

    def test_audio_through_text_stage_does_not_wait_for_text_tick(self):
        self.add('text');self.step()
        self.now=10.1;self.add('audio',path='audio');self.add('new_text')
        selected,visible=self.step()
        self.assertIn('audio',selected);self.assertNotIn('new_text',visible)

    def test_native_resume_of_previously_executed_request_is_not_new_prefill(self):
        r=self.add('a');self.step()
        self.s.running.remove(r);r.status.name='PREEMPTED';self.s.waiting.append(r)
        self.now=10.1;self.assertIn('a',self.step()[0])

    def test_one_tick_can_fund_multiple_admitted_new_prefills(self):
        for rid in 'abcd':self.add(rid,running=True,coverage=10.,executed=True)
        for rid in 'ef':self.add(rid)
        self.adapter.active_ids=set('abcd')
        selected,_=self.step()
        self.assertEqual(selected,set('cdef'))
        self.assertEqual(set(self.adapter.pauses),set('ab'))
        self.assertEqual(len([e for e,_ in self.events if e=='client_buffer_pause']),2)
        self.now=10.1
        self.assertEqual(self.step()[0],set('cdef'))
        self.assertEqual(self.adapter.decision_counts['cadence_cached_steps'],1)

    def test_conservative_buffer_factor_changes_donor_and_return_deadline(self):
        for rid in 'abcd':
            self.add(rid, running=True, coverage=2.0, executed=True)
        self.add('new')
        self.adapter.active_ids = set('abcd')
        self.write(buffer_conservativeness=2.0)
        self.assertNotIn('new', self.step(write=False)[0])
        self.assertFalse(self.adapter.pauses)
        self.now += .5
        self.credit('a', 5.)
        self.write(buffer_conservativeness=2.0)
        self.assertIn('new', self.step(write=False)[0])
        pause = self.adapter.pauses['a']
        guard = 2 * (self.adapter.resume_s + self.adapter.margin_s + .5)
        self.assertAlmostEqual(pause.resume_guard_s, guard)
        self.assertAlmostEqual(pause.resume_by, self.now + 5. - guard)

    def test_invalid_buffer_factor_rejected_before_scheduling(self):
        for factor in (.5, 'nan', True):
            with self.subTest(factor=factor):
                self.write(buffer_conservativeness=factor)
                self.assertEqual(self.adapter.read_snapshot(self.now)[1], 'snapshot_unavailable')

    def test_default_buffer_factor_accepts_same_two_second_donor(self):
        for rid in 'abcd':
            self.add(rid, running=True, coverage=2.0, executed=True)
        self.add('new')
        self.adapter.active_ids = set('abcd')
        self.assertIn('new', self.step()[0])
        self.assertTrue(self.adapter.pauses)

    def test_periodic_tick_batches_returns_before_next_safety_window(self):
        self.s.max_num_running_reqs=2
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        for rid in 'abcd':self.add(rid,running=True,coverage=10.,executed=True)
        self.pause('a',1.6);self.pause('b',1.7)
        self.adapter.active_ids=set('cd')
        selected,_=self.step()
        self.assertEqual(selected,set('ab'))
        self.assertEqual(set(self.adapter.pauses),set('cd'))
        self.assertTrue(all(p['trigger']=='resume_due' for e,p in self.events if e=='client_buffer_pause'))

    def test_emergency_return_does_not_wait_for_next_tick(self):
        self.s.max_num_running_reqs=1
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        self.add('a',running=True,coverage=1.,executed=True)
        self.add('b',running=True,coverage=10.,executed=True)
        self.pause('a',1.);self.adapter.active_ids={'b'}
        self.adapter.last_mode='buffer';self.adapter.next_policy_at=10.5
        self.assertEqual(self.step()[0],{'a'})
        self.assertEqual(self.adapter.decision_counts['urgent_policy_decisions'],1)
        self.assertLess(self.now,self.adapter.next_policy_at)

    def test_ack_and_cross_queue_recovery_priority_bypass_tick(self):
        self.s.max_num_running_reqs=2
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        a=self.add('a',coverage=1.,executed=True);a.status.name='PREEMPTED'
        self.add('b',running=True,coverage=.1,executed=True)
        c=self.add('c',executed=True);self.s.waiting.remove(c);self.s.skipped_waiting.append(c)
        self.pause('a',1.,quota=True);self.adapter.active_ids={'b'}
        self.adapter.last_mode='buffer';self.adapter.next_policy_at=10.5
        self.assertEqual(self.step()[0],{'b'})
        self.now=10.02;self.credit('a',.98)
        self.assertEqual(self.step()[0],{'b'})
        self.now=10.04;self.members['a']=self.paused.pop('a');self.acks['a']['state']='active';self.credit('a',.96)
        selected,visible=self.step()
        self.assertEqual(selected,{'a','b'});self.assertNotIn('c',visible)
        self.assertLess(self.now,self.adapter.next_policy_at)

    def test_deadline_first_waiter_beats_newcomer_in_skipped_queue(self):
        self.s.max_num_running_reqs = 2
        self.config.update(first_output_refill=True, deadline_first_recovery=True)
        self.adapter = ClientBufferScheduler(self.s, self.config,
            lambda event, **kw: self.events.append((event, kw)), clock=lambda: self.now)
        a = self.add('return', coverage=.8, executed=True)
        a.status.name = 'PREEMPTED'
        self.pause('return', .8)
        self.add('owner', running=True, coverage=10., executed=True)
        self.adapter.active_ids = {'owner'}
        new = self.add('new')
        self.s.waiting.remove(new)
        self.s.skipped_waiting.append(new)
        selected, _ = self.step()
        self.assertEqual(selected, {'return', 'owner'})
        self.assertFalse(self.adapter.pauses)
        self.s.running.remove(self.s.requests.pop('owner'))
        self.now += .05
        self.assertIn('new', self.step()[0])

    def weighted_conflict(self, credit):
        self.config.update(first_output_refill=True, deadline_first_recovery=True)
        self.s.max_num_running_reqs = 1
        self.s.max_num_scheduled_tokens = 32
        self.adapter = ClientBufferScheduler(self.s, self.config,
            lambda event, **kw: self.events.append((event, kw)), clock=lambda:self.now)
        old = self.add('old', running=True, coverage=credit, executed=True)
        self.pause('old', credit)
        new = self.add('new')
        new.num_computed_tokens = new.num_output_tokens = 0
        self.adapter.active_ids = set()
        self.write(recovery_ranking='weighted')
        return self.step(write=False)[0]

    def test_weighted_can_trade_positive_stall_for_first_output(self):
        self.assertEqual(self.weighted_conflict(.725), {'new'})
        rank = next(d for event,d in self.events if event=='client_buffer_recovery_rank')
        self.assertGreater(rank['incremental_stall_s'], 0)
        self.assertGreater(rank['startup_gain_s'], rank['incremental_stall_s'])
        self.assertTrue(self.adapter.pauses['old'].ranked_yield)
        self.assertTrue(any(event=='client_buffer_recovery_rank_result' and d['new_executed']
                            for event,d in self.events))

    def test_weighted_tie_protects_starved_old_flow(self):
        self.assertEqual(self.weighted_conflict(.1), {'old'})
        self.assertTrue(any(event=='client_buffer_recovery_rank_result' and d['old_executed']
                            for event,d in self.events))

    def test_unchanged_snapshot_and_feedback_are_not_reparsed_each_decode(self):
        self.add('a',running=True,coverage=10.,executed=True);self.adapter.active_ids={'a'}
        self.step()
        first=dict(self.adapter.decision_counts)
        for now in (10.05,10.1,10.15):
            self.now=now;self.assertEqual(self.step(write=False)[0],{'a'})
        self.assertEqual(self.adapter.decision_counts['snapshot_parses'],first['snapshot_parses'])
        self.assertEqual(self.adapter.decision_counts['feedback_parses'],first['feedback_parses'])
        self.assertEqual(self.adapter.decision_counts['cadence_cached_steps'],3)

    def test_cached_tick_keeps_newer_owner_over_older_nonheld_resident(self):
        self.s.max_num_running_reqs=1
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        self.add('older',running=True,executed=True)
        self.add('owner',running=True,executed=True)
        self.adapter.active_ids={'owner'}
        self.adapter.last_mode='buffer';self.adapter.next_policy_at=10.5
        self.assertEqual(self.step()[0],{'owner'})
        self.assertFalse(self.adapter.pauses)
        self.assertEqual(self.adapter.decision_counts['cadence_cached_steps'],1)

    def test_cached_snapshot_rechecks_expiry_and_session_on_every_read(self):
        self.add('a',running=True,executed=True);self.write()
        self.assertIsNone(self.adapter.read_snapshot(10.)[1])
        self.assertEqual(self.adapter.decision_counts['snapshot_parses'],1)
        self.assertEqual(self.adapter.read_snapshot(10.6)[1],'snapshot_invalid')
        self.assertEqual(self.adapter.decision_counts['snapshot_parses'],1)
        self.adapter.config['session']='other-session'
        self.assertEqual(self.adapter.read_snapshot(10.1)[1],'snapshot_invalid')
        self.assertEqual(self.adapter.decision_counts['snapshot_parses'],1)

    def test_native_fcfs_never_coalesces_new_requests(self):
        self.add('a');self.write(mode='fcfs',reason='baseline',engine_queue_admission=False)
        self.assertEqual(self.step(write=False)[0],{'a'})
        self.now=10.1;self.add('b');self.write(mode='fcfs',reason='baseline',engine_queue_admission=False)
        self.assertEqual(self.step(write=False)[0],{'a','b'})


if __name__=='__main__':unittest.main()
