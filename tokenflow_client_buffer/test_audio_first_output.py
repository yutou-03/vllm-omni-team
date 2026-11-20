"""Talker uses media-second credit for first-output refill and safe returns."""
import unittest
from . import test_scheduler_cadence as fixtures
from .scheduler import ClientBufferScheduler


class AudioFirstOutputTests(unittest.TestCase):
    def setUp(self):
        fixtures.PolicyCadenceTests.setUp(self)
        self.s._omni_stage_id_for_trace=lambda:1
        self.config.update(modality='audio',target_stage=1,first_output_refill=True,
                           resume_to_client_s=2.,margin_s=.5,reserve_units=.02)
        self.adapter=ClientBufferScheduler(self.s,self.config,
            lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)

    def add(self,rid,**kw):
        r=fixtures.PolicyCadenceTests.add(self,rid,**kw)
        r.client_buffer_test_metadata.update(sched_request_path='audio',sched_audio_consume_rate=1.)
        r.client_buffer_test_metadata.pop('sched_text_consume_rate_tps',None)
        return r

    def credit(self,rid,coverage):
        fixtures.PolicyCadenceTests.credit(self,rid,coverage)
        self.feedback[rid]['modality']='audio'

    def write(self,**updates):
        fixtures.PolicyCadenceTests.write(self,modality='audio',**updates)

    pause=fixtures.PolicyCadenceTests.pause
    step=fixtures.PolicyCadenceTests.step

    def test_paper_running_before_pcm_gets_safe_compute_yield(self):
        self.add('donor', running=True, coverage=12., executed=True)
        self.add('receiver', running=True, executed=True)
        self.add('protected', running=True, coverage=.2, executed=True)
        self.adapter.active_ids=set(self.members)
        self.quota=True
        self.write(recovery_ranking='paper', registered_at={'receiver':5.})
        self.assertEqual(self.step(write=False)[0], {'receiver','protected'})
        self.assertFalse(self.adapter.pauses['donor'].releases_quota)
        self.assertEqual(self.adapter.pauses['donor'].beneficiary,'receiver')
        self.assertTrue(any(e=='client_buffer_priority_exchange' and d['action']=='first_pcm'
                            for e,d in self.events))
        self.now+=.5
        self.credit('receiver',.1)
        self.credit('donor',11.5)
        self.credit('protected',.2)
        self.write(recovery_ranking='paper')
        self.assertEqual(self.step(write=False)[0], set(self.members))
        self.assertFalse(self.adapter.pauses)

    def test_reclaimed_work_cannot_reuse_unacknowledged_capacity(self):
        self.add('old',running=True,coverage=.1,executed=True)
        for name in ('one','two','three'):
            self.add(name); self.members.pop(name)
        self.adapter.active_ids={'old'}
        new=['one','two','three']
        def snapshot(gamma):
            self.write(gamma=gamma,native_fcfs_admission=False,buffer_reclaimed_ids=new,
                       registered_ids=list(self.s.requests),recovery_ranking='paper',
                       startup_members={k:dict(rate=1.) for k in self.s.requests})
        snapshot(1.1)
        self.assertEqual(self.step(write=False)[0],{'old'})
        self.now+=.1
        snapshot(1.1)
        self.assertEqual(self.step(write=False)[0],{'old'})
        snapshot(2.1)
        self.assertEqual(self.step(write=False)[0],{'old','one'})
        self.now+=.1
        self.assertEqual(self.step(write=False)[0],{'old','one'})
        self.members.pop('old');self.s.running.remove(self.s.requests.pop('old'))
        self.members['one']=dict(rate=1.);new.remove('one')
        snapshot(2.1)
        self.assertEqual(self.step(write=False)[0],{'one','two'})

    def test_vacancy_starts_audio_before_next_policy_tick(self):
        self.add('owner',running=True,coverage=.1,executed=True)
        self.adapter.active_ids={'owner'};self.step()
        self.now=10.1;self.add('new')
        self.assertEqual(self.step()[0],{'owner','new'})
        self.assertFalse(self.adapter.pauses)

    def test_media_credit_funds_audio_first_output_at_full_batch(self):
        self.add('safe',running=True,coverage=20.,executed=True)
        for i in range(3):self.add(f'protected-{i}',running=True,coverage=.1,executed=True)
        self.add('new');self.adapter.active_ids={'safe','protected-0','protected-1','protected-2'}
        self.assertEqual(self.step()[0],{'new','protected-0','protected-1','protected-2'})
        self.assertIn('safe',self.adapter.pauses)
        self.assertIsNone(self.adapter.latched_fault)

    def test_reclaimed_audio_queue_can_use_safe_donor_before_it_drains(self):
        self.add('safe',running=True,coverage=20.,executed=True)
        for i in range(3):
            self.add(f'protected-{i}',running=True,coverage=.1,executed=True)
        for rid in ('new','later'):
            self.add(rid); self.members.pop(rid)
        self.adapter.active_ids=set(self.members)
        self.write(native_fcfs_admission=False,buffer_reclaimed_ids=['new','later'],
                   registered_ids=list(self.s.requests),
                   startup_members={rid:dict(rate=1.) for rid in self.s.requests},
                   startup_feedback=self.feedback)
        selected, _ = self.step(write=False)
        self.assertEqual(selected,{'new','protected-0','protected-1','protected-2'})
        self.assertIn('safe',self.adapter.pauses)
        self.assertEqual(self.adapter.pauses['safe'].beneficiary,'new')
        self.assertEqual([r.request_id for r in self.s.waiting],['later'])
        self.assertNotIn('new',self.members)
        self.assertIsNone(self.adapter.latched_fault)

    def test_due_audio_return_and_newcomer_share_independent_safe_swaps(self):
        self.add('due',running=True,coverage=3.,executed=True);self.pause('due',3.)
        for rid in ('safe1','safe2'):self.add(rid,running=True,coverage=20.,executed=True)
        for rid in ('protected1','protected2'):self.add(rid,running=True,coverage=.1,executed=True)
        self.add('new');self.adapter.active_ids={'safe1','safe2','protected1','protected2'}
        self.assertEqual(self.step()[0],{'due','new','protected1','protected2'})
        self.assertEqual(set(self.adapter.pauses),{'safe1','safe2'})

    def test_short_audio_credit_does_not_permit_preemption(self):
        for i in range(4):self.add(f'owner-{i}',running=True,coverage=.1,executed=True)
        self.add('new');self.adapter.active_ids={f'owner-{i}' for i in range(4)}
        self.assertEqual(self.step()[0],{f'owner-{i}' for i in range(4)})
        self.assertFalse(self.adapter.pauses)

    def test_audio_quota_ack_reserves_one_slot_only(self):
        self.add('due',coverage=3.,executed=True);self.pause('due',3.,quota=True)
        self.add('owner',running=True,coverage=.1,executed=True)
        for i in range(3):self.add(f'new-{i}')
        self.adapter.active_ids={'owner'}
        self.assertEqual(self.step()[0],{'owner','new-0','new-1'})
        self.members['due']=self.paused.pop('due');self.acks['due']['state']='active'
        self.now=10.02;self.credit('due',2.98)
        self.assertEqual(self.step()[0],{'owner','due','new-0','new-1'})

    def test_safe_audio_yields_below_native_cap_then_resumes_after_first_output(self):
        self.add('donor', running=True, coverage=20., executed=True)
        self.add('new');self.adapter.active_ids = {'donor'}
        self.assertEqual(self.step()[0], {'new'})
        self.assertIn('donor', self.adapter.pauses)
        self.assertEqual(self.adapter.pauses['donor'].beneficiary, 'new')
        self.now = 10.5;self.credit('donor', 19.5);self.credit('new', .1)
        self.assertEqual(self.step()[0], {'donor', 'new'})
        self.assertFalse(self.adapter.pauses)
        self.assertTrue(any(e == 'client_buffer_resumed' for e, _ in self.events))

    def test_no_second_donor_for_same_unstarted_audio_beneficiary(self):
        for rid in ('a','b'):self.add(rid, running=True, coverage=20., executed=True)
        self.add('new');self.adapter.active_ids = {'a','b'}
        self.assertEqual(self.step()[0], {'b','new'})
        self.now = 10.5;self.credit('a', 19.5);self.credit('b', 20.)
        self.assertEqual(self.step()[0], {'b','new'})
        self.assertEqual(set(self.adapter.pauses), {'a'})

    def test_safety_blocked_ingress_can_request_specific_donor_below_cap(self):
        for rid in ('a','b'):self.add(rid, running=True, coverage=20., executed=True)
        self.add('new');self.members.pop('new')
        self.quota = True;self.adapter.active_ids = {'a','b'}
        self.write(registered_ids=['a','b','new'], admission_waiter=dict(request_id='new',
            resource_deficit={'audio':0.}, eligible_pause_sources=['b'],
            startup_latency_estimate_s=5.5), pause_release_cost_per_rate={'audio':1.})
        selected, _ = self.step(write=False)
        self.assertEqual(selected, {'a'})
        self.assertIn('b', self.adapter.pauses)
        self.assertNotIn('new', selected)  # gateway ACK is still required

    def test_full_rate_budget_uses_native_fcfs_without_safe_donor(self):
        for rid in ('a','b'):self.add(rid, running=True, coverage=.1, executed=True)
        original = self.s.running
        self.write(gamma=2.)
        with self.adapter.scheduling_context():
            self.assertIs(self.s.running, original)
            self.assertEqual([r.request_id for r in self.s.running], ['a','b'])
        modes = [d for e, d in self.events if e == 'client_buffer_mode']
        self.assertEqual(modes[-1]['mode'], 'fcfs')
        self.assertEqual(modes[-1]['reason'], 'capacity_full')
        self.assertFalse(self.adapter.pauses)

    def test_full_rate_budget_still_checks_safe_exchange_below_native_limit(self):
        self.add('donor', running=True, coverage=20., executed=True)
        self.add('new');self.members.pop('new')
        self.quota = True;self.adapter.active_ids = {'donor'}
        self.write(gamma=1., registered_ids=['donor','new'], admission_waiter=dict(request_id='new',
            resource_deficit={'audio':1.}, startup_latency_estimate_s=5.5),
            pause_release_cost_per_rate={'audio':1.})
        selected, _ = self.step(write=False)
        self.assertIn('donor', self.adapter.pauses)
        self.assertNotIn('new', selected)  # safe exchange still waits for quota ACK
