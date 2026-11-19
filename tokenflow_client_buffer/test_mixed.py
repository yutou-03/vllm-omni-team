import unittest
from .core import CapacityEstimate
from .mixed_core import MixedBook


def estimate(gamma,modality,admission=None):
    return CapacityEstimate(gamma,0.,100.,modality,'calibration','test',admission)


class MixedAdmissionTests(unittest.TestCase):
    def book(self):
        return MixedBook(estimate(100.,'text',85.),estimate(20.,'audio',17.),4.)

    def test_trace_transport_budget_does_not_change_execution_or_capacity(self):
        b = MixedBook(estimate(1000., 'text'), estimate(100., 'audio'), 4.,
                      native_limit=64, max_inflight=6263)
        t = b.views['text']
        for i in range(300):
            t.add(str(i), 1.)
        self.assertEqual(len(t.requests), 300)
        self.assertEqual(t.max_inflight, 6263)
        self.assertEqual(b.native_limit, 64)
        self.assertEqual(b.demand, dict(thinker=0., audio=0.))
        self.assertEqual(t.admit_next(b.capacity('text', 1.), 1.), '0')
        self.assertEqual(b.demand['thinker'], 1.)

    def test_units_and_joint_admission_are_not_independent_budgets(self):
        b=self.book();t,a=b.views['text'],b.views['audio']
        t.add('t',40.);self.assertEqual(t.admit_next(b.capacity('text',1.),1.),'t')
        a.add('a',8.);self.assertEqual(a.admit_next(b.capacity('audio',1.),1.),'a')
        self.assertEqual(b.demand,dict(thinker=72.,audio=8.))
        t.add('later',20.)
        self.assertIsNone(t.admit_next(b.capacity('text',1.),1.))
        self.assertEqual(b.demand['thinker'],72.)

    def test_overload_of_audio_resource_forces_both_views_to_fcfs(self):
        b=self.book();t,a=b.views['text'],b.views['audio']
        a.add('a',8.);a.admit_next(b.capacity('audio',1.),1.)
        b.estimates['audio']=estimate(7.,'audio')
        self.assertEqual(t.update(b.capacity('text',2.),2.),'fcfs')
        self.assertEqual(a.update(b.capacity('audio',2.),2.),'fcfs')
        t.add('t',1.)
        self.assertEqual(t.admit_next(b.capacity('text',2.),2.),'t')
        self.assertEqual(t.requests['t'].state, 'native_queued')
        self.assertEqual(b.reason(2.), 'overload')

    def test_pausing_preserves_demand_and_feasible_heads_keep_arrival_order(self):
        b=self.book();t,a=b.views['text'],b.views['audio']
        t.add('first',10.);a.add('second',1.)
        self.assertIsNone(a.admit_next(b.capacity('audio',1.),1.))
        t.admit_next(b.capacity('text',1.),1.)
        a.admit_next(b.capacity('audio',1.),1.)
        self.assertEqual(b.demand['thinker'],14.)
        t.complete('first')
        self.assertEqual(b.demand['thinker'],4.)

    def test_audio_rate_limit_switches_to_native_fifo_without_blocking_thinker(self):
        b=MixedBook(estimate(100.,'text'),estimate(2.,'audio'),4.,admission_policy='resource_fifo')
        t,a=b.views['text'],b.views['audio']
        for rid in ('a0','a1'):
            a.add(rid,1.);self.assertEqual(a.admit_next(b.capacity('audio',1.),1.),rid)
        a.add('head',1.);t.add('text',85.)
        self.assertIsNone(t.admit_next(b.capacity('text',1.),1.))
        self.assertTrue(b.native_fcfs)
        self.assertEqual(a.admit_next(b.capacity('audio',1.),1.),'head')
        self.assertEqual(a.requests['head'].state,'native_queued')
        self.assertEqual(t.admit_next(b.capacity('text',1.),1.),'text')
        self.assertEqual(t.requests['text'].state,'admitted')
        # The handed-off audio stays uncharged; feasible text resumes buffer admission.
        self.assertEqual(b.demand,dict(thinker=93.,audio=2.))
        self.assertEqual([r.request_id for r in b.native_queued],['head'])

    def test_resource_fifo_never_bypasses_overload_unknown_or_native_fcfs(self):
        for condition in ('overload','unknown','native'):
            with self.subTest(condition=condition):
                b=MixedBook(estimate(100.,'text'),estimate(1.,'audio'),4.,admission_policy='resource_fifo')
                t,a=b.views['text'],b.views['audio'];a.add('a',1.);a.admit_next(b.capacity('audio',1.),1.)
                a.add('head',1.);t.add('t',10.)
                if condition=='overload':b.estimates['audio']=estimate(.5,'audio')
                if condition=='unknown':b.estimates['audio']=None
                self.assertIsNone(t.admit_next(b.capacity('text',2.),2.,native=condition=='native'))
                self.assertEqual(len(b.members),1)

    def test_resource_fifo_preserves_head_blocked_by_shared_resource(self):
        b=MixedBook(estimate(100.,'text'),estimate(20.,'audio'),4.,admission_policy='resource_fifo')
        t,a=b.views['text'],b.views['audio'];t.add('running',10.);t.admit_next(b.capacity('text',1.),1.)
        t.add('head',95.);a.add('cheap',1.)
        self.assertIsNone(a.admit_next(b.capacity('audio',1.),1.))
        self.assertEqual(b.demand['thinker'],10.)

    def test_paused_service_obligation_is_not_erased_by_native_queue_handoff(self):
        b=self.book();t=b.views['text']
        t.add('paused',10.);t.admit_next(b.capacity('text',1.),1.)
        self.assertTrue(t.pause('paused'))
        t.add('waiting',10.);b.estimates['thinker']=None
        self.assertEqual(t.admit_next(None,2.),'waiting')
        self.assertEqual(t.requests['waiting'].state,'native_queued')
        self.assertEqual(t.requests['paused'].state,'paused')
        self.assertEqual(b.service_demand()['thinker'],10.)
        self.assertEqual(b.demand['thinker'],0.)
        self.assertTrue(b.native_fcfs)

    def test_unknown_capacity_delegates_queue_to_native_not_a_second_sequence_limit(self):
        b=MixedBook(None,None,4.,native_limit=2)
        t,a=b.views['text'],b.views['audio']
        t.add('t',10.);a.add('a',1.);t.add('later',10.)
        self.assertEqual(t.admit_next(None,1.),'t')
        self.assertEqual(a.admit_next(None,1.),'a')
        self.assertEqual(t.admit_next(None,1.),'later')
        self.assertEqual(len(b.members),0)
        self.assertEqual(len(b.native_queued),3)
        self.assertEqual(b.native_limit,2)
        self.assertTrue(b.mark_native_execution('text','t'))
        self.assertEqual(b.demand['thinker'],10.)

    def test_expired_capacity_hands_waiters_to_native_and_counts_no_fake_gpu_owners(self):
        b=MixedBook(estimate(930.,'text'),estimate(29.26,'audio'),4.23,native_limit=32)
        a=b.views['audio']
        for i in range(100): a.add(str(i),1.)
        for i in range(29): self.assertEqual(a.admit_next(b.capacity('audio',1.),1.),str(i))
        for i in range(9): a.complete(str(i))
        for i in range(29,100): self.assertEqual(a.admit_next(b.capacity('audio',101.),101.),str(i))
        self.assertEqual(len(b.members),20)
        self.assertEqual(len(b.native_queued),71)
        self.assertEqual(a.update(None,101.),'fcfs')

    def test_unknown_thinker_does_not_hide_known_audio_overload(self):
        b=MixedBook(None,estimate(2.,'audio'),4.,native_limit=32)
        a=b.views['audio']
        for i in range(3): a.add(str(i),1.)
        for i in range(3):self.assertEqual(a.admit_next(None,1.),str(i))
        self.assertEqual(len(b.members),0)
        self.assertEqual(len(b.native_queued),3)
        for i in range(3):b.mark_native_execution('audio',str(i))
        a.update(None,1.)
        self.assertEqual(b.reason(1.),'overload')

    def test_missing_resource_does_not_hide_other_resource_overload(self):
        b=MixedBook(estimate(100.,'text'),estimate(2.,'audio'),4.,native_limit=32)
        a,t=b.views['audio'],b.views['text']
        a.add('active',2.)
        self.assertEqual(a.admit_next(b.capacity('audio',1.),1.),'active')
        b.estimates['thinker']=None
        b.estimates['audio']=estimate(1.,'audio')
        t.add('new_text',1.)
        self.assertEqual(t.admit_next(None,2.),'new_text')
        self.assertEqual(t.requests['new_text'].state,'native_queued')
        self.assertEqual(b.reason(2.),'overload')

if __name__=='__main__':unittest.main()
