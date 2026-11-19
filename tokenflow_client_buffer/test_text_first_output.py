"""Vacant slots serve text first output without stealing a pending return's slot."""
from . import test_scheduler_cadence as fixtures
from .scheduler import ClientBufferScheduler


class TextFirstOutputTests(fixtures.PolicyCadenceTests):
    def setUp(self):
        super().setUp()
        self.config['first_output_refill'] = True
        self.adapter = ClientBufferScheduler(self.s, self.config,
            lambda event, **kw: self.events.append((event, kw)), clock=lambda: self.now)

    def test_initial_cohort_immediate_and_later_new_prefills_share_tick(self):
        self.add('a'); self.add('b')
        self.assertEqual(self.step()[0], {'a', 'b'})
        self.now = 10.1; self.add('c')
        self.assertEqual(self.step()[0], {'a', 'b', 'c'})
        self.now = 10.3; self.add('d')
        self.assertEqual(self.step()[0], {'a', 'b', 'c', 'd'})
        # New first outputs do not require another proactive policy tick.
        self.assertEqual(self.adapter.decision_counts['periodic_policy_decisions'], 1)

    def test_audio_through_text_stage_does_not_wait_for_text_tick(self):
        self.add('text'); self.step()
        self.now = 10.1; self.add('audio', path='audio'); self.add('new_text')
        selected, _ = self.step()
        self.assertEqual(selected, {'text', 'audio', 'new_text'})

    def test_safe_compute_yield_below_native_cap_serves_newcomer(self):
        self.add('owner', running=True, coverage=10., executed=True)
        self.add('new')
        self.adapter.active_ids = {'owner'}
        self.assertEqual(self.step()[0], {'new'})
        self.assertIn('owner', self.adapter.pauses)
        self.assertEqual(self.adapter.pauses['owner'].beneficiary, 'new')

    def test_reclaimed_native_waiter_benefits_from_safe_pause_before_queue_drains(self):
        self.add('owner', running=True, coverage=10., executed=True)
        self.add('new'); self.add('later')
        self.members.pop('new'); self.members.pop('later')
        self.adapter.active_ids = {'owner'}
        self.write(native_fcfs_admission=False,
                   buffer_reclaimed_ids=['new', 'later'],
                   registered_ids=['owner', 'new', 'later'],
                   startup_members={rid:dict(rate=1.) for rid in ['owner', 'new', 'later']},
                   startup_feedback=self.feedback)
        selected, _ = self.step(write=False)
        self.assertIn('new', selected)
        self.assertIn('owner', self.adapter.pauses)
        self.assertEqual(self.adapter.pauses['owner'].beneficiary, 'new')
        self.assertNotIn('new', self.members)  # no fabricated working-set charge

    def test_running_startup_without_client_output_can_benefit_from_safe_yield(self):
        self.add('owner', running=True, coverage=10., executed=True)
        self.add('first_in_flight', running=True, executed=True)
        self.adapter.active_ids = {'owner', 'first_in_flight'}
        self.assertEqual(self.step()[0], {'first_in_flight'})
        self.assertIn('owner', self.adapter.pauses)

    def test_one_pending_return_reserves_one_slot_not_all_waiting_work(self):
        old = self.add('return', coverage=1., executed=True)
        old.status.name = 'PREEMPTED'
        self.add('owner', running=True, coverage=.1, executed=True)
        for rid in ('new1', 'new2', 'new3'): self.add(rid)
        self.pause('return', 1., quota=True)
        self.adapter.active_ids = {'owner'}
        self.adapter.last_mode = 'buffer'; self.adapter.next_policy_at = 10.5
        selected, _ = self.step()
        self.assertEqual(selected, {'owner', 'new1', 'new2'})
        self.assertIn('return', self.adapter.pauses.requested_resumes)
        # An unchanged unacknowledged return keeps its reserved slot next step.
        self.now = 10.02; self.credit('return', .98)
        self.assertEqual(self.step()[0], selected)
        self.now = 10.04
        self.members['return'] = self.paused.pop('return')
        self.acks['return']['state'] = 'active'; self.credit('return', .96)
        self.assertEqual(self.step()[0], {'return', 'owner', 'new1', 'new2'})
        self.assertNotIn('return', self.adapter.pauses)

    def test_acknowledged_return_and_newcomers_can_share_free_batch(self):
        old = self.add('return', coverage=1., executed=True)
        old.status.name = 'PREEMPTED'
        self.add('owner', running=True, coverage=.1, executed=True)
        for rid in ('new1', 'new2', 'new3'): self.add(rid)
        self.pause('return', 1., quota=True)
        self.members['return'] = self.paused.pop('return')
        self.acks['return']['state'] = 'active'
        self.adapter.active_ids = {'owner'}
        self.adapter.last_mode = 'buffer'; self.adapter.next_policy_at = 10.5
        self.assertEqual(self.step()[0], {'return', 'owner', 'new1', 'new2'})

    def test_nonurgent_paused_owner_does_not_take_newcomers_free_slot(self):
        self.add('old', running=True, coverage=10., executed=True)
        self.add('owner', running=True, coverage=.1, executed=True)
        self.add('new')
        self.pause('old', 10.)
        self.adapter.active_ids = {'owner'}
        self.assertEqual(self.step()[0], {'owner', 'new'})
        self.assertIn('old', self.adapter.pauses)

    def test_due_return_does_not_forbid_an_independent_safe_first_output_swap(self):
        self.s.max_num_running_reqs = 3
        self.adapter = ClientBufferScheduler(self.s, self.config,
            lambda event, **kw: self.events.append((event, kw)), clock=lambda: self.now)
        self.add('due', running=True, coverage=1., executed=True)
        for rid in ('a', 'b'): self.add(rid, running=True, coverage=10., executed=True)
        self.pause('due', 1.)
        self.add('new')
        self.adapter.active_ids = {'a', 'b'}
        selected, _ = self.step()
        self.assertIn('due', selected)
        self.assertIn('new', selected)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(self.adapter.pauses), 1)
        self.assertNotIn('due', self.adapter.pauses)

    def test_quota_blocked_newcomer_still_gets_safe_donor_below_native_cap(self):
        self.add('owner', running=True, coverage=10., executed=True)
        self.add('new'); self.members.pop('new')
        self.quota = True; self.adapter.active_ids = {'owner'}
        self.write(gamma=1., registered_ids=['owner', 'new'],
            admission_waiter={'request_id': 'new', 'resource_deficit': {'thinker': 1.}},
            pause_release_cost_per_rate={'thinker': 1.})
        selected, _ = self.step(write=False)
        self.assertIn('owner', self.adapter.pauses)
        self.assertNotIn('new', selected)  # still needs gateway quota acknowledgement

    def test_unbuffered_full_batch_cannot_be_evicted_for_new_first_output(self):
        for rid in ('a', 'b', 'c', 'd'):
            self.add(rid, running=True, coverage=.1, executed=True)
        self.add('new'); self.adapter.active_ids = {'a', 'b', 'c', 'd'}
        self.assertEqual(self.step()[0], {'a', 'b', 'c', 'd'})
        self.assertFalse(self.adapter.pauses)

    def test_W_above_cap_allows_due_return_and_oldest_newcomer(self):
        # Four executing slots, nine admitted requests, plus a paused return.
        # Only two owners can safely yield: one to the return, one to a newcomer.
        self.add('due', running=True, coverage=1., executed=True)
        self.pause('due', 1.)
        for rid in ('safe1', 'safe2'):
            self.add(rid, running=True, coverage=10., executed=True)
        for rid in ('protected1', 'protected2'):
            self.add(rid, running=True, coverage=.1, executed=True)
        self.add('oldest_new')
        self.add('younger_resident', running=True)
        for i in range(3): self.add(f'new-{i}')
        self.adapter.active_ids = {'safe1', 'safe2', 'protected1', 'protected2'}
        selected, _ = self.step()
        self.assertEqual(selected, {'due', 'oldest_new', 'protected1', 'protected2'})
        self.assertEqual(set(self.adapter.pauses), {'safe1', 'safe2'})
        self.assertGreater(len(self.members), self.s.max_num_running_reqs)

    def test_due_return_wins_last_safe_slot_without_evicting_unbuffered_owner(self):
        self.add('due', running=True, coverage=1., executed=True)
        self.pause('due', 1.)
        self.add('safe', running=True, coverage=10., executed=True)
        for i in range(3): self.add(f'protected-{i}', running=True, coverage=.1, executed=True)
        self.add('new')
        self.adapter.active_ids = {'safe', 'protected-0', 'protected-1', 'protected-2'}
        selected, _ = self.step()
        self.assertEqual(selected, {'due', 'protected-0', 'protected-1', 'protected-2'})
        self.assertEqual(set(self.adapter.pauses), {'safe'})
