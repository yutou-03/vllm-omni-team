"""Already-started streams remain eligible for client-buffer-safe exchanges."""
from . import test_audio_first_output as fixtures


class StreamReplenishmentTests(fixtures.AudioFirstOutputTests):
    def owners(self):
        self.add('donor', running=True, coverage=20., executed=True)
        for i in range(3):
            self.add(f'protected-{i}', running=True, coverage=.1, executed=True)
        self.adapter.active_ids = {'donor', 'protected-0', 'protected-1', 'protected-2'}

    def test_started_resident_with_low_credit_gets_safe_exchange(self):
        self.owners()
        self.add('hungry', running=True, coverage=.1, executed=True)
        selected, _ = self.step()
        self.assertIn('hungry', selected)
        self.assertNotIn('donor', selected)
        self.assertEqual(self.adapter.pauses['donor'].beneficiary, 'hungry')

    def test_started_native_waiter_precedes_older_newcomer(self):
        self.owners()
        self.add('new')
        self.add('hungry', coverage=.1, executed=True)
        selected, visible = self.step()
        self.assertIn('hungry', selected)
        self.assertNotIn('new', selected)
        self.assertEqual(visible[0], 'hungry')

    def test_replenishment_donor_waits_for_new_receipt_not_started_flag(self):
        self.add('donor', running=True, coverage=20., executed=True)
        self.add('hungry', running=True, coverage=.1, executed=True)
        self.adapter.active_ids = {'donor'}
        # One native slot forces an actual exchange with a retained KV owner.
        self.s.max_num_running_reqs = 1
        self.adapter.cap = self.adapter.resident_cap = 1
        self.assertEqual(self.step()[0], {'hungry'})
        self.now = 10.5
        self.credit('donor', 19.5)
        self.feedback['hungry']['observed_at'] = self.now
        self.assertEqual(self.step()[0], {'hungry'})
        self.assertIn('donor', self.adapter.pauses)
        self.now = 11.
        self.credit('donor', 3.)
        self.credit('hungry', 20.)
        self.assertEqual(self.step()[0], {'donor'})
        self.assertNotIn('donor', self.adapter.pauses)

    def test_healthy_started_stream_does_not_take_slot_from_newcomer(self):
        self.owners()
        self.add('healthy', coverage=20., executed=True)
        self.add('new')
        selected, _ = self.step()
        self.assertIn('new', selected)
        self.assertNotIn('healthy', selected)

    def test_no_unsafe_donor_for_hungry_started_stream(self):
        for i in range(4):
            self.add(f'owner-{i}', running=True, coverage=.1, executed=True)
        self.add('hungry', coverage=.1, executed=True)
        self.adapter.active_ids = {f'owner-{i}' for i in range(4)}
        self.assertEqual(self.step()[0], self.adapter.active_ids)
        self.assertFalse(self.adapter.pauses)

    def test_exact_capacity_still_checks_starving_retained_owner(self):
        self.owners()
        self.add('hungry', running=True, coverage=.1, executed=True)
        self.write(gamma=5.)
        selected, _ = self.step(write=False)
        self.assertIn('hungry', selected)
        self.assertIn('donor', self.adapter.pauses)

    def test_selected_waiter_keeps_slot_across_native_queues(self):
        self.owners()
        other = self.add('new')
        self.s.waiting.remove(other)
        self.s.skipped_waiting.append(other)
        self.add('hungry', coverage=.1, executed=True)
        selected, _ = self.step()
        self.assertIn('hungry', selected)
        self.assertNotIn('new', selected)
