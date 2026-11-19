"""A late control ACK cannot permanently disable subsequent buffer scheduling."""
import json
from pathlib import Path
import unittest
from . import test_pause_admission as fixtures


class AckTimeoutRecoveryTests(unittest.TestCase):
    setUp = fixtures.PauseAdmissionTests.setUp
    feedback = fixtures.PauseAdmissionTests.feedback
    snapshot = fixtures.PauseAdmissionTests.snapshot
    step = fixtures.PauseAdmissionTests.step

    def timeout(self):
        self.step()
        self.g.process_pause_leases(self.now)
        self.now = 10.1
        self.feedback('a', .2)
        self.feedback('b', .1)
        self.assertNotIn('a', self.step())
        self.now = 10.7
        self.feedback('a', .2)
        self.feedback('b', .1)
        self.assertNotIn('a', self.step())
        self.assertEqual(self.adapter.latched_fault, 'pause_resume_ack_timeout')

    def test_actual_timeout_then_ack_recovers_without_uncharged_execution(self):
        self.timeout()
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.pause_acks['a']['state'], 'active')
        self.assertEqual(self.g.w.demand, 2.)
        self.now = 10.75
        self.assertIn('a', self.step())
        self.assertIsNone(self.adapter.latched_fault)
        self.assertEqual(self.adapter.last_mode, 'buffer')
        self.assertEqual(sum(k == 'client_buffer_resume_ack_recovered' for k, _ in self.events), 1)

    def test_wrong_epoch_does_not_recover_or_execute(self):
        self.timeout()
        self.g.process_pause_leases(self.now)
        self.g.pause_acks['a']['epoch'] -= 1
        self.now = 10.75
        self.assertNotIn('a', self.step())
        self.assertEqual(self.adapter.latched_fault, 'pause_resume_ack_timeout')

    def test_stale_snapshot_does_not_recover(self):
        self.timeout()
        self.g.process_pause_leases(self.now)
        d = self.snapshot()
        d['written_at'] = self.now-1.
        Path(self.g.config['snapshot_path']).write_text(json.dumps(d))
        with self.adapter.scheduling_context():
            self.assertNotIn(self.a, self.s.running)
        self.assertEqual(self.adapter.latched_fault, 'pause_resume_ack_timeout')

    def test_capacity_overload_still_falls_back_after_communication_recovers(self):
        from dataclasses import replace
        self.timeout()
        self.g.process_pause_leases(self.now)
        self.g.est = replace(self.g.est, gamma=1., admission_gamma=1.)
        self.now = 10.75
        self.step()
        self.assertIsNone(self.adapter.latched_fault)
        self.assertEqual(self.adapter.last_mode, 'fcfs')
        self.assertEqual(self.g.w.reason, 'overload')

    def test_unrelated_contract_fault_is_never_auto_cleared(self):
        self.adapter.latched_fault = 'request_rate_not_propagated'
        self.step()
        self.assertEqual(self.adapter.latched_fault, 'request_rate_not_propagated')

    def test_terminal_ack_is_pruned_but_live_restoration_ack_is_kept(self):
        self.timeout()
        self.g.process_pause_leases(self.now)
        self.assertNotIn('a', self.g.accepted_pause_leases)
        self.g.process_pause_leases(self.now)
        self.assertIn('a', self.g.pause_acks)
        self.g.w.complete('a')
        self.g.process_pause_leases(self.now)
        self.assertNotIn('a', self.g.pause_acks)

if __name__ == '__main__':
    unittest.main()
