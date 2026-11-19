"""Regressions for control timing and the decision/execution pause boundary."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from .execution import ExecutionDecision, apply_execution_decision
from .pause import PausedRequest, PausePhase
from .scheduler import ClientBufferScheduler
from .test_core import request


class PauseLifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'snapshot.json'
        self.now = 10.
        self.a, self.b = request('a', 0), request('b', 1)
        self.s = NS(running=[self.a], waiting=[], skipped_waiting=[],
                    requests={'a': self.a}, max_num_running_reqs=1,
                    kv_cache_manager=object(), _omni_stage_id_for_trace=lambda: 0)
        self.events = []
        self.adapter = ClientBufferScheduler(self.s, dict(snapshot_path=str(self.path),
            resume_to_client_s=.5, session='s', modality='text'),
            lambda event, **kw: self.events.append((event, kw)), clock=lambda: self.now)
        self.data = dict(schema_version=1, session='s', modality='text', written_at=10.,
            mode='buffer', demand=1., gamma=10., capacity_valid_until=100.,
            members={'a': dict(rate=1.)}, feedback={'a': dict(request_id='a', session='s',
                modality='text', rate=1., received=10., consumed=0., observed_at=10.,
                sequence=1, started=True, complete=False)})
        self.path.write_text(json.dumps(self.data))

    def test_publication_during_read_uses_one_post_read_decision_time(self):
        original_read = Path.read_text
        def concurrent_read(path, *args, **kwargs):
            self.data['written_at'] = 10.001
            self.data['feedback']['a']['observed_at'] = 10.001
            self.path.write_text(json.dumps(self.data))
            self.now = 10.002
            return original_read(path, *args, **kwargs)
        for interval in (0., .5):
            with self.subTest(policy_interval=interval):
                self.now = 10.
                self.adapter.policy_interval = interval
                self.adapter.snapshot_cache_key = None
                with patch.object(Path, 'read_text', concurrent_read):
                    with self.adapter.scheduling_context():
                        self.assertEqual(self.adapter.last_mode, 'buffer')
                self.assertEqual(self.adapter.last_tick, 10.002)
                self.assertAlmostEqual(self.adapter.coverage(self.a, self.data,
                    self.adapter.last_tick), 9.999)

    def test_slow_read_cannot_accept_expired_snapshot(self):
        original_read = Path.read_text
        def slow_read(path, *args, **kwargs):
            self.now = 10.6
            return original_read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', slow_read):
            self.assertEqual(self.adapter.read_snapshot()[1], 'snapshot_invalid')

    def test_future_timestamp_is_still_rejected(self):
        self.data['written_at'] = 11.
        self.path.write_text(json.dumps(self.data))
        self.assertEqual(self.adapter.read_snapshot()[1], 'snapshot_invalid')

    def test_decision_does_not_change_queues_or_kv(self):
        self.b.status.name = 'WAITING'
        self.s.waiting.append(self.b)
        self.s.requests['b'] = self.b
        kv = self.s.kv_cache_manager
        decision = self.adapter.decide(self.now, self.data, None)
        self.assertEqual(self.s.running, [self.a])
        self.assertEqual(self.s.waiting, [self.b])
        self.assertIs(self.s.kv_cache_manager, kv)
        self.assertEqual(self.a.num_computed_tokens, 10)
        self.assertNotIn(self.a, decision.running)
        self.assertIn(self.b, decision.waiting_priority)
        self.assertEqual(self.adapter.pauses['a'].phase, PausePhase.PAUSED)

    def test_restore_requires_same_epoch_and_charged_membership(self):
        self.adapter.pauses['a'] = PausedRequest(9., 10., .8, releases_quota=True)
        self.adapter.active_ids = set()
        self.data['pause_admission_protocol'] = 1
        self.data['paused_members'] = self.data.pop('members')
        self.data['members'] = {}
        self.data['demand'] = 0.
        self.data['pause_acknowledgements'] = {'a': dict(epoch=9., state='active')}
        decision = self.adapter.decide(self.now, self.data, None)
        self.assertNotIn(self.a, decision.running)  # ACK alone cannot create quota
        self.assertEqual(self.adapter.pauses['a'].phase, PausePhase.RESUME_REQUESTED)
        self.data['members'] = self.data.pop('paused_members')
        self.data['demand'] = 1.
        self.data['pause_acknowledgements']['a']['epoch'] = 8.
        self.assertNotIn(self.a, self.adapter.decide(self.now, self.data, None).running)
        self.data['pause_acknowledgements']['a']['epoch'] = 9.
        self.assertIn(self.a, self.adapter.decide(self.now, self.data, None).running)
        self.assertEqual(self.adapter.pauses['a'].phase, PausePhase.RESUME_READY)
        self.adapter.record_scheduled(NS(num_scheduled_tokens={}))
        self.assertIn('a', self.adapter.pauses)  # selection/ACK is not execution
        self.adapter.record_scheduled(NS(num_scheduled_tokens={'a': 0}))
        self.assertIn('a', self.adapter.pauses)
        self.adapter.record_scheduled(NS(num_scheduled_tokens={'a': 1}))
        self.assertNotIn('a', self.adapter.pauses)

    def test_terminal_pause_and_caches_are_retired_even_in_native_fallback(self):
        self.adapter.pauses['a'] = PausedRequest(9., 10., .8, releases_quota=True)
        self.adapter.seen_contracts.add('a')
        self.adapter.native_execution_seen.add('a')
        self.adapter.metadata_cache['a'] = (self.a, None, {})
        self.a.status.name = 'FINISHED_ABORTED'
        self.s.running.clear()
        self.data.update(mode='fcfs', reason='baseline', members={}, feedback={}, demand=0.)
        self.path.write_text(json.dumps(self.data))
        with self.adapter.scheduling_context():
            pass
        self.assertFalse(self.adapter.pauses)
        self.assertFalse(self.adapter.metadata_cache)
        self.assertFalse(self.adapter.seen_contracts)
        self.assertFalse(self.adapter.native_execution_seen)

    def test_executor_preserves_native_preemption_and_new_admission_on_error(self):
        self.s.running.append(self.b)
        self.s.requests['b'] = self.b
        waiter, newcomer = request('waiter', 2), request('new', 3)
        waiter.status.name = 'WAITING'
        self.s.waiting.append(waiter)
        self.s.requests.update(waiter=waiter, new=newcomer)
        running = self.s.running
        kv = self.s.kv_cache_manager
        decision = ExecutionDecision(running=(self.b,), blocked_waiting=frozenset({'waiter'}),
                                     max_running=2)
        with self.assertRaisesRegex(RuntimeError, 'native error'):
            with apply_execution_decision(self.s, decision, lambda r: r.request_id,
                                          lambda r: r.arrival_time) as result:
                self.assertEqual(self.s.running, [self.b])
                self.assertFalse(self.s.waiting)
                self.b.status.name = 'PREEMPTED'
                self.s.waiting.insert(0, self.b)
                self.s.running[:] = [newcomer]
                raise RuntimeError('native error')
        self.assertIs(self.s.running, running)
        self.assertEqual(self.s.running, [self.a, newcomer])
        self.assertEqual(result.active_ids, {'new'})
        self.assertEqual(self.s.max_num_running_reqs, 1)
        self.assertEqual({r.request_id for r in self.s.waiting}, {'b', 'waiter'})
        self.assertIs(self.s.kv_cache_manager, kv)
        self.assertEqual(self.a.num_computed_tokens, 10)

    def test_executor_entry_error_is_not_masked_by_result_cleanup(self):
        with patch('tokenflow_client_buffer.scheduler.apply_execution_decision',
                   side_effect=RuntimeError('executor entry failed')):
            with self.assertRaisesRegex(RuntimeError, 'executor entry failed'):
                with self.adapter.scheduling_context():
                    self.fail('native scheduling must not start')
        self.assertEqual(self.s.running, [self.a])


if __name__ == '__main__':
    unittest.main()
