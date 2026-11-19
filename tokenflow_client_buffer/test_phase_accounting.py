"""Only terminal success of the current backend attempt releases Thinker."""
import json
from pathlib import Path
import time
from .core import CapacityEstimate
from .test_recovery_barrier import MixedRecoveryBarrierTests


class PhaseAccountingTests(MixedRecoveryBarrierTests):
    def setup_audio(self, rid='audio', rate=1.):
        now = time.monotonic()
        g = self.g.gateways['audio']
        g.w.add(rid, rate);g.w.requests[rid].state = 'admitted'
        g.registered_at[rid] = now-3
        self.now = now
        return g

    def write_event(self, kind, *, rid='audio', started=None, attempt='backend', reason='stop', session='text'):
        p = Path(str(self.g.gateways['text'].config['execution_telemetry_path'])+'.completed.jsonl')
        with p.open('a') as f:
            f.write(json.dumps(dict(event=kind, stage=0, session=session, request_id=rid,
                route='audio', backend_request_id=attempt, started_at=started or self.now-2,
                completed_at=self.now-1, finish_reason=reason))+'\n')

    def test_completed_thinker_no_longer_required_by_audio_even_after_expiry(self):
        audio = self.setup_audio()
        self.write_event('started');self.write_event('completed')
        self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand, dict(thinker=0., audio=1.))
        self.g.book.estimates['thinker'] = CapacityEstimate(1000., 0., 1., 'text', 'old', 'test')
        self.assertIsNone(self.g.book.reason(self.now))
        self.assertIsNotNone(self.g.book.capacity('audio', self.now))
        snap = audio.pause_snapshot(self.now, mixed=True)
        self.assertEqual(snap['pause_release_cost_by_request']['audio'],dict(thinker=0.,audio=1.))
        self.assertEqual(self.g.book.views['audio'].requests['audio'].state, 'admitted')

    def test_failed_stale_and_wrong_session_events_do_not_release(self):
        self.setup_audio()
        self.write_event('started', session='old-session');self.write_event('completed', session='old-session')
        self.write_event('started', started=self.now-20);self.write_event('completed', started=self.now-20)
        self.write_event('started');self.write_event('completed', reason='abort')
        self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'], 4.)
        self.write_event('completed', reason='error');self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'], 4.)

    def test_restart_recharges_and_duplicate_old_completion_cannot_release(self):
        self.setup_audio()
        self.write_event('started');self.write_event('completed')
        self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'], 0.)
        self.write_event('started', started=self.now-.5, attempt='retry')
        self.write_event('completed')
        self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'], 4.)

    def test_new_request_still_needs_thinker_when_existing_audio_no_longer_does(self):
        audio = self.setup_audio()
        self.g.book.complete_thinker('audio')
        self.g.book.estimates['thinker'] = CapacityEstimate(1000., 0., 1., 'text', 'old', 'test')
        audio.w.add('new', 1.)
        self.assertEqual(audio.w.admit_next(self.g.book.capacity('audio', self.now), self.now),'new')
        self.assertEqual(audio.w.requests['new'].state, 'native_queued')
        self.assertEqual(self.g.book.reason(self.now),'capacity_unknown')

    def test_waiting_audio_with_completed_thinker_can_use_audio_capacity(self):
        audio = self.setup_audio()
        self.g.book.complete_thinker('audio')
        self.g.book.estimates['thinker'] = CapacityEstimate(1000., 0., 1., 'text', 'old', 'test')
        audio.w.add('new', 1.);self.g.book.complete_thinker('new')
        audio.w.admission_guard = lambda *_: True  # phase capacity test, separate from client safety
        self.assertEqual(audio.w.admit_next(self.g.book.capacity('audio',self.now),self.now),'new')

    def test_completion_is_idempotent_and_partial_journal_line_is_retried(self):
        self.setup_audio()
        self.write_event('started')
        path = Path(str(self.g.gateways['text'].config['execution_telemetry_path'])+'.completed.jsonl')
        self.write_event('completed');raw=path.read_bytes();path.write_bytes(raw[:-1])
        self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'],4.)
        with path.open('ab') as f:f.write(b'\n')
        self.g.process_phase_events(self.now);self.g.process_phase_events(self.now)
        self.assertEqual(self.g.book.demand['thinker'],0.)

    def test_completed_candidate_does_not_borrow_unfinished_members_thinker_cost(self):
        audio = self.setup_audio()
        self.g.book.estimates['thinker'] = CapacityEstimate(4., self.now-1, self.now+10,
                                                           'text', 'busy', 'test')
        audio.w.add('already-done', 1.); self.g.book.complete_thinker('already-done')
        audio.w.admission_guard = lambda *_: True
        self.assertEqual(audio.w.admit_next(self.g.book.capacity('audio',self.now),self.now),'already-done')
        self.assertEqual(self.g.book.demand, dict(thinker=4.,audio=2.))
        audio.w.add('needs-thinker',1.)
        self.assertEqual(audio.w.admit_next(self.g.book.capacity('audio',self.now),self.now),'needs-thinker')
        self.assertEqual(audio.w.requests['needs-thinker'].state,'native_queued')
        self.assertEqual(self.g.book.reason(self.now),'overload')
