"""Regressions for the R52 underfilled batch despite feasible capacity."""
import asyncio
from dataclasses import replace
import time

from .admission import DeliveryHistory
from .core import ClientFeedback
from . import test_admission_safety as fixtures


class AdmissionRefillTests(fixtures.MixedRecoveryBarrierTests):
    add = fixtures.AdmissionSafetyTests.add
    history = fixtures.AdmissionSafetyTests.history

    async def test_r52_existing_deficit_does_not_hide_897_waiters(self):
        g = self.add('text', 'blocker', 'admitted', rate=15.)
        for i in range(17):
            self.add('text', f'owner-{i}', 'admitted', rate=10.)
        self.add('text', 'owner-last', 'admitted', rate=5.)
        for i in range(897):
            self.add('text', f'waiting-{i}', rate=5.)
        self.g.book.estimates['thinker'] = replace(self.g.book.estimates['thinker'],
            gamma=521.5319418767233, admission_gamma=521.5319418767233)
        now = time.monotonic()
        g.config['admission_startup_s'] = 3.581977647962049
        history = g.delivery_history['blocker'] = DeliveryHistory()
        rate, gap, cover = 14.675594745350642, .2601162069477141, .23731131266492106
        for i in range(13):
            received = 100. + rate*i*gap
            f = ClientFeedback('blocker', g.session, 'text', 15., received,
                received-cover*15., now-(12-i)*gap, i+1, True, False)
            history.observe(f, self.g.book.demand)
        g.feedback['blocker'] = f
        self.assertEqual(len(g.w.members), 19)
        self.assertEqual(self.g.book.demand['thinker'], 190.)
        await self.one_tick(g)
        self.assertTrue(g.ready['waiting-0'].is_set())
        self.assertNotEqual(g.w.requests['waiting-0'].state, 'waiting')

    def test_added_contention_cannot_turn_safe_stream_into_underflow(self):
        g = self.add('audio', 'old', 'admitted')
        self.add('audio', 'new')
        g.config['admission_startup_s'] = 5.
        now = time.monotonic()
        self.history(g, 'old', now, 1., delivered_rate=1.2, gap=.1)
        g.config['capacity_window_s'] = .2
        decision, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertFalse(decision.allowed)
        forecast = checks[-1]
        self.assertGreaterEqual(forecast['baseline_minimum_buffer'], 0.)
        self.assertLess(forecast['minimum_buffer'], 0.)

    def test_existing_deficit_is_logged_not_reported_as_newcomer_harm(self):
        g = self.add('audio', 'old', 'admitted')
        self.add('audio', 'new')
        now = time.monotonic()
        self.history(g, 'old', now, .2, delivered_rate=.8)
        decision, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertTrue(decision.allowed)
        self.assertEqual(checks[-1]['observation'], 'pre_existing_underflow')
        self.assertLess(checks[-1]['baseline_minimum_buffer'], 0.)
