"""Rate-based expansion is independent of startup/buffer-backed preemption."""
import asyncio
import time
from dataclasses import replace
from .core import ClientFeedback
from .admission import DeliveryHistory, minimum_buffer
from .test_recovery_barrier import MixedRecoveryBarrierTests


class AdmissionSafetyTests(MixedRecoveryBarrierTests):
    def add(self, route, rid, state='waiting', rate=None):
        g = self.g.gateways[route]
        g.w.add(rid, rate or (1. if route == 'audio' else 5.))
        g.w.requests[rid].state = state
        g.ready[rid] = asyncio.Event()
        g.registered_at[rid] = time.monotonic()-1
        return g

    def feedback(self, g, rid, cover, *, age=0., started=True):
        now = time.monotonic();rate = g.w.requests[rid].rate
        g.feedback[rid] = ClientFeedback(rid, g.session, g.modality, rate,
            cover*rate, 0., now-age, 1, started, False)

    def history(self, g, rid, now, cover, *, delivered_rate=1.2, gap=4.):
        history = g.delivery_history[rid] = DeliveryHistory()
        rate = g.w.requests[rid].rate
        for i in range(3):
            received = cover*rate+(i+1)*delivered_rate*gap
            f = ClientFeedback(rid, g.session, g.modality, rate,
                received, received-cover*rate, now-(2-i)*gap, i+1, True, False)
            history.observe(f, self.g.book.demand)
        g.feedback[rid] = f
        return history

    def test_forecast_checks_before_chunk_not_only_endpoint(self):
        self.assertEqual(minimum_buffer(2., 1., 5., [(4., 10.)]), -2.)
        self.assertEqual(minimum_buffer(2., 1., 5., [(1., 10.)]), 1.)
        with self.assertRaises(ValueError):
            minimum_buffer(2., 1., 5., [(4., 10.), (2., 10.)])

    async def test_eight_requests_can_enter_without_serial_first_output_waits(self):
        for i in range(8):g = self.add('audio', str(i))
        await self.one_tick(g)
        self.assertEqual(len(g.w.members), 8)
        self.assertTrue(all(g.ready[str(i)].is_set() for i in range(8)))
        self.assertFalse(g.feedback)

    async def test_capacity_and_requested_rates_determine_count_not_fixed_29_or_32(self):
        self.g.book.estimates['audio'] = replace(self.g.book.estimates['audio'], gamma=7.5, admission_gamma=7.5)
        for i in range(40):g = self.add('audio', str(i), rate=1.5)
        for _ in range(40):
            await self.one_tick(g)
            if not self.g.book.pending:
                break
        self.assertEqual(len(g.w.members), 5)
        self.assertEqual(g.w.demand, 7.5)
        self.assertTrue(g.ready['5'].is_set())
        self.assertEqual(g.w.requests['5'].state,'native_queued')
        # Queue handoff does not charge execution; R49 can recover while
        # queued work stays explicitly selectable by the buffer scheduler.
        self.assertEqual(g.w.mode,'buffer')
        self.assertEqual(len(self.g.book.native_queued),35)

    async def test_small_buffer_and_slower_accumulation_do_not_block_expansion(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        now = time.monotonic();self.history(g, 'old', now, .1, delivered_rate=1.1)
        g.config['admission_startup_s'] = 500.  # irrelevant to a continuing stream
        await self.one_tick(g)
        self.assertTrue(g.ready['new'].is_set())
        self.assertNotEqual(g.w.requests['old'].state, 'paused')

    async def test_no_sample_or_stale_sample_is_unknown_not_zero_production(self):
        g = self.add('audio', 'old', 'admitted')
        self.feedback(g, 'old', .1, age=1.)
        self.add('audio', 'new1');self.add('audio', 'new2')
        await self.one_tick(g)
        self.assertTrue(g.ready['new1'].is_set());self.assertTrue(g.ready['new2'].is_set())

    async def test_temporary_shortfall_with_large_buffer_allows_expansion(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        now = time.monotonic();self.history(g, 'old', now, 100., delivered_rate=.8)
        await self.one_tick(g)
        self.assertTrue(g.ready['new'].is_set())
        self.assertEqual(g.last_admission_decision.reason, 'stage_rates_feasible')

    async def test_safe_buffer_does_not_override_capacity(self):
        g = self.add('audio', 'full', 'admitted', rate=30.)
        self.feedback(g, 'full', 100.);self.add('audio', 'waiting')
        await self.one_tick(g)
        self.assertTrue(g.ready['waiting'].is_set())
        self.assertEqual(g.w.requests['waiting'].state,'native_queued')
        self.assertEqual(g.w.mode,'buffer')
        self.assertEqual(g.w.demand,30.)

    async def test_no_W_count_cap_is_reintroduced(self):
        for i in range(34):g = self.add('text', str(i), 'admitted')
        self.add('text', 'late');await self.one_tick(g)
        self.assertTrue(g.ready['late'].is_set());self.assertEqual(len(g.w.members), 35)

    async def test_native_fcfs_not_gated_by_client_buffer(self):
        for i in range(40):g = self.add('audio', str(i))
        g.config['arm'] = 'fcfs'
        # R51 yields each 10 ms so feedback remains responsive. Draining a
        # queue is no longer guaranteed to finish in a single event-loop turn.
        for _ in range(40):
            await self.one_tick(g)
            if not self.g.book.pending:
                break
        self.assertEqual(len(g.w.members), 40)

    def test_return_before_new_first_output_allowed_when_restored_capacity_fits(self):
        g = self.add('audio', 'paused', 'paused');self.feedback(g, 'paused', 100.)
        self.add('audio', 'new');now = time.monotonic()
        g.accepted_pause_leases['paused'] = dict(epoch=now-1, resume_by=now+.1)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        # With no capacity for the returning owner, the same admission is unsafe.
        self.g.book.estimates['audio'] = replace(self.g.book.estimates['audio'], gamma=1., admission_gamma=1.)
        self.assertFalse(g.admission_safety(g.w.requests['new'], now))
        self.assertEqual(g.last_admission_decision.reason, 'recovery_before_new_first_output')

    def test_paused_audio_cannot_spend_predicted_receipts(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        g.config['admission_startup_s'] = 5.4684317899
        now = time.monotonic();self.history(g, 'old', now, 3.)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        decision, checks = g._admission_forecast(g.w.requests['new'], now, proposed_pauses={'old'})
        self.assertFalse(decision.allowed)
        pause = next(c for c in checks if c.get('paused'))
        self.assertEqual(pause['forecast_delivery_units'], 0.)
        self.assertAlmostEqual(pause['horizon_s'], 7.9684317899)

    def test_heartbeat_does_not_count_as_delivery_but_silent_time_counts(self):
        g = self.add('audio', 'old', 'admitted');now = time.monotonic()
        history = self.history(g, 'old', now-3., 3.)
        f = g.feedback['old']
        heartbeat = replace(f, observed_at=now, sequence=4)
        history.observe(heartbeat, self.g.book.demand)
        self.assertEqual(len(history.samples), 3)
        self.assertAlmostEqual(history.sustained_rate(now, 8.), 9.6/11.)

    def test_large_buffer_does_not_require_an_unnecessary_pause(self):
        g = self.add('audio', 'donor', 'admitted');self.add('audio', 'new')
        g.config['admission_startup_s'] = 5.4684317899
        now = time.monotonic();self.history(g, 'donor', now, 20., delivered_rate=.8)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        self.assertIsNone(g.safety_pause_waiter(now))
        self.assertEqual(g.w.requests['donor'].state, 'admitted')
        self.assertFalse(g.accepted_pause_leases)

    def test_pre_existing_shortfall_is_not_an_absolute_admission_veto(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        now = time.monotonic();self.history(g, 'old', now, .2, delivered_rate=.8)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        decision, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertEqual(checks[-1]['observation'], 'pre_existing_underflow')

    def test_continuing_output_is_counted_but_cannot_fund_a_pause(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        g.config['admission_startup_s'] = 5.4684317899
        now = time.monotonic();self.history(g, 'old', now, 7.5, delivered_rate=.8)
        decision, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertTrue(decision.allowed)
        forecast = checks[-1]
        self.assertGreater(forecast['forecast_delivery_units'], 0.)
        self.assertGreater(forecast['minimum_buffer'], 0.)
        paused, _ = g._admission_forecast(g.w.requests['new'], now, proposed_pauses={'old'})
        self.assertFalse(paused.allowed)
        self.assertEqual(paused.reason, 'insufficient_pause_buffer')

    def test_pre_existing_interchunk_underflow_is_identified_before_admission(self):
        g = self.add('audio', 'old', 'admitted');self.add('audio', 'new')
        g.config['admission_startup_s'] = 1.6  # total horizon 4.1 s
        now = time.monotonic();self.history(g, 'old', now, 2., delivered_rate=.8)
        decision, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertTrue(decision.allowed)
        forecast = checks[-1]
        self.assertGreater(2.+.8*4.-forecast['horizon_s'], 0.)
        self.assertLess(forecast['baseline_minimum_buffer'], 0.)
        self.assertEqual(forecast['observation'], 'pre_existing_underflow')

    def test_r37_rate_deficit_no_longer_vetoes_sufficient_client_buffer(self):
        g = self.add('audio', 'old', 'admitted', rate=.5);self.add('audio', 'new', rate=.5)
        g.config['admission_startup_s'] = 5.4684317899
        now = time.monotonic();self.history(g, 'old', now, 20., delivered_rate=.29489288197575647)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        # Exhaustion already predicted before admission is health evidence,
        # not proof that this newcomer introduced the deficit.
        f = g.feedback['old'];g.feedback['old'] = replace(f, consumed=f.received-.1)
        self.assertTrue(g.admission_safety(g.w.requests['new'], now))
        _, checks = g._admission_forecast(g.w.requests['new'], now)
        self.assertEqual(checks[-1]['observation'], 'pre_existing_underflow')

    def test_finished_thinker_audio_not_a_global_barrier_for_new_text(self):
        self.add('audio', 'audio', 'admitted');self.g.book.complete_thinker('audio')
        text = self.add('text', 'text')
        self.assertTrue(text.admission_safety(text.w.requests['text'], time.monotonic()))

    def test_each_stage_budget_must_fit_in_its_own_units(self):
        g = self.add('audio', 'old', 'admitted', rate=2.)
        self.add('audio', 'new', rate=2.)
        self.g.book.estimates['thinker'] = replace(self.g.book.estimates['thinker'], gamma=12., admission_gamma=12.)
        self.assertFalse(g.admission_safety(g.w.requests['new'], time.monotonic()))
        self.assertEqual(g.last_admission_decision.reason, 'stage_rate_capacity_exceeded')
        self.assertEqual(g.last_admission_decision.request_id, 'thinker')

    def test_newborn_stream_has_no_sustained_under_service_evidence(self):
        g = self.add('audio', 'old', 'admitted');now = time.monotonic()
        history = self.history(g, 'old', now, .1, gap=.1)
        self.assertIsNone(history.sustained_rate(now, 8.))
