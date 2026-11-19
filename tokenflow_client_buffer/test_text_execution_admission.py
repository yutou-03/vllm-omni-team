"""Admission uses rate capacity; execution slots do not cap logical W."""
import asyncio
import time
from . import test_recovery_barrier as fixtures


class TextExecutionAdmissionTests(fixtures.MixedRecoveryBarrierTests):
    def setUp(self):
        super().setUp()
        self.text = self.g.gateways['text']
        # These tests isolate the rate ledger. Client safety is exercised by
        # test_admission_safety, and does not reinstate a |W| limit.
        self.text.w.admission_guard = None
        # A legacy experiment setting must not resurrect the removed gate.
        self.text.config.update(execution_coupled_admission=True, engine_queue_admission=True)

    def add_text(self, count=35, rate=5.):
        for i in range(count):
            rid=f'new-{i}';self.text.w.add(rid,rate);self.text.ready[rid]=asyncio.Event()
        self.telemetry('text',False);self.telemetry('audio',False)

    async def test_more_than_native_batch_admitted_when_rate_fits(self):
        self.add_text()
        await self.one_tick(self.text)
        self.assertEqual(len(self.text.w.members),35)
        self.assertFalse(self.g.book.pending)
        self.assertEqual(self.text.w.demand,175.)

    async def test_restore_above_32_does_not_block_feasible_newcomer(self):
        self.add_text(count=34)
        await self.one_tick(self.text)
        self.text.w.pause('new-0')
        self.text.w.resume('new-0')
        self.text.w.add('late',5.);self.text.ready['late']=asyncio.Event()
        await self.one_tick(self.text)
        self.assertEqual(len(self.text.w.members),35)
        self.assertTrue(self.text.ready['late'].is_set())

    async def test_handed_off_newcomer_is_visible_with_W_above_32(self):
        self.add_text(count=101,rate=10.)
        await self.one_tick(self.text)
        self.assertEqual(len(self.text.w.members),100)
        self.assertEqual(self.text.w.requests['new-100'].state,'native_queued')
        import json
        snapshot=json.loads(self.text.snapshot.read_text())
        self.assertEqual(snapshot['mode'],'buffer')
        self.assertEqual(snapshot['buffer_reclaimed_ids'],['new-100'])
        self.assertIsNone(snapshot['admission_waiter'])
        self.text.w.pause('new-0')
        # Actual execution, not a second gateway submission, admits this waiter.
        self.assertTrue(self.g.book.mark_native_execution('text','new-100'))
        self.assertEqual(self.text.w.demand,1000.)
        self.assertEqual(self.g.book.service_demand()['thinker'],1010.)
        self.assertEqual(len(self.text.w.outstanding),101)

    async def test_rate_limit_hands_off_without_inventing_gpu_ownership(self):
        self.add_text(rate=100.)
        await self.one_tick(self.text)
        self.assertEqual(len(self.text.w.members),10)
        self.assertEqual(len(self.g.book.native_queued),25)
        self.assertEqual(self.g.book.service_demand()['thinker'],1000.)
        self.assertTrue(all(event.is_set() for event in self.text.ready.values()))
        self.assertTrue(self.g.book.mark_native_execution('text','new-10'))
        self.text.w.update(self.text.capacity(time.monotonic()),time.monotonic())
        self.assertEqual(self.text.w.mode,'fcfs')
        self.assertEqual(self.g.book.reason(time.monotonic()),'overload')

    async def test_restoration_preserves_overload_and_native_queue_ownership(self):
        self.add_text(count=102,rate=10.)
        await self.one_tick(self.text)
        self.text.w.pause('new-0')
        self.assertTrue(self.g.book.mark_native_execution('text','new-100'))
        self.text.w.resume('new-0');await self.one_tick(self.text)
        self.assertEqual(self.text.w.demand,1010.)
        self.assertEqual(self.text.w.requests['new-101'].state,'native_queued')
        self.assertEqual(self.text.w.mode,'fcfs')
        self.text.w.complete('new-0');await self.one_tick(self.text)
        self.assertEqual(self.text.w.mode,'buffer')
        self.assertEqual(self.text.w.requests['new-101'].state,'native_queued')

    async def test_fcfs_keeps_original_native_queue_admission(self):
        self.text.config['arm']='fcfs'
        self.add_text();await self.one_tick(self.text)
        self.assertEqual(len(self.text.w.members),35)
        self.assertFalse(self.text.pause_snapshot(time.monotonic(),mixed=True))
