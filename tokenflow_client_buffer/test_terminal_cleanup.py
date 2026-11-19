"""Finished streams leave bounded tombstones instead of live control objects."""
from pathlib import Path
import tempfile
import unittest

from .core import WorkingSet
from .mixed_core import MixedBook
from .test_core import estimate


class LedgerCleanupTests(unittest.TestCase):
    def test_history_is_bounded_and_arrival_order_never_repeats(self):
        w = WorkingSet('text', terminal_history_limit=8)
        for i in range(100):
            rid = str(i)
            w.add(rid, 1.)
            self.assertEqual(w.requests[rid].arrival_order, i)
            w.complete(rid)
            w.complete(rid)  # terminal notifications are idempotent
        self.assertFalse(w.requests)
        self.assertEqual(len(w.completed_ids), 8)
        self.assertEqual(w.demand, 0.)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            w.add('99', 1.)
        w.add('next', 1.)
        self.assertEqual(w.requests['next'].arrival_order, 100)

    def test_mixed_completion_releases_phase_and_cross_route_duplicate_guard(self):
        book = MixedBook(estimate(100.), estimate(10., modality='audio'), 2.)
        audio = book.views['audio']
        audio.add('a', 1.)
        book.complete_thinker('a')
        audio.complete('a')
        self.assertNotIn('a', book.thinker_completed)
        self.assertFalse(audio.requests)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            book.views['text'].add('a', 1.)
        book.views['text'].add('b', 1.)
        self.assertEqual(book.views['text'].requests['b'].arrival_order, 1)


class GatewayCleanupTests(unittest.TestCase):
    def test_terminal_cleanup_releases_all_gateway_objects_and_phase_attempts(self):
        from .gateway import Gateway
        from types import SimpleNamespace as NS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            g = Gateway(dict(modality='text', session='s', model=directory, arm='buffer',
                gateway_log=str(root/'gateway.jsonl'), snapshot_path=str(root/'snapshot.json')))
            try:
                g.w.add('r', 1.)
                g.coordinator = NS(phase_attempts={'r': (1., 'backend')})
                for states in (g.feedback, g.delivery_history, g.ready, g.registered_at,
                               g.pause_acks, g.accepted_pause_leases):
                    states['r'] = object()
                g.finish_request('r')
                g.finish_request('r')
                self.assertFalse(g.w.requests)
                self.assertIn('r', g.w.completed_ids)
                for states in (g.feedback, g.delivery_history, g.ready, g.registered_at,
                               g.pause_acks, g.accepted_pause_leases, g.coordinator.phase_attempts):
                    self.assertNotIn('r', states)
            finally:
                g.log.close()


class FeedbackRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_feedback_cannot_credit_a_reused_request_id(self):
        import time
        from types import SimpleNamespace as NS
        from unittest.mock import AsyncMock
        from .core import ConsumerBuffer
        from .gateway import Gateway
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            g = Gateway(dict(modality='text', session='s', model=directory, arm='buffer',
                gateway_log=str(root/'gateway.jsonl'), snapshot_path=str(root/'snapshot.json')))
            try:
                now = time.monotonic()
                g.w.add('r', 1.)
                g.registered_at['r'] = now
                buffer = ConsumerBuffer(1.)
                buffer.receive(10., now - .1)
                raw = buffer.feedback(now - .1, request_id='r', session='s',
                                      modality='text', clock_domain=g.clock_domain)
                response = await g.receive_feedback(NS(json=AsyncMock(return_value=raw)))
                self.assertEqual(response.status, 400)
                self.assertIn('predates request registration', response.text)
                self.assertNotIn('r', g.feedback)
                raw = buffer.feedback(now, request_id='r', session='s',
                                      modality='text', clock_domain=g.clock_domain)
                response = await g.receive_feedback(NS(json=AsyncMock(return_value=raw)))
                self.assertEqual(response.status, 200)
            finally:
                g.log.close()


if __name__ == '__main__':
    unittest.main()
