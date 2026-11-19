"""Feedback must be served while a large admission backlog is draining."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from aiohttp import ClientSession, ClientTimeout, web
from .core import ConsumerBuffer
from .gateway import Gateway


class FeedbackResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, recovery=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            gateway = Gateway(dict(arm='fcfs', modality='text', session='test',
                model=str(root), max_inflight=256, snapshot_path=str(root/'state.json'),
                gateway_log=str(root/'gateway.jsonl')))
            gateway.w.add('playing', 10.)
            gateway.w.requests['playing'].state = 'admitted'
            gateway.registered_at['playing'] = time.monotonic()
            count = 160
            for i in range(count):
                rid = str(i)
                gateway.w.add(rid, 10.)
                gateway.ready[rid] = asyncio.Event()
            admit = gateway.w.admit_next
            admitted = []
            blocked = False
            def slow_admit(*args, **kwargs):
                # Deterministic CPU work substitutes for per-candidate file
                # reads and full-ledger scans, not GPU production latency.
                time.sleep(.002)
                rid = admit(*args, **kwargs)
                if rid is not None:
                    admitted.append(rid)
                return rid
            gateway.w.admit_next = slow_admit
            async def feedback(request):
                nonlocal blocked
                reply = await gateway.receive_feedback(request)
                if recovery:
                    blocked = True
                return web.json_response(dict(accepted=reply.status == 200,
                    admissions=len(admitted), published=(root/'state.json').exists()))
            if recovery:
                gateway.coordinator = SimpleNamespace(config={},
                    admission_recovery_pending=lambda now: blocked)
            app = web.Application()
            app.router.add_post('/feedback', feedback)
            app.router.add_get('/contract', gateway.contract)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            url = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
            task = None
            try:
                async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                    async with session.get(url+'/contract') as response:
                        await response.read()
                    buffer = ConsumerBuffer(10.)
                    buffer.receive(10., time.monotonic())
                    raw = buffer.feedback(time.monotonic(), request_id='playing',
                        session='test', modality='text', clock_domain=gateway.clock_domain)
                    task = asyncio.create_task(gateway.loop())
                    async with session.post(url+'/feedback', json=raw) as response:
                        result = await response.json()
                    self.assertTrue(result['accepted'])
                    self.assertTrue(result['published'])
                    self.assertLess(result['admissions'], count,
                        'feedback starved until entire backlog was admitted')
                    if recovery:
                        before = len(admitted)
                        await asyncio.sleep(.06)
                        self.assertEqual(len(admitted), before,
                            'yield must recheck recovery before next admission')
                        blocked = False
                    deadline = time.monotonic()+3
                    while len(admitted)<count and time.monotonic()<deadline:
                        await asyncio.sleep(.01)
                    self.assertEqual(admitted, [str(i) for i in range(count)])
            finally:
                gateway.stopping = True
                if task:
                    await task
                await runner.cleanup()
                gateway.log.close()

    async def test_feedback_before_backlog_drains_and_fifo_is_preserved(self):
        await self.exercise()

    async def test_recovery_arriving_during_yield_blocks_further_admission(self):
        await self.exercise(recovery=True)


if __name__ == '__main__':
    unittest.main()
