"""Real HTTP/SSE loopback tests; run with aiohttp available, no model/GPU."""
import asyncio
import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import wave

from aiohttp import ClientSession, web
from tokenflow_client_buffer.client import consume_request
from tokenflow_client_buffer.core import CapacityEstimate
from tokenflow_client_buffer.gateway import Gateway
from tokenflow_client_buffer.stream import clock_domain


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def run_modality(self, modality):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer_config.json").write_text('{"added_tokens_decoder":{}}')
            captured = []
            audio = io.BytesIO()
            with wave.open(audio, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(b'\0\0' * 1200)
            encoded = base64.b64encode(audio.getvalue()).decode()

            async def fake(request):
                p = await request.json()
                captured.append(p)
                r = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
                await r.prepare(request)
                for i in range(2):
                    event = dict(modality=modality, choices=[dict(index=0,
                        delta={"content":encoded if modality == "audio" else "word "},
                        token_ids=[100+i])])
                    await r.write(("data: " + json.dumps(event) + "\n\n").encode())
                    await asyncio.sleep(.01)
                await r.write(b'data: [DONE]\n\n')
                await r.write_eof()
                return r

            upstream = web.Application()
            upstream.router.add_post('/v1/chat/completions', fake)
            runner1 = web.AppRunner(upstream)
            await runner1.setup()
            site1 = web.TCPSite(runner1, '127.0.0.1', 0)
            await site1.start()
            port1 = site1._server.sockets[0].getsockname()[1]
            now = time.monotonic()
            (root / 'capacity.json').write_text(json.dumps(asdict(CapacityEstimate(100,now,now+60,modality,'test','test'))))
            g = Gateway(dict(modality=modality,session='test-session',model=str(root),
                snapshot_path=str(root/'snapshot.json'),gateway_log=str(root/'gateway.jsonl'),
                capacity_path=str(root/'capacity.json'),arm='buffer',upstream=f'http://127.0.0.1:{port1}'))
            app = web.Application()
            app.router.add_post('/v1/chat/completions',g.stream)
            app.router.add_post('/feedback',g.receive_feedback)
            app.on_startup.append(g.startup)
            app.on_cleanup.append(g.cleanup)
            runner2 = web.AppRunner(app)
            await runner2.setup()
            site2 = web.TCPSite(runner2,'127.0.0.1',0)
            await site2.start()
            port2 = site2._server.sockets[0].getsockname()[1]
            key='text_consume_rate_tps' if modality=='text' else 'audio_consume_rate'
            rate=10. if modality=='text' else 1.
            row=dict(payload=dict(stream=True,omni_scheduling=dict(schema_version=2,
                source_request_id='r',request_path=modality,ingress_order=0,**{key:rate})))
            try:
                async with ClientSession() as session:
                    result=await consume_request(row,session=session,endpoint=f'http://127.0.0.1:{port2}',
                        contract=dict(session='test-session',modality=modality,clock_domain=clock_domain()),
                        output=root,special=[],origin=time.monotonic(),timeout=5)
                self.assertEqual(result['status'],'success',result.get('error'))
                self.assertEqual(result['remaining_queue'],0)
                self.assertEqual(captured[0]['omni_scheduling'][key],rate)
                self.assertAlmostEqual(result['received_units'],2 if modality=='text' else .1)
                self.assertEqual(g.w.demand,0)
                self.assertGreater(g.feedback['r'].consumed,0)
                self.assertEqual(result['content_sha256'],next(json.loads(line)['content_sha256']
                    for line in (root/'gateway.jsonl').read_text().splitlines()
                    if json.loads(line)['event']=='generation_complete'))
            finally:
                await runner2.cleanup()
                await runner1.cleanup()

    def test_capacity_only_ablation_preserves_gate_with_fcfs_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            g=Gateway(dict(modality='audio',session='s',model=str(root),snapshot_path=str(root/'snapshot.json'),
                gateway_log=str(root/'gateway.jsonl'),arm='buffer',capacity_only=True))
            try:
                now=time.monotonic();g.est=CapacityEstimate(2.,now,now+60.,'audio','p','test')
                g.w.add('a',1.);self.assertEqual(g.w.admit_next(g.est,now),'a');g.publish()
                self.assertEqual(json.loads((root/'snapshot.json').read_text())['mode'],'fcfs')
                g.w.add('b',2.);self.assertIsNone(g.w.admit_next(g.est,now))
                self.assertEqual(g.w.demand,1.)
            finally:g.log.close()

    async def test_large_sse_frame_and_split_utf8(self):
        from .stream import sse_events
        value = "音频" + "a" * 300000
        wire = ("data: " + value + "\r\n\r\ndata: [DONE]\n\n").encode()
        class Content:
            async def iter_chunked(self, size):
                yield wire[:7]  # split inside the first multibyte character
                for i in range(7, len(wire), size):
                    yield wire[i:i+size]
        events = [s async for s in sse_events(Content())]
        self.assertEqual(events, [value, "[DONE]"])

    async def test_text_end_to_end(self):
        await self.run_modality('text')

    async def test_audio_end_to_end(self):
        await self.run_modality('audio')


if __name__=='__main__':unittest.main()
