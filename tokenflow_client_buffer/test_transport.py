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
    async def run_modality(self, modality, legacy_startup_s=None, source="client_consumption_buffer"):
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
            g = Gateway(dict(buffer_source=source,modality=modality,session='test-session',model=str(root),
                snapshot_path=str(root/'snapshot.json'),gateway_log=str(root/'gateway.jsonl'),
                capacity_path=str(root/'capacity.json'),arm='buffer',upstream=f'http://127.0.0.1:{port1}'))
            app = web.Application()
            app.router.add_post('/v1/chat/completions',g.stream)
            app.router.add_post('/feedback',g.receive_feedback)
            app.router.add_get('/contract',g.contract)
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
            if legacy_startup_s is not None:
                row['startup_s'] = legacy_startup_s
            try:
                async with ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{port2}/contract") as response:
                        contract = await response.json()
                    self.assertEqual(contract["buffer_source"], source)
                    result=await consume_request(row,session=session,endpoint=f'http://127.0.0.1:{port2}',
                        contract=contract,
                        output=root,special=[],origin=time.monotonic(),timeout=5)
                self.assertEqual(result['status'],'success',result.get('error'))
                self.assertEqual(result['remaining_queue'],0)
                self.assertEqual(result['playback_start_policy'],'first_ready')
                self.assertEqual(result['startup_s'],result['first_ready_s'])
                received = [json.loads(line) for line in next(root.glob('*/events.jsonl')).read_text().splitlines()
                            if json.loads(line)['event'] == 'received']
                self.assertEqual(result['buffer']['started_at'],received[0]['ready_at'])
                self.assertLess(result['buffer']['started_at'],result['generation_done_at'])
                self.assertEqual(captured[0]['omni_scheduling'][key],rate)
                self.assertAlmostEqual(result['received_units'],2 if modality=='text' else .1)
                self.assertEqual(g.w.demand,0)
                # Finished streams release server feedback state; consumption
                # remains auditable in the client and accepted-feedback log.
                self.assertNotIn('r', g.feedback)
                self.assertAlmostEqual(result['buffer']['consumed'], result['received_units'])
                accepted = [json.loads(line) for line in (root/'gateway.jsonl').read_text().splitlines()
                            if json.loads(line)['event'] == 'client_feedback']
                self.assertEqual(any(e['request_id'] == 'r' for e in accepted), source == 'client_consumption_buffer')
                self.assertEqual(result['content_sha256'],next(json.loads(line)['content_sha256']
                    for line in (root/'gateway.jsonl').read_text().splitlines()
                    if json.loads(line)['event']=='generation_complete'))
            finally:
                await runner2.cleanup()
                await runner1.cleanup()

    async def test_virtual_pcm_stream_runs_without_feedback_posts(self):
        await self.run_modality("audio", source="server_virtual_output")

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

    def test_audio_empty_window_time_is_retained_with_independent_pressure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            telemetry=root/'execution.json'
            g=Gateway(dict(modality='audio',session='s',model=str(root),
                snapshot_path=str(root/'snapshot.json'),gateway_log=str(root/'gateway.jsonl'),
                execution_telemetry_path=str(telemetry),arm='buffer'))
            try:
                g.est=CapacityEstimate(20.,0.,30.,'audio','p','test')
                g.estimator.estimate=g.est
                g.window_start=g.last_capacity_check=0.
                g.previous_telemetry=dict(steps=0,full_steps=0,held_steps=0)
                telemetry.write_text(json.dumps(dict(steps=100,full_steps=100,held_steps=100,
                    written_at=8.,session='s',capacity_observed_s=8.,capacity_hardware_valid_s=8.,capacity_backend_pressure_s=8.)))
                g.update_capacity(8.)
                self.assertEqual(g.est.gamma,20.)
                self.assertEqual(g.window_start,0.)
                self.assertEqual(g.previous_telemetry['steps'],0)
                g.window_units=100.
                telemetry.write_text(json.dumps(dict(steps=120,full_steps=120,held_steps=120,
                    written_at=10.,session='s',capacity_observed_s=10.,capacity_hardware_valid_s=10.,capacity_backend_pressure_s=10.)))
                g.update_capacity(10.)
                self.assertAlmostEqual(g.est.gamma,18.)  # EWMA of 20 and 100/10, not 100/2
                self.assertAlmostEqual(g.est.admission_bound,18.)  # no extra default admission haircut
                self.assertEqual(g.window_start,10.)
                events=[json.loads(x) for x in (root/'gateway.jsonl').read_text().splitlines()]
                self.assertTrue(events[0]['deferred_empty_window'])
                self.assertTrue(events[1]['updated'])
            finally:g.log.close()

    def test_admission_gating_alone_cannot_lower_measured_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            telemetry=root/'execution.json'
            g=Gateway(dict(modality='audio',session='s',model=str(root),
                snapshot_path=str(root/'snapshot.json'),gateway_log=str(root/'gateway.jsonl'),
                execution_telemetry_path=str(telemetry),arm='buffer'))
            try:
                g.w.admission_guard=None  # isolate capacity-estimator evidence in this test
                g.est=CapacityEstimate(20.,0.,30.,'audio','p','test',17.)
                g.estimator.estimate=g.est
                for i in range(20):g.w.add(str(i),1.)
                for _ in range(17):self.assertIsNotNone(g.w.admit_next(g.est,1.))
                self.assertIsNone(g.w.admit_next(g.est,1.))
                g.window_start=g.last_capacity_check=0.;g.window_units=136.
                g.previous_telemetry=dict(steps=0,full_steps=0,held_steps=0,buffer_steps=0)
                telemetry.write_text(json.dumps(dict(steps=100,full_steps=0,held_steps=20,
                    buffer_steps=100,written_at=8.,session='s')))
                g.update_capacity(8.)
                self.assertEqual(g.est.observed_at,0.)
                self.assertEqual(g.est.gamma,20.)
                event=json.loads((root/'gateway.jsonl').read_text().splitlines()[-1])
                self.assertFalse(event['saturated'])
                self.assertTrue(event['capacity_limited'])
                self.assertEqual(event['saturation_basis'],'underfilled')
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

    async def test_legacy_audio_startup_setting_cannot_delay_first_chunk(self):
        # Even the complete 100 ms stream is below the old 200 ms threshold.
        await self.run_modality('audio', legacy_startup_s=.2)


if __name__=='__main__':unittest.main()
