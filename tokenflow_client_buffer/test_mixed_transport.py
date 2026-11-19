import asyncio
import base64
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import wave
from aiohttp import ClientSession, web
from .client import consume_request
from .mixed_gateway import MixedGateway


class MixedTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_diagnostic_controls_keep_capacity_and_preemption_separate(self):
        for safety,capacity_only,count in ((.85,True,25),(1.,False,30)):
            with self.subTest(safety=safety),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
                profile=dict(thinker=dict(raw_capacity=1000.,profile_id='t'),audio=dict(raw_capacity=30.,profile_id='a'),
                    audio_thinker_tokens_per_media_s=4.,text_thinker_tokens_per_visible_token=1.)
                (root/'profile.json').write_text(json.dumps(profile))
                outputs={m:dict(session='s-'+m,modality=m,model=str(root),
                    snapshot_path=str(root/(m+'-snapshot.json')),gateway_log=str(root/(m+'-gateway.jsonl')),
                    execution_telemetry_path=str(root/(m+'-execution.json')),upstream='http://127.0.0.1:1',
                    native_execution_limit=32,capacity_only=capacity_only) for m in ('text','audio')}
                g=MixedGateway(dict(profile=str(root/'profile.json'),arm='buffer',outputs=outputs,
                    native_execution_limit=32,admission_safety=safety,resource_log=str(root/'resources.jsonl')))
                try:
                    self.assertEqual(g.book.admission_policy,'resource_fifo')
                    now=time.monotonic();view=g.book.views['audio']
                    view.admission_guard=None  # isolate capacity-only arithmetic here
                    for i in range(40):view.add(str(i),1.)
                    for _ in range(count):self.assertIsNotNone(view.admit_next(g.book.capacity('audio',now),now))
                    next_id=view.admit_next(g.book.capacity('audio',now),now)
                    if capacity_only:
                        self.assertIsNone(next_id)
                    else:
                        self.assertEqual(next_id,str(count))
                        self.assertEqual(view.requests[next_id].state,'native_queued')
                    g.gateways['audio'].publish();state=json.loads((root/'audio-snapshot.json').read_text())
                    self.assertEqual(len(state['members']),count)
                    self.assertEqual(state['mode'],'fcfs' if capacity_only else 'buffer')
                    self.assertEqual(state['reason'],'capacity_only' if capacity_only else 'capacity_feasible')
                    if not capacity_only:
                        self.assertEqual(state['buffer_reclaimed_ids'],[str(count)])
                    self.assertEqual(state['resource_gamma']['audio'],30.)
                    self.assertLessEqual(state['demand'],state['gamma'])
                finally:
                    for output in g.gateways.values():output.log.close()
                    g.log.close()

    def test_mixed_admission_backlog_is_not_execution_saturation(self):
        from types import SimpleNamespace
        from .core import CapacityEstimator
        from .mixed_core import MixedBook
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'telemetry.json'
            estimator=CapacityEstimator('audio',safety=.85,ttl_s=900.,min_window_s=8.)
            audio=estimator.seed(20.,0.,'audio')
            other=CapacityEstimator('text',ttl_s=900.).seed(1000.,0.,'thinker')
            g=MixedGateway.__new__(MixedGateway)
            g.estimators={'audio':estimator};g.book=MixedBook(other,audio,4.)
            for i in range(20):g.book.views['audio'].add(str(i),1.)
            for _ in range(17):self.assertIsNotNone(g.book.views['audio'].admit_next(g.book.capacity('audio',0.),0.))
            previous=dict(steps=0,full_steps=0,buffer_steps=0)
            g.windows={'audio':dict(start=0.,previous=previous,units=0.,last_check=0.)}
            g.gateways={'audio':SimpleNamespace(config={'execution_telemetry_path':str(path)},session='audio',window_units=136.)}
            events=[];g.event=lambda name,**kw:events.append(kw)
            path.write_text(json.dumps(dict(steps=100,full_steps=0,buffer_steps=100,written_at=8.,session='audio')))
            g.observe('audio',8.)
            self.assertEqual(estimator.estimate.gamma,20.)
            self.assertEqual(estimator.estimate.observed_at,0.)
            self.assertFalse(events[-1]['saturated'])

    async def test_two_output_clients_share_admission_but_not_buffer_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            wav=io.BytesIO()
            with wave.open(wav,'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(b'\0\0'*1200)
            encoded=base64.b64encode(wav.getvalue()).decode()
            captured=[]
            async def upstream(request):
                p=await request.json();captured.append(p)
                m=p['omni_scheduling']['request_path']
                r=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await r.prepare(request)
                for i in range(2):
                    e=dict(modality=m,choices=[dict(index=0,delta={'content':encoded if m=='audio' else 'word '},token_ids=[100+i])])
                    await r.write(('data: '+json.dumps(e)+'\n\n').encode());await asyncio.sleep(.01)
                await r.write(b'data: [DONE]\n\n');await r.write_eof();return r
            app=web.Application();app.router.add_post('/v1/chat/completions',upstream)
            r1=web.AppRunner(app);await r1.setup();site=web.TCPSite(r1,'127.0.0.1',0);await site.start()
            port=site._server.sockets[0].getsockname()[1]
            profile=dict(thinker=dict(raw_capacity=100.,profile_id='text-cal'),audio=dict(raw_capacity=20.,profile_id='audio-cal'),
                audio_thinker_tokens_per_media_s=4.,text_thinker_tokens_per_visible_token=1.)
            (root/'profile.json').write_text(json.dumps(profile))
            outputs={m:dict(session='s-'+m,modality=m,model=str(root),
                snapshot_path=str(root/(m+'-snapshot.json')),gateway_log=str(root/(m+'-gateway.jsonl')),
                execution_telemetry_path=str(root/(m+'-execution.json')),upstream=f'http://127.0.0.1:{port}',
                native_execution_limit=32) for m in ('text','audio')}
            g=MixedGateway(dict(profile=str(root/'profile.json'),arm='buffer',outputs=outputs,
                native_execution_limit=32,resource_log=str(root/'resources.jsonl')))
            r2=web.AppRunner(g.app());await r2.setup();site2=web.TCPSite(r2,'127.0.0.1',0);await site2.start()
            endpoint='http://127.0.0.1:'+str(site2._server.sockets[0].getsockname()[1])
            try:
                async with ClientSession() as session:
                    async with session.get(endpoint+'/contract') as response:contract=await response.json()
                    rows=[]
                    for m,rate in (('text',10.),('audio',1.)):
                        key='text_consume_rate_tps' if m=='text' else 'audio_consume_rate'
                        rows.append(dict(payload=dict(stream=True,omni_scheduling=dict(schema_version=2,
                            source_request_id=m,request_path=m,**{key:rate}))))
                    origin=time.monotonic()
                    results=await asyncio.gather(*(consume_request(row,session=session,
                        endpoint=endpoint+'/'+row['payload']['omni_scheduling']['request_path'],
                        contract=contract['contracts'][row['payload']['omni_scheduling']['request_path']],
                        output=root,special=[],origin=origin,timeout=5.) for row in rows))
                self.assertTrue(all(r['status']=='success' for r in results),results)
                self.assertTrue(all(r['first_ready_s']==r['startup_s'] for r in results))
                self.assertEqual(g.gateways['text'].feedback,{})
                self.assertEqual(g.gateways['audio'].feedback,{})
                self.assertEqual(g.book.demand,dict(thinker=0.,audio=0.))
                self.assertEqual([p['omni_scheduling']['ingress_order'] for p in captured],[0,1])
                for m in ('text','audio'):
                    events=[json.loads(x) for x in (root/(m+'-gateway.jsonl')).read_text().splitlines()]
                    self.assertTrue(all(e.get('request_id', m) == m for e in events if e['event'] == 'feedback'))
                    for e in events:
                        if e['event']=='working_set' and e['mode']=='buffer':
                            self.assertLessEqual(e['resource_demand']['thinker'],e['resource_gamma']['thinker'])
                            self.assertLessEqual(e['resource_demand']['audio'],e['resource_gamma']['audio'])
            finally:
                await r2.cleanup();await r1.cleanup()

if __name__=='__main__':unittest.main()
