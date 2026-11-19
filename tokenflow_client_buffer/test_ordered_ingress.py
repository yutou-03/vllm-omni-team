import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

from .ordered_ingress import ordered_requests
from .audit_ingress import audit_pair


class OrderedIngressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.trace=self.root/'stage_events.jsonl';self.trace.write_text('')
        self.rows=[dict(arrival_s=0,payload=dict(omni_scheduling=dict(source_request_id=str(i)))) for i in range(4)]

    async def test_ack_order_is_fixed_without_waiting_for_first_output(self):
        # All stream bodies remain blocked until the last POST arrives.
        # A dispatcher waiting for first output or completion would deadlock.
        sent=[];release=asyncio.Event();write_lock=asyncio.Lock()
        self.trace.write_text(json.dumps(dict(event='stage_enqueue',stage_id=0,sched_source_request_id='0',monotonic=0))+'\n')
        async def launch(row):
            rid=row['payload']['omni_scheduling']['source_request_id'];sent.append(rid)
            await asyncio.sleep(.01)
            e=dict(event='stage_enqueue',stage_id=0,sched_source_request_id=rid,request_id='engine-'+rid,monotonic=time.monotonic())
            # Partial writes must not be mistaken for corrupt JSON.
            wire=json.dumps(e)
            async with write_lock:
                with self.trace.open('a') as f:f.write(wire[:20])
                await asyncio.sleep(.01)
                with self.trace.open('a') as f:f.write(wire[20:]+'\n')
            if len(sent)==4:release.set()
            await release.wait()
            return rid
        results=await asyncio.wait_for(ordered_requests(self.rows,launch=launch,trace_path=self.trace,output=self.root,origin=time.monotonic(),timeout=1),2)
        self.assertEqual(sent,['0','1','2','3']);self.assertEqual(results,sent)
        records=[json.loads(l) for l in (self.root/'ingress_order.jsonl').read_text().splitlines()]
        self.assertEqual([r['request_id'] for r in records],sent)
        self.assertLess(max(r['launched_monotonic'] for r in records), records[0]['acknowledged_monotonic'])

    async def test_missing_ack_fails_and_cancels_inflight_stream(self):
        cancelled=asyncio.Event()
        async def launch(row):
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        with self.assertRaises(TimeoutError):
            await ordered_requests(self.rows,launch=launch,trace_path=self.trace,output=self.root,origin=time.monotonic(),timeout=.02)
        self.assertTrue(cancelled.is_set())

    async def test_wrong_engine_order_is_rejected(self):
        async def launch(row):
            with self.trace.open('a') as f:f.write(json.dumps(dict(event='stage_enqueue',stage_id=0,sched_source_request_id='1',monotonic=time.monotonic()))+'\n')
            await asyncio.Event().wait()
        with self.assertRaisesRegex(RuntimeError,'order mismatch'):
            await ordered_requests(self.rows,launch=launch,trace_path=self.trace,output=self.root,origin=time.monotonic(),timeout=1)

    async def test_schedule_uses_arrival_then_stable_workload_order(self):
        self.rows[0]['arrival_s']=.02
        async def launch(row):
            rid=row['payload']['omni_scheduling']['source_request_id']
            with self.trace.open('a') as f:f.write(json.dumps(dict(event='stage_enqueue',stage_id=0,sched_source_request_id=rid,monotonic=time.monotonic()))+'\n')
            return rid
        result=await ordered_requests(self.rows,launch=launch,trace_path=self.trace,output=self.root,origin=time.monotonic(),timeout=1)
        records=[json.loads(l) for l in (self.root/'ingress_order.jsonl').read_text().splitlines()]
        self.assertEqual([r['request_id'] for r in records],['1','2','3','0'])
        self.assertEqual(result,['0','1','2','3'])

    async def test_client_http_path_keeps_streams_concurrent_and_records_engine_order(self):
        from argparse import Namespace
        from aiohttp import web
        from .client import run
        from .stream import clock_domain
        accepted=[];release=asyncio.Event();ingress=asyncio.Condition()
        async def contract(request):
            return web.json_response(dict(session='s',modality='text',clock_domain=clock_domain(),arm='fcfs'))
        async def feedback(request):
            return web.json_response(dict(accepted=True))
        async def stream(request):
            payload=await request.json();rid=payload['omni_scheduling']['source_request_id']
            async with ingress:
                await ingress.wait_for(lambda: payload['omni_scheduling']['ingress_order'] == len(accepted))
                accepted.append(rid)
                ingress.notify_all()
            with self.trace.open('a') as f:
                f.write(json.dumps(dict(event='stage_enqueue',stage_id=0,sched_source_request_id=rid,
                    request_id='engine-'+rid,monotonic=time.monotonic()))+'\n')
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            if len(accepted)==4:release.set()
            await release.wait()  # headers alone do not deliver any model output
            chunk=dict(choices=[dict(index=0,delta=dict(content='word '),token_ids=[100])])
            await response.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
            await response.write_eof();return response
        app=web.Application();app.router.add_get('/contract',contract)
        app.router.add_post('/feedback',feedback);app.router.add_post('/v1/chat/completions',stream)
        runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        try:
            for row in self.rows:
                row['payload']['stream']=True
                row['payload']['omni_scheduling'].update(schema_version=2,request_path='text',text_consume_rate_tps=1000.)
            workload=self.root/'workload.jsonl';workload.write_text('\n'.join(json.dumps(r) for r in self.rows))
            (self.root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            args=Namespace(workload=str(workload),output=str(self.root/'client'),model=str(self.root),timeout=5.,
                endpoint='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]),
                ingress_trace=str(self.trace),ingress_ack_timeout=1.)
            results=await asyncio.wait_for(run(args),4.)
            self.assertEqual(accepted,['0','1','2','3'])
            self.assertTrue(all(r['status']=='success' for r in results))
            manifest=json.loads((self.root/'client/manifest.json').read_text())
            self.assertEqual(manifest['ingress_order_policy'],'server_order_open_loop')
            self.assertEqual(len((self.root/'client/ingress_order.jsonl').read_text().splitlines()),4)
        finally:
            release.set();await runner.cleanup()


class IngressAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name,origin in [('a',10.),('b',100.)]:
            p=self.root/name/'client';p.mkdir(parents=True)
            (p/'manifest.json').write_text(json.dumps(dict(ingress_order_policy='stage0_acknowledged',planned_ingress_order=['r'],workload_sha256='same',origin=origin)))
            (p/'ingress_order.jsonl').write_text(json.dumps(dict(rank=0,stage_id=0,request_id='r',scheduled_monotonic=origin,launched_monotonic=origin+.1,enqueued_monotonic=origin+.2,acknowledged_monotonic=origin+.3))+'\n')

    def test_matching_order_and_relative_arrivals_pass(self):
        self.assertTrue(audit_pair(self.root/'a',self.root/'b')['valid'])

    def test_order_alone_does_not_hide_arrival_timing_drift(self):
        p=self.root/'b/client/ingress_order.jsonl';r=json.loads(p.read_text())
        r.update(enqueued_monotonic=102.,acknowledged_monotonic=102.1);p.write_text(json.dumps(r)+'\n')
        self.assertFalse(audit_pair(self.root/'a',self.root/'b')['valid'])

    def test_incomplete_order_rejected(self):
        (self.root/'b/client/ingress_order.jsonl').write_text('')
        with self.assertRaises(ValueError):audit_pair(self.root/'a',self.root/'b')


if __name__=='__main__':unittest.main()
