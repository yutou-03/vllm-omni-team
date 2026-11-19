"""Native queue and HTTP registration regressions for output-stage admission."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from .engine_queue import engine_queue_admission_context


def request(rid, state='WAITING', priority=0):
    return NS(request_id=rid,status=NS(name=state),priority=priority,arrival_time=int(rid[-1]) if rid[-1].isdigit() else 0)


class PublicQueue:
    """Expose only the native RequestQueue API used by the helper."""
    def __init__(self, rows): self.rows=list(rows)
    def __iter__(self): return iter(self.rows)
    def __bool__(self): return bool(self.rows)
    def pop_request(self): return self.rows.pop(0)
    def add_request(self, r): self.rows.append(r)
    def prepend_request(self,r): self.rows.insert(0,r)


class EngineQueueTests(unittest.TestCase):
    def make(self, running, waiting):
        return NS(running=running,waiting=waiting,requests={r.request_id:r for r in running+list(waiting)})

    def snapshot(self, registered, active=(), paused=(), enabled=True):
        return dict(engine_queue_admission=enabled,registered_ids=registered,
                    members={r:{} for r in active},paused_members={r:{} for r in paused})

    def test_waiting_hides_pending_and_preserves_native_preemption_order(self):
        a,b,c,audio=[request(x) for x in ('a','b','c','audio')]
        old=request('old','RUNNING');q=PublicQueue([a,b,c,audio]);s=self.make([old],q)
        with engine_queue_admission_context(s,self.snapshot(['a','b','c'],['b']),lambda r:r.request_id):
            self.assertEqual(list(q),[b,audio])
            self.assertIs(q.pop_request(),b);b.status.name='RUNNING';s.running.append(b)
            s.running.remove(old);old.status.name='PREEMPTED';q.prepend_request(old)
        self.assertIs(s.waiting,q)
        self.assertEqual(list(q),[old,a,c,audio])
        self.assertEqual(s.running,[b])

    def test_all_hidden_keep_original_order_and_other_modality_passes(self):
        a,b,audio=[request(x) for x in ('a','b','audio')];q=PublicQueue([a,b]);s=self.make([audio],q)
        with engine_queue_admission_context(s,self.snapshot(['a','b']),lambda r:r.request_id):
            self.assertFalse(q);self.assertEqual(s.running,[audio])
        self.assertEqual(list(q),[a,b])

    def test_running_restore_keeps_paused_candidate_and_never_resurrects_terminal(self):
        a,b,p=[request(x,'RUNNING') for x in ('a','b','p')];s=self.make([a,b,p],[])
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            with engine_queue_admission_context(s,self.snapshot(['a','b','p'],['b'],['p']),lambda r:r.request_id):
                self.assertEqual(s.running,[b,p])
                a.status.name='FINISHED_ABORTED'
                raise RuntimeError('native failure')
        self.assertEqual(s.running,[b,p])

    def test_native_fcfs_flag_is_strict_noop(self):
        a,b=request('a'),request('b');q=PublicQueue([b,a]);s=self.make([],q)
        with engine_queue_admission_context(s,self.snapshot(['a','b'],enabled=False),lambda r:r.request_id):
            self.assertEqual(list(q),[b,a]);self.assertIs(s.waiting,q)
        self.assertEqual(list(q),[b,a])

    def test_ack_handshake_hides_all_waiting_even_without_engine_mode(self):
        a,b=request('a'),request('b');q=PublicQueue([a,b]);s=self.make([],q)
        with engine_queue_admission_context(s,{},lambda r:r.request_id,hide_all_waiting=True) as mask:
            self.assertFalse(q)
            self.assertEqual(mask['hidden_waiting_ids'],{'a','b'})
            self.assertEqual(mask['hidden_source_ids'],{'a','b'})
        self.assertEqual(list(q),[a,b])

    def test_large_queue_restores_order_after_native_admission(self):
        rows=[request('r'+str(i)) for i in range(1000)]
        q=PublicQueue(rows);s=self.make([],q)
        allowed=[r.request_id for r in rows[::10]]
        with engine_queue_admission_context(s,self.snapshot([r.request_id for r in rows],allowed),lambda r:r.request_id) as mask:
            self.assertEqual(list(q),rows[::10])
            self.assertEqual(len(mask['excluded_ids']),900)
            admitted=q.pop_request();admitted.status.name='RUNNING';s.running.append(admitted)
            preempted=request('new','PREEMPTED');s.requests['new']=preempted;q.prepend_request(preempted)
        self.assertEqual(list(q),[preempted]+rows[1:])

    def test_extra_blocked_waiter_is_hidden_without_engine_flag(self):
        a,b=request('a','PREEMPTED'),request('b');q=PublicQueue([a,b]);s=self.make([],q)
        with engine_queue_admission_context(s,{},lambda r:r.request_id,blocked_waiting_ids={'a'}) as mask:
            self.assertEqual(list(q),[b]);self.assertEqual(mask['excluded_ids'],{'a'})
        self.assertEqual(list(q),[a,b])

    def test_skipped_waiting_cannot_bypass_engine_admission(self):
        a,b,c=[request(x) for x in ('a','b','c')];s=self.make([],PublicQueue([a]))
        s.skipped_waiting=PublicQueue([b,c]);s.requests.update({r.request_id:r for r in (b,c)})
        snapshot=self.snapshot(['a','b','c'],['a','c'])
        with engine_queue_admission_context(s,snapshot,lambda r:r.request_id) as mask:
            self.assertEqual(list(s.skipped_waiting),[c])
            self.assertEqual(mask['hidden_waiting_ids'],{'b'})
            selected=s.skipped_waiting.pop_request();selected.status.name='RUNNING';s.running.append(selected)
        self.assertEqual(list(s.waiting),[a]);self.assertEqual(list(s.skipped_waiting),[b])
        with engine_queue_admission_context(s,{},lambda r:r.request_id,hide_all_waiting=True) as mask:
            self.assertFalse(s.waiting);self.assertFalse(s.skipped_waiting)
            self.assertEqual(mask['excluded_ids'],{'a','b'})
        with engine_queue_admission_context(s,{},lambda r:r.request_id,blocked_waiting_ids={'b'}):
            self.assertEqual(list(s.waiting),[a]);self.assertFalse(s.skipped_waiting)
        self.assertEqual(list(s.skipped_waiting),[b])

    def test_native_fcfs_request_queue_identity_and_order(self):
        try:
            from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        except ImportError:
            self.skipTest('native vLLM RequestQueue unavailable')
        a,b,c=[request(x) for x in ('a','b','c')];q=FCFSRequestQueue([a,b,c]);s=self.make([],q)
        with engine_queue_admission_context(s,self.snapshot(['a','b','c'],['b']),lambda r:r.request_id):
            self.assertEqual(list(q),[b]);q.pop_request();b.status.name='RUNNING';s.running.append(b)
        self.assertIs(s.waiting,q);self.assertEqual(list(q),[a,c])


class EngineQueueTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_request_reaches_upstream_before_target_quota(self):
        from aiohttp import ClientSession, web
        from .core import CapacityEstimate
        from .gateway import Gateway
        import time
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
            reached=asyncio.Event();captured={};g=None
            async def upstream(req):
                payload=await req.json();rid=payload['omni_scheduling']['source_request_id']
                captured.update(state=g.w.requests[rid].state,snapshot=json.loads(g.snapshot.read_text()))
                reached.set()
                await g.ready[rid].wait()
                response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(req)
                event=dict(modality='text',choices=[dict(index=0,delta={'content':'word'},token_ids=[100])])
                await response.write(('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode())
                await response.write_eof();return response
            app=web.Application();app.router.add_post('/v1/chat/completions',upstream)
            r1=web.AppRunner(app);await r1.setup();site=web.TCPSite(r1,'127.0.0.1',0);await site.start()
            g=Gateway(dict(modality='text',session='s',arm='buffer',model=str(root),engine_queue_admission=True,
                snapshot_path=str(root/'snapshot.json'),gateway_log=str(root/'gateway.jsonl'),
                upstream='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]),max_inflight=8))
            now=time.monotonic();g.est=CapacityEstimate(1.,now,now+60.,'text','p','test')
            g.w.add('holder',1.);self.assertEqual(g.w.admit_next(g.est,now),'holder')
            app2=web.Application();app2.router.add_post('/v1/chat/completions',g.stream)
            app2.on_startup.append(g.startup);app2.on_cleanup.append(g.cleanup)
            r2=web.AppRunner(app2);await r2.setup();site2=web.TCPSite(r2,'127.0.0.1',0);await site2.start()
            payload=dict(stream=True,omni_scheduling=dict(schema_version=2,source_request_id='new',request_path='text',text_consume_rate_tps=1.))
            async def run_client():
                async with ClientSession() as client:
                    async with client.post('http://127.0.0.1:'+str(site2._server.sockets[0].getsockname()[1])+'/v1/chat/completions',json=payload,headers={'X-TokenFlow-Session':'s'}) as response:
                        return response.status,await response.text()
            task=asyncio.create_task(run_client())
            try:
                await asyncio.wait_for(reached.wait(),2.)
                self.assertEqual(captured['state'],'waiting')
                self.assertTrue(captured['snapshot']['engine_queue_admission'])
                self.assertIn('new',captured['snapshot']['registered_ids'])
                self.assertNotIn('new',captured['snapshot']['members'])
                self.assertFalse(g.ready['new'].is_set())
                g.w.complete('holder')
                status,body=await asyncio.wait_for(task,2.)
                self.assertEqual(status,200);self.assertIn('[DONE]',body)
            finally:
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                await r2.cleanup();await r1.cleanup()

    async def test_mixed_snapshot_is_route_scoped_and_native_disabled(self):
        from .mixed_gateway import MixedGateway
        for arm in ('buffer','fcfs'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
                (root/'profile.json').write_text(json.dumps(dict(thinker=dict(raw_capacity=100.,profile_id='t'),
                    audio=dict(raw_capacity=20.,profile_id='a'),audio_thinker_tokens_per_media_s=4.,text_thinker_tokens_per_visible_token=1.)))
                outputs={m:dict(session='s-'+m,modality=m,model=str(root),snapshot_path=str(root/(m+'.json')),
                    gateway_log=str(root/(m+'.jsonl')),execution_telemetry_path=str(root/(m+'-exec.json')),upstream='http://127.0.0.1:1',native_execution_limit=32) for m in ('text','audio')}
                g=MixedGateway(dict(profile=str(root/'profile.json'),arm=arm,engine_queue_admission=True,outputs=outputs,native_execution_limit=32,resource_log=str(root/'resources.jsonl')))
                try:
                    g.book.views['audio'].add('audio-a',1.)
                    for m,output in g.gateways.items():
                        output.publish();state=json.loads(output.snapshot.read_text())
                        self.assertEqual(state['engine_queue_admission'],arm=='buffer')
                        self.assertEqual(state['registered_ids'],['audio-a'] if m=='audio' and arm=='buffer' else [])
                finally:
                    for output in g.gateways.values():output.log.close()
                    g.log.close()

if __name__=='__main__':unittest.main()
