"""Capacity infeasibility delegates admission to native memory-aware FCFS."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from .mixed_core import MixedBook
from .test_mixed import estimate
from .engine_queue import engine_queue_admission_context
from .test_engine_queue import PublicQueue, request
from .test_recovery_barrier import MixedRecoveryBarrierTests


class CapacityFallbackTests(unittest.TestCase):
    def test_r47_audio_does_not_stop_at_52_and_call_it_feasible(self):
        book=MixedBook(estimate(1000.,'text'),estimate(26.151332,'audio'),4.22,
                       native_limit=64)
        audio=book.views['audio']
        for i in range(100):audio.add(str(i),.5)
        handed=[]
        for _ in range(100):
            handed.append(audio.admit_next(book.capacity('audio',1.),1.))
        self.assertEqual(handed,[str(i) for i in range(100)])
        self.assertEqual(len(book.members),52)
        self.assertEqual(len(book.native_queued),48)
        self.assertEqual(audio.update(book.capacity('audio',1.),1.),'buffer')
        self.assertIsNone(book.reason(1.))
        self.assertEqual(book.demand['audio'],26.)
        self.assertEqual(book.service_demand()['audio'],26.)
        self.assertEqual(book.service_demand(include_waiting=True)['audio'],50.)
        self.assertEqual(book.native_limit,64)

    def test_pauses_do_not_clear_overload_or_destroy_recovery_identity(self):
        book=MixedBook(estimate(1000.,'text'),estimate(1.,'audio'),4.)
        a=book.views['audio']
        for rid in ('a','b','new'):a.add(rid,.5)
        for _ in range(3):a.admit_next(book.capacity('audio',1.),1.)
        a.pause('a');a.pause('b')
        self.assertEqual(book.demand['audio'],0.)
        self.assertEqual(a.update(book.capacity('audio',2.),2.),'buffer')
        self.assertEqual(book.service_demand()['audio'],1.)
        self.assertTrue(a.resume('a'))
        self.assertEqual(book.service_demand()['audio'],1.)
        a.complete('a')
        self.assertTrue(book.mark_native_execution('audio','new'))
        self.assertEqual(a.update(book.capacity('audio',3.),3.),'buffer')
        self.assertEqual(a.requests['b'].state,'paused')
        self.assertEqual(a.requests['new'].state,'admitted')

    def test_capacity_recovery_can_exit_without_draining_whole_cohort(self):
        book=MixedBook(estimate(100.,'text'),estimate(1.,'audio'),4.,native_limit=1)
        a=book.views['audio']
        for rid in ('a','b','c'):a.add(rid,1.)
        for _ in range(3):a.admit_next(book.capacity('audio',1.),1.)
        self.assertTrue(book.native_fcfs)
        book.estimates['audio']=estimate(4.,'audio')
        self.assertEqual(a.update(book.capacity('audio',2.),2.),'buffer')
        self.assertEqual([r.request_id for r in book.native_queued], ['b','c'])
        self.assertEqual(len(book.members),1)
        for rid in ('b','c'):self.assertTrue(book.mark_native_execution('audio',rid))
        self.assertEqual(a.update(book.capacity('audio',3.),3.),'buffer')
        self.assertEqual(len(book.members),3)  # W is not capped to max_num_seqs.
        self.assertEqual(book.native_limit,1)

    def test_actual_overload_and_unknown_capacity_keep_native_ownership(self):
        book=MixedBook(estimate(100.,'text'),estimate(1.,'audio'),4.)
        a=book.views['audio']
        for rid in ('a','b','c'):
            a.add(rid,1.)
            a.admit_next(book.capacity('audio',1.),1.)
        book.mark_native_execution('audio','b')
        a.pause('a')
        self.assertEqual(a.update(book.capacity('audio',2.),2.),'fcfs')
        self.assertEqual(book.reason(2.),'overload')
        self.assertEqual(book.service_demand()['audio'],2.)
        a.complete('a'); a.complete('b')
        book.estimates['audio']=None
        self.assertEqual(a.update(book.capacity('audio',3.),3.),'fcfs')
        self.assertEqual(book.reason(3.),'capacity_unknown')
        self.assertEqual(a.requests['c'].state,'native_queued')
        book.estimates['audio']=estimate(4.,'audio')
        self.assertEqual(a.update(book.capacity('audio',4.),4.),'buffer')
        self.assertEqual(book.service_demand()['audio'],0.)
        # A delayed execution event after policy takeover is still counted once.
        self.assertTrue(book.mark_native_execution('audio','c'))
        self.assertFalse(book.mark_native_execution('audio','c'))
        self.assertEqual(book.service_demand()['audio'],1.)
        a.complete('c')
        self.assertFalse(book.mark_native_execution('audio','c'))

    def test_buffer_takeover_preserves_fifo_and_ack_masks(self):
        rows=[request('r'+str(i)) for i in range(4)]
        q=PublicQueue(rows)
        scheduler=NS(running=[],waiting=q,requests={r.request_id:r for r in rows})
        snap=dict(mode='buffer',engine_queue_admission=True,native_fcfs_admission=False,
                  registered_ids=[r.request_id for r in rows],members={},paused_members={},
                  buffer_reclaimed_ids=['r0','r1','r2'])
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id):
            self.assertEqual(list(q),rows[:3])
        self.assertEqual(list(q),rows)
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id,
                                           blocked_waiting_ids=['r0']):
            self.assertEqual(list(q),rows[1:3])
        self.assertEqual(list(q),rows)
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id,
                                           hide_all_waiting=True):
            self.assertEqual(list(q),[])
        self.assertEqual(list(q),rows)
        snap['mode']='fcfs'  # A stale takeover cannot authorize FCFS admission.
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id):
            self.assertEqual(list(q),[])

    def test_native_queue_handoff_preserves_memory_ownership_and_fifo(self):
        rows=[request('r'+str(i)) for i in range(6)]
        q=PublicQueue(rows)
        scheduler=NS(running=[],waiting=q,requests={r.request_id:r for r in rows},max_num_running_reqs=2)
        snap=dict(mode='fcfs',engine_queue_admission=True,native_fcfs_admission=True,
                  registered_ids=[r.request_id for r in rows],members={},paused_members={},
                  native_fcfs_ids=[r.request_id for r in rows[:5]])
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id):
            self.assertEqual(list(q),rows[:5])
            self.assertEqual(scheduler.running,[])
            self.assertEqual(scheduler.max_num_running_reqs,2)
        self.assertEqual(list(q),rows)
        # Unvalidated/no-mode handoff must not accidentally open the gate.
        snap['mode']='buffer'
        with engine_queue_admission_context(scheduler,snap,lambda r:r.request_id):
            self.assertEqual(list(q),[])


class FallbackGatewayTests(MixedRecoveryBarrierTests):
    async def test_native_handoff_keeps_recovery_barrier_and_publishes_fcfs(self):
        a=self.g.gateways['audio'];self.g.book.estimates['audio']=estimate(.25,'audio')
        # Test clock-based estimates must be valid at this gateway's clock.
        import time
        from .core import CapacityEstimate
        now=time.monotonic()
        self.g.book.estimates['audio']=CapacityEstimate(.25,now-1,now+60,'audio','p','test')
        a.w.add('new',.5);a.ready['new']=asyncio.Event()
        self.telemetry('audio',True);self.telemetry('text',False)
        await self.one_tick(a)
        self.assertFalse(a.ready['new'].is_set())
        self.telemetry('audio',False)
        await self.one_tick(a)
        self.assertTrue(a.ready['new'].is_set())
        self.assertEqual(a.w.requests['new'].state,'native_queued')
        state=json.loads(a.snapshot.read_text())
        self.assertEqual(state['mode'],'buffer')
        self.assertEqual(state['native_fcfs_ids'],[])
        self.assertEqual(state['buffer_reclaimed_ids'],['new'])
        self.assertEqual(state['members'],{})

    async def test_capacity_only_diagnostic_keeps_queued_work_visible(self):
        a=self.g.gateways['audio'];a.config['capacity_only']=True
        a.w.add('new',.5);a.w.requests['new'].state='native_queued'
        a.publish()
        state=json.loads(a.snapshot.read_text())
        self.assertEqual(state['mode'],'fcfs')
        self.assertTrue(state['native_fcfs_admission'])
        self.assertEqual(state['native_fcfs_ids'],['new'])
        self.assertEqual(state['buffer_reclaimed_ids'],[])
        self.assertEqual(state['members'],{})

    def test_execution_journal_partial_line_session_and_terminal_are_safe(self):
        a=self.g.gateways['audio'];a.w.add('new',.5)
        a.w.requests['new'].state='native_queued'
        p=Path(a.config['execution_telemetry_path']+'.admissions.jsonl')
        import time
        now=time.monotonic();a.registered_at['new']=now-1
        event=dict(session=a.session,request_id='new',started_at=now)
        p.write_text(json.dumps({**event,'started_at':now-2})+'\n')
        self.g.process_native_execution()
        self.assertEqual(a.w.requests['new'].state,'native_queued')
        with p.open('a') as f:f.write(json.dumps({**event,'session':'old'})+'\n'+json.dumps(event))
        self.g.process_native_execution()
        self.assertEqual(a.w.requests['new'].state,'native_queued')
        with p.open('a') as f:f.write('\n')
        self.g.process_native_execution()
        self.assertEqual(a.w.requests['new'].state,'admitted')
        a.w.complete('new')
        with p.open('a') as f:f.write(json.dumps(event)+'\n')
        self.g.process_native_execution()
        self.assertNotIn('new', a.w.requests)
        self.assertIn('new', a.w.completed_ids)



class InstalledNativeCapacityTests(unittest.TestCase):
    def native(self, kv_slots):
        from .test_native_fallback_transition import NativeFallbackTransitionTests
        from vllm.v1.request import Request
        from vllm import SamplingParams
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        helper=NativeFallbackTransitionTests();self.addCleanup(helper.doCleanups)
        s,a,_=helper.make()
        rows=[Request(request_id='n'+str(i),prompt_token_ids=[1,2,3],sampling_params=SamplingParams(max_tokens=8),pooling_params=None,arrival_time=float(i))
              for i in range(5)]
        s.running=[];s.waiting=FCFSRequestQueue();s.skipped_waiting=FCFSRequestQueue()
        for request in rows:s.waiting.add_request(request)
        s.requests={r.request_id:r for r in rows};s.prev_step_scheduled_req_ids=set()
        s.scheduler_config.enable_chunked_prefill=True
        s.is_encoder_decoder=False;s.scheduler_reserve_full_isl=False;s.log_stats=False
        s.use_eagle=False
        blocks=NS(get_block_ids=lambda **kw:([],))
        allocated=[]
        def allocate(request,*args,**kwargs):
            if len(allocated)>=kv_slots:return None
            allocated.append(request.request_id);return blocks
        s.kv_cache_manager=NS(new_step_starts=lambda:None,allocate_slots=allocate,
            get_computed_blocks=lambda request:(blocks,0),get_blocks=lambda rid:blocks,
            get_num_common_prefix_blocks=lambda rid:[],empty_kv_cache_blocks=blocks)
        a.active_ids=None
        a.metadata=lambda r:dict(sched_source_request_id=r.request_id,
            sched_request_path='text',sched_ingress_order=r.arrival_time,sched_text_consume_rate_tps=1.)
        a.snapshot_path.write_text(json.dumps(dict(schema_version=1,session='s',modality='text',
            written_at=10.,mode='fcfs',reason='overload',gamma=1.,demand=0.,members={},feedback={},
            engine_queue_admission=True,native_fcfs_admission=True,
            native_fcfs_ids=[r.request_id for r in rows],registered_ids=[r.request_id for r in rows])))
        return s,a,allocated

    def test_native_fcfs_handoff_obeys_sequence_limit(self):
        s,a,allocated=self.native(10)
        with a.scheduling_context():out=s.schedule()
        self.assertEqual(list(out.num_scheduled_tokens),['n0','n1'])
        self.assertEqual(allocated,['n0','n1'])
        self.assertEqual([r.request_id for r in s.waiting],['n2','n3','n4'])
        self.assertEqual(s.max_num_running_reqs,2)

    def test_native_fcfs_handoff_obeys_kv_limit_before_sequence_limit(self):
        s,a,allocated=self.native(1)
        with a.scheduling_context():out=s.schedule()
        self.assertEqual(list(out.num_scheduled_tokens),['n0'])
        self.assertEqual(allocated,['n0'])
        self.assertEqual([r.request_id for r in s.waiting],['n1','n2','n3','n4'])

class InstalledBufferTakeoverTests(InstalledNativeCapacityTests):
    def native(self, kv_slots):
        s,a,allocated=super().native(kv_slots)
        state=json.loads(a.snapshot_path.read_text())
        state.update(mode='buffer',reason='capacity_feasible',gamma=10.,
                     capacity_valid_until=100.,native_fcfs_admission=False,
                     buffer_reclaimed_ids=state.pop('native_fcfs_ids'))
        a.snapshot_path.write_text(json.dumps(state))
        return s,a,allocated


if __name__=='__main__':unittest.main()
