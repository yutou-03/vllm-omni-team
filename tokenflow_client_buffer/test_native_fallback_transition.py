"""Exercise the installed vLLM Scheduler.schedule, not a queue-only mock."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from .scheduler import ClientBufferScheduler
from .test_core import request


class NativeFallbackTransitionTests(unittest.TestCase):
    def make(self):
        from vllm.v1.core.sched.scheduler import Scheduler, PauseState, SchedulingPolicy
        from vllm.v1.request import RequestStatus
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'snapshot.json'
        owners = [request('late',2),request('first',0),request('middle',1)]
        for r in owners:
            r.status=RequestStatus.RUNNING
            r.num_tokens_with_spec=11; r.has_encoder_inputs=False
            r.all_token_ids=list(range(11)); r.use_structured_output=False
        s=Scheduler.__new__(Scheduler)
        s.running=owners.copy();s.waiting=[];s.skipped_waiting=[]
        s.requests={r.request_id:r for r in owners}
        s.max_num_running_reqs=2;s.max_num_scheduled_tokens=64;s.max_num_encoder_input_tokens=0
        s._pause_state=PauseState.UNPAUSED;s.policy=SchedulingPolicy.FCFS
        s.max_model_len=4096;s.scheduler_config=NS(long_prefill_token_threshold=0,async_scheduling=False)
        s.need_mamba_block_aligned_split=False;s.num_lookahead_tokens=0;s.lora_config=None
        s.kv_cache_config=NS(kv_cache_groups=[]);s.use_v2_model_runner=False;s.use_pp=False
        s.prev_step_scheduled_req_ids={'late','middle'};s.needs_kv_cache_zeroing=False
        s.finished_req_ids=set();s.connector=None;s.ec_connector=None
        s.encoder_cache_manager=NS(get_freed_mm_hashes=lambda:[])
        s.kv_cache_manager=NS(new_step_starts=lambda:None,
            allocate_slots=lambda *a,**k:NS(get_block_ids=lambda **kw:([],)),
            get_num_common_prefix_blocks=lambda rid:[])
        s._omni_stage_id_for_trace=lambda:1
        path.write_text(json.dumps(dict(schema_version=1,session='s',modality='text',written_at=10.,
            mode='fcfs',reason='capacity_unknown',gamma=None,demand=3.,members={},feedback={})))
        adapter=ClientBufferScheduler(s,dict(snapshot_path=str(path),session='s',modality='text',resume_to_client_s=.5),lambda *a,**k:None,clock=lambda:10.)
        adapter.active_ids={'late','middle'}
        return s,adapter,owners

    def test_raw_excess_residency_reproduces_native_assertion(self):
        s,_,_=self.make()
        with self.assertRaises(AssertionError):s.schedule()

    def test_last_pause_gone_does_not_pass_excess_residents_to_native(self):
        s,a,owners=self.make()
        self.assertFalse(a.pauses)
        with a.scheduling_context():
            output=s.schedule()
            self.assertEqual(set(output.num_scheduled_tokens),{'late','middle'})
            self.assertEqual(len(s.running),2)
            self.assertEqual(s.max_num_running_reqs,2)
        self.assertEqual(len(s.running),3)
        self.assertEqual(a.active_ids,{'late','middle'})
        self.assertEqual(owners[1].num_computed_tokens,10)
        self.assertEqual(owners[0].num_computed_tokens,11)
        self.assertFalse(output.preempted_req_ids)
        # One current owner finishes. Native handoff is now safe and the
        # retained request continues from its computed position without prefill.
        s.running.remove(owners[0]);s.requests.pop('late')
        with a.scheduling_context():
            next_output=s.schedule()
        self.assertIsNone(a.active_ids)
        self.assertEqual(next_output.num_scheduled_tokens,{'first':1})
        self.assertFalse(next_output.scheduled_new_reqs)
        self.assertFalse(next_output.preempted_req_ids)


if __name__=='__main__':unittest.main()
