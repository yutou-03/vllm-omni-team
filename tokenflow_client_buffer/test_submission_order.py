import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from .submission_order import SubmissionOrder, submit_benchmark_message

class SubmissionOrderTests(unittest.TestCase):
    def test_all_requests_enqueued_without_any_engine_ack(self):
        s=SubmissionOrder();out=[]
        for i in reversed(range(64)):s.submit('case',i,dict(request_id=str(i)),out.append)
        self.assertEqual([x['request_id'] for x in out],list(map(str,range(64))))
    def test_duplicate_fails_and_sessions_do_not_block_each_other(self):
        s=SubmissionOrder();out=[]
        s.submit('a',1,dict(request_id='a1'),out.append)
        s.submit('b',0,dict(request_id='b0'),out.append)
        self.assertEqual(len(out),1)
        with self.assertRaises(ValueError):s.submit('a',1,{},out.append)
    def test_aborted_pending_cannot_be_enqueued_later(self):
        s=SubmissionOrder();out=[]
        s.submit('a',1,dict(request_id='a1'),out.append);s.abort(['a1'])
        with self.assertRaises(RuntimeError):s.submit('a',0,dict(request_id='a0'),out.append)
        self.assertEqual(out,[])
    def test_opt_in_server_hook_and_normal_traffic(self):
        out=[];e=NS(prompt_expand_func=None,request_queue=NS(sync_q=NS(put_nowait=out.append)))
        def msg(rank):return dict(request_id=str(rank),original_prompt=dict(additional_information=dict(meta=dict(sched_service_class='tokenflow_benchmark:case',sched_ingress_order=rank))))
        with patch.dict('os.environ',{'VLLM_OMNI_BENCHMARK_ORDERED_SUBMIT':'0'}):self.assertFalse(submit_benchmark_message(e,msg(0)))
        with patch.dict('os.environ',{'VLLM_OMNI_BENCHMARK_ORDERED_SUBMIT':'1'}):
            self.assertFalse(submit_benchmark_message(e,dict(original_prompt={})))
            self.assertTrue(submit_benchmark_message(e,msg(1)))
            self.assertEqual(out,[])
            self.assertTrue(submit_benchmark_message(e,msg(0)))
        self.assertEqual([x['request_id'] for x in out],['0','1'])
