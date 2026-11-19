"""Retained client pauses must not become synthetic native preemptions."""
from collections import deque
from types import SimpleNamespace as NS
import unittest


class ChunkResidencyTests(unittest.TestCase):
    def make(self):
        from vllm.v1.request import RequestStatus
        from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
        adapter = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
        adapter._chunk_wait_queue_lock = __import__("threading").RLock()
        adapter.connector = NS(stage_id=1)
        adapter.scheduler_max_num_seqs = 32
        adapter.waiting_for_chunk_waiting_requests = deque()
        adapter.waiting_for_chunk_running_requests = deque()
        adapter._finished_load_reqs = set()
        adapter._process_chunk_queue = lambda *args: None
        running = [NS(request_id=str(i), status=RequestStatus.RUNNING,
                      num_computed_tokens=100+i) for i in range(54)]
        class Queue(list):
            def prepend_requests(self, requests): self[:0] = requests
        return adapter, Queue(), running

    def test_retained_54_owners_are_not_moved_or_recomputed(self):
        from vllm.v1.request import RequestStatus
        adapter, waiting, running = self.make()
        before = list(running)
        adapter.process_pending_chunks(waiting, running, enforce_running_limit=False)
        self.assertEqual(running, before)
        self.assertFalse(waiting)
        self.assertTrue(all(r.status == RequestStatus.RUNNING for r in running))
        self.assertEqual([r.num_computed_tokens for r in running], list(range(100,154)))
        self.assertEqual(adapter.scheduler_max_num_seqs, 32)

    def test_default_native_limit_behavior_is_unchanged(self):
        from vllm.v1.request import RequestStatus
        adapter, waiting, running = self.make()
        adapter.process_pending_chunks(waiting, running)
        self.assertEqual(len(running),32)
        self.assertEqual(len(waiting),22)
        self.assertTrue(all(r.status == RequestStatus.PREEMPTED for r in waiting))

    def test_old_trimming_reproduces_native_previous_step_assertion(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        adapter, waiting, running = self.make()
        owner = running[-1]
        adapter.process_pending_chunks(waiting, running)
        native = NS(use_pp=False, prev_step_scheduled_req_ids={owner.request_id})
        with self.assertRaises(AssertionError):
            Scheduler._make_cached_request_data(native, [], [owner],
                                                {owner.request_id:1}, {}, {})


if __name__ == '__main__': unittest.main()
