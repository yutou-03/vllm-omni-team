"""A sender's terminal cleanup must not detach scheduler chunk-wait queues."""
from collections import deque
from types import SimpleNamespace as NS
import threading
import unittest

class ChunkQueueRaceTests(unittest.TestCase):
    def test_background_cleanup_preserves_waiting_and_running_batches(self):
        from vllm.v1.request import RequestStatus
        from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
        for source in ('waiting', 'running'):
            with self.subTest(source=source):
                a = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
                a.connector = NS(stage_id=1)
                a.scheduler_max_num_seqs = 64
                a._receiver_buffer = None
                a._drain_state_lock = threading.Lock()
                a._chunk_wait_queue_lock = threading.RLock()
                a._qwen3_talker_conditioning_gate = False
                for name in ('_pending_load_reqs', 'waiting_for_chunk_waiting_requests',
                             'waiting_for_chunk_running_requests'):
                    setattr(a, name, deque())
                for name in ('finished_requests', 'requests_with_ready_chunks',
                             '_qwen3_bootstrap_published', '_qwen3_entity_layout_invalid',
                             '_cancelled_load_reqs', '_finished_load_reqs'):
                    setattr(a, name, set())
                for name in ('get_req_chunk', '_qwen3_received_entity_rows', '_qwen3_prompt_rows',
                             'request_ids_mapping', 'requests_origin_status', 'request_payload'):
                    setattr(a, name, {})
                status = RequestStatus.WAITING if source == 'waiting' else RequestStatus.RUNNING
                requests = [NS(request_id=str(i), status=status) for i in range(6)]
                class Queue(list):
                    def add_request(self, request): self.append(request)
                    def prepend_requests(self, requests): self[:0] = requests
                waiting = Queue(requests if source == 'waiting' else [])
                running = list(requests if source == 'running' else [])
                attempted, completed = threading.Event(), threading.Event()
                errors = []
                def cleanup():
                    attempted.set()
                    try: a.cleanup_receiver('unrelated-finished-request')
                    except BaseException as error: errors.append(error)
                    finally: completed.set()
                worker = threading.Thread(target=cleanup, daemon=True)
                def load(request):
                    if request is requests[0]:
                        worker.start()
                        self.assertTrue(attempted.wait(2))
                        completed.wait(.1)
                a.load_async = load
                a.process_pending_chunks(waiting, running)
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                a.restore_queues(waiting, running)
                self.assertEqual(list(waiting if source == 'waiting' else running), requests)

if __name__ == '__main__': unittest.main()
