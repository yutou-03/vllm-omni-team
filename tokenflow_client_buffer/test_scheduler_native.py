"""A clean FCFS adapter must preserve the native scheduler's queue decisions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from .pause import PausedRequest

from .scheduler import ClientBufferScheduler
from .test_core import request


class NativeFCFSPassthroughTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "snapshot.json"
        # Deliberately disagree with ingress order and exceed max_num_seqs.
        # Native chunk bookkeeping, rather than this adapter, owns the queue.
        self.requests = [request("later", 2), request("first", 0), request("middle", 1)]
        self.s = NS(running=list(self.requests), waiting=[],
                    requests={r.request_id: r for r in self.requests},
                    max_num_running_reqs=2, kv_cache_manager=NS(),
                    _omni_stage_id_for_trace=lambda: 0)
        self.events = []
        self.adapter = ClientBufferScheduler(self.s,
            dict(snapshot_path=str(self.path), resume_to_client_s=.5,
                 session="s", modality="text"),
            lambda event, **fields: self.events.append((event, fields)), clock=lambda: 10.)
        self.path.write_text(json.dumps(dict(schema_version=1, session="s", modality="text",
            written_at=10., mode="fcfs", reason="baseline", demand=3., gamma=10.,
            members={r.request_id: dict(rate=1.) for r in self.requests}, feedback={})))

    def test_preserves_excess_native_queue_and_native_post_schedule_order(self):
        running = self.s.running
        with self.adapter.scheduling_context():
            self.assertIs(self.s.running, running)
            self.assertEqual(self.s.running, self.requests)
            self.assertEqual(self.s.max_num_running_reqs, 2)
            # The native scheduler may defer the first entry and reorder ready
            # entries. Context exit must neither reinsert nor re-sort it.
            self.s.running[:] = [self.requests[2], self.requests[1]]
        self.assertEqual(self.s.running, [self.requests[2], self.requests[1]])
        self.assertIsNone(self.adapter.active_ids)
        self.adapter.record_scheduled(NS(num_scheduled_tokens={"middle": 1, "first": 1}))
        self.assertEqual(self.adapter.telemetry["fcfs_steps"], 1)
        self.assertEqual(len([e for e, _ in self.events if e == "client_buffer_contract"]), 3)

    def test_native_queue_changes_survive_schedule_exception(self):
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            with self.adapter.scheduling_context():
                self.s.running[:] = [self.requests[2]]
                raise RuntimeError("native failure")
        self.assertEqual(self.s.running, [self.requests[2]])

    def test_outstanding_pause_still_requires_gateway_resume_ack(self):
        self.adapter.pauses["first"] = PausedRequest(9., float("inf"), .8)
        self.adapter.pauses["first"].resume_by = 10.2
        self.adapter.pauses["first"].releases_quota = True
        self.adapter.active_ids = {"first"}
        with self.adapter.scheduling_context():
            self.assertNotIn(self.requests[1], self.s.running)
        self.assertIn("first", self.adapter.pauses.requested_resumes)

    def test_legacy_execution_cap_is_still_enforced(self):
        self.adapter = ClientBufferScheduler(self.s,
            dict(snapshot_path=str(self.path), resume_to_client_s=.5,
                 session="s", modality="text", execution_cap=1),
            lambda *_args, **_kwargs: None, clock=lambda: 10.)
        with self.adapter.scheduling_context():
            self.assertEqual(len(self.s.running), 1)


if __name__ == "__main__":
    unittest.main()
