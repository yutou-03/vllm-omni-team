"""Exercise the real adapter writer, including native FinishReason enums."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from .test_scheduler_native import NativeFCFSPassthroughTests


class PhaseEventTests(NativeFCFSPassthroughTests):
    def test_native_terminal_success_is_written_without_another_schedule(self):
        from vllm.v1.engine import FinishReason
        path = self.path.parent/'execution.json'
        self.adapter.config['execution_telemetry_path'] = str(path)
        request = self.requests[0]
        request.client_buffer_test_metadata['sched_request_path'] = 'audio'
        with self.adapter.scheduling_context():
            pass
        self.adapter.record_scheduled(NS(num_scheduled_tokens={request.request_id:1}))
        self.adapter.record_phase_completed(request, FinishReason.LENGTH)
        records = [json.loads(x) for x in Path(str(path)+'.completed.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in records], ['started','completed'])
        self.assertEqual(records[-1]['route'], 'audio')
        self.assertEqual(records[-1]['finish_reason'], 'length')
        self.adapter.record_phase_completed(request, FinishReason.LENGTH)
        self.assertEqual(len(Path(str(path)+'.completed.jsonl').read_text().splitlines()), 2)

    def test_abort_and_error_cannot_publish_success(self):
        from vllm.v1.engine import FinishReason
        path = self.path.parent/'execution.json'
        self.adapter.config['execution_telemetry_path'] = str(path)
        for request, reason in zip(self.requests, (FinishReason.ABORT, FinishReason.ERROR)):
            self.adapter.phase_started[request.request_id] = 9.
            self.adapter.record_phase_completed(request, reason)
        self.assertFalse(Path(str(path)+'.completed.jsonl').exists())
