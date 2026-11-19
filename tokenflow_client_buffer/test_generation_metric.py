import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .analyze import analyze
from .generation_metric import score_request


class GenerationMetricTest(unittest.TestCase):
    def test_reader_ticks_stalls_and_ties(self):
        # 10 tokens -> full weight at depth one and zero at depth two.
        times = [0., .05, .1, .3, .4, .5, .6, .7, .8, 1.2]
        score = score_request(list(enumerate(times, 1)), 10, 10, 0.)
        self.assertEqual(score["utility"], 8.)
        self.assertAlmostEqual(score["stall_s"], .3)
        self.assertEqual(score["ttft_s"], 0.)

    def test_missing_or_duplicate_events_fail(self):
        for sequence in ([(1, 1.)], [(1, 1.), (1, 2.)], [(1, 2.), (2, 1.)]):
            with self.assertRaises(ValueError):
                score_request(sequence, 2, 5, 0.)

    def test_analysis_uses_generation_and_common_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = root/"client"
            folder = client/hashlib.sha256(b"test").hexdigest()[:16]
            folder.mkdir(parents=True)
            row = dict(request_id="test", modality="text", status="success", rate=1.,
                       scheduled_at=0., ended_at=22., generation_done_at=12.,
                       received_units=10., startup_s=11., first_ready_s=11.,
                       send_lag_s=0., stall_s=0., max_stall_s=0.,
                       workload={"payload":{"omni_scheduling":{}}})
            (client/"results.json").write_text(json.dumps([row]))
            (folder/"events.jsonl").write_text(json.dumps(dict(event="received", buffered_before=0., units=10)))
            (folder/"stream.sse").write_text('data: {"usage":{"completion_tokens":10}}\n')
            events = [dict(event="request_ingress", request_id="backend", sched_source_request_id="test")]
            events += [dict(event="stage_token_output", request_id="backend",
                            stage_id=0, monotonic=float(i), output_len=i, num_new_tokens=1)
                       for i in range(1, 11)]
            trace = root/"stage_events.jsonl"
            trace.write_text("".join(json.dumps(e)+"\n" for e in events))
            self.assertIsNone(analyze(client)["paper_qos"])
            window = dict(trace=str(root), start_offset=0, end_offset=trace.stat().st_size)
            (root/"generation_trace_window.json").write_text(json.dumps(window))
            summary = analyze(client)
            self.assertEqual(summary["client_receipt"]["utility"], 1.)
            self.assertEqual(summary["effective_units_per_s"], 10/12)
            self.assertEqual(summary["paper_qos"], (10-1)/12)
            # A step can accept two tokens. Both have one generation time.
            events[1].update(num_new_tokens=2, output_len=2)
            events[2].update(num_new_tokens=2)
            trace.write_text("".join(json.dumps(e)+"\n" for e in events))
            window["end_offset"] = trace.stat().st_size
            paired = analyze(client, trace_window=window)
            self.assertEqual(paired["utility"], 9.)
            events.pop(2)
            trace.write_text("".join(json.dumps(e)+"\n" for e in events))
            window["end_offset"] = trace.stat().st_size
            with self.assertRaises(ValueError):
                analyze(client, trace_window=window)
            self.assertEqual(summary["throughput_by_window"]["generation"]["raw_units_per_s"], 10/12)
