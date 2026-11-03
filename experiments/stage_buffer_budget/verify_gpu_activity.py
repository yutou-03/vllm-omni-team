"""CPU checks for measurement denominators, missing data, and replay cleanup."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gpu_activity import integrate, stage_map
import run_frozen_stream as replay


class ActivityTests(unittest.TestCase):
    def test_weighted_window_and_thresholds(self):
        rows = [{'monotonic_s': t, 'gpu_busy_pct': u, 'power_w': 20}
                for t, u in [(0, 100), (1, 0), (3, 50)]]
        result = integrate(rows, .5, 2, 3)
        self.assertAlmostEqual(result['mean_gpu_busy_pct'], 100 / 3)
        self.assertAlmostEqual(result['coverage_fraction'], 1)
        self.assertAlmostEqual(result['high_activity_fraction'], 1 / 3)
        self.assertAlmostEqual(result['sampled_energy_j'], 30)

    def test_missing_data_is_not_zero_busy(self):
        rows = [{'monotonic_s': t, 'gpu_busy_pct': u} for t, u in [(0, None), (1, 90), (2, 0), (6, 0)]]
        result = integrate(rows, 0, 6, 1.5)
        self.assertAlmostEqual(result['mean_gpu_busy_pct'], 90)
        self.assertAlmostEqual(result['coverage_fraction'], 1 / 6)

    def test_reject_shared_device(self):
        with self.assertRaises(ValueError):
            stage_map('{"0":[1,2],"1":[2]}')

    def run_replay(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / 'workload.jsonl'
            workload.write_text(json.dumps({'source_request_id': 'r', 'arrival_offset_s': 0}) + '\n')
            collector = Mock()
            collector.start.return_value = collector
            collector.errors = []
            argv = ['replay', '--workload', str(workload), '--output', str(root / 'run'),
                    '--window-s', '1', '--gpu-stage-map', '{"0":[0]}']
            with patch('sys.argv', argv), patch.object(replay, 'GPUCollector', return_value=collector), \
                 patch.object(replay, 'drain', return_value={'drained': True}), \
                 patch.object(replay.time, 'sleep'), \
                 patch.object(replay, 'send_request', side_effect=RuntimeError('injected') if failure else None,
                              return_value={'source_request_id': 'r', 'status': 'success'}):
                if failure:
                    with self.assertRaises(RuntimeError):
                        replay.main()
                else:
                    replay.main()
            collector.stop.assert_called_once()
            manifest = json.loads((root / 'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed' if failure else 'completed')
            self.assertIn('replay_end_monotonic_s', manifest)

    def test_success_stops_sampler(self):
        self.run_replay(False)

    def test_failure_stops_sampler(self):
        self.run_replay(True)


if __name__ == '__main__':
    unittest.main()
