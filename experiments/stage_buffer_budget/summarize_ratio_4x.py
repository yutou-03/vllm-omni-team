#!/usr/bin/env python3
"""Create a cross-run CSV from measured 4:1, 4:2, 4:3 artifacts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, metavar='RATIO=RUN_DIR')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = []
    for spec in a.run:
        ratio, directory = spec.split('=', 1)
        run = Path(directory)
        summary = json.loads((run / 'ratio_summary.json').read_text())
        raw = json.loads((run / 'results.json').read_text())
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if manifest['ratio'] != ratio:
            raise ValueError(f'ratio mismatch in {run}')
        pools = summary['pools']
        thinker = [v for k, v in pools.items() if k.startswith('0:')]
        talker = [v for k, v in pools.items() if k.startswith('1:')]
        if len(thinker) != 1 or len(talker) != int(ratio.split(':')[1]):
            raise ValueError(f'unexpected stage/replica count in {run}')
        events = summary['events_by_pool']
        event_sum = lambda stage, name: sum(v.get(name, 0) for k, v in events.items()
                                            if k.startswith(f'{stage}:'))
        c = summary['client']
        updates = []
        with (run / 'kv_pools_by_engine.csv').open(newline='') as stream:
            for sample in csv.DictReader(stream):
                if sample['stage_id'] == '1':
                    updates.append((float(sample['relative_time_s']),
                                    int(sample['engine_pid']),
                                    int(sample['used_bytes_per_rank'])))
        updates.sort()
        latest = {}
        last_t = 0.0
        area = 0.0
        peak = 0
        for t, pid, used in updates:
            if not 0 <= t <= c['observed_end_s']:
                continue
            area += sum(latest.values()) * (t - last_t)
            latest[pid] = used
            peak = max(peak, sum(latest.values()))
            last_t = t
        if summary['drained_after']:
            area += sum(latest.values()) * (c['observed_end_s'] - last_t)
        result.append({
            'ratio': ratio,
            'run_dir': str(run.resolve()),
            'workload_sha256': manifest['workload_sha256'],
            'source_commit': manifest['source_commit'],
            'attempted': raw['attempted_requests'],
            'success': raw['success'],
            'timeout': raw['timeout'],
            'error': raw['error'],
            'run_error': raw['run_error'],
            'drained_after': summary['drained_after'],
            'fixed_horizon_s': c['fixed_horizon_s'],
            'observed_completion_span_s': c['observed_end_s'],
            'completed_req_per_s_observed_span': c['observed_successes_per_s'],
            'audio_content_s_per_wall_s_observed_span': c['accepted_audio_content_seconds_per_wall_s_observed_span'],
            'all_requests_completed_without_timeout': c['all_requests_completed_without_timeout'],
            'successes_by_fixed_horizon': c['successes_by_fixed_horizon'],
            'success_req_per_s_fixed_horizon': c['successes_per_s_fixed_horizon'],
            'audio_frames_per_s_fixed_horizon': c['accepted_audio_frames_per_s_fixed_horizon'],
            'audio_content_s_per_wall_s_fixed_horizon': c['accepted_audio_content_seconds_per_wall_s_fixed_horizon'],
            'send_lateness_p95_s': raw['send_lateness_p95_s'],
            'ttft_p50_s_success_only': c['successful_request_ttft_p50_s'],
            'ttft_p95_s_success_only': c['successful_request_ttft_p95_s'],
            'ttfp_p50_s_success_only': c['successful_request_ttfp_p50_s'],
            'ttfp_p95_s_success_only': c['successful_request_ttfp_p95_s'],
            'e2e_p50_s_success_only': c['successful_request_e2e_p50_s'],
            'e2e_p95_s_success_only': c['successful_request_e2e_p95_s'],
            'thinker_pool_gb_per_rank': thinker[0]['pool_bytes_per_rank'] / 1e9,
            'thinker_peak_native_pct': thinker[0]['peak_native_usage_pct'],
            'thinker_gt90_s': thinker[0]['gt90_time_s_observed'],
            'thinker_passive_preemptions': event_sum(0, 'passive_preemption'),
            'talker_replicas': len(talker),
            'talker_pool_gb_per_replica': talker[0]['pool_bytes_per_rank'] / 1e9,
            'talker_total_pool_gb_across_gpus': sum(v['pool_bytes_per_rank'] for v in talker) / 1e9,
            'talker_aggregate_peak_used_gb': peak / 1e9,
            'talker_aggregate_mean_used_gb_observed_span': (
                area / c['observed_end_s'] / 1e9 if summary['drained_after'] else ''),
            'talker_mean_native_pct_each': json.dumps([
                round(v['mean_native_usage_pct_observed'], 3) for v in talker]),
            'talker_peak_native_pct_each': json.dumps([round(v['peak_native_usage_pct'], 3)
                                                       for v in talker]),
            'talker_gt90_s_each': json.dumps([round(v['gt90_time_s_observed'], 3)
                                            for v in talker]),
            'talker_allocation_failures': event_sum(1, 'allocation_failure'),
            'talker_passive_preemptions': event_sum(1, 'passive_preemption'),
            'slo_goodput': '',
            'slo_missing_reason': c['slo_missing_reason'],
        })
    if [row['ratio'] for row in result] != ['4:1', '4:2', '4:3']:
        p.error('pass runs in 4:1, 4:2, 4:3 order')
    if len({row['workload_sha256'] for row in result}) != 1:
        raise ValueError('workload hashes differ')
    if len({row['source_commit'] for row in result}) != 1:
        raise ValueError('service source commits differ')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    print(a.output)


if __name__ == '__main__':
    main()
