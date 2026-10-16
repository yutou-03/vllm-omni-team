#!/usr/bin/env python3
"""Measure consecutive schedule() start gaps from a real mixed_pressure replay.

The interval includes scheduling, model execution, output handling and waiting;
it is not GPU kernel time. The separate schedule_ms measures schedule() itself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def read_iterations(path: Path) -> dict[str, list[dict]]:
    stages = defaultdict(list)
    for line_number, raw in enumerate(path.open('rb'), 1):
        event = json.loads(raw)
        if event.get('event') != 'iteration':
            continue
        if 'schedule_monotonic_start_s' not in event:
            raise ValueError(f'{path}:{line_number}: exact monotonic timing missing')
        stage = str(event['stage_id'])
        stages[stage].append({
            'start': float(event['schedule_monotonic_start_s']),
            'end': float(event['schedule_monotonic_end_s']),
            'iteration_id': event.get('iteration_id'),
            'schedule_ms': float(event['schedule_duration_monotonic_ms']),
            'batch_tokens': int(event.get('batch_num_tokens') or 0),
            'batch_seqs': int(event.get('batch_num_seqs') or 0),
            'running_before': int(event.get('num_running_reqs_before') or 0),
            'waiting_before': int(event.get('num_waiting_reqs_before') or 0),
            'kv_used_blocks': event.get('kv_used_blocks'),
            'kv_free_blocks': event.get('kv_free_blocks'),
            'preemptions': len(event.get('preempted_req_ids') or ()),
        })
    for events in stages.values():
        events.sort(key=lambda row: row['start'])
    return stages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    return {
        'count': len(values),
        'mean_ms': sum(values) / len(values) if values else None,
        'p50_ms': quantile(values, .50),
        'p90_ms': quantile(values, .90),
        'p95_ms': quantile(values, .95),
        'p99_ms': quantile(values, .99),
        'p999_ms': quantile(values, .999),
        'max_ms': max(values) if values else None,
        'above_50ms_fraction': sum(value > 50 for value in values) / len(values) if values else None,
        'above_100ms_fraction': sum(value > 100 for value in values) / len(values) if values else None,
        'above_250ms_fraction': sum(value > 250 for value in values) / len(values) if values else None,
        'above_1000ms_fraction': sum(value > 1000 for value in values) / len(values) if values else None,
    }


def plot(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stages = sorted({row['stage_id'] for row in rows}, key=int)
    fig, axes = plt.subplots(len(stages), 1, figsize=(11, 2.5 * len(stages)), sharex=True)
    if len(stages) == 1:
        axes = [axes]
    for stage, axis in zip(stages, axes):
        stage_rows = [row for row in rows if row['stage_id'] == stage and row['batch_tokens'] > 0]
        axis.scatter([row['relative_time_s'] for row in stage_rows],
                     [row['interval_ms'] for row in stage_rows],
                     s=3, alpha=.35, rasterized=True)
        axis.axhline(100, color='gray', linestyle='--', linewidth=1)
        axis.set_yscale('log')
        axis.set_ylabel(f'Stage {stage}\nGap (ms)')
        axis.grid(True, alpha=.2)
    axes[-1].set_xlabel('Time since injection start (s)')
    fig.suptitle('Scheduler start-to-start gaps after nonempty batches | mixed_pressure')
    fig.tight_layout()
    fig.savefig(output_dir / 'iteration_gap_timeline.png', dpi=300)
    fig.savefig(output_dir / 'iteration_gap_timeline.pdf')
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for stage in stages:
        values = sorted(row['interval_ms'] for row in rows
                        if row['stage_id'] == stage and row['batch_tokens'] > 0)
        if values:
            axis.step(values, [(index + 1) / len(values) for index in range(len(values))],
                      where='post', label=f'Stage {stage} (n={len(values)})')
    for threshold in (50, 100, 250):
        axis.axvline(threshold, color='gray', linestyle=':', linewidth=1)
    axis.set_xscale('log')
    axis.set_xlim(left=.1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel('Scheduler gap after nonempty batch (ms)')
    axis.set_ylabel('Empirical CDF')
    axis.legend()
    axis.grid(True, alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / 'iteration_gap_ecdf.png', dpi=300)
    fig.savefig(output_dir / 'iteration_gap_ecdf.pdf')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((run_dir / 'results.json').read_text())
    origin = float(result['origin_monotonic_s'])
    injection_end = origin + float(result['arrival_duration_s'])
    trace_path = run_dir / 'traces/iteration_events.jsonl'
    iterations = read_iterations(trace_path)
    rows = []
    audit = {}
    for stage, events in sorted(iterations.items(), key=lambda item: int(item[0])):
        count = 0
        for index, (current, following) in enumerate(zip(events, events[1:])):
            start = current['start']
            if start < origin:
                continue
            interval_ms = (following['start'] - start) * 1000
            if interval_ms < 0:
                raise ValueError(f'negative monotonic interval in stage {stage}')
            rows.append({
                'stage_id': stage,
                'sequence_index': index,
                'relative_time_s': start - origin,
                'start_monotonic_s': start,
                'next_start_monotonic_s': following['start'],
                'interval_ms': interval_ms,
                'phase': 'injection' if start < injection_end else 'drain',
                'iteration_id': current['iteration_id'],
                'schedule_ms': current['schedule_ms'],
                'post_schedule_to_next_start_ms': max(0.0, (following['start'] - current['end']) * 1000),
                'batch_tokens': current['batch_tokens'],
                'batch_seqs': current['batch_seqs'],
                'running_before': current['running_before'],
                'waiting_before': current['waiting_before'],
                'kv_used_blocks': current['kv_used_blocks'],
                'kv_free_blocks': current['kv_free_blocks'],
                'reactive_preemptions': current['preemptions'],
            })
            count += 1
        audit[stage] = {'iteration_events': len(events), 'valid_intervals': count}
    if not rows:
        raise ValueError('no between-schedule gaps found after injection origin')
    columns = list(rows[0])
    write_csv(output_dir / 'iteration_intervals.csv', rows, columns)
    summary_rows = []
    for stage in sorted({row['stage_id'] for row in rows}, key=int):
        for phase in ('injection', 'drain', 'full'):
            for population in ('all', 'nonempty_batch', 'running_at_start'):
                subset = [row for row in rows if row['stage_id'] == stage
                          and (phase == 'full' or row['phase'] == phase)
                          and (population == 'all'
                               or (population == 'nonempty_batch' and row['batch_tokens'] > 0)
                               or (population == 'running_at_start' and row['running_before'] > 0))]
                summary_rows.append({'stage_id': stage, 'phase': phase,
                                     'population': population,
                                     **summarize([row['interval_ms'] for row in subset]),
                                     'schedule_p50_ms': quantile([row['schedule_ms'] for row in subset], .5),
                                     'schedule_p99_ms': quantile([row['schedule_ms'] for row in subset], .99)})
    write_csv(output_dir / 'iteration_summary.csv', summary_rows, list(summary_rows[0]))
    tail = []
    for stage in sorted({row['stage_id'] for row in rows}, key=int):
        tail.extend(sorted((row for row in rows if row['stage_id'] == stage
                            and row['batch_tokens'] > 0),
                           key=lambda row: row['interval_ms'], reverse=True)[:20])
    write_csv(output_dir / 'longest_intervals.csv', tail, columns)
    plot(rows, output_dir)
    manifest = {
        'run_id': result['run_id'],
        'origin_monotonic_s': origin,
        'arrival_duration_s': result['arrival_duration_s'],
        'planned_requests': result['planned_requests'],
        'attempted_requests': result['attempted_requests'],
        'successful_requests': result['success'],
        'run_error': result['run_error'],
        'iteration_trace_sha256': sha256_file(trace_path),
        'trace_audit': audit,
        'metric_definition': 'monotonic difference between consecutive schedule() starts on the same stage; includes execution, output handling and waiting',
        'quantile_method': 'linear interpolation of sorted per-iteration values',
    }
    (output_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output_dir': str(output_dir), 'trace_audit': audit,
                      'successful_requests': result['success']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
