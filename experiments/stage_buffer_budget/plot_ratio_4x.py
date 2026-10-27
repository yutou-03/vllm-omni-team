#!/usr/bin/env python3
"""Plot measured native KV-pool occupancy for the fixed 4:x replica study."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, metavar='RATIO=RUN_DIR')
    p.add_argument('--output-prefix', type=Path, required=True)
    a = p.parse_args()
    runs = {}
    for spec in a.run:
        ratio, path = spec.split('=', 1)
        if ratio in runs:
            p.error(f'duplicate ratio: {ratio}')
        runs[ratio] = Path(path)
    if tuple(runs) != ('4:1', '4:2', '4:3'):
        p.error('pass 4:1, 4:2, 4:3 in that order')
    fig, axs = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True, layout='constrained')
    aggregate = {}
    for row_index, (ratio, run) in enumerate(runs.items()):
        summary = json.loads((run / 'ratio_summary.json').read_text())
        series = defaultdict(list)
        with (run / 'kv_pools_by_engine.csv').open(newline='') as f:
            for sample in csv.DictReader(f):
                key = int(sample['stage_id']), int(sample['engine_pid'])
                series[key].append((float(sample['relative_time_s']),
                                    int(sample['used_bytes_per_rank']) / 1e9,
                                    int(sample['pool_bytes_per_rank']) / 1e9))
        talker_updates = sorted((t, pid, used) for (stage, pid), samples in series.items()
                                if stage == 1 for t, used, _ in samples)
        latest = {}
        curve = [(0.0, 0.0)]
        for t, pid, used in talker_updates:
            if t < 0 or t > 270:
                continue
            latest[pid] = used
            curve.append((t, sum(latest.values())))
        if summary.get('drained_after') is True and curve:
            curve.append((270.0, curve[-1][1]))
        aggregate[ratio] = curve
        for (stage, pid), samples in sorted(series.items()):
            samples.sort()
            # A final allocator-free event records the pool state after its last schedule.
            # Once the service reports drained, carry that measured state to the
            # common 270 s comparison horizon.
            if summary['client']['fixed_horizon_s'] > samples[-1][0] and summary.get('drained_after') is True:
                samples.append((summary['client']['fixed_horizon_s'], samples[-1][1], samples[-1][2]))
            ax = axs[row_index, stage]
            gpu = summary['gpu_by_engine_pid'].get(str(pid))
            label = f'GPU {gpu}' if gpu is not None else f'engine {pid}'
            ax.step([s[0] for s in samples], [s[1] for s in samples],
                    where='post', linewidth=1.25, label=label)
            capacity = samples[0][2]
            ax.axhline(capacity, color='0.5', linestyle=':', linewidth=0.8)
            ax.set_ylim(0, capacity * 1.06)
        for stage, title in ((0, 'Thinker TP4: per GPU rank'),
                             (1, 'Talker TP1: separate GPU pools')):
            ax = axs[row_index, stage]
            ax.set_title(f'{ratio}  |  {title}', fontsize=10)
            ax.set_ylabel('Native KV used (GB)')
            ax.axvline(30, color='0.4', linestyle='--', linewidth=0.8)
            ax.set_xlim(0, 270)
            ax.grid(axis='y', alpha=0.2)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ax in axs[-1]:
        ax.set_xlabel('Time since frozen replay start (s)')
    fig.suptitle('Measured KV-pool occupancy | 280 requests in 30 s | native FCFS', fontsize=12)
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output_prefix.with_suffix('.png'), dpi=320)
    fig.savefig(a.output_prefix.with_suffix('.pdf'))
    plt.close(fig)
    fig2, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    for ratio, curve in aggregate.items():
        ax.step([t for t, _ in curve], [gb for _, gb in curve], where='post',
                label=ratio, linewidth=1.5)
    ax.set(xlim=(0, 270), ylim=(0, None), xlabel='Time since frozen replay start (s)',
           ylabel='Talker native KV used across GPUs (GB)',
           title='Measured Talker KV residency across independent replica pools')
    ax.axvline(30, color='0.4', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(frameon=False)
    fig2.savefig(a.output_prefix.with_name('talker_aggregate_kv').with_suffix('.png'), dpi=320)
    fig2.savefig(a.output_prefix.with_name('talker_aggregate_kv').with_suffix('.pdf'))
    plt.close(fig2)


if __name__ == '__main__':
    main()
