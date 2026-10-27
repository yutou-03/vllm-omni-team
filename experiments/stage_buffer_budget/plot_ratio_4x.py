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
    common_horizon = max(json.loads((run / 'ratio_summary.json').read_text())
                         ['client']['observed_end_s'] for run in runs.values())
    fig, axs = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True, layout='constrained')
    aggregate = {}
    throughput = {}
    for row_index, (ratio, run) in enumerate(runs.items()):
        summary = json.loads((run / 'ratio_summary.json').read_text())
        throughput[ratio] = summary['client']
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
            if t < 0 or t > common_horizon:
                continue
            latest[pid] = used
            curve.append((t, sum(latest.values())))
        if summary.get('drained_after') is True and curve:
            curve.append((common_horizon, curve[-1][1]))
        aggregate[ratio] = curve
        for (stage, pid), samples in sorted(series.items()):
            samples.sort()
            # A final allocator-free event records the pool state after its last schedule.
            # Once the service reports drained, carry that measured state to the
            # common observed comparison horizon.
            if common_horizon > samples[-1][0] and summary.get('drained_after') is True:
                samples.append((common_horizon, samples[-1][1], samples[-1][2]))
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
            ax.set_xlim(0, common_horizon)
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
    ax.set(xlim=(0, common_horizon), ylim=(0, None), xlabel='Time since frozen replay start (s)',
           ylabel='Talker native KV used across GPUs (GB)',
           title='Measured Talker KV residency across independent replica pools')
    ax.axvline(30, color='0.4', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(frameon=False)
    fig2.savefig(a.output_prefix.with_name('talker_aggregate_kv').with_suffix('.png'), dpi=320)
    fig2.savefig(a.output_prefix.with_name('talker_aggregate_kv').with_suffix('.pdf'))
    plt.close(fig2)
    fig3, out_axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    labels = list(throughput)
    rates = [throughput[r]['observed_successes_per_s'] for r in labels]
    audio = [throughput[r]['accepted_audio_content_seconds_per_wall_s_observed_span']
             for r in labels]
    for ax, values, ylabel in (
        (out_axes[0], rates, 'Completed requests / wall s'),
        (out_axes[1], audio, 'Accepted audio seconds / wall s')):
        bars = ax.bar(labels, values, width=0.6)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set(ylabel=ylabel, ylim=(0, max(values) * 1.18 if values else 1))
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig3.suptitle('Measured finite-trace output / each run observed completion span')
    fig3.savefig(a.output_prefix.with_name('useful_output_ratio_4x').with_suffix('.png'), dpi=320)
    fig3.savefig(a.output_prefix.with_name('useful_output_ratio_4x').with_suffix('.pdf'))
    plt.close(fig3)

    # A request can be preempted more than once.  Count distinct logical
    # requests with a completed memory-triggered restart, not victim picks or
    # the number of preemption events, for the right-hand percentage axis.
    latency = {
        'TTFT': [throughput[r]['successful_request_ttft_p95_s'] for r in labels],
        'TTFP': [throughput[r]['successful_request_ttfp_p95_s'] for r in labels],
        'E2E': [throughput[r]['successful_request_e2e_p95_s'] for r in labels],
    }
    preempted_ratio_pct = []
    for ratio in labels:
        client = throughput[ratio]
        denominator = client['planned']
        if denominator != 280 or client['attempted'] != denominator or (
            client['statuses'].get('success') != denominator
        ):
            raise ValueError(f'{ratio}: figure requires all 280 requests completed')
        preempted_ids = set()
        restart_events = 0
        with (runs[ratio] / 'traces' / 'stage_events.jsonl').open() as stream:
            for line in stream:
                if 'full_pipeline_restarted' not in line:
                    continue
                event = json.loads(line)
                if (event.get('event') == 'full_pipeline_restarted'
                        and event.get('stage_id') == 1
                        and event.get('preemption_cause') == 'allocation_failure'):
                    preempted_ids.add(event['logical_request_id'])
                    restart_events += 1
        summary = json.loads((runs[ratio] / 'ratio_summary.json').read_text())
        recorded_events = sum(
            counts.get('passive_preemption', 0)
            for pool, counts in summary['events_by_pool'].items()
            if pool.startswith('1:')
        )
        if restart_events != recorded_events:
            raise ValueError(f'{ratio}: restart/preemption event count mismatch')
        preempted_ratio_pct.append(len(preempted_ids) / denominator * 100)

    fig4, lat_ax = plt.subplots(figsize=(11.5, 5), layout='constrained')
    ratio_ax = lat_ax.twinx()
    centers = list(range(len(labels)))
    width = 0.24
    for index, (metric, values) in enumerate(latency.items()):
        bars = lat_ax.bar(
            [center + (index - 1) * width for center in centers],
            values, width=width, label=metric,
        )
        bar_labels = [f'{value:.1f}' for value in values]
        if metric == 'TTFP':
            # The 4:2 red point falls near the top of this orange bar.
            bar_labels[1] = ''
        lat_ax.bar_label(bars, labels=bar_labels, padding=3, fontsize=9)
        if metric == 'TTFP':
            bar = bars[1]
            lat_ax.annotate(
                f'{values[1]:.1f}',
                (bar.get_x() + bar.get_width() / 2, values[1]),
                xytext=(0, -5), textcoords='offset points',
                ha='center', va='top', fontsize=9,
            )
    lat_ax.set_xticks(centers, labels)
    lat_ax.set_ylim(0, max(max(values) for values in latency.values()) * 1.19)
    lat_ax.set(xlabel='Thinker:Talker GPU ratio', ylabel='P95 latency (s)')
    lat_ax.grid(axis='y', alpha=0.2)
    lat_ax.set_axisbelow(True)

    ratio_ax.plot(
        centers, preempted_ratio_pct, color='tab:red', marker='o',
        markersize=8, linewidth=2, label='Preempted', zorder=5,
    )
    ratio_ax.set_ylim(0, max(10.0, max(preempted_ratio_pct) * 1.25))
    ratio_ax.set_ylabel('Preempted Ratio (%)', color='tab:red')
    ratio_ax.tick_params(axis='y', colors='tab:red')
    ratio_ax.spines['right'].set_color('tab:red')
    for x, value in zip(centers, preempted_ratio_pct):
        ratio_ax.annotate(
            f'{value:.2f}%', (x, value), xytext=(0, 10),
            textcoords='offset points', ha='center',
            color='tab:red', fontsize=9, fontweight='bold',
        )
    handles, names = lat_ax.get_legend_handles_labels()
    ratio_handles, ratio_names = ratio_ax.get_legend_handles_labels()
    lat_ax.legend(handles + ratio_handles, names + ratio_names,
                  frameon=False, ncol=4, loc='upper right')
    fig4.suptitle(
        'Measured P95 latency and preempted request ratio | 280 completed requests per run\n'
        'Preempted = distinct Talker requests restarted after KV allocation failure / 280'
    )
    fig4.savefig(a.output_prefix.with_name('latency_p95_and_preemptions').with_suffix('.png'), dpi=320)
    fig4.savefig(a.output_prefix.with_name('latency_p95_and_preemptions').with_suffix('.pdf'))
    plt.close(fig4)


if __name__ == '__main__':
    main()
