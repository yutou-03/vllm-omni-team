"""Independent hardware sampling; never infer GPU busy time from scheduler gaps.

The NVIDIA driver utilization window is coarser than the collection period.
This measures device-wide kernel activity, not SM occupancy or useful FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from pathlib import Path


def stage_map(value):
    mapping = {str(k): [int(v) for v in values] for k, values in json.loads(value).items()}
    devices = [v for values in mapping.values() for v in values]
    if not mapping or any(not v for v in mapping.values()) or len(set(devices)) != len(devices):
        raise ValueError('each stage needs distinct, exclusive physical GPU indices')
    return mapping


class GPUCollector:
    def __init__(self, output: Path, mapping: dict, interval_ms: int = 200):
        if interval_ms < 100:
            raise ValueError('collection interval must be at least 100 ms')
        self.output, self.mapping, self.interval_ms = output, mapping, interval_ms
        self.reverse = {int(gpu): str(stage) for stage, gpus in mapping.items() for gpu in gpus}
        self.ready = threading.Event()
        self.seen, self.errors, self.samples = set(), [], 0
        self.process = self.thread = None
        self.stopped = False

    def start(self):
        self.output.mkdir(parents=True, exist_ok=False)
        command = ['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,power.draw,memory.used',
                   '--format=csv,noheader,nounits', '-i', ','.join(map(str, sorted(self.reverse))),
                   '--loop-ms=' + str(self.interval_ms)]
        self.manifest = {'source': 'nvidia-smi device-wide driver utilization', 'command': command,
                         'stage_physical_gpu_mapping': self.mapping, 'interval_ms': self.interval_ms,
                         'start_monotonic_s': time.monotonic(), 'start_wall_s': time.time(),
                         'scope': 'hardware_activity_not_SM_occupancy_or_effective_work',
                         'resolution_note': 'driver averaging window is not the collection interval',
                         'exclusive_devices_required': True}
        self.stderr = (self.output / 'stderr.log').open('w')
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.stderr,
                                            text=True, bufsize=1)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            if not self.ready.wait(30):
                raise RuntimeError('not all selected GPUs produced valid samples within 30 s')
            self._save()
            return self
        except BaseException:
            self.stop()
            raise

    def _read(self):
        def number(raw):
            try:
                value = float(raw.strip())
                return value if math.isfinite(value) else None
            except ValueError:
                return None
        with (self.output / 'samples.jsonl').open('x') as stream:
            try:
                for line in self.process.stdout:
                    values = next(csv.reader([line]))
                    if len(values) != 7:
                        self.errors.append('unparseable_line:' + line.strip())
                        continue
                    stamp, index, uuid, gpu, memory, power, used = values
                    index = int(index)
                    row = {'monotonic_s': time.monotonic(), 'wall_s': time.time(),
                           'driver_timestamp': stamp.strip(), 'gpu_index': index,
                           'gpu_uuid': uuid.strip(), 'stage_id': self.reverse[index],
                           'gpu_busy_pct': number(gpu), 'memory_busy_pct': number(memory),
                           'power_w': number(power), 'memory_used_mib': number(used)}
                    if row['gpu_busy_pct'] is not None and 0 <= row['gpu_busy_pct'] <= 100:
                        self.seen.add(index)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    self.samples += 1
                    if self.seen == set(self.reverse):
                        self.ready.set()
            except Exception as error:
                self.errors.append(repr(error))

    def _save(self):
        self.manifest.update(samples=self.samples, errors=self.errors)
        (self.output / 'manifest.json').write_text(json.dumps(self.manifest, indent=2) + '\n')

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.process is not None:
            self.manifest['unexpected_exit_code'] = self.process.poll()
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    self.errors.append('reader_not_stopped')
            self.process.stdout.close()
        if hasattr(self, 'stderr'):
            self.stderr.close()
        if hasattr(self, 'manifest'):
            self.manifest.update(end_monotonic_s=time.monotonic(), end_wall_s=time.time())
            self._save()


def integrate(rows, start, end, max_gap_s, high=80, low=10):
    covered = busy = high_s = low_s = energy = 0.0
    power_covered = 0.0
    for a, b in zip(rows, rows[1:]):
        duration = min(end, b['monotonic_s']) - max(start, a['monotonic_s'])
        if duration <= 0 or b['monotonic_s'] - a['monotonic_s'] > max_gap_s:
            continue
        value = a['gpu_busy_pct']
        if value is None or not 0 <= value <= 100:
            continue
        covered += duration
        busy += duration * value
        high_s += duration * (value >= high)
        low_s += duration * (value <= low)
        if a.get('power_w') is not None:
            energy += duration * a['power_w']
            power_covered += duration
    return {'coverage_fraction': covered / (end - start), 'covered_s': covered,
            'mean_gpu_busy_pct': busy / covered if covered else None,
            'high_activity_fraction': high_s / covered if covered else None,
            'low_activity_fraction': low_s / covered if covered else None,
            'sampled_energy_j': energy if power_covered else None,
            'power_coverage_s': power_covered}


def analyze(directory, start, end, output, high=80, low=10, max_gap_s=1.0):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('analysis needs a finite positive measurement window')
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (directory / 'samples.jsonl').open()]
    groups = {str(gpu): sorted([r for r in rows if r['gpu_index'] == gpu], key=lambda r: r['monotonic_s'])
              for values in manifest['stage_physical_gpu_mapping'].values() for gpu in values}
    summary = {'measurement_start_monotonic_s': start, 'measurement_end_monotonic_s': end,
               'scope': 'coarse device activity; not useful GPU compute or exact starvation time',
               'high_threshold_pct': high, 'low_threshold_pct': low, 'max_interpolation_gap_s': max_gap_s,
               'sampler_errors': manifest.get('errors', []),
               'sampler_unexpected_exit_code': manifest.get('unexpected_exit_code'),
               'gpus': {}, 'stages': {}}
    for gpu, values in groups.items():
        summary['gpus'][gpu] = integrate(values, start, end, max_gap_s, high, low)
    for stage, gpus in manifest['stage_physical_gpu_mapping'].items():
        stats = [summary['gpus'][str(gpu)] for gpu in gpus]
        valid = all(s['coverage_fraction'] >= .95 for s in stats)
        means = [s['mean_gpu_busy_pct'] for s in stats]
        summary['stages'][stage] = {'gpus': gpus, 'valid_coverage': valid,
            'mean_across_ranks_gpu_busy_pct': sum(means) / len(means) if valid else None,
            'minimum_rank_mean_gpu_busy_pct': min(means) if valid else None}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages = manifest['stage_physical_gpu_mapping']
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 2.5 * len(stages)), sharex=True, squeeze=False)
    for axis, (stage, gpus) in zip(axes[:, 0], stages.items()):
        for gpu in gpus:
            data = [r for r in groups[str(gpu)] if start <= r['monotonic_s'] <= end]
            axis.plot([r['monotonic_s'] - start for r in data], [r['gpu_busy_pct'] for r in data],
                      label='GPU ' + str(gpu), linewidth=.8)
        axis.set_ylim(-2, 102)
        axis.set_ylabel('Stage ' + stage + '\nGPU busy (%)')
        axis.legend(loc='upper right', ncol=len(gpus))
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel('Seconds from measurement start')
    fig.suptitle('Device activity from driver samples (not SM occupancy)')
    fig.tight_layout()
    fig.savefig(output / 'gpu_activity.png', dpi=180)
    fig.savefig(output / 'gpu_activity.svg')
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('--output', type=Path, required=True)
    collect.add_argument('--stage-map', type=stage_map, required=True)
    collect.add_argument('--duration-s', type=float, required=True)
    collect.add_argument('--interval-ms', type=int, default=200)
    report = sub.add_parser('analyze')
    report.add_argument('--telemetry', type=Path, required=True)
    report.add_argument('--start', type=float, required=True)
    report.add_argument('--end', type=float, required=True)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        if not math.isfinite(args.duration_s) or args.duration_s <= 0:
            parser.error('duration-s must be finite and positive')
        collector = GPUCollector(args.output, args.stage_map, args.interval_ms)
        try:
            collector.start()
            time.sleep(args.duration_s)
        finally:
            collector.stop()
    else:
        print(json.dumps(analyze(args.telemetry, args.start, args.end, args.output), indent=2))


if __name__ == '__main__':
    main()
