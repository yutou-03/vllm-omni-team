#!/usr/bin/env python3
"""Freeze provenance for one measured 4:x open-loop replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--session-dir', type=Path, required=True)
    p.add_argument('--ratio', choices=('4:1', '4:2', '4:3'), required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--pid-gpu-map', type=Path)
    a = p.parse_args()
    run = a.run_dir.resolve()
    results = json.loads((run / 'results.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    summary = json.loads((run / 'ratio_summary.json').read_text())
    session = json.loads((a.session_dir / 'manifest.json').read_text())
    workload = Path(protocol['workload'])
    if sha256(workload) != protocol['workload_sha256']:
        raise ValueError('workload changed during replay')
    for name, source in (('workload.jsonl', workload),
                         ('workload.manifest.json', workload.with_suffix('.manifest.json'))):
        target = run / name
        if not target.exists():
            shutil.copyfile(source, target)
        elif sha256(target) != sha256(source):
            raise ValueError(f'existing {name} differs from source')
    if a.pid_gpu_map:
        target = run / 'pid_gpu_observation.json'
        if not target.exists():
            shutil.copyfile(a.pid_gpu_map, target)
    artifacts = ('workload.jsonl', 'workload.manifest.json', 'results.json',
                 'protocol.json', 'traces/stage_events.jsonl',
                 'traces/iteration_events.jsonl', 'request_summary.csv',
                 'kv_pools_by_engine.csv', 'ratio_summary.json')
    manifest = {
        'schema_version': 1,
        'kind': 'measured_4_to_x_stage_gpu_ratio_replay',
        'ratio': a.ratio,
        'source_commit': a.source_commit,
        'source_worktree': str(Path(protocol['service']['framework']).resolve()),
        'model': protocol['model'],
        'service_session': str(a.session_dir.resolve()),
        'service_manifest_sha256': sha256(a.session_dir / 'manifest.json'),
        'deploy_config_sha256': session['deploy_sha256'],
        'actual_experiment_env': session['experiment_env'],
        'devices': protocol['service']['devices'],
        'workload_sha256': protocol['workload_sha256'],
        'planned_requests': results['planned_requests'],
        'attempted_requests': results['attempted_requests'],
        'arrival_window_s': results['arrival_window_s'],
        'safety_timeout_s': protocol['timeout_s'],
        'safety_timeout_is_slo': False,
        'slo_goodput': None,
        'successful_requests': results['success'],
        'timeouts': results['timeout'],
        'errors': results['error'],
        'run_error': results['run_error'],
        'drained_after': results['drained_after'],
        'client_metrics': summary['client'],
        'kv_pools': summary['pools'],
        'kv_events_by_pool': summary['events_by_pool'],
        'gpu_by_engine_pid': summary['gpu_by_engine_pid'],
        'artifacts_sha256': {name: sha256(run / name) for name in artifacts if (run / name).is_file()},
    }
    (run / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'run_dir': str(run), 'success': results['success'],
                      'timeout': results['timeout'], 'run_error': results['run_error']}, indent=2))


if __name__ == '__main__':
    main()
