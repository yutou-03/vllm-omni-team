#!/usr/bin/env python3
"""Record host GPU indices and container PID namespace IDs for KV-pool labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess


def nvidia_smi(*args: str) -> list[str]:
    return subprocess.check_output(('nvidia-smi', *args), text=True).splitlines()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', required=True, help='comma-separated host GPU indices')
    a = p.parse_args()
    devices = {int(x) for x in a.devices.split(',')}
    uuid_to_index = {uuid.strip(): i for i, uuid in enumerate(
        nvidia_smi('--query-gpu=uuid', '--format=csv,noheader'))}
    processes = []
    for line in nvidia_smi('--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'):
        uuid, pid_raw, memory_raw = (item.strip() for item in line.split(','))
        gpu_index = uuid_to_index[uuid]
        if gpu_index not in devices:
            continue
        pid = int(pid_raw)
        status = Path(f'/proc/{pid}/status').read_text().splitlines()
        ns_line = next(item for item in status if item.startswith('NSpid:'))
        chain = [int(x) for x in ns_line.split(':', 1)[1].split()]
        processes.append({'pid': pid, 'gpu_index': gpu_index,
                          'used_gpu_memory_mib': int(memory_raw),
                          'container_pid': chain[-1], 'namespace_pid_chain': chain})
    if not processes:
        raise RuntimeError('no GPU process found on requested devices')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'source': 'nvidia-smi during replay',
                                    'processes': sorted(processes, key=lambda x: x['gpu_index'])},
                                   indent=2) + '\n')
    print(str(a.output))


if __name__ == '__main__':
    main()
