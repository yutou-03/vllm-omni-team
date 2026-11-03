"""Freeze a larger 16-loose/16-urgent arrival burst, preserving V3 metrics."""
import argparse, copy, hashlib, json
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    base, root = a.base_root, a.root
    root.mkdir(parents=True, exist_ok=False)
    old = [json.loads(line) for line in (base / 'workload.jsonl').read_text().splitlines()]
    assert len(old) == 20 and all(x['request_path'] == 'text' for x in old)
    assert all(x['payload']['max_tokens'] == 512 for x in old[:16])
    assert all(x['payload']['max_tokens'] == 32 for x in old[16:])
    rows = [copy.deepcopy(x) for x in old[:16]]
    for index in range(16, 32):
        row = copy.deepcopy(old[16 + (index - 16) % 4])
        rid = f'v3-burst32-r{index}'
        row['source_request_id'] = row['request_id'] = rid
        row['arrival_offset_s'] = 6. + (index - 16) * .005
        row['payload']['omni_scheduling']['source_request_id'] = rid
        row['payload']['omni_scheduling']['ingress_order'] = index
        rows.append(row)
    workload = root / 'workload.jsonl'
    workload.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    for name in ('text_delivery.json', 'metric.json'):
        (root / name).write_bytes((base / name).read_bytes())
    dump(root / 'contracts.json', {row['source_request_id']:
         {'ttft_slo_ms': row['ttft_slo_ms'], 'first_output_service_ms': 400.}
         for row in rows})
    for policy in ('dynamic', 'fixed'):
        config = json.loads((base / f'{policy}_controller.json').read_text())
        config['contracts_path'] = str(root / 'contracts.json')
        dump(root / f'{policy}_controller.json', config)
    plan = json.loads((base / 'plan.json').read_text())
    plan.update(scope='larger_text_burst_single_seed_sensitivity_not_formal',
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        offered=len(rows), seed=1104, arms=['dynamic16', 'nohold16', 'buffer16'],
        limits='32 text requests; 16 urgent arrived at 6s; no audio or K64 generalization; same metric and limits as V3',
        gate='first dynamic arm must pass multiple real hold chains, retained KV and exact client integrity before controls')
    dump(root / 'plan.json', plan)
    print(json.dumps({'root': str(root), 'workload_sha256': plan['workload_sha256'],
                      'offered': len(rows), 'arms': plan['arms']}))

if __name__ == '__main__':
    main()
