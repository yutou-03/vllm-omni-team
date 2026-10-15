#!/usr/bin/env python3
"""Build two real multimodal requests with explicit stage SLOs for a correctness probe."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--background-index', type=int, default=4)
    p.add_argument('--urgent-index', type=int, default=5)
    p.add_argument('--urgent-delay-s', type=float, default=10.0)
    p.add_argument('--urgent-slo-ms', type=float, default=9000.0)
    a = p.parse_args()
    if (not 0 < a.urgent_delay_s < 240 or a.background_index == a.urgent_index
            or not 0 < a.urgent_slo_ms < 120000):
        p.error('require distinct indices, delay in (0,240), and urgent SLO in (0,120000) ms')
    originals = [json.loads(line) for line in a.source.read_text().splitlines() if line]
    rows = []
    for order, (index, kind, arrival, slo) in enumerate((
            (a.background_index, 'background', 0., 120000.),
            (a.urgent_index, 'urgent', a.urgent_delay_s, a.urgent_slo_ms))):
        row = copy.deepcopy(originals[index])
        if 'audio' not in row['payload'].get('modalities', ()):
            raise ValueError('both requests must generate audio and traverse Talker')
        rid = f'urgentkeep-{kind}-{index:04d}'
        row['original_source_request_id'] = row['source_request_id']
        row['source_request_id'] = row['request_id'] = rid
        row['arrival_offset_s'] = arrival
        row['payload']['omni_scheduling'] = {
            'schema_version': 1, 'source_request_id': rid,
            'ingress_order': order, 'slo_ms': slo,
            'request_path': 'audio',
            'predicted_stage_ms': [3000., 5000., 1000.],
        }
        row['test_scope'] = 'two_request_urgent_keep_correctness_not_performance'
        rows.append(row)
    content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                      for row in rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        f.write(content)
    print(json.dumps({'path':str(a.output),'sha256':hashlib.sha256(content.encode()).hexdigest(),
                      'requests':len(rows),'urgent_delay_s':a.urgent_delay_s,
                      'urgent_slo_ms':a.urgent_slo_ms}))


if __name__ == '__main__':
    main()
