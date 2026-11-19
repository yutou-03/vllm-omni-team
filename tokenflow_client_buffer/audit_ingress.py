"""Fail a paired benchmark when engine ordering or arrival timing is unmatched."""
import argparse
import json
from pathlib import Path


def read_case(case):
    client = Path(case)/'client'
    manifest = json.loads((client/'manifest.json').read_text())
    if manifest.get('ingress_order_policy') not in ('stage0_acknowledged', 'server_order_open_loop'):
        raise ValueError('case did not use acknowledged stage-0 ingress ordering')
    rows = [json.loads(line) for line in (client/'ingress_order.jsonl').read_text().splitlines()]
    expected = manifest['planned_ingress_order']
    if [r['request_id'] for r in rows] != expected or len(set(expected)) != len(expected):
        raise ValueError('engine ingress sequence is incomplete, duplicated or reordered')
    for i, r in enumerate(rows):
        if r['rank'] != i or str(r['stage_id']) != '0':
            raise ValueError('invalid ingress rank or stage')
        if not r['launched_monotonic'] <= r['enqueued_monotonic'] <= r['acknowledged_monotonic']:
            raise ValueError('invalid acknowledgement clock order')
        if manifest.get('ingress_order_policy') == 'stage0_acknowledged' and i and rows[i-1]['acknowledged_monotonic'] > r['launched_monotonic']:
            raise ValueError('next request launched before previous engine acknowledgement')
    return manifest, rows


def audit_pair(first, second, max_skew_s=.5):
    if max_skew_s < 0:
        raise ValueError('negative arrival skew tolerance')
    ma, a = read_case(first);mb, b = read_case(second)
    if ma['workload_sha256'] != mb['workload_sha256']:
        raise ValueError('paired workloads differ')
    if ma['planned_ingress_order'] != mb['planned_ingress_order']:
        raise ValueError('paired engine ingress orders differ')
    if ma.get('ingress_order_policy') != mb.get('ingress_order_policy'):
        raise ValueError('paired protocols differ')
    open_loop = ma.get('ingress_order_policy') == 'server_order_open_loop'
    skews = []; dispatch_skews = []; lags = []
    for x, y in zip(a, b):
        if abs((x['scheduled_monotonic']-ma['origin'])-(y['scheduled_monotonic']-mb['origin'])) > 1e-6:
            raise ValueError('paired planned arrival times differ')
        skews.append(abs((x['enqueued_monotonic']-ma['origin'])-(y['enqueued_monotonic']-mb['origin'])))
        dispatch_skews.append(abs((x['launched_monotonic']-ma['origin'])-(y['launched_monotonic']-mb['origin'])))
        lags.extend([x['launched_monotonic']-x['scheduled_monotonic'], y['launched_monotonic']-y['scheduled_monotonic']])
    maximum = max(dispatch_skews if open_loop else skews, default=0.)
    valid = maximum <= max_skew_s and (not open_loop or max(lags, default=0.) <= max_skew_s)
    return dict(valid=valid, requests=len(a), exact_ingress_order_match=True,
        max_dispatch_skew_s=max(dispatch_skews, default=0.), max_dispatch_lag_s=max(lags, default=0.),
        matching_scope="source_clock_and_native_order" if open_loop else "native_enqueue",
        max_enqueue_skew_s=max(skews, default=0.), max_allowed_skew_s=max_skew_s,
        reason='matched' if valid else 'actual_arrival_timing_mismatch',
        source='stage0_enqueue_acknowledgements',
        send_lag_included_in_ttfp=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second')
    p.add_argument('--max-skew-s',type=float,default=.5);p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        report=audit_pair(a.first,a.second,a.max_skew_s)
    except (ValueError,KeyError,OSError) as e:
        report=dict(valid=False,reason=str(e))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['valid']:raise SystemExit(1)


if __name__=='__main__':main()
