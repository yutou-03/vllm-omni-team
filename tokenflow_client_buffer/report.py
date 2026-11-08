"""Paired pilot aggregation and exportable figures; failures remain visible."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics


METRICS = ('paper_qos', 'startup_mean_s', 'startup_p95_s', 'stall_total_s',
           'generated_units_per_s', 'observation_s', 'slo_goodput_requests_per_s')


def interval(values):
    if len(values) < 2:
        return None
    rng = random.Random(4517)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [means[249], means[9749]]


def aggregate(root, prefix, count=5):
    root = Path(root)
    pairs, rows = [], []
    for rep in range(count):
        arms = {}
        for arm in ('fcfs', 'buffer'):
            case = root/'runs'/f'{prefix}_{rep}_{arm}'
            if not (case/'complete.json').exists():
                raise ValueError(f'incomplete case: {case}')
            s = json.loads((case/'client/summary.json').read_text())
            results = json.loads((case/'client/results.json').read_text())
            arms[arm] = dict(summary=s, results=results, case=str(case))
            rows.append(dict(repetition=rep, arm=arm, completed=s['completed'],
                             failures=s['failures'], units=s['units'], **{k:s[k] for k in METRICS}))
        hashes = {r['request_id']:r['content_sha256'] for r in arms['fcfs']['results']}
        equal = sum(hashes.get(r['request_id']) == r['content_sha256'] for r in arms['buffer']['results'])
        pairs.append(dict(repetition=rep, identical_outputs=equal,
                          requests=len(arms['buffer']['results']), arms=arms))
    metrics = {}
    for key in METRICS:
        a = [p['arms']['fcfs']['summary'][key] for p in pairs]
        b = [p['arms']['buffer']['summary'][key] for p in pairs]
        if any(v is None for v in a+b):
            metrics[key] = dict(error='failed run; aggregate unavailable')
            continue
        differences = [y-x for x,y in zip(a,b)]
        metrics[key] = dict(fcfs_mean=statistics.mean(a), buffer_mean=statistics.mean(b),
                            paired_difference_mean=statistics.mean(differences),
                            paired_difference_bootstrap_95=interval(differences),
                            per_pair_fcfs=a, per_pair_buffer=b)
        if all(x > 0 for x in a):
            relative = [100*(y/x-1) for x,y in zip(a,b)]
            metrics[key].update(paired_relative_percent_mean=statistics.mean(relative),
                                paired_relative_bootstrap_95=interval(relative))
    summary = dict(prefix=prefix, repetitions=count, metrics=metrics,
                   identical_outputs=sum(p['identical_outputs'] for p in pairs),
                   paired_requests=sum(p['requests'] for p in pairs),
                   failures=sum(r['failures'] for r in rows),
                   caveat='Small fixed-profile pilot. Bootstrap across paired repetitions only; not a stable P99 or general workload claim.',
                   cases=[{k:p[k] for k in ('repetition','identical_outputs','requests')} for p in pairs])
    out = root/'reports'; out.mkdir(exist_ok=True)
    (out/f'{prefix}_comparison.json').write_text(json.dumps(summary,indent=2))
    with (out/f'{prefix}_runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary, pairs


def plots(root, prefix, summary, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, grid=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    axes=grid.flat
    for ax,key,title in zip(axes,('startup_mean_s','stall_total_s','paper_qos','generated_units_per_s'),
                            ('Mean startup (s)','Total rebuffering per run (s)','Streaming QoS',
                             'Raw generation (tokens/s)' if prefix.startswith('text') else 'Raw generation (media seconds/s)')):
        metric=summary['metrics'][key]
        for x,y in zip(metric['per_pair_fcfs'],metric['per_pair_buffer']):
            ax.plot([0,1],[x,y],marker='o',alpha=.65)
        ax.set_xticks([0,1],['FCFS\nK=4, R=16','Client buffer\nK=4, R=16']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle(prefix+' — paired repetitions')
    out=Path(root)/'reports'
    fig.savefig(out/f'{prefix}_paired.png',dpi=180);fig.savefig(out/f'{prefix}_paired.pdf');plt.close(fig)
    # One explicitly identified representative run, not the best observed run.
    case=Path(pairs[0]['arms']['buffer']['case'])
    events=[json.loads(x) for x in (case/'gateway.jsonl').read_text().splitlines()]
    start=min(e['monotonic'] for e in events if e['event']=='arrived')
    ws=[e for e in events if e['event']=='working_set']
    fig,axes=plt.subplots(3,1,figsize=(9,6.5),sharex=True,layout='constrained')
    t=[e['monotonic']-start for e in ws]
    axes[0].step(t,[e['demand'] for e in ws],where='post',label='Demand: sum r_i')
    axes[0].step(t,[e['gamma'] if e['gamma'] is not None else float('nan') for e in ws],where='post',label='Estimated capacity Gamma')
    axes[0].legend();axes[0].set_ylabel('Output units / second')
    axes[1].step(t,[len(e['members']) for e in ws],where='post',label='Working set')
    axes[1].step(t,[e['waiting'] for e in ws],where='post',label='Waiting')
    axes[1].legend();axes[1].set_ylabel('Logical requests')
    axes[2].step(t,[int(e['mode']=='buffer') for e in ws],where='post')
    axes[2].set_yticks([0,1],['FCFS','Buffer']);axes[2].set_xlabel('Seconds since first arrival');axes[2].set_ylabel('Ledger mode')
    fig.suptitle(prefix+' — repetition 0 capacity ledger')
    fig.savefig(out/f'{prefix}_capacity.png',dpi=180);fig.savefig(out/f'{prefix}_capacity.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('prefix');p.add_argument('--count',type=int,default=5)
    a=p.parse_args();summary,pairs=aggregate(a.root,a.prefix,a.count);plots(a.root,a.prefix,summary,pairs)
    print(json.dumps(summary,indent=2))
