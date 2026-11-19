"""Separate auxiliary attribution from the primary two-policy matrix."""
import argparse
import csv
import json
from pathlib import Path
from .audit_mixed import audit_case


def summarize(root):
    root=Path(root);trace=Path(json.loads((root/'runtime/active_launch.json').read_text())['trace'])
    flat=[]
    for rep in (0,1):
        cases={'fcfs':f'audio_formal_{rep}_fcfs','buffer':f'audio_formal_{rep}_buffer',
               'capacity_only':f'audio_diag_capacity_only_{rep}', 'no_reserve':f'audio_diag_no_reserve_{rep}'}
        for policy,name in cases.items():
            case=root/'runs'/name;p=case/'attribution.json'
            if not (case/'complete.json').exists() or not (case/'audio/mechanism.json').exists():continue
            a=json.loads(p.read_text()) if p.exists() else {}
            if a.get('analysis_schema_version')!=4:a=audit_case(case,trace)
            v=a['outputs']['audio'];s=v['summary']
            flat.append(dict(policy=policy,repetition=rep,mean_first_ready_s=s['first_ready_mean_s'],
                p95_first_ready_s=s['startup_p95_s'],cumulative_stall_s=s['stall_total_s'],
                generated_units_per_s=s['generated_units_per_s'],
                mean_admission_wait_s=v['first_ready_components']['ingress_admission_wait_s']['mean'],
                pauses=v['pauses'],max_working_set=v['working_set_size']['maximum'],
                startup_gate_requests=a['startup_gate_requests'],errors=a['errors']))
    contrasts=[]
    for baseline,variant,meaning in (
            ('fcfs','capacity_only','Effect of capacity admission with stage execution remaining FCFS'),
            ('capacity_only','buffer','Effect of preemption at identical 85-percent admission headroom'),
            ('buffer','no_reserve','Effect of removing extra admission headroom while retaining full scheduling and demand <= Gamma')):
        reps=sorted({r['repetition'] for r in flat if r['policy']==baseline}&{r['repetition'] for r in flat if r['policy']==variant})
        if not reps:continue
        metrics={}
        for key in ('mean_first_ready_s','p95_first_ready_s','cumulative_stall_s','generated_units_per_s','mean_admission_wait_s','pauses'):
            means={p:sum(r[key] for r in flat if r['policy']==p and r['repetition'] in reps)/len(reps) for p in (baseline,variant)}
            metrics[key]=dict(**means,change_percent=100*(means[variant]/means[baseline]-1) if means[baseline] else None)
        contrasts.append(dict(baseline=baseline,variant=variant,meaning=meaning,paired_repetitions=reps,metrics=metrics))
    result=dict(per_case=flat,contrasts=contrasts,
        limitations=['Auxiliary cases run later on the same loaded model and repeat the exact workloads and calibrated Gamma. They are not temporally interleaved with the primary arms; time drift is not completely controlled.',
                     'These diagnostics are not additional primary policies and must not replace the original two-policy result.',
                     'No-reserve uses 100 percent of the measured Gamma for admission; it does not relax the demand <= Gamma constraint.'])
    report=root/'reports';report.mkdir(exist_ok=True)
    (report/'audio_diagnostics.json').write_text(json.dumps(result,indent=2))
    if flat:
        with (report/'audio_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.root),indent=2))
