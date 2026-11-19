"""Matched 1:1 validation of resource-aware FIFO without changing buffer credit."""
import argparse
import json
from pathlib import Path
from .audit_mixed import audit_case


def summarize(root):
    root=Path(root);trace=Path(json.loads((root/'runtime/active_launch.json').read_text())['trace'])
    rows=[]
    for rep in (0,1):
        cases={'fcfs':f'mixed_t100_formal_{rep}_fcfs','buffer_global_fifo':f'mixed_t100_formal_{rep}_buffer',
               'buffer_resource_fifo':f'mixed_diag_resource_fifo_{rep}'}
        for policy,name in cases.items():
            c=root/'runs'/name
            if not (c/'complete.json').exists() or not (c/'text/mechanism.json').exists():continue
            p=c/'attribution.json';a=json.loads(p.read_text()) if p.exists() else audit_case(c,trace)
            for m in ('text','audio'):
                v=a['outputs'][m];s=v['summary']
                rows.append(dict(policy=policy,repetition=rep,modality=m,mean_first_ready_s=s['first_ready_mean_s'],
                    p95_first_ready_s=s['startup_p95_s'],cumulative_stall_s=s['stall_total_s'],generated_units_per_s=s['generated_units_per_s'],
                    admission_wait_mean_s=v['first_ready_components']['ingress_admission_wait_s']['mean'],
                    pauses=v['pauses'],active_mode_seconds=v['active_mode_seconds'],
                    thinker_idle_with_resource_feasible_text_waiting_s=a['admission_interference'].get('shared_thinker_no_live_requests_while_feasible_text_blocked_s'),
                    actual_pause_stall_s=a['continuity_exposure'][m].get('stall_during_actual_pause_s'),
                    errors=a['errors'],startup_gate_requests=a['startup_gate_requests']))
    comparisons=[]
    for m in ('text','audio'):
        for base in ('fcfs','buffer_global_fifo'):
            variant='buffer_resource_fifo'
            reps=sorted({r['repetition'] for r in rows if r['policy']==base and r['modality']==m}&
                        {r['repetition'] for r in rows if r['policy']==variant and r['modality']==m})
            if not reps:continue
            metrics={}
            for key in ('mean_first_ready_s','p95_first_ready_s','cumulative_stall_s','generated_units_per_s','admission_wait_mean_s'):
                values={p:sum(r[key] for r in rows if r['policy']==p and r['modality']==m and r['repetition'] in reps)/len(reps) for p in (base,variant)}
                metrics[key]=dict(**values,change_percent=100*(values[variant]-values[base])/abs(values[base]) if values[base] else None)
            comparisons.append(dict(modality=m,baseline=base,variant=variant,paired_repetitions=reps,metrics=metrics))
    report=dict(per_case=rows,comparisons=comparisons,
        changed='Only admission candidate selection; preserve audio-head Thinker cost when bypassing audio-only scarcity. Client buffering, rates, Gamma, headroom, native execution limit and pause rules unchanged.',
        limitations='Later validation on the same model and exact primary 1:1 workloads, not a re-run of every mixture. Temporal drift remains a limitation. Residual feasible-text waiting may include the event-loop interval and the deliberate shared-resource reservation for the audio head.')
    (root/'reports/resource_fifo_validation.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.root),indent=2))
