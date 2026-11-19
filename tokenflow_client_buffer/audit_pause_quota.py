"""Audit active-W quota transfers separately from per-iteration execution pauses."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    root=Path(root);cases=[];errors=[]
    for case in sorted((root/'runs').iterdir()):
        if not (case/'complete.json').exists():continue
        rows=json.loads((case/'client/results.json').read_text())
        by_id={r['request_id']:r for r in rows}
        config=json.loads((case/'gateway_config.json').read_text())
        record=dict(case=case.name,arm=config['arm'],requests=len(rows),outputs={})
        for m in ('text','audio'):
            selected=[r for r in rows if r['modality']==m]
            if not selected:continue
            proof_path=case/m/'mechanism.json'
            if not proof_path.exists():continue
            proof=json.loads(proof_path.read_text())
            events=[json.loads(x) for x in (case/m/'gateway.jsonl').open()]
            snapshots=[e for e in events if e['event']=='working_set']
            arrivals=[e for e in events if e['event']=='arrived']
            admitted={e['request_id']:e['monotonic'] for e in events if e['event']=='admitted'}
            stale=0.;fits_but_blocked=0.
            for a,b in zip(snapshots,snapshots[1:]):
                if (a['mode']=='fcfs' and a['reason']=='overload' and a.get('gamma')
                        and 0<a['demand']<=a['gamma'] and a.get('waiting',0)):
                    dt=b['monotonic']-a['monotonic'];stale+=dt
                    pending=[e for e in arrivals if e['monotonic']<=a['monotonic']
                             and admitted.get(e['request_id'],float('inf'))>a['monotonic']]
                    if pending and a['demand']+pending[0]['rate']<=a['gamma']+1e-9:fits_but_blocked+=dt
            witnesses=[]
            for w in proof.get('admission_quota_witnesses',[]):
                r=by_id.get(w['beneficiary']);first=r['buffer']['first_ready'] if r else None
                witnesses.append(dict(w,beneficiary_first_client_ready=first,
                    client_first_before_victim_resume=bool(first and w.get('resumed_at')
                        and w['released_at']<=first<w['resumed_at'])))
            firsts=sorted(r['first_ready_s'] for r in selected)
            record['outputs'][m]=dict(requests=len(selected),
                mean_first_ready_s=sum(firsts)/len(firsts),
                cumulative_stall_s=sum(r['buffer']['stalled_s'] for r in selected),
                execution_pause_resume_pairs=proof['complete_pause_resume_pairs'],
                quota_releases=proof.get('admission_quota_releases',0),
                early_admissions=sum(w['admitted_before_victim_generation_done'] for w in witnesses),
                complete_client_handoffs=sum(w['client_first_before_victim_resume'] for w in witnesses),
                max_active_requests=max(len(s['members']) for s in snapshots),
                max_paused_requests=max(len(s.get('paused_members',{})) for s in snapshots),
                max_outstanding_requests=max(len(s['members'])+len(s.get('paused_members',{})) for s in snapshots),
                feasible_but_stale_overload_s=stale,
                next_request_fits_but_stale_overload_blocks_s=fits_but_blocked,
                startup_gate_requests=sum(r['first_ready_s']!=r['startup_s'] for r in selected),
                witnesses=witnesses,errors=proof['errors'])
            errors.extend(case.name+': '+e for e in proof['errors'])
            if any(r['status']!='success' for r in selected):errors.append(case.name+': unsuccessful client')
        cases.append(record)
    result=dict(cases=cases,errors=sorted(set(errors)),
        working_set_definition='Active W excludes confirmed client-safe pauses; P retains request/KV state. Restores are charged before execution.',
        first_output_clock='Client first usable receipt, including decode, not server send time.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'reports/pause_quota_audit.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    result=audit(a.root)
    print(json.dumps(dict(cases=len(result['cases']),errors=result['errors'])))
