"""Audit the controlled capacity-drop case separately from performance runs."""
import argparse
import json
from pathlib import Path

from .verify_run import read, verify


def verify_drop(case, trace):
    case, trace = Path(case), Path(trace)
    basic=verify(case,trace)
    injection=json.loads((case/'fault_injection.json').read_text())
    when=injection['injected_at']
    events=read(case/'gateway.jsonl')
    start=min(e['monotonic'] for e in events if e['event']=='arrived')
    end=max(e['monotonic'] for e in events if e['event']=='terminal')
    scheduler=[e for e in read(trace/'stage_events.jsonl') if start<=e['monotonic']<=end+.5]
    errors=list(basic['errors'])
    gamma=None; rates={}; members=set(); violations=[]
    for e in events:
        event=e['event']
        if event=='arrived':rates[e['request_id']]=e['rate']
        elif event in ('capacity_updated','capacity_window'):
            gamma=e['estimate']['gamma']
        elif event=='admitted':
            demand=sum(rates[rid] for rid in members)
            if gamma is not None and demand>gamma+1e-7:
                violations.append(dict(request_id=e['request_id'],demand=demand,gamma=gamma))
            members.add(e['request_id'])
        elif event=='terminal':members.discard(e['request_id'])
    overload=[e for e in events if e['event']=='working_set' and e['reason']=='overload' and e['monotonic']>=when]
    fallback=[e for e in scheduler if e['event']=='client_buffer_mode' and e.get('mode')=='fcfs'
              and e.get('reason')=='overload' and e['monotonic']>=when]
    released=[e for e in scheduler if e['event']=='client_buffer_fallback_release' and e.get('reason')=='overload'
              and e['monotonic']>=when]
    iterations=[e for e in read(trace/'iteration_events.jsonl') if start<=e['monotonic']<=end+.5]
    mode=None; forbidden_pauses=[]; same_iteration_mode_corrections=0
    for e in scheduler:
        if e['event']=='client_buffer_mode':mode=e['mode']
        elif e['event']=='client_buffer_pause' and mode=='fcfs' and e['monotonic']>=when:
            # The adapter emits pause decisions before its mode summary inside
            # one schedule() call. Resolve the actual iteration, not line order.
            iteration=next((i for i in iterations if i['stage_id']==e['stage_id'] and
                i['schedule_monotonic_start_s']<=e['monotonic']<=i['schedule_monotonic_end_s']),None)
            contemporaneous=[m for m in scheduler if iteration and m['event']=='client_buffer_mode'
                and m['stage_id']==e['stage_id'] and iteration['schedule_monotonic_start_s']<=m['monotonic']
                <=iteration['schedule_monotonic_end_s']]
            actual=contemporaneous[-1]['mode'] if contemporaneous else mode
            if actual=='fcfs':forbidden_pauses.append(e['request_id'])
            else:same_iteration_mode_corrections+=1
    if not overload:errors.append('no gateway overload observed')
    if not fallback:errors.append('no actual scheduler overload fallback observed')
    if not released:errors.append('no held request released by overload fallback')
    if violations:errors.append('new admission while existing W exceeded capacity')
    if forbidden_pauses:errors.append('active pause during FCFS mode')
    summary=dict(errors=errors,requests=basic['requests'],injected_gamma=injection['estimate']['gamma'],
        ledger_detection_s=overload[0]['monotonic']-when if overload else None,
        scheduler_fallback_s=fallback[0]['monotonic']-when if fallback else None,
        paused_requests_released=len(released),overload_admission_violations=violations,
        fcfs_active_pause_violations=forbidden_pauses,
        same_iteration_mode_corrections=same_iteration_mode_corrections,
        late_arrivals=sum(e['event']=='arrived' and e['monotonic']>when for e in events),
        note='Controlled estimate drop, not a physical GPU slowdown; online capacity estimator remains enabled.')
    (case/'capacity_drop_verification.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('trace');a=p.parse_args()
    result=verify_drop(a.case,a.trace);print(json.dumps(result,indent=2))
    if result['errors']:raise SystemExit(1)
