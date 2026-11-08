"""Join gateway, scheduler and client evidence without counting decisions as execution."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def verify(case, trace):
    case, trace = Path(case), Path(trace)
    gateway = read(case/'gateway.jsonl')
    results = json.loads((case/'client/results.json').read_text())
    source_ids = {r['request_id'] for r in results}
    start = min(e['monotonic'] for e in gateway if e['event']=='arrived')
    end = max(e['monotonic'] for e in gateway if e['event']=='terminal')
    events = [e for e in read(trace/'stage_events.jsonl') if start <= e['monotonic'] <= end+.5]
    iterations = [e for e in read(trace/'iteration_events.jsonl') if start <= e['monotonic'] <= end+.5]
    contracts = {e['request_id']:e for e in events if e['event']=='client_buffer_contract'
                 and e.get('source_request_id') in source_ids}
    outputs=defaultdict(list)
    for e in gateway:
        if e['event']=='server_output':outputs[e['request_id']].append(e['monotonic'])
    complete={e['request_id']:e for e in gateway if e['event']=='generation_complete'}
    errors=[]
    manifest_path=case/'service_manifest.json'
    if manifest_path.exists():
        config=json.loads(manifest_path.read_text()).get('scheduler_config',{})
        if results and config.get('modality')==results[0]['modality'] and len(contracts)!=len(source_ids):
            errors.append(f'missing scheduler rate contracts: {len(contracts)}/{len(source_ids)}')
    for result in results:
        rid=result['request_id']
        if result['status']!='success':errors.append(f'{rid}: client failure')
        if rid not in complete or complete[rid]['content_sha256']!=result['content_sha256']:
            errors.append(f'{rid}: content hash mismatch')
    for e in contracts.values():
        if e['requested_rate']!=e['observed_rate']:errors.append('rate propagation mismatch')
    for e in gateway:
        if e['event']=='working_set':
            actual=sum(r['rate'] for r in e['members'].values())
            if abs(actual-e['demand'])>1e-7:errors.append('W demand mismatch')
            if e['mode']=='buffer' and (e['gamma'] is None or actual>e['gamma']+1e-7):
                errors.append('buffer mode above capacity')
    pauses=[]
    fallback_releases=[]
    pending={}
    for e in events:
        if e['event']=='client_buffer_pause':
            if e.get('inventory_source')!='client_consumption_buffer':errors.append('nonclient pause credit')
            if e['coverage_s']<=0:errors.append('active pause without positive client credit')
            pending[e['request_id']]=e
        elif e['event']=='client_buffer_fallback_release':
            if e['request_id'] in pending:
                fallback_releases.append(dict(request_id=e['request_id'],reason=e['reason']))
                pending.pop(e['request_id'])
        elif e['event']=='client_buffer_resumed'  and e['request_id'] in pending:
            pause=pending.pop(e['request_id'])
            rid=e['request_id'];source=contracts.get(rid,{}).get('source_request_id')
            substituted=[i for i in iterations if pause['monotonic']<=i.get('schedule_monotonic_end_s',i['monotonic'])<e['monotonic']
                         and i['stage_id']==pause['stage_id'] and rid not in i['scheduled_req_ids'] and i['scheduled_req_ids']]
            later=[t for t in outputs[source] if t>=e['monotonic']]
            pauses.append(dict(request_id=rid,source_request_id=source,held_s=e['held_s'],
                coverage_s=pause['coverage_s'],substitution_steps=len(substituted),
                new_output_after_resume=bool(later),
                first_output_after_resume_s=min(later)-e['monotonic'] if later else None))
    modes=Counter((e['mode'],e.get('reason')) for e in events if e['event']=='client_buffer_mode')
    summary=dict(requests=len(results),contracts_observed=len(contracts),errors=sorted(set(errors)),
        complete_pause_resume_pairs=len(pauses),unique_paused_requests=len({p['source_request_id'] for p in pauses}),
        evidenced_substitutions=sum(p['substitution_steps']>0 and p['new_output_after_resume'] for p in pauses),
        unresolved_pauses=len(pending),fallback_releases=fallback_releases,modes={str(k):v for k,v in modes.items()},pauses=pauses,
        limitations='server-output timestamp is downstream observation; not a per-token GPU causal timestamp')
    (case/'mechanism.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('trace');a=p.parse_args()
    s=verify(a.case,a.trace)
    print(json.dumps({k:v for k,v in s.items() if k not in ('pauses','fallback_releases')},indent=2))
    if s['errors']:raise SystemExit(1)
