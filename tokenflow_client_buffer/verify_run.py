"""Join gateway, scheduler and client evidence without counting decisions as execution."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def iter_read(path, *, start=None, end=None):
    with Path(path).open() as stream:
        for line in stream:
            if start is not None:
                raw=line.partition('"monotonic": ')[2]
                if raw:
                    timestamp=float(raw.partition(',')[0].partition('}')[0])
                    if timestamp<start or end is not None and timestamp>end:continue
            if line.strip():
                yield json.loads(line)


def read(path):
    return list(iter_read(path))


def verify(case, trace):
    case, trace = Path(case), Path(trace)
    gateway = read(case/'gateway.jsonl')
    results = json.loads((case/'client/results.json').read_text())
    source_ids = {r['request_id'] for r in results}
    start = min(e['monotonic'] for e in gateway if e['event']=='arrived')
    end = max(e['monotonic'] for e in gateway if e['event']=='terminal')
    # Detailed stage-study snapshots can be gigabytes; mechanism validation
    # only needs adapter contracts/modes/pause/resume events. Filter the raw
    # event marker before decoding unrelated tensor/KV observations.
    events = []
    with (trace/'stage_events.jsonl').open() as stream:
        for line in stream:
            if '"event": "client_buffer_' not in line and '"event":"client_buffer_' not in line:
                continue
            e = json.loads(line)
            if start <= e['monotonic'] <= end+.5:
                events.append(e)
    iterations = [e for e in iter_read(trace/'iteration_events.jsonl',start=start,end=end+.5) if start <= e['monotonic'] <= end+.5]
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
        if 'by_stage' in config and results:
            stage=0 if results[0]['modality']=='text' else 1
            config=config['by_stage'][str(stage)]
            events=[e for e in events if not e['event'].startswith('client_buffer_') or e['stage_id']==stage]
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
    gateway_cfg = json.loads((case/'gateway_config.json').read_text()) if (case/'gateway_config.json').exists() else {}
    expected_source = gateway_cfg.get('buffer_source', 'client_consumption_buffer')
    pauses=[]
    fallback_releases=[]
    pending={}
    for e in events:
        if e['event']=='client_buffer_pause':
            if e.get('inventory_source') != expected_source:errors.append('unexpected pause credit source')
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
    quota_releases = [e for e in gateway if e['event'] == 'pause_admission_released']
    quota_restores = [e for e in gateway if e['event'] == 'pause_admission_restored']
    quota_witnesses = []
    for release in quota_releases:
        source, epoch = release['request_id'], release['epoch']
        pause = next((e for e in events if e['event'] == 'client_buffer_pause'
                      and e.get('source_request_id') == source and e.get('lease_epoch') == epoch), None)
        if pause is None:
            errors.append('quota release without scheduler pause proof')
            continue
        restore = next((e for e in quota_restores if e['request_id'] == source and e['epoch'] == epoch), None)
        resume = next((e for e in events if e['event'] == 'client_buffer_resumed'
                       and e['request_id'] == pause['request_id'] and e['monotonic'] >= release['monotonic']), None)
        if resume and (restore is None or restore['monotonic'] > resume['monotonic']):
            errors.append('execution resumed before quota restoration')
        beneficiary = pause.get('beneficiary_source_request_id')
        admission = next((e for e in gateway if e['event'] == 'admitted'
                          and e['request_id'] == beneficiary and e['monotonic'] >= release['monotonic']), None)
        first_server_output = min(outputs.get(beneficiary, []), default=None)
        beneficiary_result = next((r for r in results if r['request_id'] == beneficiary), None)
        first_output = (beneficiary_result.get('buffer', {}).get('first_ready')
                        if beneficiary_result else None)
        quota_witnesses.append(dict(source_request_id=source, beneficiary=beneficiary,
            epoch=epoch, released_at=release['monotonic'],
            restored_at=restore['monotonic'] if restore else None,
            resumed_at=resume['monotonic'] if resume else None,
            beneficiary_admitted_at=admission['monotonic'] if admission else None,
            beneficiary_first_output=first_output,
            beneficiary_first_server_output=first_server_output,
            admitted_before_victim_generation_done=bool(admission and source in complete
                and admission['monotonic'] < complete[source]['monotonic']),
            first_output_before_victim_resumed=bool(first_output and resume and
                release['monotonic'] <= first_output < resume['monotonic'])))
    if any(e['event'] == 'client_buffer_resume_ack_timeout' for e in events):
        errors.append('resume quota acknowledgement timeout')
    modes=Counter((e['mode'],e.get('reason')) for e in events if e['event']=='client_buffer_mode')
    iteration_stats={}
    for stage in sorted({i['stage_id'] for i in iterations}):
        active=[i for i in iterations if i['stage_id']==stage and i['scheduled_req_ids']]
        if active:
            iteration_stats[str(stage)]=dict(iterations=len(active),
                max_scheduled_requests=max(len(i['scheduled_req_ids']) for i in active),
                mean_scheduled_requests=sum(len(i['scheduled_req_ids']) for i in active)/len(active),
                max_kv_usage_fraction=max(i.get('kv_cache_usage') or 0. for i in active))
    snapshots=[e for e in gateway if e['event']=='working_set']
    dwell=Counter()
    for previous,current in zip(snapshots,snapshots[1:]):
        if previous['members']:
            dwell[(previous['mode'],previous['reason'])]+=max(0.,min(current['monotonic'],end)-previous['monotonic'])
    windows=[e for e in gateway if e['event']=='capacity_window']
    summary=dict(requests=len(results),contracts_observed=len(contracts),errors=sorted(set(errors)),
        iteration_statistics=iteration_stats,
        active_working_set_mode_seconds={str(k):v for k,v in dwell.items()},
        capacity_windows=len(windows),
        saturated_capacity_windows=sum(e['saturated'] for e in windows),
        accepted_capacity_windows=sum(e.get('updated', e['saturated']) for e in windows),
        complete_pause_resume_pairs=len(pauses),unique_paused_requests=len({p['source_request_id'] for p in pauses}),
        evidenced_substitutions=sum(p['substitution_steps']>0 and p['new_output_after_resume'] for p in pauses),
        unresolved_pauses=len(pending),fallback_releases=fallback_releases,modes={str(k):v for k,v in modes.items()},pauses=pauses,
        admission_quota_releases=len(quota_releases), admission_quota_restores=len(quota_restores),
        admission_quota_witnesses=quota_witnesses,
        admission_first_output_clock='client first usable receipt; server output retained separately',
        limitations='server-output timestamp is downstream observation; not a per-token GPU causal timestamp')
    (case/'mechanism.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('trace');a=p.parse_args()
    s=verify(a.case,a.trace)
    print(json.dumps({k:v for k,v in s.items() if k not in ('pauses','fallback_releases')},indent=2))
    if s['errors']:raise SystemExit(1)
