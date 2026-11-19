"""Read-only attribution of matched isolated/mixed streaming experiments.

Does not tune workloads or policy. Keeps modalities separate and decomposes
first-ready latency using gateway and shared-stage timestamps.
"""
import argparse
from collections import Counter, defaultdict
import csv
import bisect
import hashlib
import json
from pathlib import Path
from .analyze import quantile


def records(path, skip_feedback=False, time_bounds=None):
    with Path(path).open() as stream:
        for line in stream:
            if skip_feedback and ('"event": "client_feedback"' in line or '"event": "server_output"' in line):
                continue
            if time_bounds is not None:
                raw=line.partition('"monotonic": ')[2]
                if raw:
                    timestamp=float(raw.partition(',')[0].partition('}')[0])
                    if not time_bounds[0]<=timestamp<=time_bounds[1]:continue
            if line.strip():
                yield json.loads(line)


def distribution(values):
    values=list(values)
    return dict(n=len(values),mean=sum(values)/len(values) if values else None,
                p50=quantile(values,.5),p95=quantile(values,.95),maximum=max(values) if values else None)


def admission_interference(gateway):
    # Reconstruct the server-owned global FIFO at event boundaries. Credit is
    # never inferred from interstage data; this is offline resource attribution.
    timeline=sorted((e for events in gateway.values() for e in events
                     if e['event'] in ('arrived','admitted','terminal','working_set')),
                    key=lambda e:e['monotonic'])
    pending={};members={};bounds={};weights={};mode=None;last=None
    wall=Counter();request_wait=Counter();longest=0.;episode_start=None;text_blocked=[]
    for e in timeline:
        now=e['monotonic']
        eligible=[];blocked_by=None
        if last is not None and pending and mode=='buffer' and bounds:
            queue=sorted(pending.values(),key=lambda r:r['order'])
            demand={'thinker':sum(r['rate']*weights[r['modality']] for r in members.values()),
                    'audio':sum(r['rate'] for r in members.values() if r['modality']=='audio')}
            def fits(row):
                return (demand['thinker']+row['rate']*weights[row['modality']]<=bounds['thinker']+1e-7
                        and (row['modality']=='text' or demand['audio']+row['rate']<=bounds['audio']+1e-7))
            if not fits(queue[0]):
                blocked_by=queue[0]['modality']
                eligible=[r for r in queue[1:] if r['modality']!=blocked_by and fits(r)]
            if eligible:
                dt=now-last
                wall[blocked_by+'_head_blocks_'+eligible[0]['modality']]+=dt
                request_wait[eligible[0]['modality']]+=dt*len(eligible)
                if eligible[0]['modality']=='text':
                    if text_blocked and abs(text_blocked[-1][1]-last)<1e-8:text_blocked[-1][1]=now
                    else:text_blocked.append([last,now])
                if episode_start is None:episode_start=last
                longest=max(longest,now-episode_start)
            else:episode_start=None
        event=e['event'];rid=e.get('request_id')
        if event=='arrived':
            meta=e['payload']['omni_scheduling']
            pending[rid]=dict(modality=meta['request_path'],rate=e['rate'],order=meta['ingress_order'])
        elif event=='admitted' and rid in pending:members[rid]=pending.pop(rid)
        elif event=='terminal':members.pop(rid,None);pending.pop(rid,None)
        elif event=='working_set':
            bounds=e.get('resource_admission_gamma',{});weights=e.get('resource_coefficients',{});mode=e['mode']
        last=now
    return dict(fifo_head_blocking_wall_s=dict(wall),otherwise_resource_feasible_request_wait_s=dict(request_wait),
                longest_episode_s=longest,resource_feasible_text_blocked_intervals=text_blocked,
                interpretation='Time when a request of the other modality fits the estimated resource budget but global FIFO blocks it behind an infeasible head. This is admission policy coupling, not measured GPU execution contention.')


def continuity_exposure(case,rows,pairs,modes,end):
    held=defaultdict(list);recover=defaultdict(list);receipts={};latencies=defaultdict(list)
    byid={r['request_id']:r for r in rows}
    fallback=defaultdict(list)
    for modality,events in modes.items():
        events=sorted(events,key=lambda e:e['monotonic'])
        for i,e in enumerate(events):
            if e['mode']=='fcfs':fallback[modality].append((e['monotonic'],events[i+1]['monotonic'] if i+1<len(events) else end))
    for r in rows:
        rid=r['request_id'];key=hashlib.sha256(rid.encode()).hexdigest()[:16]
        receipts[rid]=[e for e in records(case/'client'/key/'events.jsonl') if e['event']=='received']
    for pause,resume in pairs:
        rid=pause.get('source_request_id')
        if rid not in receipts:continue
        a,b=pause['monotonic'],resume['monotonic'];held[rid].append((a,b))
        ready=[e['ready_at'] for e in receipts[rid]];i=bisect.bisect_left(ready,b)
        if i<len(ready):
            recover[rid].append((b,ready[i]));latencies[byid[rid]['modality']].append(ready[i]-b)
    def union(spans):
        result=[]
        for a,b in sorted(spans):
            if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
            else:result.append((a,b))
        return result
    def overlap(a,b,spans):return sum(max(0.,min(b,y)-max(a,x)) for x,y in spans)
    total=defaultdict(Counter)
    for r in rows:
        rid=r['request_id'];m=r['modality'];events=receipts[rid]
        exposures=union(held[rid]+recover[rid]);recoveries=union(recover[rid])
        for i,e in enumerate(events):
            a=e['ready_at']+(e['buffered_before']+e['units'])/r['rate']
            b=events[i+1]['ready_at'] if i+1<len(events) else r['generation_done_at']
            if b>a:
                total[m]['reconstructed_stall_s']+=b-a
                total[m]['stall_during_adapter_fcfs_s']+=overlap(a,b,fallback[m])
                total[m]['stall_during_actual_pause_s']+=overlap(a,b,held[rid])
                total[m]['stall_between_resume_and_next_receipt_s']+=overlap(a,b,recoveries)
                total[m]['stall_during_pause_or_replenishment_s']+=overlap(a,b,exposures)
        total[m]['reported_stall_s']+=r['stall_s']
    return {m:dict(v,resume_to_next_client_receipt_s=distribution(latencies[m]),
                    note='Pause/replenishment overlap is exposure, not a counterfactual causal attribution; next receipt can be in-flight output.') for m,v in total.items()}


def audit_case(case,trace):
    case=Path(case)
    rows=json.loads((case/'client/results.json').read_text())
    gateway={m:list(records(case/m/'gateway.jsonl',skip_feedback=True)) for m in ('text','audio')}
    arrival={};admission={};terminal={};lookup={r['request_id']:r for r in rows}
    for events in gateway.values():
        for e in events:
            target={'arrived':arrival,'admitted':admission,'terminal':terminal}.get(e['event'])
            if target is not None:target[e['request_id']]=e['monotonic']
    begin=min(arrival.values());end=max(terminal.values())
    source={};ready={};forward={};done={};pauses={};pause_pairs=[];modes=defaultdict(list)
    for e in records(trace/'stage_events.jsonl',time_bounds=(begin,end+.5)):
        if not begin<=e['monotonic']<=end+.5:continue
        event=e['event'];rid=e.get('request_id');stage=e.get('stage_id');key=(stage,rid)
        if e.get('sched_source_request_id') in lookup:source[rid]=e['sched_source_request_id']
        if event=='client_buffer_mode' and stage in (0,1):modes['text' if stage==0 else 'audio'].append(e)
        if event=='stage_data_ready':ready[key]=min(ready.get(key,float('inf')),e['monotonic'])
        elif event=='stage_done_metrics':done[key]=max(done.get(key,0.),e['monotonic'])
        elif event=='stage_first_forward':forward[key]=min(forward.get(key,float('inf')),e['monotonic'])
        elif event=='client_buffer_pause':pauses[key]=e
        elif event=='client_buffer_resumed' and key in pauses:
            p=pauses.pop(key);pause_pairs.append((p,e))
    physical={src:rid for rid,src in source.items()}
    request_rows=[]
    for r in rows:
        rid=r['request_id'];gpu_id=physical.get(rid);t=forward.get((0,gpu_id));rd=ready.get((0,gpu_id))
        row=dict(request_id=rid,modality=r['modality'],status=r['status'],
            first_ready_s=r['first_ready_s'],send_lag_s=r['send_lag_s'],stall_s=r['stall_s'],
            ingress_admission_wait_s=admission[rid]-arrival[rid],
            admitted_to_thinker_first_forward_s=t-admission[rid] if t else None,
            thinker_ready_to_first_forward_s=t-rd if t and rd else None,
            thinker_forward_to_client_first_ready_s=r['scheduled_at']+r['first_ready_s']-t if t else None,
            generation_s=r['generation_done_at']-r['scheduled_at'])
        for stage in (1,2):
            s=forward.get((stage,gpu_id));data=ready.get((stage,gpu_id))
            row[f'stage{stage}_ready_to_first_forward_s']=s-data if s and data else None
            row[f'thinker_to_stage{stage}_ready_s']=data-t if data and t else None
        request_rows.append(row)
    iterations=defaultdict(list)
    for e in records(trace/'iteration_events.jsonl',time_bounds=(begin,end+.5)):
        if not begin<=e['monotonic']<=end+.5 or not e['scheduled_req_ids']:continue
        n=Counter(lookup[source[rid]]['modality'] for rid in e['scheduled_req_ids'] if rid in source)
        if n:iterations[e['stage_id']].append((e,n))
    stage_stats={}
    for stage,items in iterations.items():
        prefix=[e.get('schedule_prefix_monotonic_ms',0)/1000 for e,n in items]
        stage_stats[str(stage)]=dict(iterations=len(items),scheduler_wall_total_s=sum(prefix),
            scheduler_wall_s=distribution(prefix),batch_size=distribution(sum(n.values()) for e,n in items),
            both_modalities_iterations=sum(n['text']>0 and n['audio']>0 for e,n in items),
            scheduled_request_steps={m:sum(n[m] for e,n in items) for m in ('text','audio')})
    outputs={};errors=[]
    for m in ('text','audio'):
        rr=[r for r in request_rows if r['modality']==m]
        if not rr:continue
        s=json.loads((case/m/'client/summary.json').read_text())
        mechanism=json.loads((case/m/'mechanism.json').read_text())
        errors.extend(mechanism['errors'])
        state=[e for e in gateway[m] if e['event']=='working_set']
        dwell=Counter();gamma=[];sizes=[]
        for e in state:
            sizes.append(len(e['members']))
            if e['gamma']:gamma.append(e['gamma'])
            if e['mode']=='buffer':
                for resource,demand in e['resource_demand'].items():
                    cap=e['resource_gamma'].get(resource)
                    if demand and (cap is None or demand>cap+1e-7):errors.append('vector capacity violation '+resource)
        for old,new in zip(state,state[1:]):
            if old['members']:dwell[old['mode']+':'+old['reason']]+=max(0.,min(new['monotonic'],end)-old['monotonic'])
        outputs[m]=dict(summary={k:v for k,v in s.items() if k not in ('request_scores',)},
            first_ready_components={k:distribution(r[k] for r in rr if r[k] is not None)
                for k in rr[0] if k.endswith('_s')},
            pauses=mechanism['complete_pause_resume_pairs'],unique_paused_requests=mechanism['unique_paused_requests'],
            evidenced_substitutions=mechanism['evidenced_substitutions'],
            resume_to_server_output_s=distribution(p['first_output_after_resume_s'] for p in mechanism['pauses'] if p['first_output_after_resume_s'] is not None),
            held_s=distribution(p['held_s'] for p in mechanism['pauses']),
            active_mode_seconds=dict(dwell),working_set_size=distribution(sizes),residual_gamma=distribution(gamma))
    # Beneficiary attribution is explicit in pause records when available;
    # raw stage overlap is descriptive, not proof of cross-modal causation.
    benefits=Counter();beneficiary_stage_state=Counter();late_benefit_examples=[]
    for p,r in pause_pairs:
        victim=lookup.get(p.get('source_request_id'),{})
        beneficiary=lookup.get(p.get('beneficiary_source_request_id'),{})
        benefits[victim.get('modality','unknown')+'->'+beneficiary.get('modality','resume_due_or_unknown')] += 1
        if beneficiary.get('modality')=='audio':
            bid=physical.get(p.get('beneficiary_source_request_id'));timepoint=p['monotonic']
            prefix=victim.get('modality','unknown')+'->audio'
            beneficiary_stage_state[prefix+':all']+=1
            ready2=ready.get((2,bid));forward2=forward.get((2,bid))
            first=beneficiary['scheduled_at']+beneficiary['first_ready_s']
            if ready2 is not None and ready2<=timepoint:
                beneficiary_stage_state[prefix+':first_waveform_input_already_ready']+=1
                if len(late_benefit_examples)<10:late_benefit_examples.append(dict(victim=p.get('source_request_id'),beneficiary=p.get('beneficiary_source_request_id'),pause_after_waveform_ready_s=timepoint-ready2,client_first_ready_after_pause_s=first-timepoint))
            if forward2 is not None and forward2<=timepoint:beneficiary_stage_state[prefix+':first_waveform_execution_already_started']+=1
            if first<=timepoint:beneficiary_stage_state[prefix+':client_already_started_feedback_not_yet_seen']+=1
    admission=admission_interference(gateway)
    active=[]
    for (stage,rid),start in sorted(ready.items(),key=lambda item:item[1]):
        if stage!=0 or rid not in source:continue
        finish=done.get((0,rid),end)
        if active and start<=active[-1][1]:active[-1][1]=max(active[-1][1],finish)
        else:active.append([start,finish])
    blocked=admission['resource_feasible_text_blocked_intervals']
    admission['shared_thinker_no_live_requests_while_feasible_text_blocked_s']=sum(
        b-a-sum(max(0.,min(b,y)-max(a,x)) for x,y in active) for a,b in blocked)
    admission['idle_definition']='No request alive at stage 0 (first data-ready to stage-done interval), while a queued text request fits the resource budget but is blocked by global FIFO. Conservative, not a hardware-utilization sample.'
    result=dict(analysis_schema_version=4,case=case.name,errors=sorted(set(errors)),outputs=outputs,stage_statistics=stage_stats,
        startup_gate_requests=sum(r['startup_s']!=r['first_ready_s'] for r in rows),
        pause_beneficiary_modalities=dict(benefits),
        beneficiary_critical_stage_observations=dict(beneficiary_stage_state),beneficiary_stage_examples=late_benefit_examples,
        admission_interference=admission,
        continuity_exposure=continuity_exposure(case,rows,pause_pairs,modes,end),
        notes=['Audio first-ready includes receipt and decoding of the first playable chunk.',
            'Admission waiting is measured at the gateway and is included in TTFT.',
            'Shared-stage occupancy and stage waits are observational; pure/mixed request counts differ.',
            'Resume-to-server-output includes pipeline/transport delay, not only GPU execution.',
            'Working-set and gamma distributions summarize logged states, not time-weighted samples.'])
    report=case/'attribution.json';report.write_text(json.dumps(result,indent=2))
    with (case/'request_attribution.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(request_rows[0]));writer.writeheader();writer.writerows(request_rows)
    return result


def summarize(root):
    root=Path(root);reports=root/'reports';reports.mkdir(exist_ok=True)
    manifest=json.loads((root/'runtime/active_launch.json').read_text())
    command=manifest['command'];trace=Path(next(x.split('=',1)[1] for x in command if x.startswith('STAGE_QUEUE_TRACE_DIR=')))
    completed=[]
    for case in sorted((root/'runs').glob('*formal*')):
        if not (case/'complete.json').exists():continue
        if not all((case/m/'mechanism.json').exists() for m in ('text','audio') if (case/m/'client').exists()):continue
        file=case/'attribution.json'
        a=json.loads(file.read_text()) if file.exists() else {}
        if a.get('analysis_schema_version')!=4 or 'beneficiary_critical_stage_observations' not in a:a=audit_case(case,trace)
        completed.append(a)
    flat=[]
    for a in completed:
        kind,rep,arm=a['case'].rsplit('_formal_',1)[0],*a['case'].rsplit('_formal_',1)[1].split('_',1)
        for m,data in a['outputs'].items():
            s=data['summary'];c=data['first_ready_components']
            flat.append(dict(kind=kind,repetition=int(rep),arm=arm,modality=m,requests=s['requests'],
                mean_first_ready_s=s['first_ready_mean_s'],p95_first_ready_s=s['startup_p95_s'],p99_first_ready_s=s['startup_p99_s'],
                max_first_ready_s=c['first_ready_s']['maximum'],
                cumulative_stall_s=s['stall_total_s'],generated_units_per_s=s['generated_units_per_s'],
                generation_window_s=s['generated_window_s'],paper_qos=s['paper_qos'],effective_useful_units_per_s=s['effective_units_per_s'],
                mean_admission_wait_s=c['ingress_admission_wait_s']['mean'],
                mean_thinker_queue_s=c['thinker_ready_to_first_forward_s']['mean'],
                pauses=data['pauses'],unique_paused_requests=data['unique_paused_requests'],
                errors=len(a['errors']),startup_gate_requests=a['startup_gate_requests']))
    if flat:
        with (reports/'formal_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    comparisons=[]
    for kind,m in sorted({(r['kind'],r['modality']) for r in flat}):
        by={arm:[r for r in flat if r['kind']==kind and r['modality']==m and r['arm']==arm] for arm in ('fcfs','buffer')}
        paired=sorted({r['repetition'] for r in by['fcfs']} & {r['repetition'] for r in by['buffer']})
        if not paired:continue
        c=dict(kind=kind,modality=m,paired_repetitions=paired,metrics={})
        for key in ('mean_first_ready_s','p95_first_ready_s','p99_first_ready_s','max_first_ready_s','cumulative_stall_s','generated_units_per_s','mean_admission_wait_s','mean_thinker_queue_s','pauses','paper_qos','effective_useful_units_per_s'):
            if any(r[key] is None for vals in by.values() for r in vals if r['repetition'] in paired):
                c['metrics'][key]=dict(error="metric unavailable for paired comparison")
                continue
            means={arm:sum(r[key] for r in vals if r['repetition'] in paired)/len(paired) for arm,vals in by.items()}
            c['metrics'][key]=dict(**means,absolute_delta=means['buffer']-means['fcfs'],change_percent=100*(means['buffer']-means['fcfs'])/abs(means['fcfs']) if means['fcfs'] else None)
        comparisons.append(c)
    result=dict(completed_formal_cases=len(completed),per_case=flat,comparisons=comparisons,
        quantile_convention='Per-case empirical P95; comparison averages repetition P95s (not pooled).',
        caveat='Two repetitions; fixed prompts, rate mix, bounded output and six-GPU deployment. No statistical significance claim.')
    (reports/'formal_results.json').write_text(json.dumps(result,indent=2))
    (reports/'analysis_source_manifest.json').write_text(json.dumps({str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=summarize(a.root)
    print(json.dumps(dict(completed_formal_cases=result['completed_formal_cases'],comparisons=result['comparisons']),indent=2))
