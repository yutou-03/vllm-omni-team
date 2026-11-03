"""Mixed-path cohort metrics, complete-offer CDFs, and paired content audits."""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path


def stats(values):
    values=sorted(v for v in values if v is not None and math.isfinite(v))
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
        'p50':values[math.ceil(.5*len(values))-1] if values else None,
        'p95':values[math.ceil(.95*len(values))-1] if values else None,
        'max':max(values) if values else None}


def trace_summary(path, response_ids):
    produced=collections.defaultdict(list);chains=[];current={};active=set();peak=0;errors=[]
    queues={};reasons=collections.Counter();caps={};final_delivery={}
    sent=collections.defaultdict(set);received=collections.defaultdict(set);consumed=collections.defaultdict(set)
    for line in path.open():
        e=json.loads(line);name=e['event'];sid=e.get('stage_id');rid=e.get('physical_request_id') or e.get('request_id');t=e['monotonic'];key=(sid,rid)
        transfer=(rid,e.get('pipeline_attempt_id',0),e.get('stage_attempt_id',0))
        if name in ['stage_study_text_payload_sent','stage_study_chunk_sent']:sent[(sid,transfer)].add(e['chunk_id'])
        if name in ['stage_study_text_payload_received','stage_study_chunk_received']:received[(sid-1,transfer)].add(e['chunk_id'])
        if name=='stage_study_chunk_conversion_started':consumed[(1,transfer)].add(e['chunk_id'])
        if name=='small_hold_capacity_configured':caps[str(sid)]={k:e.get(k) for k in ['execution_cap','resident_cap','hold_enabled']}
        if name=='stage_study_thinker_output_produced' and rid in response_ids:produced[rid].append((t,e['generated_output_tokens_total']))
        if name=='stage_buffer_decision':
            reasons[str(sid)+':'+e['reason']]+=1
            if e['reason']=='recovery_margin_exhausted':errors.append({'reason':'recovery_margin_exhausted','stage':sid,'request':rid})
        if name=='stage_buffer_policy_suspended':errors.append({'reason':'policy_suspended','stage':sid,'request':rid})
        if name=='small_hold_requested':
            c={'stage':sid,'victim':rid,'requested':t,'evidence':e['evidence'],'kv_before':e['kv_block_ids']}
            chains.append(c);current[key]=c;active.add(key);peak=max(peak,len(active))
        elif name=='urgent_keep_admitted':
            c=current.get((sid,e['victim_request_id']))
            if c:c.update(admitted=t,urgent=e['urgent_request_id'],kv_admitted=e['victim_kv_block_ids'])
        elif name in ['small_hold_released','small_hold_first_rescheduled']:
            c=current.get(key)
            if c:c['released' if name=='small_hold_released' else 'rescheduled']=t
            if name=='small_hold_first_rescheduled':active.discard(key)
        if name in ['stage_study_thinker_output_produced','stage_study_chunk_sent']:
            c=current.get(key)
            if c and 'rescheduled' in c and t>=c['rescheduled']:c.setdefault('new_output',t)
        if name=='stage_study_receiver_buffer_enqueued':
            q=queues.setdefault(key,{'enqueued':[],'released':[],'max_units':0})
            q['enqueued'].append(e['chunk_id']);q['max_units']=max(q['max_units'],e['buffered_units'])
        if name=='stage_study_receiver_buffer_released':queues.setdefault(key,{'enqueued':[],'released':[],'max_units':0})['released'].append(e['chunk_id'])
        if name=='stage_buffer_text_inventory' and e.get('closed'):
            final_delivery[rid]=e
            if e.get('action')!='completed' or e['received_wire_sha256']!=e['released_wire_sha256'] or e['received_chars']!=e['released_chars']:
                errors.append({'reason':'text_delivery_conservation','request':rid})
    for key,q in queues.items():
        if q['enqueued']!=list(range(len(q['enqueued']))) or q['released']!=q['enqueued']:
            errors.append({'reason':'receiver_enqueue_release_sequence','stage':key[0],'request':key[1],
                'enqueued':len(q['enqueued']),'released':len(q['released'])})
    for key,ids in received.items():
        if ids-sent[key]:errors.append({'reason':'received_without_published','stage':key[0],'request':key[1][0]})
    for key,ids in consumed.items():
        if ids-received[key]:errors.append({'reason':'consumed_without_received','stage':key[0],'request':key[1][0]})
    for c in chains:
        keys=['requested','admitted','released','rescheduled','new_output']
        c['complete']=all(k in c for k in keys) and all(c[x]<=c[y] for x,y in zip(keys,keys[1:]))
        c['kv_retained']=c.get('kv_admitted')==c['kv_before']
        if 'rescheduled' in c:c['actual_pause_s']=c['rescheduled']-c['requested']
        if 'new_output' in c:c['resume_to_new_output_s']=c['new_output']-c['rescheduled']
    return produced, {'requested':len(chains),'complete_chains':sum(c['complete'] for c in chains),
        'kv_retained_chains':sum(c['kv_retained'] for c in chains),'peak_simultaneous':peak,
        'by_stage':{str(s):{'requested':sum(c['stage']==s for c in chains),'complete':sum(c['stage']==s and c['complete'] for c in chains)} for s in [0,1]},
        'actual_pause_s':stats([c.get('actual_pause_s') for c in chains]),
        'nominal_100ms_overruns':sum(c.get('actual_pause_s',0)>.1 for c in chains),
        'resume_to_new_output_s':stats([c.get('resume_to_new_output_s') for c in chains]),
        'capacity':caps,'errors':errors,'reasons':dict(reasons),'chains':chains,
        'receiver_streams':len(queues),'enqueued_chunks':sum(len(q['enqueued']) for q in queues.values()),
        'released_chunks':sum(len(q['released']) for q in queues.values())}


def analyze_arm(root,arm,tokenizer):
    from sse_client_extended import add_visible_token_metrics
    from stream_goodput import request_score
    path=root/arm;trace=arm.split('_',1)[1]
    manifest=json.loads((path/'manifest.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if manifest['status']!='completed':raise ValueError('incomplete replay')
    workload=root/(trace+'.jsonl')
    if hashlib.sha256(workload.read_bytes()).hexdigest()!=plan['workload_sha256'][trace]:raise ValueError('workload hash mismatch')
    rows=[json.loads(x) for x in workload.read_text().splitlines()]
    received=json.loads((path/'results.json').read_text());results={x['source_request_id']:x for x in received}
    if len(results)!=len(rows):raise ValueError('offered cohort lost requests')
    scores=json.loads((path/'stream_score.json').read_text());score_byid={x['request_id']:x for x in scores['requests']}
    config=json.loads((root/'metric.json').read_text());without_w=dict(config,buffer_full_credit_s=1e12)
    produced,audit=trace_summary(root/(arm+'_service/traces/stage_events.jsonl'),{r['response_request_id'] for r in received})
    origin=min(r['scheduled_arrival_monotonic_s'] for r in received)
    last=max((x[0] for events in produced.values() for x in events),default=origin)
    generation_window=last-origin
    if generation_window<=0:raise ValueError('missing generation timestamps')
    delivery_window=max(r['client_completion_monotonic_s'] for r in received)-origin
    per=[];integrity=[]
    for row in rows:
        r=results[row['source_request_id']];s=score_byid[row['source_request_id']]
        p=sorted(produced.get(r['response_request_id'],[]));n=max((v for _,v in p),default=0)
        counts=[v for _,v in p]
        if counts and (counts!=sorted(set(counts)) or counts[-1]!=len(r['text_token_ids'])):integrity.append(row['source_request_id'])
        if r['status']=='success' and not counts:integrity.append(row['source_request_id'])
        add_visible_token_metrics(r,tokenizer);arr=r['visible_token_arrival_monotonic_s']
        tpot=(arr[-1]-arr[0])/(len(arr)-1) if len(arr)>1 else 0.
        production_tpot=(p[-1][0]-p[0][0])/(n-1) if n>1 else 0.
        ttft_ok=r['ttft_s'] is not None and r['ttft_s']<=row['ttft_slo_ms']/1000
        first_ok=s['first_slo_met'];complete=r['status']=='success' and r['done_received']
        continuity=(s['max_rebuffer_s']<=row['max_stall_slo_s'] and s['rebuffer_s']<=row['total_stall_slo_s'])
        tpot_ok=tpot<=row['tpot_slo_s']+1e-6 if row['request_path']=='text' else True
        e2e_ok=r['e2e_s']<=row['slo_ms']/1000
        full=complete and ttft_ok and first_ok and continuity and tpot_ok and e2e_ok
        nw=request_score(row,r,without_w,tokenizer)
        pcm_hash=None
        if row['request_path']=='audio':
            import soundfile as sf
            pcm=hashlib.sha256()
            for chunk in r['audio_chunks']:
                samples,sample_rate=sf.read(Path(r['artifact_directory'])/chunk['path'],dtype='float64',always_2d=True)
                pcm.update(samples.astype('<f8').tobytes())
            pcm_hash=pcm.hexdigest()
        per.append({'request_id':row['source_request_id'],'path':row['request_path'],'tier':row['slo_tier'],
            'complete':complete,'generated_tokens':n,'wire_tokens':len(r['text_token_ids']),'full_slo_met':full,
            'ttft_s':r['ttft_s'],'ttfp_s':r['ttfp_s'],'ttft_slo_met':ttft_ok,'first_slo_met':first_ok,
            'tpot_s':tpot,'production_tpot_s':production_tpot,'tpot_slo_met':tpot_ok,'continuity_met':continuity,
            'e2e_s':r['e2e_s'],'e2e_met':e2e_ok,'stall_s':s['rebuffer_s'],'max_stall_s':s['max_rebuffer_s'],
            'audio_duration_s':r['audio_duration_s'],'audio_pcm_sha256':pcm_hash,'G_contribution':s['contribution'],'G_without_w_contribution':nw['contribution'],
            'utility':s['utility'],'first_penalty':s.get('first_penalty',0.),'stall_penalty':config['mu']*s['rebuffer_s'],
            'send_lateness_s':r['send_lateness_s'],'token_sha256':r['text_token_ids_sha256'],
            'audio_signature':[(x['frames'],x['sample_rate'],x['decoded_sha256']) for x in r['audio_chunks']]})
    groups={}
    selectors={'all':lambda x:True,'text':lambda x:x['path']=='text','audio':lambda x:x['path']=='audio'}
    selectors.update({p+'_'+t:(lambda x,p=p,t=t:x['path']==p and x['tier']==t) for p in ['text','audio'] for t in ['standard','loose']})
    for key,select in selectors.items():
        group=[x for x in per if select(x)]
        groups[key]={'offered':len(group),'completed':sum(x['complete'] for x in group),'full_slo_met':sum(x['full_slo_met'] for x in group),
            'ttft_s':stats([x['ttft_s'] for x in group]),'ttfp_s':stats([x['ttfp_s'] for x in group]),
            'ttft_slo_met':sum(x['ttft_slo_met'] for x in group),'first_slo_met':sum(x['first_slo_met'] for x in group),
            'tpot_s':stats([x['tpot_s'] for x in group]),'production_tpot_s':stats([x['production_tpot_s'] for x in group]),
            'e2e_s':stats([x['e2e_s'] for x in group]),'stall_s':stats([x['stall_s'] for x in group]),
            'generated_tokens':sum(x['generated_tokens'] for x in group),'slo_tokens':sum(x['generated_tokens'] for x in group if x['full_slo_met']),
            'Gstream':sum(x['G_contribution'] for x in group)/manifest['arrival_window_s']}
    totals=groups['all']
    out={'arm':arm,'scope':'paired pilot, 200 mixed requests; not repeated-seed capacity result',
        'Gstream':scores['G_stream'],'Gstream_without_buffer_weight':sum(x['G_without_w_contribution'] for x in per)/manifest['arrival_window_s'],
        'generated_tokens_per_s':totals['generated_tokens']/delivery_window,
        'SLO_generated_tokens_per_s':totals['slo_tokens']/delivery_window,
        'engine_generated_tokens_per_s':totals['generated_tokens']/generation_window,
        'engine_SLO_generated_tokens_per_s':totals['slo_tokens']/generation_window,
        'delivery_tokens_per_s':sum(x['wire_tokens'] for x in per)/delivery_window,
        'audio_playable_seconds_per_s':sum(x['audio_duration_s'] for x in per)/delivery_window,
        'SLO_audio_playable_seconds_per_s':sum(x['audio_duration_s'] for x in per if x['full_slo_met'])/delivery_window,
        'arrival_window_s':manifest['arrival_window_s'],'generation_window_s':generation_window,'delivery_window_s':delivery_window,
        'max_send_lateness_s':max(x['send_lateness_s'] for x in per),'generation_wire_count_mismatches':integrity,
        'groups':groups,'audit':audit,'requests':per}
    (path/'benchmark.json').write_text(json.dumps(out,indent=2)+'\n')
    with (path/'requests.csv').open('w') as f:
        fields=[k for k in per[0] if k not in ['audio_signature']]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(per)
    print(json.dumps({k:v for k,v in out.items() if k not in ['groups','audit','requests']},indent=2))
    return out


def report(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/benchmark.json')}
    result={};lines=['# Mixed 200-request paired comparison','','One frozen seed per arrival trace. Every arm: 120 text + 80 audio.','',
        '| Trace | Policy | Gstream | Generated tokens/s | SLO tokens/s | Mean TTFT (s) | TTFT P95 (s) | Full SLO | Pauses / chains |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    cdf_rows=[]
    for i,trace in enumerate(['poisson','burstgpt']):
        for policy in ['native','ours']:
            a=arms.get(policy+'_'+trace)
            if a is None:continue
            g=a['groups']['all'];audit=a['audit']
            lines.append(f"| {trace} | {policy} | {a['Gstream']:.3f} | {a['generated_tokens_per_s']:.3f} | {a['SLO_generated_tokens_per_s']:.3f} | {g['ttft_s']['mean']:.3f} | {g['ttft_s']['p95']:.3f} | {g['full_slo_met']}/200 | {audit['requested']}/{audit['complete_chains']} |")
            for j,group in enumerate(['all','text','audio']):
                requests=[x for x in a['requests'] if group=='all' or x['path']==group]
                values=sorted(x['ttft_s'] for x in requests if x['ttft_s'] is not None)
                y=[(k+1)/len(requests) for k in range(len(values))]
                axes[i,j].step([0]+values,[0]+y,where='post',label=policy)
                cdf_rows.extend({'trace':trace,'policy':policy,'group':group,'ttft_s':t,'cdf_all_offered':v} for t,v in zip(values,y))
                axes[i,j].set_title(trace+' / '+group);axes[i,j].set_xlabel('TTFT (s)');axes[i,j].set_ylabel('Fraction of offered requests');axes[i,j].set_ylim(0,1.02);axes[i,j].grid(alpha=.25);axes[i,j].legend()
        if 'native_'+trace in arms and 'ours_'+trace in arms:
            b,o=arms['native_'+trace],arms['ours_'+trace]
            bp={x['request_id']:x for x in b['requests']};op={x['request_id']:x for x in o['requests']}
            result[trace]={'relative_changes_pct':{k:100*(o[k]/b[k]-1) if b[k] else None for k in ['Gstream','Gstream_without_buffer_weight','generated_tokens_per_s','SLO_generated_tokens_per_s']},
                'identical_text_token_hashes':sum(bp[k]['token_sha256']==op[k]['token_sha256'] for k in bp),
                'audio_same_chunk_samples_and_hashes':sum(bp[k]['audio_signature']==op[k]['audio_signature'] for k in bp if bp[k]['path']=='audio'),
                'audio_same_concatenated_pcm':sum(bp[k]['audio_pcm_sha256']==op[k]['audio_pcm_sha256'] for k in bp if bp[k]['path']=='audio'),
                'audio_pairs':80}
    fig.tight_layout();fig.savefig(root/'ttft_cdf.png',dpi=180);fig.savefig(root/'ttft_cdf.svg');plt.close(fig)
    with (root/'ttft_cdf.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['trace','policy','group','ttft_s','cdf_all_offered']);w.writeheader();w.writerows(cdf_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for axis,trace in zip(axes,['poisson','burstgpt']):
        for policy in ['native','ours']:
            a=arms.get(policy+'_'+trace)
            if a is None:continue
            values=sorted(x['ttfp_s'] for x in a['requests'] if x['path']=='audio' and x['ttfp_s'] is not None)
            axis.step([0]+values,[0]+[(i+1)/80 for i in range(len(values))],where='post',label=policy)
        axis.set_title(trace+' / audio');axis.set_xlabel('First playable audio (s)');axis.set_ylabel('Fraction of offered audio requests');axis.set_ylim(0,1.02);axis.grid(alpha=.25)
        if axis.lines:axis.legend()
    fig.tight_layout();fig.savefig(root/'ttfp_cdf.png',dpi=180);fig.savefig(root/'ttfp_cdf.svg');plt.close(fig)
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines+=['','## Definitions and validity','',
        '- Gstream retains the frozen continuous TTFT violation penalty and client-buffer weight; Gstream without that weight is also reported.',
        '- Generated token throughput counts Thinker model output IDs once, including audio-path accompanying text; codec/audio samples are not added.',
        '- Main throughput denominator: first scheduled arrival through final client completion (includes pacing and drain); engine-only generation rate is a separate diagnostic.',
        '- SLO token throughput counts all generated tokens of complete requests passing TTFT, path-first-output, E2E and continuity; text also passes visible average TPOT.',
        '- Mean TTFT is over observed first outputs; CDF denominator includes all 200 offered requests. Audio TTFP is separately available in benchmark.json.',
        '- Baseline is native FIFO with matching K16/R64 capacity configuration and instrumentation, with added queues/pacing and proactive holds disabled.',
        '- Exact output mismatches, policy suspension, missing hold chains or missing timestamps limit attribution; inspect comparison.json and each benchmark.json.',
        '- A single Poisson trace and one selected BurstGPT segment do not establish generalization or maximum sustainable capacity.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm');p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.report:report(a.root)
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/data/models/Qwen3-Omni-30B-A3B-Instruct',local_files_only=True)
        analyze_arm(a.root,a.arm,tokenizer)
