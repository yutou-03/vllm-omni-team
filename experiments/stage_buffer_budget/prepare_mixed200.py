"""Freeze two mixed cohorts before paired performance runs; real BurstGPT arrivals."""
import argparse,copy,csv,hashlib,json,math,random,statistics
from pathlib import Path

E=Path(__file__).resolve().parent

def write(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def prepare(root,rate=1.2,seed=20261103):
    if not (root/'calibration/completed.json').exists():raise ValueError('complete independent calibration first')
    cal=[r for r in json.loads((root/'calibration/results.json').read_text()) if not r['warmup']]
    byclass={}
    for r in cal:byclass.setdefault(r['metadata']['stratum'],[]).append(r)
    thresholds={}
    for path in ['text','audio']:
        first=[r['ttfp_s' if path=='audio' else 'ttft_s'] for r in cal if r['metadata']['request_path']==path]
        base=max(3. if path=='audio' else 1.3,math.ceil(1.25*max(first)*10)/10)
        for tier in ['standard','loose']:
            key=path+'_'+tier;data=byclass[key]
            slots=512 if key=='text_loose' else 128 if key in ['text_standard','audio_loose'] else 64
            end=max((160. if tier=='loose' else 45.) if path=='text' else (180. if tier=='loose' else 90.), 3*max(r['e2e_s'] for r in data))
            first_s=base*(4 if path=='text' and tier=='loose' else 2 if tier=='loose' else 1)
            stall=max(.25,math.ceil(1.25*max(r.get('audio_max_stall_s',0.) for r in data)*100)/100) if path=='audio' else .1
            thresholds[key]={'max_tokens':slots,'first_slo_s':first_s,'e2e_slo_s':end,'max_stall_s':stall,
                'total_stall_s':max(.5,2*stall),'tpot_slo_s':1/4.8,
                'stage0_prediction_ms':max(6000. if tier=='standard' else 20000.,2000*max((r['text_finish_monotonic_s'] or r['client_completion_monotonic_s'])-r['send_monotonic_s'] for r in data)),
                'stage1_prediction_ms':max(10000.,2000*max(r['e2e_s'] for r in data)) if path=='audio' else None}
    write(root/'slo_calibration.json',{'samples':len(cal),'scope':'pilot, two per stratum; independent of comparison results','thresholds':thresholds})
    source=root/'sources/BurstGPT_1.csv'
    allrows=list(csv.DictReader(source.open()))
    rows=[dict(r,source_csv_line=i+2) for i,r in enumerate(allrows) if r['Model']=='ChatGPT' and r['Log Type']=='API log' and int(r['Response tokens'])>0]
    rows.sort(key=lambda r:int(r['Timestamp']))
    blocks=[]
    for i in range(0,len(rows)-200,200):
        ts=[int(r['Timestamp']) for r in rows[i:i+200]]
        gaps=[b-a for a,b in zip(ts,ts[1:])]
        if ts[-1]>ts[0]:blocks.append((statistics.pstdev(gaps)/statistics.mean(gaps),i,ts[-1]-ts[0]))
    blocks.sort();cv,start,span=blocks[int(.75*(len(blocks)-1))]
    selected=rows[start:start+200];window=200/rate
    write(root/'sources/burstgpt_selected_rows.json',selected)
    write(root/'sources/burstgpt_provenance.json',{'url':'https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv',
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'filter':'ChatGPT, API log, positive response tokens; stable timestamp sort',
        'selection':'nonoverlapping 200-request blocks, 75th percentile interarrival CV; chosen without model performance',
        'cv':cv,'filtered_start_index':start,'source_span_s':span,'time_scale':(window-1)/span,
        'replay_scope':'arrival timestamps only; synthetic mixed prompts and SLOs; not native multimodal BurstGPT content'})
    rng=random.Random(seed)
    kinds=['text_standard']*60+['text_loose']*60+['audio_standard']*40+['audio_loose']*40
    rng.shuffle(kinds)
    topics=['how solar panels produce electricity','why leaves change color','how maps represent Earth','how airplanes stay in the air',
        'how bees communicate','how water filters work','how bridges support weight','how a compass works',
        'why the ocean has tides','how batteries store energy','how wind turbines work','how clouds produce lightning',
        'how plants obtain nutrients','how sound travels','how telescopes collect light','how glaciers move',
        'how paper is recycled','how satellites remain in orbit','how soap removes dirt','how insulation keeps houses warm']
    arrivals={}
    prng=random.Random(seed+1);arrivals['poisson']=sorted(prng.uniform(0,window) for _ in range(200))
    t0=int(selected[0]['Timestamp']);arrivals['burstgpt']=[(int(r['Timestamp'])-t0)*(window-1)/span for r in selected]
    contracts={};hashes={}; workloads={}
    for trace,times in arrivals.items():
        cohort=[]
        for i,(kind,t) in enumerate(zip(kinds,times)):
            path,tier=kind.split('_');c=thresholds[kind];rid=f'm200-{trace}-{i:03d}'
            words=450 if kind=='text_loose' else 90 if kind in ['text_standard','audio_loose'] else 40
            prompt=f'Explain {topics[i%len(topics)]} to a curious beginner in about {words} words. Use clear concrete examples.'
            row={'source_request_id':rid,'request_id':rid,'arrival_offset_s':t,'request_path':path,'slo_tier':tier,
                'stratum':kind,'random_seed':42,'text_consume_rate_tps':4.8,'audio_equivalent_rate_tps':3.3,
                'slo_ms':c['e2e_slo_s']*1000,'ttft_slo_ms':(c['first_slo_s'] if path=='text' else (2.6 if tier=='standard' else 5.2))*1000,
                'ttfp_slo_ms':c['first_slo_s']*1000 if path=='audio' else None,
                'tpot_slo_s':c['tpot_slo_s'],'max_stall_slo_s':c['max_stall_s'],'total_stall_slo_s':c['total_stall_s'],
                'payload':{'model':'/data/models/Qwen3-Omni-30B-A3B-Instruct','messages':[{'role':'user','content':prompt}],
                    'modalities':['text','audio'] if path=='audio' else ['text'],'max_tokens':c['max_tokens'],'ignore_eos':False,
                    'temperature':0.,'seed':42,'return_token_ids':True,'stream':True,'stream_options':{'include_usage':True},
                    'omni_scheduling':{'schema_version':1,'source_request_id':rid,'request_path':path,'ingress_order':i,
                        'slo_ms':c['e2e_slo_s']*1000,'predicted_stage_ms':[c['stage0_prediction_ms'],c['stage1_prediction_ms'],100. if path=='audio' else None]}}}
            if path=='audio':row['payload']['audio']={'format':'wav','voice':'Chelsie'}
            contracts[rid]={'ttft_slo_ms':row['ttft_slo_ms'],'first_output_service_ms':400.}
            if path=='audio':contracts[rid].update(ttfp_slo_ms=row['ttfp_slo_ms'],first_output_service_ms=2000.)
            cohort.append(row)
        p=root/(trace+'.jsonl')
        if p.exists():raise FileExistsError(p)
        p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cohort))
        hashes[trace]=hashlib.sha256(p.read_bytes()).hexdigest();workloads[trace]=cohort
    smoke=[]
    specifications=[('text_loose',i*.015) for i in range(16)]+[('text_standard',6.+i*.005) for i in range(4)]+[('audio_loose',8.),('audio_standard',8.1),('audio_loose',9.),('audio_standard',9.1)]
    for i,(kind,arrival) in enumerate(specifications):
        row=copy.deepcopy(next(x for x in workloads['poisson'] if x['stratum']==kind))
        old=row['source_request_id'];rid='m200-smoke-'+str(i)
        row.update(source_request_id=rid,request_id=rid,arrival_offset_s=arrival)
        row['payload']['omni_scheduling'].update(source_request_id=rid,ingress_order=i)
        contracts[rid]=copy.deepcopy(contracts[old]);smoke.append(row)
    p=root/'smoke.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in smoke))
    hashes['smoke']=hashlib.sha256(p.read_bytes()).hexdigest()
    write(root/'contracts.json',contracts)
    write(root/'text_delivery.json',{'tokens_per_s':4.8,'max_chars':131072,'max_events':4096,'boundary_reserve_tokens':32,'delivery_lead_tokens':1.})
    write(root/'receiver.json',{'stages':{'1':{'max_chunks':1024,'max_units':2048,'max_bytes_per_request':67108864,'max_total_bytes':536870912,
        'build_after_chunks':32,'build_delay_s':.5,'build_target_units':64,'minimum_slo_ms':150000.,'remaining_guard_s':10.},
        '2':{'max_chunks':256,'max_units':8192,'max_bytes_per_request':16777216,'max_total_bytes':268435456,
            'build_delay_s':0.,'release_units_per_s':30.,'minimum_slo_ms':150000.}}})
    config=json.loads((E/'controller_budget_config.json').read_text())
    config.update(max_concurrent_holds=16,max_holds_per_request=0,text_delivery_inventory=True,max_text_observation_age_s=1.,
        guard_full_urgent_service=True,inventory_deadline_lease=True,contracts_path=str(root/'contracts.json'),urgent_guard_ms=250.,
        allow_paced_receiver_budget=True,performance_claim=False)
    write(root/'dynamic_controller.json',config)
    write(root/'metric.json',{'metric_version':'soft_ttft_v2','beta':3.3,'mu':3.3,'text_tps':4.8,'audio_equivalent_tps':3.3,
        'audio_unit_s':.02,'lateness_decay_s':1.,'buffer_full_credit_s':10.,'buffer_decay_s':20.,'gate':'none'})
    write(root/'plan.json',{'scope':'200-request mixed-path paired pilots, one seed per trace, not capacity proof','seed':seed,
        'offered_per_trace':200,'path_counts':{'text':120,'audio':80},'tier_counts':{'standard':100,'loose':100},
        'K_by_stage':{'0':16,'1':16},'R_by_stage':{'0':64,'1':64},'window_s':window,'offered_qps':rate,'timeout_s':360.,
        'workload_sha256':hashes,'poisson_definition':'homogeneous Poisson process conditioned on exactly 200 arrivals in the fixed window (uniform order statistics)',
        'arms':['native_poisson','ours_poisson','ours_burstgpt','native_burstgpt'],
        'baseline':'native FIFO, same K/R adapter and tracing, no added receiver/text buffer or active pause',
        'metric_denominators':{'Gstream':'fixed offered-arrival window','generated_tps':'first scheduled arrival to final Stage0 production event',
            'SLO_generated_tps':'same generation interval; complete requests satisfying their full frozen path contract',
            'delivery_tps':'first scheduled arrival to final client completion, also reported'},
        'token_unit':'Thinker model output token IDs, counted once per request including audio-path accompanying text; codec frames excluded',
        'audio_output':'playable seconds/s reported separately; mixed token/s is not total three-stage computational work',
        'CDF':'all offered requests in denominator; missing first output remains censored beyond finite axis',
        'safety':'do not interpret score differences as pause benefit without full event chains; no weight or threshold tuning after results'})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for a,(name,rows) in zip(ax,workloads.items()):
        for kind in thresholds:
            ts=[r['arrival_offset_s'] for r in rows if r['stratum']==kind]
            a.hist(ts,bins=list(range(0,math.ceil(window)+5,5)),histtype='step',label=kind)
        a.set_title(name+' / 200 requests');a.set_ylabel('Requests / 5 s');a.legend(ncol=4,fontsize=8)
    ax[-1].set_xlabel('Scheduled arrival (s)');fig.tight_layout();fig.savefig(root/'arrival_traces.png',dpi=160);plt.close(fig)
    print(json.dumps({'root':str(root),'window_s':window,'thresholds':thresholds,'hashes':hashes},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rate',type=float,default=1.2);a=p.parse_args();prepare(a.root,a.rate)
