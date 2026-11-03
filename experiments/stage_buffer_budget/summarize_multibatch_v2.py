"""Summarize frozen multi-execution replays and soft first-output penalties."""
import argparse,collections,json,statistics,hashlib
from pathlib import Path
from audit_receiver_mechanism import audit


def percentile(v,p):
 v=sorted(v);return v[min(len(v)-1,int((len(v)-1)*p))] if v else None


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
 arms=['native64','nohold16','dynamic16','fixed16','nohold8','dynamic8','guard16'];summary={'metric_version':'soft_ttft_v2','beta':3.3,'scope':'multi-execution controlled burst pilot; ordered arms; natural prompt templates; not production capacity proof','arms':{},'comparisons':{}}
 for arm in arms:
  runs=[];rows=[]
  for seed in [1103,2207,3301]:
   path=root/f'{arm}_s{seed}';score=path/'stream_score.json'
   if not score.exists():continue
   man=json.loads((path/'manifest.json').read_text());d=json.loads(score.read_text())
   if man['status']!='completed':raise ValueError('not drained')
   if hashlib.sha256(Path(man['workload']).read_bytes()).hexdigest()!=man['workload_sha256']:raise ValueError('workload modified')
   runs.append({'seed':seed,'G':d['G_stream'],'T':d['arrival_window_s'],'beta_G':{str(b):sum(max(0.,x['utility']-b*x['first_violation_s']-d['metric_config']['mu']*x['rebuffer_s']) for x in d['requests'])/d['arrival_window_s'] for b in [1.65,3.3,6.6,13.2]}});rows.extend(d['requests'])
  if not runs:continue
  val={'runs':runs,'G_mean':statistics.mean(x['G'] for x in runs),'groups':{},'beta_mean':{str(b):statistics.mean(x['beta_G'][str(b)] for x in runs) for b in [1.65,3.3,6.6,13.2]}}
  for group in ['all','text_standard','text_loose','audio_loose']:
   xs=[x for x in rows if group=='all' or x['path']+'_'+x['tier']==group]
   if not xs:continue
   val['groups'][group]={'requests':len(xs),'complete':sum(x['complete'] for x in xs),'first_slo_met':sum(x['first_slo_met'] for x in xs),'e2e_slo_met':sum(x['e2e_slo_met'] for x in xs),'first_mean_s':statistics.mean(x['first_output_s'] for x in xs if x['first_output_s'] is not None),'first_p95_s':percentile([x['first_output_s'] for x in xs if x['first_output_s'] is not None],.95),'e2e_p95_s':percentile([x['e2e_s'] for x in xs if x['e2e_s'] is not None],.95),'mean_rebuffer_s':statistics.mean(x['rebuffer_s'] for x in xs),'max_rebuffer_p95_s':percentile([x['max_rebuffer_s'] for x in xs],.95),'mean_raw_units':statistics.mean(x['raw_units'] for x in xs),'mean_content_s':statistics.mean(x.get('content_s',0) for x in xs)}
  trace=root/f'{arm}_service/traces/stage_events.jsonl'
  if trace.exists():
   allowed_ids={x['request_id'] for x in rows};allowed_ids|={'chatcmpl-'+x for x in allowed_ids}
   keep=[];names=collections.Counter();reject=collections.Counter();batch=collections.defaultdict(list);counts=collections.defaultdict(collections.Counter);produced_text=collections.defaultdict(list);converted_audio=collections.defaultdict(list)
   needed={'small_hold_requested','small_hold_released','small_hold_first_rescheduled','urgent_keep_admitted','stage_first_forward','stage_study_receiver_buffer_enqueued','stage_study_receiver_buffer_released','stage_study_text_payload_sent','stage_study_text_payload_received','stage_study_chunk_sent','stage_study_chunk_received','stage_study_chunk_conversion_started','stage_buffer_policy_suspended','stage_study_text_conditioning_selected'}
   with trace.open() as f:
    for line in f:
     if not line.endswith('\n'):continue
     ev=json.loads(line)
     if not any(str(ev.get(k,'')) in allowed_ids for k in ['request_id','physical_request_id','urgent_request_id','victim_request_id']):continue
     n=ev['event'];names[n]+=1
     rid=ev.get('physical_request_id') or ev.get('request_id')
     if n=='stage_study_thinker_output_produced':produced_text[rid].append(ev['generated_output_tokens_total'])
     if n=='stage_study_chunk_conversion_completed':converted_audio[rid].append((ev['chunk_id'],ev['audio_samples']))
     if n=='stage_first_forward' and ev.get('batch_num_seqs') is not None:batch[ev.get('stage_id')].append(ev['batch_num_seqs'])
     if n=='stage_buffer_decision':
      reject[ev.get('reason','unknown')]+=1
      for k,v in ev.get('rejection_counts',{}).items():reject['candidate:'+k]+=v
     if n in needed:keep.append(ev)
     if n in ['stage_study_text_payload_sent','stage_study_text_payload_received','stage_study_chunk_sent','stage_study_chunk_received']:
      rid=ev.get('physical_request_id') or ev.get('request_id');counts[n][(rid,ev.get('chunk_id'))]+=1
   mechanism=audit(keep)
   received=collections.defaultdict(float);selected=collections.defaultdict(set);windows=[];active={}
   for ev in sorted(keep,key=lambda x:x['monotonic']):
    n=ev['event'];rid=ev.get('physical_request_id') or ev.get('request_id');sid=ev.get('stage_id')
    if n=='stage_study_text_payload_received':received[rid]+=ev['raw_entity_embedding_rows_total']-(ev.get('raw_ids_prompt_count',0) if ev['chunk_id']==0 else 0)
    elif n=='stage_study_text_conditioning_selected' and ev.get('conditioning_kind')=='output_embedding':selected[rid].update(range(ev['start_index'],ev['end_index']))
    if n=='small_hold_requested' and sid==0:
     w={'request_id':rid,'requested':ev['monotonic'],'minimum_received_units_until_new_chunk':received[rid]-len(selected[rid]),'requested_hold_s':ev.get('evidence',{}).get('requested_hold_s')};windows.append(w);active[rid]=w
    if rid in active:
     w=active[rid]
     if 'new_chunk' not in w:w['minimum_received_units_until_new_chunk']=min(w['minimum_received_units_until_new_chunk'],received[rid]-len(selected[rid]))
     if n=='small_hold_first_rescheduled':w['rescheduled']=ev['monotonic']
     if n=='stage_study_text_payload_sent' and 'rescheduled' in w:w.setdefault('new_chunk',ev['monotonic'])
   all_pairs=[]
   for send,receive in [('stage_study_text_payload_sent','stage_study_text_payload_received'),('stage_study_chunk_sent','stage_study_chunk_received')]:
    all_pairs.append({'event':send,'sent':sum(counts[send].values()),'received':sum(counts[receive].values()),'sent_not_received':len(counts[send]-counts[receive]),'received_not_sent':len(counts[receive]-counts[send]),'duplicate_send_keys':sum(v>1 for v in counts[send].values()),'duplicate_receive_keys':sum(v>1 for v in counts[receive].values())})
   delivery=[]
   for run in runs:
    for result in json.loads((root/f"{arm}_s{run['seed']}"/'results.json').read_text()):
     rid=result['response_request_id'];text_counts=produced_text[rid];expected=sorted(converted_audio[rid]);actual=[x['frames'] for x in result['audio_chunks']]
     delivery.append({'request_id':result['source_request_id'],'generated_text_tokens':max(text_counts,default=0),'received_text_tokens':len(result['text_token_ids']),'text_count_equal':max(text_counts,default=0)==len(result['text_token_ids']),'text_counter_contiguous':text_counts==list(range(1,len(text_counts)+1)),'audio_chunk_lengths_equal':actual==[x[1] for x in expected],'audio_chunk_ids_contiguous':[x[0] for x in expected]==list(range(len(expected))),'generated_audio_samples':sum(x[1] for x in expected),'received_audio_samples':sum(actual)})
   (root/f'{arm}_delivery_audit.json').write_text(json.dumps({'scope':'transport count/order and per-chunk sample lengths; not cross-run semantic equality','requests':delivery,'all_passed':all(x['text_count_equal'] and x['text_counter_contiguous'] and x['audio_chunk_lengths_equal'] and x['audio_chunk_ids_contiguous'] for x in delivery)},indent=2))
   (root/f'{arm}_mechanism_audit.json').write_text(json.dumps({'chain_audit':mechanism,'inventory_windows':windows,'event_counts':dict(names),'decision_reasons':dict(reject),'transport_conservation':all_pairs,'actual_first_forward_peak_batch':{str(k):max(v) for k,v in batch.items()}},indent=2))
   val['mechanism']={'retained_KV_at_urgent_admission':sum(c.get('victim_kv')==c['request_event'].get('kv_block_ids') for c in mechanism['chains'] if c.get('victim_kv') is not None),'urgent_first_batch_sizes':[c.get('urgent_first_batch_num_seqs') for c in mechanism['chains']],'release_overrun_ms':[max(0.,1000*(c['released']-c['request_event']['deadline_monotonic_s'])) for c in mechanism['chains'] if 'released' in c],'delivery_count_order_passed':all(x['text_count_equal'] and x['text_counter_contiguous'] and x['audio_chunk_lengths_equal'] and x['audio_chunk_ids_contiguous'] for x in delivery),'inventory_windows':windows,'requested':len(mechanism['chains']),'complete_chains':mechanism['complete_chains'],'integrity_errors':mechanism['integrity_errors'],'event_counts':dict(names),'decision_reasons':dict(reject),'transport_conservation':all_pairs,'actual_first_forward_peak_batch':{str(k):max(v) for k,v in batch.items()},'actual_pause_ms':[x['actual_to_rescheduled_ms'] for x in mechanism['chains'] if x['complete_chain']],'reschedule_to_new_chunk_ms':[x['reschedule_to_first_chunk_ms'] for x in mechanism['chains'] if x['complete_chain']]}
  summary['arms'][arm]=val
 for policy,base in [('dynamic16','nohold16'),('fixed16','nohold16'),('dynamic8','nohold8'),('dynamic16','native64'),('nohold16','native64'),('guard16','nohold16'),('guard16','native64'),('guard16','dynamic16')]:
  if policy not in summary['arms'] or base not in summary['arms']:continue
  a1=summary['arms'][policy];a0=summary['arms'][base];r1={x['seed']:x for x in a1['runs']};r0={x['seed']:x for x in a0['runs']};seeds=sorted(set(r1)&set(r0))
  summary['comparisons'][policy+'_vs_'+base]={'seeds':seeds,'gain_percent':100*(statistics.mean(r1[s]['G'] for s in seeds)/statistics.mean(r0[s]['G'] for s in seeds)-1),'beta_gain_percent':{str(b):100*(statistics.mean(r1[s]['beta_G'][str(b)] for s in seeds)/statistics.mean(r0[s]['beta_G'][str(b)] for s in seeds)-1) for b in [1.65,3.3,6.6,13.2]}}
 (root/'multibatch_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'arms':{k:{'G':v['G_mean'],'seeds':len(v['runs']),'holds':v.get('mechanism',{}).get('requested')} for k,v in summary['arms'].items()},'comparisons':summary['comparisons']},indent=2))

if __name__=='__main__':main()
