"""Filter real pilot requests and audit conservation, safety windows, and content drift."""
import argparse,collections,json,statistics
from pathlib import Path
from audit_receiver_mechanism import audit as chain_audit
from audit_stage_buffer_trace import audit as inventory_audit


def qtile(values,p):
 values=sorted(values);return values[min(len(values)-1,int((len(values)-1)*p))] if values else None


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args()
 service=a.root/('native_k1_service2' if a.arm=='native_k1' else a.arm+'_service')
 trace=service/'traces/stage_events.jsonl';out=a.root/(a.arm+'_pilot_events.jsonl')
 keep=[];names=collections.Counter();peaks={};receives=collections.defaultdict(float);selected=collections.defaultdict(set);codec_received=collections.defaultdict(float);codec_started=collections.defaultdict(float)
 with trace.open() as f,out.open('w') as g:
  for line in f:
   e=json.loads(line)
   if not any('exp2-pilot-' in str(e.get(k,'')) for k in ('request_id','physical_request_id','victim_request_id','urgent_request_id')):continue
   g.write(line);keep.append(e);names[e['event']]+=1
 integrity=inventory_audit(out);chains=chain_audit(keep)
 active={};safety=[]
 def rid(e):return e.get('physical_request_id') or e.get('request_id')
 def ready(stage,r):return receives[r]-len(selected[r]) if stage==0 else codec_received[r]-codec_started[r]
 for e in sorted(keep,key=lambda e:e['monotonic']):
  n=e['event'];r=rid(e);sid=e.get('stage_id');t=e['monotonic']
  if n=='stage_study_text_payload_received':receives[r]+=e['raw_entity_embedding_rows_total']-(e.get('raw_ids_prompt_count',0) if e['chunk_id']==0 else 0)
  elif n=='stage_study_text_conditioning_selected' and e.get('conditioning_kind')=='output_embedding':selected[r].update(range(e['start_index'],e['end_index']))
  elif n=='stage_study_chunk_received':codec_received[r]+=e['new_frames']
  elif n=='stage_study_chunk_conversion_started':codec_started[r]+=e['new_frames']
  if n=='small_hold_requested':
   v={'stage_id':sid,'request_id':r,'hold_start':t,'min_ready_until_new_chunk':ready(sid,r),'at_hold_ready':ready(sid,r),'last_new_chunk':None};active[(sid,r)]=v;safety.append(v)
  for (stage,request),v in active.items():
   if request==r and v['last_new_chunk'] is None:v['min_ready_until_new_chunk']=min(v['min_ready_until_new_chunk'],ready(stage,r))
  if n=='small_hold_first_rescheduled' and (sid,r) in active:active[(sid,r)]['rescheduled']=t
  if n in ('stage_study_text_payload_sent','stage_study_chunk_sent') and (sid,r) in active:
   v=active[(sid,r)]
   if 'rescheduled' in v and v['last_new_chunk'] is None:v['last_new_chunk']=t
  if n=='stage_study_receiver_buffer_enqueued':
   for k in ['buffered_units','buffered_bytes','total_buffered_bytes','buffered_chunks']:peaks[f'stage{sid}_{k}']=max(peaks.get(f'stage{sid}_{k}',0),e.get(k,0))
 release_over=[];reexecute_over=[];new_chunk=[];actual=[]
 for c in chains['chains']:
  deadline=c['request_event']['deadline_monotonic_s']
  if 'released' in c:release_over.append(max(0,1000*(c['released']-deadline)))
  if 'rescheduled' in c:reexecute_over.append(max(0,1000*(c['rescheduled']-deadline)))
  if c['complete_chain']:actual.append(c['actual_to_rescheduled_ms']);new_chunk.append(c['reschedule_to_first_chunk_ms'])
 counts={'requested':len(chains['chains']),'complete':chains['complete_chains'],
  'release_past_100ms_request_count':sum(v>0 for v in release_over),'resume_past_request_count':sum(v>0 for v in reexecute_over),
  'actual_pause_ms_p50':qtile(actual,.5),'actual_pause_ms_p95':qtile(actual,.95),'actual_pause_ms_max':max(actual,default=None),
  'release_overrun_ms_max':max(release_over,default=None),'resume_overrun_ms_max':max(reexecute_over,default=None),
  'resume_to_new_chunk_ms_p95':qtile(new_chunk,.95),'resume_to_new_chunk_ms_max':max(new_chunk,default=None)}
 result={'arm':a.arm,'scope':'pilot real requests only; unit-test and warmup events excluded, raw trace retained',
  'chain_summary':counts,'chains':chains,'inventory':integrity,'minimum_inventory_windows':safety,'receiver_peaks':peaks,
  'peak_resident_requests':max((e.get('resident_count',0) for e in keep),default=0),
  'all_holds_keep_positive_receiver_stock':all(v['min_ready_until_new_chunk']>0 for v in safety)}
 (a.root/(a.arm+'_pilot_audit.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps({'arm':a.arm,'chains':counts,'conservation_errors':len(integrity['conservation_violations']),
  'receiver_errors':chains['integrity_errors'],'stock_positive':result['all_holds_keep_positive_receiver_stock']},indent=2))
if __name__=='__main__':main()
