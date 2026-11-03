"""Independent sequential prefill-length validation; no performance policy tuning."""
import argparse,copy,hashlib,json,random,statistics,time
from pathlib import Path
from sse_client_extended import send_request
from run_frozen_stream import drain


def quantile(xs,q):
 xs=sorted(xs);return xs[min(len(xs)-1,int((len(xs)-1)*q))]


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);a=p.parse_args()
 out=a.root/'prefill_prediction';out.mkdir(exist_ok=False)
 template=json.loads((a.root/'mechanism_workload.jsonl').read_text().splitlines()[1]);rows=[]
 lengths=[64,128,256,384,448,640,768,1024,1536,2048];rng=random.Random(5121102)
 for rep in range(10):
  order=lengths[:];rng.shuffle(order)
  for n in order:
   row=copy.deepcopy(template);rid=f'exp2-prefill-n{n}-r{rep}'
   row.update(source_request_id=rid,request_id=rid,arrival_offset_s=0,slo_ms=120000,ttft_slo_ms=10000,
     stratum='prefill_validation',repetition=rep,repeated_words=n,split='warmup' if rep<2 else ('train' if rep<6 else 'validation'))
   row['payload']['messages']=[{'role':'user','content':'Read this list and answer with one word: '+('apple '*n)}]
   row['payload']['max_tokens']=4;row['payload']['omni_scheduling'].update(source_request_id=rid,slo_ms=120000,predicted_stage_ms=[1000.,None,None])
   rows.append(row)
 data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows);(out/'workload.jsonl').write_text(data)
 manifest={'scope':'isolated prefill-to-first-output prediction validation; includes output handling, not pure GPU kernel time',
   'workload_sha256':hashlib.sha256(data.encode()).hexdigest(),'seed':5121102,'expected_requests':len(rows),
   'fit':'ordinary least squares, intercept + actual prompt token count; train 4 repetitions/length, validation 4; warmup 2 excluded',
   'score_metric_changed':False,'status':'running'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));drain('http://127.0.0.1:8095');results=[]
 for row in rows:
  if time.time()>1793620506:break # 11:55:06 UTC; leave time for reporting.
  result=send_request(row,out/'client','http://127.0.0.1:8095/v1/chat/completions',timeout_s=60)
  results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2))
  if result['status']!='success':raise RuntimeError('prefill validation request failed')
 drain('http://127.0.0.1:8095');events={}
 with a.trace.open() as f:
  for line in f:
   e=json.loads(line);rid=e.get('request_id') or ''
   if 'exp2-prefill-' not in rid or e.get('stage_id')!=0:continue
   name=e['event'];v=events.setdefault(rid,{})
   if name=='stage_first_forward':v.setdefault('first_forward',e['monotonic'])
   elif name=='stage_study_thinker_output_produced':v.setdefault('first_output',e['monotonic'])
 lookup={r['source_request_id']:r for r in rows};observations=[]
 for result in results:
  r=lookup[result['source_request_id']];e=events.get(result['response_request_id'],{})
  if not {'first_forward','first_output'}<=e.keys():raise ValueError('missing measured prefill events')
  observations.append({'request_id':r['source_request_id'],'split':r['split'],'prompt_tokens':result['actual_prompt_tokens'],
    'repetition':r['repetition'],'service_ms':1000*(e['first_output']-e['first_forward']),'client_ttft_ms':1000*result['ttft_s']})
 train=[x for x in observations if x['split']=='train'];valid=[x for x in observations if x['split']=='validation']
 xm=statistics.mean(x['prompt_tokens'] for x in train);ym=statistics.mean(x['service_ms'] for x in train)
 slope=sum((x['prompt_tokens']-xm)*(x['service_ms']-ym) for x in train)/sum((x['prompt_tokens']-xm)**2 for x in train);intercept=ym-slope*xm
 for x in valid:
  x['predicted_ms']=intercept+slope*x['prompt_tokens'];x['absolute_error_ms']=abs(x['predicted_ms']-x['service_ms']);x['absolute_percentage_error']=x['absolute_error_ms']/x['service_ms']
 groups={}
 for name,group in [('at_most_512',[x for x in valid if x['prompt_tokens']<=512]),('over_512',[x for x in valid if x['prompt_tokens']>512])]:
  if group:groups[name]={'n':len(group),'MAPE_percent':100*statistics.mean(x['absolute_percentage_error'] for x in group),
     'MAE_ms':statistics.mean(x['absolute_error_ms'] for x in group),'p95_APE_percent':100*quantile([x['absolute_percentage_error'] for x in group],.95)}
 summary={'scope':manifest['scope'],'fit':{'intercept_ms':intercept,'slope_ms_per_token':slope},'groups':groups,
   'interpretation':'Single text prompt family and isolated K=1 service; cannot infer busy batch checkpoint prediction or an exact universal 512-token breakpoint.',
   'observations':observations}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));manifest.update(status='completed' if len(results)==len(rows) else 'time_limited',completed=len(results));(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k!='observations'},indent=2))
if __name__=='__main__':main()
