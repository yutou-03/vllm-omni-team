"""Independent low-load check before freezing mixed-path 200-request workloads."""
import argparse, copy, json, sys, time
from pathlib import Path
from mixed200_service import Service
from run_frozen_stream import drain
from sse_client_extended import send_request

def make_row(index, path, tier):
    loose = tier == 'loose'
    tokens = (512 if loose else 128) if path == 'text' else (128 if loose else 64)
    topic = ['how rain forms','how bicycles change gears','how a camera works','why seasons change'][index % 4]
    prompt = ('Explain '+topic+' in detail, with examples and numbered steps.' if loose else 'Explain '+topic+' in three concise sentences.')
    rid = 'mixed200-cal-'+str(index)
    row = {'source_request_id':rid,'request_id':rid,'request_path':path,'slo_tier':tier,'stratum':path+'_'+tier,
        'arrival_offset_s':0.,'slo_ms':240000.,'ttft_slo_ms':5200.,'ttfp_slo_ms':6000. if path=='audio' else None,
        'payload':{'model':'/data/models/Qwen3-Omni-30B-A3B-Instruct','messages':[{'role':'user','content':prompt}],
            'modalities':['text','audio'] if path=='audio' else ['text'], 'max_tokens':tokens,
            'ignore_eos':False,'temperature':0.,'seed':4103+index,'return_token_ids':True,
            'stream':True,'stream_options':{'include_usage':True},
            'omni_scheduling':{'schema_version':1,'source_request_id':rid,'request_path':path,
                'ingress_order':index,'slo_ms':240000.,'predicted_stage_ms':[20000.,40000.,100.] if path=='audio' else [20000.,None,None]}}}
    if path=='audio':row['payload']['audio']={'format':'wav','voice':'Chelsie'}
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    service=Service(a.root,'calibration')
    output=a.root/'calibration';output.mkdir()
    results=[];rows=[]
    try:
        service.start()
        for index,(path,tier) in enumerate([('text','standard'),('audio','standard')]+[(p,t) for p in ['text','audio'] for t in ['standard','loose'] for _ in range(2)]):
            row=make_row(index,path,tier);rows.append(row)
            result=send_request(row,output/'client','http://127.0.0.1:8095/v1/chat/completions',timeout_s=240)
            result['warmup']=index<2;results.append(result)
            (output/'workload.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (output/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:result.get(k) for k in ['source_request_id','status','ttft_s','ttfp_s','e2e_s','audio_duration_s','audio_max_stall_s','error']}),flush=True)
            if result['status']!='success':raise RuntimeError('calibration failed')
            drain('http://127.0.0.1:8095')
        (output/'completed.json').write_text(json.dumps({'completed':True,'time':time.time(),'scope':'8 independent observations; pilot calibration only'}))
    finally:service.stop()
if __name__=='__main__':main()
