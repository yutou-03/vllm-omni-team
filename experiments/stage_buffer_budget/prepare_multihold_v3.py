"""Freeze a K16 saturated-slot mechanism workload before observing GPU results."""
import argparse,copy,hashlib,json,random
from pathlib import Path
E=Path(__file__).resolve().parent

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=False)
 topics=['how bicycles change gears','how rain forms','how libraries lend books','how a vegetable garden grows','how a camera works','how bread rises','how trains change tracks','how rivers shape valleys']
 rows=[]
 for i in range(20):
  loose=i<16;rid=f'v3-s1103-r{i}';max_tokens=512 if loose else 32
  prompt=(f'Write a detailed tutorial of about 450 words about {topics[i%8]}. Include six numbered steps and examples.' if loose else f'Briefly explain {topics[i%8]} in one sentence.')
  slo=160000. if loose else 20000.;first=5200. if loose else 1300.
  row={'source_request_id':rid,'request_id':rid,'arrival_offset_s':i*.015 if loose else 6.,
    'request_path':'text','slo_tier':'loose' if loose else 'standard','slo_ms':slo,'ttft_slo_ms':first,'ttfp_slo_ms':None,
    'text_consume_rate_tps':4.8,'audio_equivalent_rate_tps':3.3,'random_seed':42,'stratum':'text_loose' if loose else 'text_urgent',
    'payload':{'model':'/data/models/Qwen3-Omni-30B-A3B-Instruct','messages':[{'role':'user','content':prompt}],
      'modalities':['text'],'max_tokens':max_tokens,'ignore_eos':False,'temperature':0.,'seed':42,'return_token_ids':True,
      'stream':True,'stream_options':{'include_usage':True},'omni_scheduling':{'schema_version':1,'source_request_id':rid,
        'request_path':'text','ingress_order':i,'slo_ms':slo,'predicted_stage_ms':[30000. if loose else 6000.,None,None]}}}
  rows.append(row)
 workload=r/'workload.jsonl';workload.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 write(r/'contracts.json',{x['source_request_id']:{'ttft_slo_ms':x['ttft_slo_ms'],'first_output_service_ms':400.} for x in rows})
 write(r/'text_delivery.json',{'tokens_per_s':4.8,'max_chars':131072,'max_events':4096,'boundary_reserve_tokens':32})
 metric={'metric_version':'soft_ttft_v2','beta':3.3,'mu':3.3,'text_tps':4.8,'audio_equivalent_tps':3.3,'audio_unit_s':.02,'lateness_decay_s':1.,'buffer_full_credit_s':10.,'buffer_decay_s':20.,'gate':'none','sensitivity_beta':[3.3,13.2,26.4]}
 write(r/'metric.json',metric)
 c=json.loads((E/'controller_budget_config.json').read_text());c.update(max_concurrent_holds=16,text_delivery_inventory=True,
    guard_full_urgent_service=True,contracts_path=str(r/'contracts.json'),urgent_guard_ms=250.,performance_claim=False)
 write(r/'dynamic_controller.json',c);fixed=copy.deepcopy(c);fixed['fixed_urgent_budget_s']=6.;write(r/'fixed_controller.json',fixed)
 write(r/'plan.json',{'scope':'mechanism_and_single_seed_pilot_not_formal','K':16,'R':64,'window_s':30.,'timeout_s':180.,
    'workload_sha256':hashlib.sha256(workload.read_bytes()).hexdigest(),'offered':len(rows),'seed':1103,
    'arms':['dynamic16_smoke','nohold16','buffer16','dynamic16','fixed16'],
    'main_metric':'soft_ttft_v2 beta=3.3; old gate and beta sensitivity also reported',
    'gate':'at least 3 simultaneous real holds; all chains complete; retained KV; exact text/wire conservation; positive owned stock',
    'limits':'text-only first validation; no audio or K64 generalization; service predictions conservative uncalibrated pilot values',
    'SLO_basis':'TTFT 1.3s urgent, 5.2s loose; E2E accommodates 512/4.8=106.7s text delivery plus margin',
    'pause_count':'derived from FIFO urgent backlog, safe candidates and R headroom; never fixed to two'})
 print(r)
if __name__=='__main__':main()
