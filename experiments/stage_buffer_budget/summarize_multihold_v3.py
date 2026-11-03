"""Separate pacing gains from pause gains; never call a failed smoke performance evidence."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

def stat(values):
    v=sorted(x for x in values if x is not None)
    return {'n':len(v),'mean':statistics.mean(v) if v else None,'p95':v[math.ceil(.95*len(v))-1] if v else None,'max':max(v) if v else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root;arms={};raw={}
 for arm in ['dynamic16_smoke','nohold16','buffer16','dynamic16','fixed16']:
  score_path=r/arm/'stream_score.json'
  if not score_path.exists():continue
  score=json.loads(score_path.read_text());rr=score['requests'];T=score['arrival_window_s'];mu=score['metric_config']['mu']
  actual=json.loads((r/arm/'results.json').read_text());raw[arm]={x['source_request_id']:x for x in actual}
  by_tier={}
  for tier in ['loose','standard']:
   rows=[x for x in rr if x['tier']==tier]
   by_tier[tier]={'requests':len(rows),'first_slo_met':sum(x['first_slo_met'] for x in rows),
     'ttf_s':stat([x['first_output_s'] for x in rows]),'e2e_s':stat([x['e2e_s'] for x in rows]),
     'e2e_met':sum(x['e2e_slo_met'] for x in rows),'rebuffer_s':stat([x['rebuffer_s'] for x in rows]),
     'max_rebuffer_s':stat([x['max_rebuffer_s'] for x in rows])}
  audit_path=r/(arm+'_audit.json');audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
  chains=audit.get('chains',[])
  arms[arm]={'scope':'mechanism_only' if arm.endswith('smoke') else 'single_seed_pilot',
    'offered':len(rr),'complete':sum(x['complete'] for x in rr),'G':score['G_stream'],
    'old_gate_G':sum(x['net'] for x in rr if x['complete'] and x['first_slo_met'])/T,
    'beta_G':{str(b):sum(max(0,x['utility']-b*x['first_violation_s']-mu*x['rebuffer_s']) for x in rr)/T for b in [3.3,13.2,26.4]},
    'utility_per_T':sum(x['utility'] for x in rr)/T,'first_penalty_per_T':sum(x['first_penalty'] for x in rr)/T,
    'rebuffer_penalty_per_T':sum(mu*x['rebuffer_s'] for x in rr)/T,'by_tier':by_tier,
    'mechanism':{k:audit.get(k) for k in ['requested','complete_chains','kv_retained_chains','peak_concurrent_holds','integrity_errors','client_text_hash_verified']},
    'actual_pause_ms':stat([c.get('actual_pause_ms') for c in chains]),
    'requested_100ms_exceeded':sum(c.get('actual_pause_ms',0)>100 for c in chains),
    'resume_to_new_chunk_ms':stat([c.get('resume_to_first_chunk_ms') for c in chains]),
    'client_send_lateness_s':stat([x['send_lateness_s'] for x in actual]),
    'score_sha256':hashlib.sha256(score_path.read_bytes()).hexdigest()}
 comparisons={}
 for arm,base in [('buffer16','nohold16'),('dynamic16','nohold16'),('dynamic16','buffer16'),('fixed16','buffer16')]:
  if arm not in arms or base not in arms:continue
  ids=set(raw[arm])&set(raw[base]);hash_same=sum(raw[arm][i]['text_sha256']==raw[base][i]['text_sha256'] for i in ids)
  comparisons[arm+'_vs_'+base]={'relative_G_pct':100*(arms[arm]['G']/arms[base]['G']-1),
    'old_gate_relative_pct':100*(arms[arm]['old_gate_G']/arms[base]['old_gate_G']-1) if arms[base]['old_gate_G'] else None,
    'paired':len(ids),'identical_text':hash_same,'identical_token_ids':sum(raw[arm][i]['text_token_ids_sha256']==raw[base][i]['text_token_ids_sha256'] for i in ids),
    'beta_gain_pct':{b:100*(v/arms[base]['beta_G'][b]-1) if arms[base]['beta_G'][b] else None for b,v in arms[arm]['beta_G'].items()},
    'causal_pause_comparison':base=='buffer16','output_comparable':hash_same==len(ids)}
 result={'scope':'text-only K16/R64 single seed; not formal production or two-stage conclusion',
   'caveats':['buffer-only isolates delivery pacing from pause effects','all urgent requests currently share one predicted cost; fixed and dynamic budgets coincide','beta=3.3 primary; other beta values frozen sensitivity','server text is not client-owned inventory','100ms is requested slice, not actual wall-clock bound'],
   'arms':arms,'comparisons':comparisons}
 (r/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'scores':{k:v['G'] for k,v in arms.items()},'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
