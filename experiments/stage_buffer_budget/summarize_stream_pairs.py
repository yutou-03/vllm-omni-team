"""Pair frozen seeds, retain class guardrails, and report uncertainty and scope."""
import argparse,collections,json,random,statistics
from pathlib import Path


def pct(values,p):
 values=sorted(values)
 return values[min(len(values)-1,max(0,int((len(values)-1)*p)))] if values else None


def summarize(root):
 arms={}; seeds=[1103,2207,3301,4409,5519]
 for arm in ['native_k1','fixed','dynamic','matched','paced','buffer_only']:
  runs={}
  for seed in seeds:
   path=root/f'{arm}_pilot_s{seed}/stream_score.json'
   if path.exists():
    manifest=json.loads((path.parent/'manifest.json').read_text())
    if manifest.get('status')!='completed':raise ValueError('incomplete run cannot enter paired summary')
    runs[seed]=json.loads(path.read_text())
  if not runs:continue
  reqs=[r for x in runs.values() for r in x['requests']]
  groups={}
  for group in ['all','text_standard','text_loose','audio_standard','audio_loose']:
   rows=[r for r in reqs if group=='all' or r['path']+'_'+r['tier']==group]
   if rows:groups[group]={'requests':len(rows),'completed':sum(r['complete'] for r in rows),'first_slo_met':sum(r['first_slo_met'] for r in rows),
    'e2e_slo_met':sum(r['e2e_slo_met'] for r in rows),'mean_first_s':statistics.mean(r['first_output_s'] for r in rows if r['first_output_s'] is not None),
    'p95_first_s':pct([r['first_output_s'] for r in rows if r['first_output_s'] is not None],.95),'p95_e2e_s':pct([r['e2e_s'] for r in rows if r['e2e_s'] is not None],.95),
    'mean_rebuffer_s':statistics.mean(r['rebuffer_s'] for r in rows),
    'p95_max_rebuffer_s':pct([r['max_rebuffer_s'] for r in rows],.95),
    'max_rebuffer_s':max(r['max_rebuffer_s'] for r in rows),
    'mean_content_s':statistics.mean(r.get('content_s',0) for r in rows),
    'mean_raw_equivalent_units':statistics.mean(r['raw_units'] for r in rows),
    'G_contribution':sum(r['contribution'] for r in rows)/(60*len(runs))}
  arms[arm]={'seeds':list(runs),'G_stream_mean':statistics.mean(r['G_stream'] for r in runs.values()),
    'G_by_seed':{str(s):r['G_stream'] for s,r in runs.items()},'groups':groups}
 comparisons={}
 for policy in ['fixed','dynamic','paced']:
  for base in ['native_k1','matched','buffer_only']:
   if policy not in arms or base not in arms:continue
   paired=sorted(set(arms[policy]['seeds'])&set(arms[base]['seeds']))
   for seed in paired:
    left=json.loads((root/f'{policy}_pilot_s{seed}/stream_score.json').read_text())
    right=json.loads((root/f'{base}_pilot_s{seed}/stream_score.json').read_text())
    if (left['workload_sha256'],left['metric_config_sha256'])!=(right['workload_sha256'],right['metric_config_sha256']):
     raise ValueError('paired artifacts differ')
   pairs=[(arms[policy]['G_by_seed'][str(s)],arms[base]['G_by_seed'][str(s)]) for s in paired]
   if not pairs:continue
   ratio=sum(x for x,y in pairs)/sum(y for x,y in pairs)-1
   rng=random.Random(613); boot=[]
   for _ in range(10000):
    sample=rng.choices(pairs,k=len(pairs));boot.append(sum(x for x,y in sample)/sum(y for x,y in sample)-1)
   comparisons[policy+'_vs_'+base]={'paired_seeds':paired,'gain_percent':100*ratio,
      'paired_bootstrap_95_percent_interval':[100*pct(boot,.025),100*pct(boot,.975)] if len(paired)>1 else None,
      'absolute_G_difference':statistics.mean(x-y for x,y in pairs)}
 result={'scope':'controlled K=1 exploratory pilot; ordered arms, repeated profile contents, calibration N=3/stratum; not production capacity claim',
  'arms':arms,'comparisons':comparisons,'expected_seed_count':5,'metric_frozen_before_comparison':True}
 (root/'paired_summary.json').write_text(json.dumps(result,indent=2))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(summarize(a.root),indent=2))
