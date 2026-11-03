"""Freeze a moderate mixed-path pilot before examining policy comparisons."""
import copy,hashlib,json,random
from pathlib import Path
ROOT=Path('/home/zhongyu/research_reports/stage_buffer_experiment_20261102/two_hour_1758')
source=ROOT.parent/'calibration_pilot/frozen_pilot_workload.jsonl'
profiles={r['stratum']:r for r in map(json.loads,source.read_text().splitlines())}
contracts={}; manifests=[]
for seed in [1103,2207,3301,4409,5519]:
 rng=random.Random(seed);rows=[]
 for cycle in range(3):
  specifications=[('audio_long','loose',256,0.,120000.,1300.,3000.),
     ('text_short','standard',64,1.5,3000.,1300.,None),
     ('text_long','loose',64,7.,6000.,2600.,None),
     ('audio_short','standard',64,9.,45000.,1300.,2500.)]
  for stratum,tier,tokens,offset,slo,ttft,ttfp in specifications:
   row=copy.deepcopy(profiles[stratum]);rid=f'exp2-pilot-s{seed}-c{cycle}-{stratum}'
   path=row['request_path'];arrival=20*cycle+offset+rng.uniform(0,.08)
   row.update(source_request_id=rid,request_id=rid,arrival_offset_s=arrival,slo_tier=tier,slo_ms=slo,
     ttft_slo_ms=ttft,ttfp_slo_ms=ttfp,random_seed=seed,test_scope='paired_pilot_not_formal_calibration',
     text_consume_rate_tps=4.8,audio_equivalent_rate_tps=3.3)
   row['payload']['max_tokens']=tokens
   pred=[5500.,60000.,100.] if tokens==256 else ([1400.,20000.,100.] if path=='audio' else [1400. if stratum=='text_short' else 1600.,None,None])
   row['payload']['omni_scheduling']={'schema_version':1,'source_request_id':rid,'ingress_order':len(rows),
      'slo_ms':slo,'request_path':path,'predicted_stage_ms':pred}
   contracts[rid]={'ttft_slo_ms':ttft,'first_output_service_ms':200.}
   rows.append(row)
 target=ROOT/f'pilot_s{seed}.jsonl';data=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
 if target.exists():raise FileExistsError(target)
 target.write_text(data);manifests.append({'seed':seed,'path':str(target),'sha256':hashlib.sha256(data.encode()).hexdigest(),'requests':len(rows),'arrival_window_s':60.})
(ROOT/'pilot_contracts.json').write_text(json.dumps(contracts,indent=2))
(ROOT/'pilot_workload_manifest.json').write_text(json.dumps({'scope':'normal-rate moderate-burst mixed-path pilot','seeds':manifests,
  'offered_qps':.2,'loose_fraction':.5,'request_path_audio_fraction':.5,'score_config_sha256':hashlib.sha256((ROOT/'metric_pilot.json').read_bytes()).hexdigest(),
  'slo_source':'3-per-stratum sequential pilot; 256-token reference, not >=20 sample independent calibration',
  'policy_result_used_to_choose_workload':False,'execution_cap':1,'native_resident_cap':1,'policy_resident_cap':64,
  'resource_matched_diagnostic_required':True},indent=2))
config=json.loads(Path(__file__).with_name('controller_budget_config.json').read_text())
config.update(guard_full_urgent_service=True,contracts_path=str(ROOT/'pilot_contracts.json'))
for name in ['dynamic','fixed']:
 c=copy.deepcopy(config)
 if name=='fixed':
  c['fixed_urgent_budget_s']=1.6;c['stages']['0']['pause_above_s']=.3;c['stages']['1']['pause_above_s']=.1
 (ROOT/f'{name}_controller.json').write_text(json.dumps(c,indent=2))
 env=json.loads((ROOT/'stage0_env.json').read_text());env['VLLM_OMNI_STAGE_BUFFER_CONFIG']=str(ROOT/f'{name}_controller.json')
 (ROOT/f'{name}_env.json').write_text(json.dumps(env,indent=2))
print(json.dumps(manifests,indent=2))
