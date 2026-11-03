"""Sequential paired seeds; each run drains, and preserves failed offered requests."""
import argparse,hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
from sse_client_extended import send_request
from run_frozen_stream import drain

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
p.add_argument('--seeds',default='1103,2207,3301,4409,5519');p.add_argument('--mechanism-first',action='store_true');a=p.parse_args()
for _ in range(300):
 try:
  with urllib.request.urlopen('http://127.0.0.1:8095/health',timeout=2) as response:
   if response.status==200:break
 except Exception:time.sleep(1)
else:raise RuntimeError('server did not become ready')
rows=[json.loads(l) for l in (a.root/'mechanism_workload.jsonl').read_text().splitlines()]
warm=rows[1];rid='exp2-warmup-'+a.arm;warm['source_request_id']=warm['request_id']=rid
warm['payload']['omni_scheduling'].update(source_request_id=rid,slo_ms=120000)
if not (a.root/(a.arm+'_warmup')).exists():
 print('warmup',send_request(warm,a.root/(a.arm+'_warmup'),'http://127.0.0.1:8095/v1/chat/completions')['status'],flush=True)
else:print('warmup already completed; resuming',flush=True)
drain('http://127.0.0.1:8095')
for name in (['mechanism'] if a.mechanism_first else [])+['pilot_s'+s for s in a.seeds.split(',') if s]:
 path=a.root/(name+'_workload.jsonl' if name=='mechanism' else name+'.jsonl')
 output=a.root/(a.arm+'_'+name)
 command=[sys.executable,str(Path(__file__).with_name('run_frozen_stream.py')),'--workload',str(path),'--output',str(output),'--window-s','3' if name=='mechanism' else '60']
 print('START',name,flush=True);subprocess.run(command,check=True);print('DONE',name,flush=True)
 if name=='mechanism' and a.arm in {'fixed','dynamic','paced'}:
  from audit_receiver_mechanism import audit
  trace=a.root/(a.arm+'_service')/'traces/stage_events.jsonl'
  with trace.open() as f:evidence=audit([json.loads(l) for l in f])
  (a.root/(a.arm+'_mechanism_audit.json')).write_text(json.dumps(evidence,indent=2))
  if not evidence['complete_chains'] or evidence['integrity_errors']:
   raise RuntimeError('mechanism gate failed; performance replay prohibited')
  held=next(r for r in json.loads((output/'results.json').read_text()) if 'victim' in r['source_request_id'])
  reference_dir=a.root/(a.arm+'_local_output_reference')
  subprocess.run([sys.executable,str(Path(__file__).with_name('run_frozen_stream.py')),
   '--workload',str(a.root/'solo_workload.jsonl'),'--output',str(reference_dir),'--window-s','1'],check=True)
  reference=json.loads((reference_dir/'results.json').read_text())[0]
  def wav_hash(row):
   h=hashlib.sha256()
   for path in sorted(Path(row['artifact_directory']).glob('*.wav')):h.update(path.read_bytes())
   return h.hexdigest()
  proof={'reference_scope':'same live service, solo without a pause trigger',
   'text_tokens_equal':held['text_token_ids']==reference['text_token_ids'],
   'audio_bytes_equal':wav_hash(held)==wav_hash(reference),
   'held_audio_sha256':wav_hash(held),'reference_audio_sha256':wav_hash(reference)}
  (a.root/(a.arm+'_output_proof.json')).write_text(json.dumps(proof,indent=2))
  if not proof['text_tokens_equal'] or not proof['audio_bytes_equal']:
   raise RuntimeError('mechanism output differs from same-process nohold reference')

