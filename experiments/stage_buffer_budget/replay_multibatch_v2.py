"""Replay immutable v2 workloads/configs into a new directory; never overwrite evidence."""
import argparse,hashlib,json,subprocess,sys,time,urllib.request,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--gpus',required=True);p.add_argument('--arms',default='native64,nohold16,dynamic16,fixed16');p.add_argument('--seeds',default='1103,2207,3301');a=p.parse_args()
 arms=a.arms.split(',');seeds=[int(x) for x in a.seeds.split(',')]
 if not set(arms)<={'native64','nohold16','dynamic16','fixed16','nohold8','dynamic8','guard16'}:raise ValueError('unknown arm')
 if not set(seeds)<={1103,2207,3301}:raise ValueError('only frozen seeds allowed')
 a.output_root.mkdir(parents=True,exist_ok=False);e=Path(__file__).parent;r=a.frozen_root;tag=hashlib.sha256(str(a.output_root).encode()).hexdigest()[:8]
 for n in ['metric_v2.json','workload_manifest.json','legacy_k1_v2_summary.json','calibration_summary.json']:
  shutil.copy2(r/n,a.output_root/n)
 (a.output_root/'replay_provenance.json').write_text(json.dumps({'frozen_root':str(r),'argv':sys.argv,'time':time.time(),'arms':arms,'seeds':seeds},indent=2))
 from run_frozen_stream import drain
 from sse_client_extended import send_request
 warm=[json.loads(x) for x in (r/'calibration_workload.jsonl').read_text().splitlines()]
 for arm in arms:
  service=a.output_root/(arm+'_service');name='stage-buffer-exp2-replay-'+tag+'-'+arm
  command=[sys.executable,str(e/'launch_exp2_container.py'),'--run',str(service),'--name',name,'--source',str(r/'paired_framework'),'--gpus',a.gpus,'--entrypoint-script',str(r/'external_entrypoint/server_entry.py')]
  if arm!='native64':command+=['--env',str(r/(arm+'_env.json'))]
  try:
   subprocess.run(command,check=True)
   for _ in range(900):
    try:
     with urllib.request.urlopen('http://127.0.0.1:8095/health',timeout=2) as response:
      if response.status==200:break
    except Exception:time.sleep(1)
   else:raise TimeoutError('startup')
   for index in [0,2]:
    x=json.loads(json.dumps(warm[index]));rid='replay-warm-'+arm+'-'+str(index);x['source_request_id']=x['request_id']=rid;x['payload']['omni_scheduling']['source_request_id']=rid
    result=send_request(x,a.output_root/(arm+'_warmup'),'http://127.0.0.1:8095/v1/chat/completions',timeout_s=240)
    if result['status']!='success':raise RuntimeError('warmup failed')
   drain('http://127.0.0.1:8095')
   for seed in seeds:
    workload=r/f'workload_s{seed}.jsonl';out=a.output_root/f'{arm}_s{seed}'
    subprocess.run([sys.executable,str(e/'run_frozen_stream.py'),'--workload',str(workload),'--output',str(out),'--window-s','75','--timeout-s','240'],check=True)
    scorer=r/'paired_framework/experiments/stage_buffer_budget/stream_goodput.py'
    subprocess.run(['docker','exec',name,'python',str(scorer),'--workload',str(workload),'--run',str(out),'--config',str(r/'metric_v2.json')],check=True)
  finally:
   # server.log is written directly to the host; never truncate it during cleanup.
   exists=subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   if exists.returncode==0:
    stopped=subprocess.run(['docker','stop','--timeout','30',name],capture_output=True,text=True)
    if stopped.returncode and subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
     raise RuntimeError('container cleanup failed: '+stopped.stderr)
 subprocess.run([sys.executable,str(e/'summarize_multibatch_v2.py'),str(a.output_root)],check=True)

if __name__=='__main__':main()
