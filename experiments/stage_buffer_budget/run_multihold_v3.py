"""One reusable Docker container; one isolated service process per comparison arm."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
E=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3,4,5');p.add_argument('--arms',default='dynamic16_smoke,nohold16,buffer16,dynamic16,fixed16');a=p.parse_args();r=a.root
 plan=json.loads((r/'plan.json').read_text());assert hashlib.sha256((r/'workload.jsonl').read_bytes()).hexdigest()==plan['workload_sha256']
 name='stage-buffer-exp2-v3-'+hashlib.sha256(str(r).encode()).hexdigest()[:8]
 (r/'runtime_cache').mkdir(exist_ok=True)
 service_process=None;service_log=None;pid_file=None;created=False
 def progress(**fields):
  fields.update(wall_time=time.time(),container=name);(r/'progress.json').write_text(json.dumps(fields,indent=2));print(json.dumps(fields),flush=True)
 def stop_service():
  nonlocal service_process,service_log
  if service_process is None:return
  if pid_file and pid_file.exists():
   pid=int(pid_file.read_text())
   cleanup="import psutil,sys; p=psutil.Process(int(sys.argv[1])); children=p.children(recursive=True); targets=children+[p]; [x.terminate() for x in targets if x.is_running()]; gone,alive=psutil.wait_procs(targets,timeout=15); [x.kill() for x in alive if x.is_running()]"
   subprocess.run(['docker','exec',name,'python','-c',cleanup,str(pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
  try:service_process.wait(timeout=30)
  except subprocess.TimeoutExpired:raise RuntimeError('service failed to stop; container will be removed')
  service_log.close();service_process=None;service_log=None
 def interrupt(signum,frame):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,interrupt)
 try:
  subprocess.run([sys.executable,str(E/'launch_exp2_container.py'),'--run',str(r/'worker'),'--name',name,'--gpus',a.gpus,'--entrypoint-script',str(E/'service_entry.py'),'--worker-only'],check=True)
  created=True;snap=r/'worker/framework';deploy=r/'worker/deploy.yaml'
  for arm in a.arms.split(','):
   service=r/(arm+'_service');service.mkdir();(service/'traces').mkdir();pid_file=service/'server.pid'
   env={'PYTHONPATH':str(snap),'STAGE_STUDY_FRAMEWORK':str(snap),'PYTHONDONTWRITEBYTECODE':'1','VLLM_BATCH_INVARIANT':'1',
     'VLLM_OMNI_BASELINE_POLICY':'native_fcfs','VLLM_OMNI_STAGE_STUDY_TRACE':'1','STAGE_QUEUE_TRACE_MODE':'stage_study',
     'STAGE_QUEUE_TRACE_DIR':str(service/'traces'),'STAGE_QUEUE_RUN_ID':arm,'VLLM_OMNI_SMALL_HOLD_EXECUTION_CAP':'16',
     'VLLM_OMNI_SMALL_HOLD_STAGES':'0','VLLM_OMNI_SLO_ACTIVE_PREEMPTION':'0',
     'TORCHINDUCTOR_CACHE_DIR':str(r/'runtime_cache/inductor'),'TRITON_CACHE_DIR':str(r/'runtime_cache/triton')}
   if arm!='nohold16':env['VLLM_OMNI_TEXT_DELIVERY_CONFIG']=str(r/'text_delivery.json')
   if arm.startswith(('dynamic','fixed')):
    env.update(VLLM_OMNI_SMALL_HOLD_PROBE='1',VLLM_OMNI_URGENT_KEEP_ADMISSION='1',VLLM_OMNI_STAGE_BUFFER_POLICY='1',
      VLLM_OMNI_SMALL_HOLD_CONTROLLER='stage_buffer_controller_budget:decide',
      VLLM_OMNI_STAGE_BUFFER_CONFIG=str(r/('fixed_controller.json' if arm.startswith('fixed') else 'dynamic_controller.json')))
   args=['docker','exec']
   for k,v in env.items():args+=['-e',f'{k}={v}']
   wrapper="import os,sys,runpy; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
   args += [name,'python','-u','-c',wrapper,str(pid_file),str(snap/'experiments/stage_buffer_budget/service_entry.py'),'serve','/data/models/Qwen3-Omni-30B-A3B-Instruct','--omni','--host','0.0.0.0','--port','8095','--deploy-config',str(deploy)]
   (service/'launch.json').write_text(json.dumps({'command':args,'environment':env,'source_manifest':str(r/'worker/manifest.json')},indent=2))
   service_log=(service/'server.log').open('w');service_process=subprocess.Popen(args,stdout=service_log,stderr=subprocess.STDOUT)
   progress(arm=arm,status='starting')
   for t in range(900):
    if service_process.poll() is not None:raise RuntimeError(f'{arm} server exited {service_process.returncode}')
    try:
     with urllib.request.urlopen('http://127.0.0.1:8095/health',timeout=2) as response:
      if response.status==200:break
    except Exception:time.sleep(1)
   else:raise TimeoutError('server startup')
   from sse_client_extended import send_request
   from run_frozen_stream import drain
   warm=json.loads((r/'workload.jsonl').read_text().splitlines()[0]);rid='warm-'+arm;warm['source_request_id']=warm['request_id']=rid
   warm['payload']['max_tokens']=16;warm['payload']['omni_scheduling']['source_request_id']=rid
   result=send_request(warm,r/(arm+'_warmup'),'http://127.0.0.1:8095/v1/chat/completions',timeout_s=180)
   if result['status']!='success':raise RuntimeError('warmup failed '+str(result.get('error')))
   drain('http://127.0.0.1:8095');progress(arm=arm,status='replaying')
   out=r/arm
   subprocess.run([sys.executable,str(E/'run_frozen_stream.py'),'--workload',str(r/'workload.jsonl'),'--output',str(out),'--window-s',str(plan['window_s']),'--timeout-s',str(plan['timeout_s'])],check=True)
   subprocess.run(['docker','exec',name,'python',str(snap/'experiments/stage_buffer_budget/stream_goodput.py'),'--workload',str(r/'workload.jsonl'),'--run',str(out),'--config',str(r/'metric.json')],check=True)
   from audit_multihold_v3 import audit
   result=audit(service/'traces/stage_events.jsonl');(r/(arm+'_audit.json')).write_text(json.dumps(result,indent=2))
   progress(arm=arm,status='audited',holds=result['requested'],complete=result['complete_chains'],peak=result['peak_concurrent_holds'],errors=result['integrity_errors'])
   stop_service()
   if arm=='dynamic16_smoke':
    positive=all(v>0 for v in result['minimum_owned_text_tokens_by_victim'].values())
    if result['peak_concurrent_holds']<3 or result['requested']!=result['complete_chains'] or result['kv_retained_chains']!=result['requested'] or result['integrity_errors'] or not positive:
     raise RuntimeError('mechanism gate failed; performance replay stopped')
  progress(status='completed')
 except BaseException as error:
  progress(status='failed',error=str(error));raise
 finally:
  try:stop_service()
  finally:
   if created:subprocess.run(['docker','stop','--timeout','20',name],check=False)
   (r/'container_cleanup.json').write_text(json.dumps({'name':name,'auto_remove':True,'time':time.time(),'remaining':subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0},indent=2))
if __name__=='__main__':main()
