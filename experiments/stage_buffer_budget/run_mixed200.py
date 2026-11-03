"""Four paired 200-request runs following a mixed mechanism gate; one worker."""
import argparse,copy,hashlib,json,signal,subprocess,sys,time
from pathlib import Path
from mixed200_service import Service
from run_frozen_stream import drain
from sse_client_extended import send_request
from calibrate_mixed200 import make_row
from gpu_activity import analyze as gpu_analyze

E=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arms');p.add_argument('--keep-worker',action='store_true');a=p.parse_args();root=a.root
    plan=json.loads((root/'plan.json').read_text());arms=a.arms.split(',') if a.arms else ['ours_smoke']+plan['arms']
    service=None
    def progress(**fields):
        fields['time']=time.time();(root/'progress.json').write_text(json.dumps(fields,indent=2));print(json.dumps(fields),flush=True)
    def interrupt(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for arm in arms:
            policy,trace=arm.split('_',1);workload=root/(trace+'.jsonl')
            if hashlib.sha256(workload.read_bytes()).hexdigest()!=plan['workload_sha256'][trace]:raise ValueError('frozen workload changed')
            service=Service(root,arm);progress(arm=arm,status='starting');service.start()
            for j,path in enumerate(['text','audio']):
                warm=make_row(100+j,path,'standard');rid='warm-'+arm+'-'+path
                warm['source_request_id']=warm['request_id']=rid;warm['payload']['omni_scheduling']['source_request_id']=rid
                warm['payload']['max_tokens']=16
                result=send_request(warm,root/(arm+'_warmup'),'http://127.0.0.1:8095/v1/chat/completions',timeout_s=240)
                if result['status']!='success':raise RuntimeError('warmup failed: '+str(result.get('error')))
                drain('http://127.0.0.1:8095')
            window=30. if trace=='smoke' else plan['window_s'];out=root/arm
            progress(arm=arm,status='replaying',offered=24 if trace=='smoke' else 200)
            subprocess.run([sys.executable,str(E/'run_frozen_stream.py'),'--workload',str(workload),'--output',str(out),
                '--window-s',str(window),'--timeout-s',str(plan['timeout_s']),'--max-client-workers','256',
                '--feedback-min-interval-s','.2','--gpu-stage-map','{"0":[0,1,2,3],"1":[4],"2":[5]}'],check=True)
            for command in [
                ['stream_goodput.py','--workload',str(workload),'--run',str(out),'--config',str(root/'metric.json')],
                ['analyze_mixed200.py','--root',str(root),'--arm',arm]]:
                subprocess.run(['docker','exec','stage-buffer-mixed200','python',str(E/command[0])]+command[1:],check=True)
            bench=json.loads((out/'benchmark.json').read_text());audit=bench['audit']
            manifest=json.loads((out/'manifest.json').read_text())
            gpu_analyze(out/'gpu_telemetry',manifest['origin_monotonic_s'],manifest['drained_monotonic_s'],out/'gpu_analysis')
            progress(arm=arm,status='analyzed',Gstream=bench['Gstream'],generated_tps=bench['generated_tokens_per_s'],
                SLO_tps=bench['SLO_generated_tokens_per_s'],holds=audit['requested'],complete_chains=audit['complete_chains'],errors=len(audit['errors']))
            service.stop();service=None
            if trace=='smoke':
                if audit['requested']==0 or audit['complete_chains']!=audit['requested'] or audit['kv_retained_chains']!=audit['requested'] or audit['errors'] or bench['generation_wire_count_mismatches'] or bench['groups']['all']['completed']!=24:
                    raise RuntimeError('mixed mechanism gate failed; do not claim pause benefit or continue performance comparison')
            if trace!='smoke':
                subprocess.run([sys.executable,str(E/'analyze_mixed200.py'),'--root',str(root),'--report'],check=True)
        progress(status='completed')
    except BaseException as error:
        progress(status='failed',error=repr(error));raise
    finally:
        try:
            if service:service.stop()
        finally:
            if not a.keep_worker:
                subprocess.run(['docker','stop','--timeout','20','stage-buffer-mixed200'],capture_output=True)
                (root/'container_cleanup.json').write_text(json.dumps({'auto_remove':True,'name':'stage-buffer-mixed200',
                    'remaining':subprocess.run(['docker','inspect','stage-buffer-mixed200'],capture_output=True).returncode==0,'time':time.time()},indent=2))
if __name__=='__main__':main()
