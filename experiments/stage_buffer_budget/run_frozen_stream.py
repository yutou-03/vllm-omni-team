"""Open-loop replay of explicit experiment-two contracts; retain every SSE event."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from sse_client_extended import send_request, FeedbackWriter
from gpu_activity import GPUCollector, stage_map


def drain(endpoint, timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        with urllib.request.urlopen(endpoint+'/v1/omni/drain_status?timeout=2', timeout=8) as r:
            state=json.load(r)
        if state.get('healthy') and state.get('drained'):
            return state
        time.sleep(.25)
    raise TimeoutError(str(state))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--endpoint',default='http://127.0.0.1:8095')
    p.add_argument('--window-s',type=float,required=True); p.add_argument('--timeout-s',type=float,default=180)
    p.add_argument('--feedback-min-interval-s',type=float,default=0.0)
    p.add_argument('--max-client-workers',type=int,default=256,help='must cover all offered requests to preserve open-loop arrivals')
    p.add_argument('--gpu-stage-map',type=stage_map,help='JSON mapping stage IDs to exclusive physical GPU indices')
    a=p.parse_args(); rows=[json.loads(l) for l in a.workload.read_text().splitlines() if l]
    if not rows or len(rows)>a.max_client_workers:
        raise ValueError('open-loop replay requires one available client worker per offered request')
    assert len({r['source_request_id'] for r in rows})==len(rows)
    assert all(0<=r['arrival_offset_s']<a.window_s for r in rows)
    a.output.mkdir(parents=True,exist_ok=False)
    start_state=drain(a.endpoint)
    collector=GPUCollector(a.output/'gpu_telemetry',a.gpu_stage_map) if a.gpu_stage_map else None
    origin=None
    feedback=FeedbackWriter(a.output/'client_feedback.json',min_interval_s=a.feedback_min_interval_s)
    manifest={'workload':str(a.workload),'workload_sha256':hashlib.sha256(a.workload.read_bytes()).hexdigest(),
       'client_sha256':hashlib.sha256(Path(__file__).with_name('sse_client_extended.py').read_bytes()).hexdigest(),
       'arrival_window_s':a.window_s,'origin_monotonic_s':origin,'offered':len(rows),'drained_before':start_state,
       'status':'running','scope':'mechanism_or_pilot_not_formal',
       'feedback_min_interval_s':a.feedback_min_interval_s,'client_workers':len(rows),'gpu_stage_map':a.gpu_stage_map,'gpu_telemetry_enabled':collector is not None}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def task(row):
        planned=origin+row['arrival_offset_s'];time.sleep(max(0,planned-time.monotonic()))
        return send_request(row,a.output/'client',a.endpoint+'/v1/chat/completions',feedback=feedback,
            scheduled_arrival_monotonic_s=planned,timeout_s=a.timeout_s)
    results=[]
    try:
        if collector is not None:
            collector.start()
        origin=time.monotonic()+1
        manifest['origin_monotonic_s']=origin
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            jobs=[pool.submit(task,r) for r in rows]
            for f in as_completed(jobs):
                value=f.result();results.append(value)
                (a.output/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps({k:value.get(k) for k in ('source_request_id','status','ttft_s','ttfp_s','e2e_s','audio_stall_s','error')}),flush=True)
        manifest['client_completed_monotonic_s']=time.monotonic()
        manifest['max_send_lateness_s']=max(r['send_monotonic_s']-r['scheduled_arrival_monotonic_s'] for r in results) if all('send_monotonic_s' in r for r in results) else None
        manifest['drained_after']=drain(a.endpoint)
        manifest['drained_monotonic_s']=time.monotonic()
        manifest['status']='completed'; manifest['success']=sum(r['status']=='success' for r in results)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error))
        raise
    finally:
        if collector is not None:
            collector.stop()
            manifest['gpu_sampler_errors']=collector.errors
        manifest['replay_end_monotonic_s']=time.monotonic()
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
