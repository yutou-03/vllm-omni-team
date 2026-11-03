"""Open-loop replay of explicit experiment-two contracts; retain every SSE event."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from sse_client_extended import send_request, FeedbackWriter


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
    a=p.parse_args(); rows=[json.loads(l) for l in a.workload.read_text().splitlines() if l]
    assert len({r['source_request_id'] for r in rows})==len(rows)
    assert all(0<=r['arrival_offset_s']<a.window_s for r in rows)
    a.output.mkdir(parents=True,exist_ok=False)
    start_state=drain(a.endpoint)
    origin=time.monotonic()+1
    feedback=FeedbackWriter(a.output/'client_feedback.json')
    manifest={'workload':str(a.workload),'workload_sha256':hashlib.sha256(a.workload.read_bytes()).hexdigest(),
       'client_sha256':hashlib.sha256(Path(__file__).with_name('sse_client_extended.py').read_bytes()).hexdigest(),
       'arrival_window_s':a.window_s,'origin_monotonic_s':origin,'offered':len(rows),'drained_before':start_state,
       'status':'running','scope':'mechanism_or_pilot_not_formal'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def task(row):
        planned=origin+row['arrival_offset_s'];time.sleep(max(0,planned-time.monotonic()))
        return send_request(row,a.output/'client',a.endpoint+'/v1/chat/completions',feedback=feedback,
            scheduled_arrival_monotonic_s=planned,timeout_s=a.timeout_s)
    results=[]
    with ThreadPoolExecutor(max_workers=min(len(rows),64)) as pool:
        jobs=[pool.submit(task,r) for r in rows]
        for f in as_completed(jobs):
            value=f.result();results.append(value)
            (a.output/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:value.get(k) for k in ('source_request_id','status','ttft_s','ttfp_s','e2e_s','audio_stall_s','error')}),flush=True)
    manifest['drained_after']=drain(a.endpoint)
    manifest['status']='completed'; manifest['success']=sum(r['status']=='success' for r in results)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
