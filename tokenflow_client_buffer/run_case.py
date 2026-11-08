"""Host-side pilot runner. Owns only its gateway/client processes, not the model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--modality',choices=['text','audio'],required=True)
    p.add_argument('--arm',choices=['fcfs','buffer'],required=True)
    p.add_argument('--workload',required=True)
    p.add_argument('--capacity-profile')
    p.add_argument('--capacity-only',action='store_true')
    p.add_argument('--transport-root')
    p.add_argument('--capacity-drop-after',type=float)
    p.add_argument('--capacity-drop-gamma',type=float)
    p.add_argument('--container',default='zy-tokenflow-client-20261106')
    a=p.parse_args()
    if a.capacity_only and (a.arm!='buffer' or not a.capacity_profile):
        p.error('capacity-only ablation requires buffer arm and calibrated profile')
    if a.capacity_drop_after is not None and (not a.capacity_profile or
            a.capacity_drop_gamma is None or a.capacity_drop_gamma <= 0):
        p.error('capacity drop requires a calibrated profile and positive gamma')
    root=Path(a.root).resolve();snap=Path(a.transport_root).resolve() if a.transport_root else root/'framework';case=root/'runs'/a.name
    case.mkdir(exist_ok=False)
    active = root/'runtime/active_launch.json'
    if active.exists():(case/'service_manifest.json').write_bytes(active.read_bytes())
    (case/'transport_source.json').write_text(json.dumps({str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (snap/'tokenflow_client_buffer').glob('*.py')},indent=2))
    config=dict(session=f'tokenflow-{a.modality}-pilot',modality=a.modality,arm=a.arm,capacity_only=a.capacity_only,
        model='/data/models/Qwen3-Omni-30B-A3B-Instruct',max_inflight=32,recovery_samples=1,
        snapshot_path=str(root/'runtime/snapshot.json'),execution_telemetry_path=str(root/'runtime/execution.json'),
        gateway_log=str(case/'gateway.jsonl'),upstream='http://127.0.0.1:18195',capacity_ttl_s=30.,capacity_safety=.85)
    if a.capacity_profile:
        profile=json.loads(Path(a.capacity_profile).read_text())
        if profile['modality']!=a.modality:raise ValueError('capacity profile modality mismatch')
        now=time.monotonic()
        capacity=dict(gamma=profile['raw_capacity']*.85,observed_at=now,valid_until=now+60.,modality=a.modality,
            profile_id=hashlib.sha256(Path(a.capacity_profile).read_bytes()).hexdigest(),source='heldout_calibration')
        (case/'capacity.json').write_text(json.dumps(capacity,indent=2))
        config['capacity_path']=str(case/'capacity.json')
    (case/'gateway_config.json').write_text(json.dumps(config,indent=2))
    base=['docker','exec','-e','PYTHONPATH='+str(snap),'-w',str(snap),a.container,'python','-u']
    wrapper='import os,sys,runpy;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));sys.argv=sys.argv[2:];runpy.run_module("tokenflow_client_buffer.gateway",run_name="__main__")'
    cmd=base+['-c',wrapper,str(case/'gateway.pid'),'gateway','--config',str(case/'gateway_config.json'),'--port','18196']
    # Require actual scheduler/KV/connector drain, not merely HTTP health.
    drain_url='http://127.0.0.1:18195/v1/omni/drain_status?timeout=5'
    for _ in range(60):
        with urllib.request.urlopen(drain_url,timeout=8) as response:
            drain=json.load(response)
        if drain.get('drained'):
            (case/'drain_before.json').write_text(json.dumps(drain,indent=2))
            break
        time.sleep(.5)
    else:raise RuntimeError('model was not drained before case')
    stop_fault = threading.Event()
    fault_thread = None
    def inject_capacity_drop():
        first = None
        while not stop_fault.wait(.025):
            try:
                if first is None:
                    arrivals = [json.loads(x) for x in (case/'gateway.jsonl').read_text().splitlines()
                                if '"event": "arrived"' in x]
                    if arrivals:first = arrivals[0]['monotonic']
                if first is not None and time.monotonic() >= first+a.capacity_drop_after:
                    from .stream import atomic_json
                    now = time.monotonic()
                    raw = json.loads((case/'capacity.json').read_text())
                    raw.update(gamma=a.capacity_drop_gamma, observed_at=now, valid_until=now+60.,
                               source='controlled_capacity_drop')
                    atomic_json(case/'capacity.json',raw)
                    atomic_json(case/'fault_injection.json',dict(injected_at=now,
                        first_arrival=first,estimate=raw,kind='capacity_drop; not performance measurement'))
                    return
            except (OSError, ValueError):
                pass
    with (case/'gateway.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if proc.poll() is not None:raise RuntimeError('gateway failed; see '+str(case/'gateway.log'))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18196/contract',timeout=1) as r:
                        if r.status==200:break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('gateway readiness')
            client=base+['-m','tokenflow_client_buffer.client','--endpoint','http://127.0.0.1:18196',
                '--workload',str(Path(a.workload).resolve()),'--output',str(case/'client')]
            (case/'launch.json').write_text(json.dumps(dict(gateway=cmd,client=client,config=config),indent=2))
            if a.capacity_drop_after is not None:
                fault_thread = threading.Thread(target=inject_capacity_drop,daemon=True)
                fault_thread.start()
            result=subprocess.run(client,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=660)
            (case/'client.log').write_text(result.stdout)
            print(result.stdout,flush=True)
            if (case/'client/results.json').exists():
                analysis=subprocess.run(base+['-m','tokenflow_client_buffer.analyze',str(case/'client')],capture_output=True,text=True)
                print(analysis.stdout,flush=True)
                if analysis.returncode:raise RuntimeError(analysis.stderr)
            if result.returncode:raise RuntimeError('client failed; see '+str(case/'client.log'))
            for _ in range(60):
                with urllib.request.urlopen(drain_url,timeout=8) as response:
                    drain=json.load(response)
                if drain.get('drained'):
                    (case/'drain_after.json').write_text(json.dumps(drain,indent=2))
                    break
                time.sleep(.5)
            else:raise RuntimeError('model not drained after case')
            (case/'complete.json').write_text(json.dumps(dict(completed=True,wall=time.time())))
        finally:
            stop_fault.set()
            if fault_thread:fault_thread.join(timeout=2)
            if (case/'gateway.pid').exists():
                stop='import os,signal;from pathlib import Path;p=int(Path('+repr(str(case/'gateway.pid'))+').read_text());os.kill(p,signal.SIGTERM)'
                subprocess.run(['docker','exec',a.container,'python','-c',stop],capture_output=True,timeout=10)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=10)
            subprocess.run(['docker','exec',a.container,'chown','-R',
                f'{os.getuid()}:{os.getgid()}',str(case)],check=True,timeout=30)


if __name__=='__main__':main()
