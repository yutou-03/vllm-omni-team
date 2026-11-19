"""Run a single mixed case against an already loaded, drained model instance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from .core import buffer_conservativeness
from .generation_metric import configured_trace, trace_offset
from .native_suite import CONTAINER, save


def drain(case,name):
    for _ in range(60):
        with urllib.request.urlopen('http://127.0.0.1:18195/v1/omni/drain_status?timeout=5',timeout=8) as response:
            state=json.load(response)
        if state.get('drained'):
            save(case/name,state);return
        time.sleep(.5)
    raise RuntimeError('model did not drain')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--name',required=True);p.add_argument('--arm',choices=['fcfs','buffer'],required=True)
    p.add_argument('--workload',required=True,type=Path);p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--container',default=CONTAINER)
    p.add_argument('--gateway-framework',type=Path)
    p.add_argument('--service-root',type=Path,help='Existing loaded service runtime; request artifacts remain under root')
    p.add_argument('--capacity-only',action='store_true')
    p.add_argument('--admission-safety',type=float,default=1.0)
    p.add_argument('--pause-releases-admission',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--admission-policy',choices=['global_fifo','resource_fifo'],default='resource_fifo')
    p.add_argument('--ordered-ingress', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--recovery-ranking',choices=['deadline','weighted'],default='deadline')
    p.add_argument('--buffer-source', choices=('server_virtual_output', 'client_consumption_buffer'),
                   default='server_virtual_output', help='Origin of unread output for policy decisions.')
    p.add_argument('--buffer-conservativeness', type=buffer_conservativeness, default=1.0,
                   help='Multiply the client-buffer time reserved for resuming a paused stream (>= 1).')
    a=p.parse_args()
    root=a.root.resolve();case=root/'runs'/a.name;case.mkdir(exist_ok=False)
    service_root=a.service_root.resolve() if a.service_root else root
    service_manifest=json.loads((service_root/'runtime/active_launch.json').read_text())
    save(case/'service_manifest.json',service_manifest)
    execution_limit=int(service_manifest['max_num_seqs'])
    if execution_limit <= 0:raise ValueError('invalid native execution limit')
    config=dict(arm=a.arm,profile=str(a.profile.resolve()),native_execution_limit=execution_limit,
                resource_log=str(case/'resources.jsonl'),capacity_ttl_s=30.,admission_safety=a.admission_safety,admission_policy=a.admission_policy,outputs={})
    experiment=json.loads((root/'experiment.json').read_text())
    ingress_trace = None
    if a.ordered_ingress:
        launch = json.loads((service_root/'runtime/active_launch.json').read_text())
        trace_dirs = [arg.split('=', 1)[1] for arg in launch['command']
                      if arg.startswith('STAGE_QUEUE_TRACE_DIR=')]
        if len(trace_dirs) != 1:
            raise ValueError('ordered ingress requires an unambiguous live STAGE_QUEUE_TRACE_DIR')
        ingress_trace = Path(trace_dirs[0])/'stage_events.jsonl'
        # The event writer opens lazily on the first request. An empty file
        # is valid at cold start; the client still requires real enqueue ACKs.
        ingress_trace.parent.mkdir(parents=True, exist_ok=True)
        # Existing container-owned traces need only be readable, not writable.
        if not ingress_trace.exists():
            ingress_trace.touch(exist_ok=True)
    measurement_timeout=float(experiment.get('measurement_timeout_s',1800.))
    engine_queue=bool(experiment.get('engine_queue_admission',False)) and a.arm=='buffer'
    if engine_queue:
        modalities={json.loads(line)['payload']['omni_scheduling']['request_path'] for line in a.workload.read_text().splitlines() if line.strip()}
        if len(modalities)!=1:
            raise ValueError('engine queue admission requires isolated output until mixed phase resource accounting is validated')
    startup_profile=json.loads(a.profile.read_text())
    for m in ('text','audio'):
        (case/m).mkdir()
        config['outputs'][m]=dict(buffer_source=a.buffer_source,buffer_conservativeness=a.buffer_conservativeness,recovery_ranking=a.recovery_ranking,session='tokenflow-'+m+'-pilot',modality=m,
            model='/data/models/Qwen3-Omni-30B-A3B-Instruct',max_inflight=int(experiment.get("transport_max_inflight",256)),native_execution_limit=execution_limit,capacity_only=a.capacity_only,pause_releases_admission=a.pause_releases_admission,
            engine_queue_admission=engine_queue,
            admission_startup_s=startup_profile.get(m+'_startup_bound_s', .5),
            snapshot_path=str(service_root/'runtime'/('snapshot_'+m+'.json')),
            execution_telemetry_path=str(service_root/'runtime'/('execution_'+m+'.json')),
            ordered_backend_trace=str(ingress_trace) if ingress_trace is not None else None,
            gateway_log=str(case/m/'gateway.jsonl'),upstream='http://127.0.0.1:18195',upstream_timeout_s=measurement_timeout)
    for m, stage in (('text', '0'), ('audio', '1')):
        scheduler_path = service_root/'runtime'/('scheduler_'+m+'_'+str(execution_limit)+'.json')
        if scheduler_path.exists():
            scheduler = json.loads(scheduler_path.read_text())
            scheduler = scheduler.get('by_stage', {}).get(stage, scheduler)
            config['outputs'][m]['admission_resume_s'] = scheduler['resume_to_client_s']
            config['outputs'][m]['admission_margin_s'] = scheduler.get('margin_s', .5)
    # Allow an explicitly documented profile-validity policy, never an implicit
    # per-modality change. Short TTL remains the default.
    config['capacity_ttl_s']=json.loads((root/'experiment.json').read_text()).get('capacity_ttl_s',30.)
    save(case/'gateway_config.json',config)
    snap=a.gateway_framework.resolve() if a.gateway_framework else root/'framework'
    save(case/'gateway_source_manifest.json',{str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (snap/'tokenflow_client_buffer').glob('*.py')})
    save(case/'diagnostic_settings.json',dict(gateway_framework=str(snap),capacity_only=a.capacity_only,admission_safety=a.admission_safety,admission_policy=a.admission_policy,pause_releases_admission=a.pause_releases_admission,engine_queue_admission=engine_queue,service_root=str(service_root)))
    base=['docker','exec','-e','PYTHONPATH='+str(snap),'-w',str(snap),a.container,'python','-u']
    wrapper='import os,sys,runpy;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));sys.argv=sys.argv[2:];runpy.run_module("tokenflow_client_buffer.mixed_gateway",run_name="__main__")'
    command=base+['-c',wrapper,str(case/'gateway.pid'),'mixed_gateway','--config',str(case/'gateway_config.json')]
    drain(case,'drain_before.json')
    with (case/'gateway.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if proc.poll() is not None:raise RuntimeError('mixed gateway exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18196/contract',timeout=1) as reply:
                        if reply.status==200:break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('mixed gateway readiness')
            client=base+['-m','tokenflow_client_buffer.client','--endpoint','http://127.0.0.1:18196',
                         '--workload',str(a.workload.resolve()),'--output',str(case/'client'),'--timeout',str(measurement_timeout)]
            if ingress_trace is not None:
                client += ['--ingress-trace', str(ingress_trace)]
            save(case/'launch.json',dict(gateway=command,client=client))
            generation_trace = configured_trace(service_manifest)
            start_offset = trace_offset(generation_trace)
            with (case/'client.log').open('w') as output:
                subprocess.run(client,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=measurement_timeout+60)
            subprocess.run(['docker','exec',a.container,'chown','-R',f'{os.getuid()}:{os.getgid()}',str(case)],check=True)
            if generation_trace and generation_trace.exists():
                save(case/'generation_trace_window.json', dict(trace=str(generation_trace.parent),
                     start_offset=start_offset, end_offset=trace_offset(generation_trace)))
            rows=json.loads((case/'client/results.json').read_text())
            from .analyze import analyze
            for m in ('text','audio'):
                subset=[r for r in rows if r['modality']==m]
                if not subset:continue
                path=case/m/'client';path.mkdir()
                save(path/'results.json',subset)
                for row in subset:
                    key=hashlib.sha256(row['request_id'].encode()).hexdigest()[:16]
                    (path/key).symlink_to(case/'client'/key,target_is_directory=True)
                analyze(path)
                save(case/m/'service_manifest.json',json.loads((case/'service_manifest.json').read_text()))
            drain(case,'drain_after.json')
            save(case/'complete.json',dict(completed=True,wall=time.time()))
        finally:
            if (case/'gateway.pid').exists():
                stop='import os,signal;from pathlib import Path;os.kill(int(Path('+repr(str(case/'gateway.pid'))+').read_text()),signal.SIGTERM)'
                subprocess.run(['docker','exec',a.container,'python','-c',stop],capture_output=True,timeout=10)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
            subprocess.run(['docker','exec',a.container,'chown','-R',f'{os.getuid()}:{os.getgid()}',str(case)],check=True,timeout=30)

if __name__=='__main__':main()
