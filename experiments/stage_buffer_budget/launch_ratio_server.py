"""Start an isolated 4:x ratio service after verifying every requested GPU is idle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/zhongyu/research_reports/servegen_active_buffer_20261012')
CONFIG_ROOT = Path(__file__).resolve().parent
FRAMEWORK = Path('/home/zhongyu/project/vllm-omni-servegen-active-20261012')
MODEL = '/data/models/Qwen3-Omni-30B-A3B-Instruct'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--framework', default=str(FRAMEWORK))
    p.add_argument('--deploy', default=str(CONFIG_ROOT/'deploy_ratio_4_1.yaml'))
    p.add_argument('--extra-env-file')
    p.add_argument('--port', type=int, default=8095)
    p.add_argument('--devices', default='0,1,2,3,4,5')
    p.add_argument('--trace', choices=['on','off'], default='on')
    a=p.parse_args()
    if not a.session.replace('_','').isalnum(): raise ValueError('bad session')
    framework=Path(a.framework).resolve()
    deploy=Path(a.deploy).resolve()
    if not framework.is_dir() or not deploy.is_file(): raise ValueError('missing framework or deploy config')
    devices=[int(x) for x in a.devices.split(',')]
    if len(devices) not in (6, 7, 8) or len(set(devices)) != len(devices):
        raise ValueError('six to eight distinct devices required')
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for row in info.splitlines():
        idx,mem,util=[int(x.strip()) for x in row.split(',')]
        if idx in devices and (mem>100 or util>5): raise RuntimeError(f'GPU {idx} already occupied: {mem} MiB, {util}%')
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('127.0.0.1',a.port))
    run=ROOT/'sessions'/a.session
    run.mkdir(parents=True,exist_ok=False)
    (run/'traces').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('VLLM_OMNI_','STAGE_QUEUE_','SMALL_'))}
    runtime=Path('/home/zhongyu/research_reports/stage_study_20261002')/'.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib'
    env.update(CUDA_VISIBLE_DEVICES=a.devices, PYTHONPATH=str(framework)+':'+str(ROOT),
        PYTHONDONTWRITEBYTECODE='1', VLLM_BATCH_INVARIANT='1',
        LD_LIBRARY_PATH=str(runtime)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
        VLLM_OMNI_BASELINE_POLICY='native_fcfs',
        VLLM_OMNI_STAGE_STUDY_TRACE='1' if a.trace=='on' else '0',
        VLLM_OMNI_STAGE_STUDY_SAMPLE_MS='50',
        STAGE_QUEUE_TRACE_MODE='stage_study' if a.trace=='on' else 'off',
        STAGE_QUEUE_TRACE_DIR=str(run/'traces'), STAGE_QUEUE_RUN_ID=a.session,
        STAGE_STUDY_FRAMEWORK=str(framework))
    if a.extra_env_file:
        extra=json.loads(Path(a.extra_env_file).read_text())
        if not isinstance(extra,dict) or not all(isinstance(v,str) for v in extra.values()): raise ValueError('invalid extra env')
        if any(not k.startswith(('VLLM_OMNI_', 'STAGE_QUEUE_', 'SMALL_')) for k in extra): raise ValueError('unexpected experiment env key')
        env.update(extra)
    command=[str(Path('/home/zhongyu/research_reports/stage_study_20261002')/'.venv/bin/python'),str(ROOT/'server_entry.py'),'serve',MODEL,'--omni',
             '--host','127.0.0.1','--port',str(a.port),'--deploy-config',str(deploy)]
    files={str(p.relative_to(framework)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in framework.rglob('*.py') if '__pycache__' not in p.parts}
    (run/'manifest.json').write_text(json.dumps({'args':vars(a),'command':command,'framework':str(framework),
        'source_hashes':files,'deploy_sha256':hashlib.sha256(deploy.read_bytes()).hexdigest(),
        'experiment_env':{k:v for k,v in env.items() if k.startswith(('VLLM_OMNI_', 'STAGE_QUEUE_', 'SMALL_'))},'deploy_path':str(deploy),
        'model':MODEL,'gpu_preflight':info,'created_wall_s':time.time()},indent=2))
    with (run/'server.log').open('xb') as log:
        server=subprocess.Popen(command,cwd=framework,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state={'session':a.session,'pid':server.pid,'pgid':server.pid,'port':a.port,'devices':devices,
           'framework':str(framework),'log':str(run/'server.log'),'started_wall_s':time.time(),'trace':a.trace}
    (run/'service.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)

if __name__=='__main__': main()
