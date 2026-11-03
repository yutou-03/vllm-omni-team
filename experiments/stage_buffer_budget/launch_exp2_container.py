"""Freeze this worktree and launch one isolated experiment service on idle GPUs."""
import argparse, hashlib, json, os, shutil, socket, subprocess, time
from pathlib import Path

IMAGE='exp1/vllm-omni-stage01:20261028-6840e2c9-pandas'
MODEL='/data/models/Qwen3-Omni-30B-A3B-Instruct'
REPO=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True)
 p.add_argument('--env',type=Path);p.add_argument('--native-cap',type=int);p.add_argument('--source',type=Path);p.add_argument('--gpus',default='0,1,2,3,4,5');p.add_argument('--entrypoint-script',type=Path,default=Path('/home/zhongyu/research_reports/servegen_active_buffer_20261012/server_entry.py'))
 p.add_argument('--worker-only',action='store_true',help='one reusable container; services run via docker exec')
 a=p.parse_args(); selected=[int(x) for x in a.gpus.split(',')]
 if len(selected)!=6 or len(set(selected))!=6:raise ValueError('exactly six distinct GPUs required')
 a.run.mkdir(parents=True,exist_ok=False);(a.run/'traces').mkdir()
 info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 for line in info.splitlines():
  idx,mem,util=[int(x.strip()) for x in line.split(',')]
  if idx in selected and (mem>100 or util>10):raise RuntimeError(f'GPU {idx} occupied: {mem} MiB {util}%')
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',8095))
 source=a.source or REPO;snap=a.run/'framework'
 if a.source:
  shutil.copytree(source,snap)
 else:
  names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=REPO).decode().split('\0')
  for name in names:
   src=REPO/name
   if name and src.is_file():
    dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 deploy=a.run/'deploy.yaml'
 base=snap/'experiments/stage_buffer_budget/deploy_ratio_4_1.yaml'
 # Resolve base_config relative to this run, and keep the reference stage mapping.
 data=base.read_text().replace('base_config: ../../vllm_omni/deploy/qwen3_omni_moe.yaml',f'base_config: {snap}/vllm_omni/deploy/qwen3_omni_moe.yaml')
 if a.native_cap is not None:data=data.replace('  - stage_id: 0\n',f'  - stage_id: 0\n    max_num_seqs: {a.native_cap}\n',1)
 deploy.write_text(data)
 env={'PYTHONPATH':str(snap),'STAGE_STUDY_FRAMEWORK':str(snap),'PYTHONDONTWRITEBYTECODE':'1','VLLM_BATCH_INVARIANT':'1',
 'VLLM_OMNI_BASELINE_POLICY':'native_fcfs','VLLM_OMNI_STAGE_STUDY_TRACE':'1','STAGE_QUEUE_TRACE_MODE':'stage_study',
 'STAGE_QUEUE_TRACE_DIR':str(a.run/'traces'),'STAGE_QUEUE_RUN_ID':a.name}
 if a.env:env.update(json.loads(a.env.read_text()))
 command=['docker','run','--rm','-d','--name',a.name,'--gpus','"device='+a.gpus+'"','--shm-size','8g','-p','127.0.0.1:8095:8095',
 '-v','/home/zhongyu:/home/zhongyu','-v','/data/models:/data/models:ro','-v',f'{snap}:{snap}:ro']
 cache=a.run.parent/'runtime_cache'
 if cache.exists():command+=['-v',f'{cache}:/root/.cache']
 for k,v in env.items():command+=['-e',f'{k}={v}']
 # Persist stdout/stderr (including child processes) before Docker auto-removes the container.
 service_wrapper="import os,runpy,sys; fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600); os.dup2(fd,1); os.dup2(fd,2); os.close(fd); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
 command+=['--entrypoint','python',IMAGE,'-u','-c',service_wrapper,str(a.run/'server.log'),str(a.entrypoint_script),'serve',MODEL,'--omni','--host','0.0.0.0','--port','8095','--deploy-config',str(deploy)]
 if a.worker_only:
  i=command.index('--entrypoint');command=command[:i]+['--entrypoint','sleep',IMAGE,'infinity']
 manifest={'command':command,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
 'image_id':subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip(),'environment':env,
 'source_sha256':{str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*.py')},
 'deploy_sha256':hashlib.sha256(deploy.read_bytes()).hexdigest(),'entrypoint_script':str(a.entrypoint_script),'entrypoint_sha256':hashlib.sha256(a.entrypoint_script.read_bytes()).hexdigest(),'physical_gpu_mapping':selected,'gpu_preflight':info,'started_wall_s':time.time(),
 'referenced_configs_sha256':{v:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in env.items() if (k.endswith('_CONFIG') or k.endswith('_FILE')) and v and Path(v).is_file()}}
 (a.run/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(subprocess.check_output(command,text=True).strip(),flush=True)

if __name__=='__main__':main()
