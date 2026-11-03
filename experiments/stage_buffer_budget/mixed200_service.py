"""Service lifecycle inside one explicitly owned, auto-removed worker container."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

class Service:
    def __init__(self, root, arm, name='stage-buffer-mixed200'):
        self.root, self.arm, self.name = Path(root), arm, name
        self.directory = self.root / (arm + '_service')
        self.directory.mkdir(exist_ok=False)
        (self.directory / 'traces').mkdir()
        self.pid_file = self.directory / 'server.pid'
        self.process = self.log = None

    def start(self):
        r = self.root
        snap = r / 'worker/framework'
        env = {'PYTHONPATH':str(snap), 'STAGE_STUDY_FRAMEWORK':str(snap),
            'PYTHONDONTWRITEBYTECODE':'1', 'VLLM_BATCH_INVARIANT':'1',
            'VLLM_OMNI_BASELINE_POLICY':'native_fcfs', 'VLLM_OMNI_STAGE_STUDY_TRACE':'1',
            'STAGE_QUEUE_TRACE_MODE':'stage_study', 'STAGE_QUEUE_TRACE_DIR':str(self.directory/'traces'),
            'STAGE_QUEUE_RUN_ID':self.arm, 'VLLM_OMNI_SMALL_HOLD_EXECUTION_CAP':'16',
            'VLLM_OMNI_SMALL_HOLD_STAGES':'0,1', 'VLLM_OMNI_SLO_ACTIVE_PREEMPTION':'0',
            'TORCHINDUCTOR_CACHE_DIR':str(r/'runtime_cache/inductor'),
            'TRITON_CACHE_DIR':str(r/'runtime_cache/triton')}
        if self.arm.startswith('ours'):
            env.update(VLLM_OMNI_TEXT_DELIVERY_CONFIG=str(r/'text_delivery.json'),
                VLLM_OMNI_RECEIVER_BUFFER_CONFIG=str(r/'receiver.json'),
                VLLM_OMNI_SMALL_HOLD_PROBE='1', VLLM_OMNI_URGENT_KEEP_ADMISSION='1',
                VLLM_OMNI_STAGE_BUFFER_POLICY='1',
                VLLM_OMNI_SMALL_HOLD_CONTROLLER='stage_buffer_controller_budget:decide',
                VLLM_OMNI_STAGE_BUFFER_CONFIG=str(r/'dynamic_controller.json'))
        command = ['docker','exec']
        for k,v in env.items(): command += ['-e',k+'='+v]
        wrapper = "import os,sys,runpy; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        command += [self.name,'python','-u','-c',wrapper,str(self.pid_file),str(snap/'experiments/stage_buffer_budget/service_entry.py'),
            'serve','/data/models/Qwen3-Omni-30B-A3B-Instruct','--omni','--host','0.0.0.0','--port','8095',
            '--deploy-config',str(r/'worker/deploy.yaml')]
        (self.directory/'launch.json').write_text(json.dumps({'environment':env,'command':command},indent=2))
        self.log = (self.directory/'server.log').open('w')
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(900):
            if self.process.poll() is not None: raise RuntimeError('server exited: '+str(self.directory))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8095/health',timeout=2) as response:
                    if response.status == 200: return self
            except Exception: time.sleep(1)
        raise TimeoutError('startup: '+str(self.directory))

    def stop(self):
        if self.process is None: return
        # The dedicated worker contains no unrelated jobs. Reparented GPU
        # workers can outlive the API process and disappear from its child tree.
        cleanup = """import os,psutil
me=psutil.Process(os.getpid())
protected={1,me.pid}|{p.pid for p in me.parents()}
targets=[p for p in psutil.process_iter() if p.pid not in protected]
for p in targets:
 try:p.terminate()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(targets,timeout=10)
for p in alive:
 try:p.kill()
 except psutil.NoSuchProcess:pass
psutil.wait_procs(alive,timeout=10)
remaining=[p.pid for p in psutil.process_iter() if p.pid not in protected and p.status()!=psutil.STATUS_ZOMBIE]
if remaining:raise RuntimeError('worker processes remain: '+str(remaining))
"""
        result=subprocess.run(['docker','exec',self.name,'python','-c',cleanup],capture_output=True,text=True,timeout=35)
        if result.returncode:raise RuntimeError('owned worker cleanup failed: '+result.stderr)
        self.process.wait(timeout=45)
        self.log.close()
        self.process = None
