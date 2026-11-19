"""Matched isolated/mixed comparisons on one six-GPU model instance."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from .native_suite import Suite,prepare as prepare_native,save,CONTAINER


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,prior,repo):
    prepare_native(root,Path('/home/zhongyu/research_reports/tokenflow_client_20261106_4h'),repo)
    configs={}
    for stage,m in ((0,'text'),(1,'audio')):
        c=json.loads((root/'runtime'/f'scheduler_{m}_32.json').read_text())
        c.update(snapshot_path=str(root/'runtime'/f'snapshot_{m}.json'),
            execution_telemetry_path=str(root/'runtime'/f'execution_{m}.json'),resource_telemetry=True,
            capacity_device_ids=[0,1,2,3] if stage==0 else [4,5],
            capacity_gpu_aggregation='mean' if stage==0 else 'max')
        configs[str(stage)]=c
    save(root/'runtime/scheduler_audio_32.json',dict(by_stage=configs))
    workloads={}
    for rep in range(2):
        originals={m:[json.loads(x) for x in (root/'workloads'/f'{m}_{rep}.jsonl').read_text().splitlines()]
                   for m in ('text','audio')}
        for ntext in (150,100,50):
            rows=copy.deepcopy(originals['text'][:ntext]+originals['audio'][:200-ntext])
            random.Random(32100+ntext+rep).shuffle(rows)
            name=f'mixed_t{ntext}_{rep}'
            for i,row in enumerate(rows):
                row['arrival_s']=i*.005 if i<50 else 1.5+(i-50)*.005
                row['payload']['omni_scheduling'].update(source_request_id=f'{name}-{i}',ingress_order=i)
            path=root/'workloads'/(name+'.jsonl')
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows));workloads[path.name]=digest(path)
    if (prior/'workloads/resource_profile.json').exists():
        profile=json.loads((prior/'workloads/resource_profile.json').read_text())
        profile.update(provenance=str(prior),bootstrap_only=True)
    else:
        text=prior/'workloads/text_32_capacity.json';audio=prior/'workloads/audio_32_capacity.json'
        audio_rows=json.loads((prior/'runs/audio_32_calibration/client/results.json').read_text())
        rho=max(r['workload']['payload']['max_tokens']/r['received_units'] for r in audio_rows)
        profile=dict(thinker=dict(raw_capacity=json.loads(text.read_text())['raw_capacity'],profile_id=digest(text)),
            audio=dict(raw_capacity=json.loads(audio.read_text())['raw_capacity'],profile_id=digest(audio)),
            audio_thinker_tokens_per_media_s=rho,text_thinker_tokens_per_visible_token=1.,
            coefficient_basis='maximum configured Thinker output-token bound / actual media duration in independent native audio calibration; not buffer inventory',
            provenance=str(prior),bootstrap_only=True)
    save(root/'workloads/resource_profile.json',profile)
    e=json.loads((root/'experiment.json').read_text())
    e.update(max_num_seqs=[32],repetitions=2,arms=['fcfs','buffer'],total_formal_cases=20,
        isolated_formal_cases=8,mixed_formal_cases=12,mixed_text_counts=[150,100,50],
        detailed_stage_trace=0,capacity_ttl_s=900.,prior=str(prior),
        admission_policy='resource_fifo',admission_safety=1.0,pause_releases_admission=True,
        comparison='Same instance and code for isolated and mixed workloads; both arms retain client, iteration, pause/resume and ingress traces. Detailed tensor/KV snapshots disabled equally.',
        capacity_prior_scope='Calibrated prior valid for this bounded case on unchanged isolated hardware; saturated observations update it. Longer validity avoids treating an underloaded non-bottleneck stage as failed merely for being idle. Log actual observation age; this is not a hardware-failure detector.',
        scheduling_scope='Text client credit pauses Thinker text requests; audio client credit pauses Talker audio requests. Both compete for the same Thinker execution slots. Shared resource admission uses a vector budget; no summed token/media units.')
    e['workload_sha256'].update(workloads);save(root/'experiment.json',e)
    save(root/'progress.json',dict(status='prepared',completed_formal_cases=0,planned_formal_cases=20))


class MixedSuite(Suite):
    def progress(self,status,**extra):
        save(self.root/'progress.json',dict(status=status,updated_at=time.time(),controller_pid=os.getpid(),
            cell=self.cell,completed_formal_cases=len(self.completed),completed_cases=self.completed,
            planned_formal_cases=20,**extra))

    def case(self,name,arm,workload,trace,formal=False):
        self.progress('running_case',case=name)
        command=[sys.executable,'-m','tokenflow_client_buffer.run_mixed_case','--root',str(self.root),
            '--name',name,'--arm',arm,'--workload',str(workload),'--profile',str(self.root/'workloads/resource_profile.json')]
        with (self.root/'runtime'/(name+'.runner.log')).open('w') as log:
            subprocess.run(command,cwd=self.root/'framework',stdout=log,stderr=subprocess.STDOUT,check=True)
        case=self.root/'runs'/name
        from .verify_run import verify
        rows=json.loads((case/'client/results.json').read_text())
        for modality in ('text','audio'):
            if not any(r['modality']==modality for r in rows):continue
            evidence=verify(case/modality,trace)
            if evidence['errors'] or evidence['unresolved_pauses']:
                raise RuntimeError('mixed mechanism error: '+name+'/'+modality)
        if any(r['first_ready_s']!=r['startup_s'] for r in rows):raise RuntimeError('startup gate detected')
        if formal:self.completed.append(name)
        self.progress('case_complete',case=name)
        return case

    @staticmethod
    def hashes(case):
        return {r['request_id']:r['content_sha256'] for r in json.loads((case/'client/results.json').read_text())}

    def calibrate(self,modality,trace):
        workload=self.small_workload(modality,32,'resource_calibration',64)
        before={}
        telemetry=self.root/'runtime/execution_text.json'
        if telemetry.exists():before=json.loads(telemetry.read_text())
        case=self.case(modality+'_resource_calibration','fcfs',workload,trace)
        after=json.loads(telemetry.read_text())
        save(case/'resource_counter_delta.json',dict(before=before,after=after))
        stats=json.loads((case/modality/'client/summary.json').read_text())
        rows=json.loads((case/'client/results.json').read_text())
        path=self.root/'workloads/resource_profile.json';profile=json.loads(path.read_text())
        first_batch=sorted(rows,key=lambda r:r['scheduled_at'])[:32]
        profile[modality+'_startup_bound_s']=max(r['first_ready_s'] for r in first_batch)+.25
        profile[modality+'_startup_bound_basis']='Maximum first-ready latency of first native batch in independent calibration, plus 0.25s; an empirical bound, not a hard guarantee.'
        if modality=='text':
            units=after['resource_decode_tokens']-before.get('resource_decode_tokens',0)
            raw=min(stats['generated_units_per_s'],units/stats['generated_window_s'])
            profile['thinker']=dict(raw_capacity=raw,profile_id=digest(case/'resource_counter_delta.json'),source_case=str(case),
                resource_decode_units=units,window_s=stats['generated_window_s'])
        else:
            profile['audio']=dict(raw_capacity=stats['generated_units_per_s'],profile_id=digest(case/modality/'client/summary.json'),source_case=str(case))
            profile['audio_thinker_tokens_per_media_s']=max(r['workload']['payload']['max_tokens']/r['received_units'] for r in rows)
            profile['bootstrap_only']=False
        save(path,profile)

    def run_matched(self):
        prior=Path(json.loads((self.root/'experiment.json').read_text())['prior'])
        while True:
            status=json.loads((prior/'progress.json').read_text())
            if status['status'] in ('complete','diagnostic_complete_superseded'):break
            if status['status']=='stopped_with_error':
                handoff=prior/'runtime/superseded.json'
                transition=json.loads(handoff.read_text()) if handoff.exists() else {}
                expected=(transition.get('next_root')==str(self.root)
                          and status.get('error','').startswith('KeyboardInterrupt'))
                if not expected:raise RuntimeError('prior run failed; inspect before next experiment')
            self.progress('waiting_for_single_modality_run',prior_progress=status.get('completed_formal_cases',0))
            time.sleep(30)
        manifest=json.loads((self.root/'runtime/source_manifest.json').read_text())
        for name,sha in manifest.items():
            if digest(self.root/'framework'/name)!=sha:raise RuntimeError('frozen source changed: '+name)
        try:
            self.cell='matched_single_instance';trace=self.start('audio',32,True)
            for m in ('text','audio'):
                warm=self.small_workload(m,32,'warmup',4)
                self.case(m+'_shared_warmup','fcfs',warm,trace)
                self.calibrate(m,trace)
            # A mixed loopback/small GPU pair catches routing/content issues
            # before the formal pressure matrix.
            source=self.root/'workloads/mixed_t100_0.jsonl'
            rows=[json.loads(x) for x in source.read_text().splitlines()][:32]
            for i,row in enumerate(rows):
                row['arrival_s']=i*.005;row['payload']['omni_scheduling']['source_request_id']=f'mixed-smoke-{i}'
            smoke=self.root/'workloads/mixed_smoke.jsonl';smoke.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            native=self.case('mixed_smoke_fcfs','fcfs',smoke,trace)
            buffered=self.case('mixed_smoke_buffer','buffer',smoke,trace)
            if self.hashes(native)!=self.hashes(buffered):raise RuntimeError('mixed smoke output mismatch')
            for kind in ('text','audio','mixed_t150','mixed_t100','mixed_t50'):
                self.cell=kind
                for rep in range(2):
                    arms=['fcfs','buffer'];random.Random(6470+rep).shuffle(arms);paired={}
                    for arm in arms:
                        name=f'{kind}_formal_{rep}_{arm}'
                        paired[arm]=self.case(name,arm,self.root/'workloads'/f'{kind}_{rep}.jsonl',trace,True)
                    if self.hashes(paired['fcfs'])!=self.hashes(paired['buffer']):raise RuntimeError('formal output mismatch '+kind)
            self.progress('complete')
        except BaseException as error:
            self.progress('stopped_with_error',error=repr(error));raise
        finally:self.stop()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--prior',type=Path)
    p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root.resolve(),a.prior.resolve(),Path(__file__).resolve().parents[1]);return
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate);MixedSuite(a.root.resolve()).run_matched()

if __name__=='__main__':main()
