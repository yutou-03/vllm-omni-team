"""One process, two client output buffers, one shared resource admission book."""
import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from aiohttp import web
from .core import CapacityEstimator
from .capacity_signals import execution_window_evidence
from .gateway import Gateway
from .phases import PhaseAccounting
from .mixed_core import MixedBook
from .stream import atomic_json, clock_domain


class OutputGateway(Gateway):
    def __init__(self,config,coordinator):
        super().__init__(config)
        self.coordinator=coordinator
        self.w=coordinator.book.views[self.modality]
        self.w.admission_guard = self.admission_safety

    def capacity(self,now):
        return self.coordinator.book.capacity(self.modality,now)

    def update_capacity(self,now):
        pass  # measured once centrally in resource units

    def publish(self):
        now=time.monotonic();est=self.capacity(now)
        self.refresh_virtual_output(now)
        state=self.w.snapshot(est,now,native=self.config['arm']=='fcfs')
        book=self.coordinator.book
        state.update(schema_version=1,session=self.session,modality=self.modality,
            buffer_source=self.buffer_source, buffer_conservativeness=self.buffer_conservativeness,
            recovery_ranking=self.config.get('recovery_ranking', 'deadline'),
            clock_domain=self.clock_domain,written_at=now,capacity_valid_until=est.valid_until if est else 0.,
            feedback={rid:asdict(f) for rid,f in self.feedback.items() if rid in state['members'] or rid in state['paused_members']},
            startup_members={r.request_id:dict(rate=r.rate) for v in book.views.values()
                             for r in v.requests.values() if r.state != 'finished'},
            # Cross-modal first-output status can identify a beneficiary, but
            # only this modality's client feedback above supplies pause credit.
            startup_feedback={rid:dict(started=f.started,received=f.received) for g in self.coordinator.gateways.values()
                              for rid,f in g.feedback.items()},
            **book.snapshot(now), **self.pause_snapshot(now, mixed=True),
            **self.engine_queue_snapshot())
        # Keep the local mode/reason from the ledger; global_resource_reason is
        # separate because a feasible global book must not erase 'baseline'.
        state['reason']='baseline' if self.config['arm']=='fcfs' else self.w.reason
        queued = [r.request_id for r in self.w.requests.values()
                  if r.state == 'native_queued']
        if self.config['arm'] != 'fcfs' and book.native_fcfs:
            state['mode'] = 'fcfs'
            state['reason'] = book.native_fcfs_reason
        if self.config.get('capacity_only') and self.config['arm']!='fcfs':
            state['mode'],state['reason']='fcfs','capacity_only'
            if queued:
                state['native_fcfs_admission'] = True
        # Queue ownership changes with the final policy; physical FIFO entries
        # and execution identities do not. Never resubmit or bulk-admit.
        state['native_fcfs_ids'] = queued if state['mode'] == 'fcfs' else []
        state['buffer_reclaimed_ids'] = queued if state['mode'] == 'buffer' else []
        atomic_json(self.snapshot,state)
        status=(state['mode'],state['reason'],state['demand'],state['gamma'],state['waiting'])
        if status!=self.last_status or now-self.last_publish>1.:
            self.event('working_set',**{k:v for k,v in state.items()
                if k not in ('feedback','startup_members','startup_feedback')})
            self.last_status,self.last_publish=status,now


class MixedGateway(PhaseAccounting):
    def __init__(self,config):
        self.config=config;self.stopping=False
        self.init_phase_accounting()
        profile=json.loads(Path(config['profile']).read_text())
        self.profile=profile
        now=time.monotonic()
        self.estimators={k:CapacityEstimator('text' if k=='thinker' else 'audio',
            safety=config.get('admission_safety',1.0),ttl_s=config.get('capacity_ttl_s',30.),min_window_s=2. if k=='thinker' else 8.) for k in ('thinker','audio')}
        estimates={k:self.estimators[k].seed(profile[k]['raw_capacity'],now,profile[k]['profile_id'])
                   for k in self.estimators}
        # Independent calibration seed stays valid through setup/initial chunk.
        from .core import CapacityEstimate
        estimates={k:CapacityEstimate(e.gamma,e.observed_at,now+max(60.,config.get('capacity_ttl_s',30.)),e.modality,e.profile_id,e.source,e.admission_gamma)
                   for k,e in estimates.items()}
        for k,e in estimates.items():self.estimators[k].estimate=e
        self.book=MixedBook(estimates['thinker'],estimates['audio'],
            profile['audio_thinker_tokens_per_media_s'],
            text_thinker_tokens_per_visible_token=profile['text_thinker_tokens_per_visible_token'],
            native_limit=config['native_execution_limit'],admission_policy=config.get('admission_policy','resource_fifo'),
            max_inflight=max(v.get('max_inflight',256) for v in config['outputs'].values()))
        self.native_execution_offsets = {}
        self.native_execution_inodes = {}
        self.gateways={m:OutputGateway(dict(config['outputs'][m],arm=config['arm']),self)
                       for m in ('text','audio')}
        self.log=Path(config['resource_log']).open('a',buffering=1)
        self.windows={k:dict(start=now,previous=None,units=None,last_check=now) for k in self.estimators}

    def event(self,name,**values):
        self.log.write(json.dumps(dict(event=name,monotonic=time.monotonic(),**values))+'\n')

    async def contract(self,request):
        return web.json_response(dict(modality='mixed',clock_domain=clock_domain(),
            contracts={m:dict(session=g.session,modality=m,clock_domain=clock_domain(),arm=self.config['arm'],buffer_source=g.buffer_source)
                       for m,g in self.gateways.items()}))

    def process_native_execution(self):
        # Append-only events preserve first executions between telemetry samples.
        # Requests not yet executed remain native_queued, never pretend to own KV.
        offsets = self.native_execution_offsets
        for modality, gateway in self.gateways.items():
            path = Path(gateway.config['execution_telemetry_path'] + '.admissions.jsonl')
            try:
                with path.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    if (self.native_execution_inodes.get(modality) != identity
                            or stat.st_size < offsets.get(modality, 0)):
                        self.native_execution_inodes[modality] = identity
                        offsets[modality] = 0
                    stream.seek(offsets.get(modality, 0))
                    for _ in range(4096):
                        position = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            stream.seek(position)
                            break
                        try:
                            event = json.loads(line)
                            rid = event['request_id']
                            started = float(event['started_at'])
                            registered = gateway.registered_at.get(rid)
                            if (event.get('session') != gateway.session
                                    or registered is None or not math.isfinite(started)
                                    or not registered <= started <= time.monotonic()):
                                continue
                            if self.book.mark_native_execution(modality, rid):
                                gateway.event('native_execution_admitted',
                                    request_id=rid, execution_started_at=started)
                        except (ValueError, KeyError, TypeError):
                            continue
                    offsets[modality] = stream.tell()
            except OSError:
                continue

    def admission_recovery_pending(self, now):
        # No await: restore/release both ledgers before exposing any shared
        # capacity to either output gateway. A feasible W after a safe swap
        # does not imply that the outstanding restoration ACK has completed.
        self.process_phase_events(now)
        self.process_native_execution()
        for gateway in self.gateways.values():
            gateway.process_pause_leases(now)
        return any(gateway.recovery_pending for gateway in self.gateways.values())

    def observe(self,key,now):
        w=self.windows[key];estimator=self.estimators[key]
        if now-w['start']<estimator.min_window_s or now-w['last_check']<2.:return
        w['last_check']=now
        modality='text' if key=='thinker' else 'audio';g=self.gateways[modality]
        try:
            t=json.loads(Path(g.config['execution_telemetry_path']).read_text())
            total=t['resource_decode_tokens'] if key=='thinker' else self.gateways['audio'].window_units
            prev=w['previous']
            if prev is None or w['units'] is None:
                w.update(start=now,previous=t,units=total);return
            steps=t['steps']-prev['steps'];full=t['full_steps']-prev['full_steps']
            if steps<=0 or not 0<=now-t['written_at']<=.5 or t['session']!=g.session:return
            units=total-w['units'];duration=now-w['start']
            need=self.book.demand[key]
            pending=self.book.pending
            capacity_limited=False
            if pending and need>0:
                first=pending[0]
                route=next(m for m,v in self.book.views.items() if first.request_id in v.requests)
                cost=self.book.costs(route, first)[key]
                capacity_limited=cost>0 and need+cost>estimator.estimate.admission_bound
            # Runnable backend pressure plus real hardware/token-budget work
            # certifies downward updates. Neither a gateway queue nor 32
            # resident/executed identities establishes hardware saturation.
            evidence=execution_window_evidence(t,prev,duration)
            saturated=evidence['saturated']
            buffer_steps=t['buffer_steps']-prev['buffer_steps']
            policy='buffer' if buffer_steps==steps else 'fcfs' if buffer_steps==0 else 'mixed'
            before=estimator.estimate
            estimator.observe(generated_units=units,duration_s=duration,now=now,
                saturated=saturated,profile_id=before.profile_id,policy=policy)
            self.book.estimates[key]=estimator.estimate
            self.event('resource_capacity_window',resource=key,units=units,duration_s=duration,
                policy=policy,saturated=saturated,full_steps=full,steps=steps,
                capacity_limited=capacity_limited,
                **{k:v for k,v in evidence.items() if k!='saturated'},
                observed_output_per_s=units/duration,
                deferred_empty_window=units==0,
                updated=before is not estimator.estimate,
                capacity_scope='current_delivered_service; no peak extrapolation; policy costs remain in wall time',
                estimate=asdict(estimator.estimate),**self.book.snapshot(now))
            # Retain silent elapsed time even below max_num_seqs. Otherwise
            # a later media chunk is incorrectly divided only by its final gap.
            if units!=0:w.update(start=now,previous=t,units=total)
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.event('resource_observation_error',resource=key,error=repr(error))

    async def loop(self):
        while not self.stopping:
            now=time.monotonic()
            for key in self.estimators:self.observe(key,now)
            await asyncio.sleep(.025)

    async def startup(self,app):
        for g in self.gateways.values():await g.startup(app)
        self.task=asyncio.create_task(self.loop())

    async def cleanup(self,app):
        self.stopping=True
        await self.task
        for g in self.gateways.values():await g.cleanup(app)
        self.log.close()

    def app(self):
        app=web.Application(client_max_size=16*1024*1024)
        app.router.add_get('/contract',self.contract)
        for m,g in self.gateways.items():
            app.router.add_get('/'+m+'/contract',g.contract)
            app.router.add_post('/'+m+'/feedback',g.receive_feedback)
            app.router.add_post('/'+m+'/v1/chat/completions',g.stream)
        app.on_startup.append(self.startup);app.on_cleanup.append(self.cleanup)
        return app


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--port',type=int,default=18196)
    a=p.parse_args();g=MixedGateway(json.loads(Path(a.config).read_text()))
    web.run_app(g.app(),host='127.0.0.1',port=a.port,print=None)

if __name__=='__main__':main()
