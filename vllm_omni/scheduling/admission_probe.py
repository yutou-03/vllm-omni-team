"""Bounded, auditable admission mechanism experiment (disabled by default)."""
from __future__ import annotations
import hashlib
import heapq
import json
import os
from pathlib import Path
import time

from vllm_omni.scheduling.policy import request_scheduling_metadata
from vllm_omni.scheduling.request_queue import PolicyOrderedRequestQueue


def choose(candidates, releases, *, now, mode):
    """One-step lookahead over visible requests, with common milestone costs.

    All running slots and all free slots appear in releases. No future arrival
    or observed test completion time is available to the decision function.
    """
    def urgency(c):
        return (min(c['deadlines'][m]-v for m,v in c['costs'].items()), c['order'], c['id'])
    ordered=sorted(candidates,key=urgency)
    if mode=='shortest':
        return min(ordered,key=lambda c:(c['occupancy'],urgency(c)))['id'], []
    if mode=='milestone':
        return ordered[0]['id'], []
    evaluations=[]
    for first in ordered[:8]:
        slots=list(releases);heapq.heapify(slots)
        forecast=[]
        for c in [first]+[c for c in ordered if c is not first]:
            start=max(now,heapq.heappop(slots))
            missed=[m for m,v in c['costs'].items() if start+v>c['deadlines'][m]]
            occupancy=c['costs']['ttft'] if mode=='no_occupancy' else c['occupancy']
            heapq.heappush(slots,start+occupancy)
            forecast.append({'id':c['id'],'start':start,'missed':missed,'release':start+occupancy})
        evaluations.append({'first':first['id'],'missed_requests':sum(bool(x['missed']) for x in forecast),
                            'forecast':forecast})
    best=min(evaluations,key=lambda e:e['missed_requests'])
    return best['first'], evaluations


class AdmissionProbe:
    def __init__(self, scheduler, artifact, mode):
        if mode not in {'milestone','occupancy','no_occupancy','shortest'}:
            raise ValueError('unknown admission probe mode')
        self.scheduler=scheduler
        self.model=artifact
        self.mode=mode
        self.preferred=None
        scheduler.waiting=PolicyOrderedRequestQueue(key=self.key)
        scheduler.skipped_waiting=PolicyOrderedRequestQueue(key=self.key)

    @classmethod
    def from_environment(cls,scheduler):
        mode=os.environ.get('VLLM_OMNI_ADMISSION_PROBE_MODE')
        if not mode or int(scheduler._omni_stage_id_for_trace())!=0:
            return None
        path=Path(os.environ['VLLM_OMNI_ADMISSION_PROBE_MODEL'])
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=os.environ['VLLM_OMNI_ADMISSION_PROBE_SHA256']:
            raise ValueError('admission cost artifact hash mismatch')
        if scheduler.waiting or scheduler.skipped_waiting or scheduler._path_rr is not None:
            raise ValueError('admission probe requires empty queues and no RR')
        return cls(scheduler,json.loads(payload),mode)

    def info(self,r):
        meta=request_scheduling_metadata(r)
        if meta.get('sched_schema_version')!=2:
            return None  # Shared warmup stays in native arrival order.
        path=meta['sched_request_path']
        cost=self.model['profiles'][f'{path}:{r.max_tokens}']
        return {'id':str(r.request_id),'order':meta['sched_ingress_order'],
                'deadlines':meta['sched_milestone_deadlines_monotonic_s'],
                'costs':cost['milestones_s'],'occupancy':cost['occupancy_s']}

    def key(self,r):
        c=self.info(r)
        if c is None:
            return (str(r.request_id)!=self.preferred,0,str(r.request_id))
        return (c['id']!=self.preferred,
                min(c['deadlines'][m]-v for m,v in c['costs'].items()),c['order'],c['id'])

    def select(self,emit):
        s=self.scheduler
        queues=[q for q in [s.waiting,s.skipped_waiting] if q]
        if not queues:return None
        candidates=[self.info(r) for q in queues for r in q]
        if any(c is None for c in candidates):
            return queues[0]
        now=time.monotonic()
        free=max(0,s.max_num_running_reqs-len(s.running))
        releases=[now]*free
        running=[]
        for r in s.running:
            c=self.info(r)
            if c is None:raise ValueError('warmup overlapped measurement')
            remaining=max(0,1-r.num_output_tokens/r.max_tokens)
            release=now+c['occupancy']*remaining
            releases.append(release)
            running.append({'id':c['id'],'remaining_fraction':remaining,'predicted_release':release})
        self.preferred,forecast=choose(candidates,releases,now=now,mode=self.mode)
        emit('admission_probe_decision',stage_id=0,mode=self.mode,request_id=self.preferred,
             running_count=len(s.running),free_slots=free,candidates=candidates,
             running_release_estimates=running,forecasts=forecast)
        return next(q for q in queues if any(str(r.request_id)==self.preferred for r in q))
