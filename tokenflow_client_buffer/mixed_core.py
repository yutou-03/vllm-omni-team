"""Atomic mixed admission in Thinker-token and playable-audio resource units.

Resource coefficients come from independent calibration. They are NEVER
client buffer credit. Each client's buffer remains in its own output units.
"""
import math
from .core import CapacityEstimate, WorkingSet


class MixedBook:
    def __init__(self, thinker, audio, audio_thinker_tokens_per_media_s, *,
                 text_thinker_tokens_per_visible_token=1., native_limit=32, admission_policy='resource_fifo', max_inflight=256):
        self.estimates = dict(thinker=thinker, audio=audio)
        self.weights = dict(text=float(text_thinker_tokens_per_visible_token),
                            audio=float(audio_thinker_tokens_per_media_s))
        if any(not math.isfinite(x) or x <= 0 for x in self.weights.values()):
            raise ValueError('positive calibrated resource coefficients required')
        if admission_policy not in ('global_fifo','resource_fifo'):raise ValueError('unknown mixed admission policy')
        self.admission_policy=admission_policy
        self.native_limit = native_limit
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("positive transport budget required")
        self.max_inflight = max_inflight
        self.sequence = 0
        # Native queue ownership is distinct from buffer working-set admission.
        # A queued request has no GPU KV allocation merely because it is visible
        # to native FCFS. The engine owns memory admission and execution limits.
        self.native_fcfs = False
        self.native_fcfs_reason = None
        self.native_fcfs_epoch = 0
        self.native_fcfs_trigger = None
        self.thinker_completed = set()
        self.views = {m: MixedView(self, m) for m in ('text', 'audio')}

    @property
    def native_queued(self):
        return [r for v in self.views.values() for r in v.requests.values()
                if r.state == 'native_queued']

    def service_demand(self, *, include_waiting=False):
        """Sustained obligations; a temporary pause never erases consumption."""
        return {key: math.fsum(self.costs(m, r)[key]
                    for m, v in self.views.items() for r in v.requests.values()
                    if r.state != 'finished' and (include_waiting or r.state in {'admitted', 'paused'}))
                for key in ('thinker', 'audio')}

    def demand_reason(self, demand, now):
        unknown = False
        for key, need in demand.items():
            if need <= 0:
                continue
            e = self.estimates[key]
            if e is None or not e.valid(now, 'text' if key == 'thinker' else 'audio'):
                unknown = True
            elif need > e.gamma + 1e-9:
                return 'overload'
        return 'capacity_unknown' if unknown else None

    def enter_native_fcfs(self, reason):
        if not self.native_fcfs:
            self.native_fcfs_epoch += 1
            self.native_fcfs_trigger = reason
        self.native_fcfs = True
        self.native_fcfs_reason = reason

    def refresh_native_fcfs(self, now):
        # Evaluate the service pool BEFORE filtering by the active quota book.
        # Otherwise paused owners vanish from the feasibility test and a
        # capacity-rejected newcomer can never trigger the advertised fallback.
        reason = self.demand_reason(self.service_demand(), now)
        # Empty W does not certify the capacity of resources queued work will
        # need. Check estimate validity without charging queue entries as W.
        if reason is None:
            for modality, view in self.views.items():
                for request in view.requests.values():
                    if request.state != 'native_queued':
                        continue
                    for key, need in self.costs(modality, request).items():
                        e = self.estimates[key]
                        if need > 0 and (e is None or not e.valid(
                                now, 'text' if key == 'thinker' else 'audio')):
                            reason = 'capacity_unknown'
                            break
                    if reason:
                        break
                if reason:
                    break
        if reason:
            self.enter_native_fcfs(reason)
        if self.native_fcfs and reason is None:
            # Recover on service feasibility, not on an empty engine queue.
            # Already handed-off requests stay in place and remain uncharged
            # until their first execution journal event. The snapshot explicitly
            # hands these waiting identities to buffer scheduling, so a policy
            # switch neither hides them nor falsely promotes them into W.
            self.native_fcfs = False
            self.native_fcfs_reason = None
            self.native_fcfs_trigger = None
        return self.native_fcfs

    def mark_native_execution(self, modality, request_id):
        request = self.views[modality].requests.get(request_id)
        if request is not None and request.state == 'native_queued':
            request.state = 'admitted'
            return True
        return False

    def complete_thinker(self, request_id):
        if not any(request_id in v.requests for v in self.views.values()):
            return False
        before = request_id in self.thinker_completed
        self.thinker_completed.add(request_id)
        return not before

    def costs(self, route, request):
        return dict(thinker=0. if request.request_id in self.thinker_completed else
                    self.weights[route] * request.rate,
                    audio=request.rate if route == 'audio' else 0.)

    @property
    def members(self):
        return [r for v in self.views.values() for r in v.members]

    @property
    def outstanding(self):
        return [r for v in self.views.values() for r in v.outstanding]

    def admission_waiter(self, now):
        pending = self.pending
        if not pending or self.reason(now) is not None:
            return None
        r = pending[0]
        route = next(m for m, v in self.views.items() if r.request_id in v.requests)
        costs = self.costs(route, r)
        deficits = {}
        for key, cost in costs.items():
            if not cost:
                continue
            e = self.estimates[key]
            if e is None or not e.valid(now, 'text' if key == 'thinker' else 'audio'):
                return None
            deficits[key] = max(0., self.demand[key] + cost - e.admission_bound)
        if not any(x > 1e-9 for x in deficits.values()):
            return None
        return dict(request_id=r.request_id, modality=route, rate=r.rate,
                    resource_cost=costs, resource_deficit=deficits)

    @property
    def pending(self):
        return sorted((r for v in self.views.values() for r in v.requests.values()
                       if r.state == 'waiting'), key=lambda r:r.arrival_order)

    @property
    def demand(self):
        return dict(thinker=math.fsum(self.costs(m, r)['thinker']
                    for m, v in self.views.items() for r in v.members),
                    audio=self.views['audio'].demand)

    def reason(self, now):
        if self.native_fcfs:
            return self.native_fcfs_reason
        return self.demand_reason(self.service_demand(), now)

    def capacity(self, modality, now, candidate=None):
        # Convert remaining *live-phase* capacity into this route's rate units.
        # Completed Thinker work has zero coefficient; an idle expired Thinker
        # must not disable audio-only continuation. Newcomers are checked with
        # their own phase costs separately in admit_next.
        view = self.views[modality]
        candidates = ([candidate] if candidate is not None else
                      view.members or [r for r in view.requests.values() if r.state == 'waiting'][:1])
        if not candidates:
            candidates = view.outstanding
        coefficients = dict(thinker=max((self.costs(modality, r)['thinker'] / r.rate
                                         for r in candidates), default=0.),
                            audio=1. if modality == 'audio' else 0.)
        bounds, admissions, required = [], [], []
        for key, coefficient in coefficients.items():
            if coefficient <= 0:
                continue
            e = self.estimates[key]
            if e is None or not e.valid(now, 'text' if key == 'thinker' else 'audio'):
                return None
            required.append(e)
            bounds.append(view.demand + (e.gamma - self.demand[key]) / coefficient)
            admissions.append(view.demand + (e.admission_bound - self.demand[key]) / coefficient)
        if not required:
            return None
        return CapacityEstimate(max(1e-12, min(bounds)), max(e.observed_at for e in required),
            min(e.valid_until for e in required), modality,
            ':'.join(e.profile_id for e in required), 'mixed_live_phase_residual',
            max(1e-12, min(admissions)))

    def snapshot(self, now):
        return dict(resource_demand=self.demand,
                    resource_service_demand=self.service_demand(),
                    native_fcfs_admission=self.native_fcfs,
                    native_fcfs_reason=self.native_fcfs_reason,
                    native_fcfs_epoch=self.native_fcfs_epoch,
                    native_fcfs_trigger=self.native_fcfs_trigger,
                    native_queued=len(self.native_queued),
                    resource_gamma={k:e.gamma if e is not None and e.valid(now,'text' if k=='thinker' else 'audio') else None
                                    for k,e in self.estimates.items()},
                    resource_admission_gamma={k:e.admission_bound if e else None for k,e in self.estimates.items()},
                    resource_coefficients=self.weights, reason=self.reason(now),
                    global_waiting=len(self.pending), global_members=len(self.members),
                    global_paused=len(self.outstanding)-len(self.members))


class MixedView(WorkingSet):
    def __init__(self, book, modality):
        super().__init__(modality,max_inflight=book.max_inflight,recovery_samples=1,
                         native_execution_limit=book.native_limit)
        self.book = book

    def add(self,rid,rate):
        if any(v.has_request_id(rid) for v in self.book.views.values()):
            raise ValueError('duplicate mixed logical request')
        super().add(rid,rate)
        self.requests[rid].arrival_order=self.book.sequence
        self.book.sequence+=1

    def complete(self, rid):
        super().complete(rid)
        self.book.thinker_completed.discard(rid)

    def update(self,estimate,now):
        self.book.refresh_native_fcfs(now)
        super().update(estimate,now)
        reason=self.book.reason(now)
        if reason:
            self.mode,self.reason,self._safe_samples='fcfs',reason,0
        return self.mode

    def admit_next(self,estimate,now,*,native=False):
        pending=self.book.pending
        if not pending:return None
        candidate=pending[0]
        if not native:
            self.book.refresh_native_fcfs(now)
            route = next(m for m, v in self.book.views.items()
                         if candidate.request_id in v.requests)
            prospective = self.book.service_demand()
            for key, cost in self.book.costs(route, candidate).items():
                prospective[key] += cost
            reason = self.book.demand_reason(prospective, now)
            if reason:
                self.book.enter_native_fcfs(reason)
            if self.book.native_fcfs:
                if candidate.request_id not in self.requests:
                    return None
                # Enqueue, do not claim a running slot or GPU memory. Native
                # FCFS allocates KV and admits to execution when resources fit.
                candidate.state = 'native_queued'
                self.mode, self.reason = 'fcfs', self.book.native_fcfs_reason
                return candidate.request_id
        if candidate.request_id not in self.requests:
            # Buffer-mode escape from a modality-specific bottleneck. Never
            # skip a head blocked by the SHARED Thinker resource, and never
            # expand an overloaded/unknown working set. Reserve the audio
            # head's Thinker requirement so bypassing text cannot starve it
            # when its audio slot becomes available.
            can_bypass=(not native and self.book.admission_policy=='resource_fifo'
                        and self.modality=='text' and self.book.reason(now) is None
                        and candidate.request_id in self.book.views['audio'].requests)
            if not can_bypass:return None
            audio_est=self.book.estimates['audio'];thinker_est=self.book.estimates['thinker']
            if (audio_est is None or thinker_est is None or
                not audio_est.valid(now,'audio') or not thinker_est.valid(now,'text')):return None
            audio_need=self.book.views['audio'].demand+candidate.rate
            reserved=self.book.costs('audio',candidate)['thinker']
            head_blocked_only_by_audio=(audio_need>audio_est.admission_bound+1e-9
                and self.book.demand['thinker']+reserved<=thinker_est.admission_bound+1e-9)
            text_head=next((r for r in pending if r.request_id in self.requests),None)
            if (not head_blocked_only_by_audio or text_head is None or
                self.book.demand['thinker']+reserved+self.book.costs('text',text_head)['thinker']>
                    thinker_est.admission_bound+1e-9):return None
            candidate=text_head
        if not native:
            # Per-candidate live phases matter: completed Thinker work has no
            # Thinker cost. Unknown/overloaded service has already handed off
            # above, never add a second len(W) vs max_num_seqs gate here.
            estimate = self.book.capacity(self.modality, now, candidate=candidate)
            for resource, cost in self.book.costs(self.modality, candidate).items():
                if not cost:
                    continue
                e = self.book.estimates[resource]
                if e is None or not e.valid(now, 'text' if resource == 'thinker' else 'audio'):
                    return None
                if self.book.demand[resource] + cost > e.admission_bound + 1e-9 and self.book.members:
                    return None
        return super().admit_next(estimate,now,native=native)
