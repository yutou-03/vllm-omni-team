"""Rate-feasible admission and client-credit-backed pauses are separate decisions.

Normal admission uses live-phase resource demands and calibrated capacity, with
a before/after delivery forecast for incremental admission risk. Missing samples,
first-output startup and a small client buffer are NOT admission barriers.
Capacity allocation is a rate budget, not a measured per-request performance
prediction. A pre-existing delivery deficit is logged, not attributed to a newcomer.
Continuing streams can spend buffer on temporary shortfalls; only
a paused request must survive with zero predicted delivery.
"""
from collections import deque
from dataclasses import dataclass
import math

from .core import number


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    request_id: str | None = None
    minimum_buffer: float | None = None


def minimum_buffer(buffered, rate, horizon, deliveries=()):
    """Minimum of B(t)=B(0)+G(t)-rate*t, including BEFORE every delivery jump."""
    if not all(math.isfinite(v) and v >= 0 for v in (buffered, rate, horizon)):
        raise ValueError('invalid buffer forecast')
    current, previous, minimum = buffered, 0., buffered
    for at, units in deliveries:
        if not all(math.isfinite(v) for v in (at, units)) or not previous <= at <= horizon or units < 0:
            raise ValueError('invalid delivery envelope')
        current -= rate * (at - previous)
        minimum = min(minimum, current)
        current += units
        previous = at
    return min(minimum, current - rate * (horizon - previous))


class DeliveryHistory:
    """Bounded history of actual receipt jumps; heartbeat polls are not chunks."""
    def __init__(self):
        self.samples = deque(maxlen=512)

    def observe(self, feedback, demand, active=True):
        if not active or feedback.complete:
            self.samples.clear()
            return
        if not feedback.started:
            return
        previous = self.samples[-1] if self.samples else None
        if previous is None or feedback.received > previous[1]:
            self.samples.append((feedback.observed_at, feedback.received, dict(demand)))

    def sustained_rate(self, now, window):
        """Delivered units per wall second, including the current silent tail.

        A single chunk or startup gap is not evidence of sustained under-service.
        Keep an observation bracketing the window's beginning, and require two
        completed delivery intervals. Unknown is None, never zero production.
        """
        if len(self.samples) < 3 or self.samples[0][0] > now-window:
            return None
        samples = list(self.samples)
        start = max(i for i, sample in enumerate(samples) if sample[0] <= now-window)
        if len(samples)-start < 3:
            return None
        duration = now-samples[start][0]
        if duration <= 0:
            return None
        return (samples[-1][1]-samples[start][1])/duration


    def forecast_deliveries(self, now, window, horizon, rate):
        """Extrapolate receipt jumps, retaining the largest recent chunk gap.

        This is an adaptation heuristic, not a guaranteed service curve or an
        assertion that Gamma/N is actual per-request production. An overdue
        chunk gets another full gap, never instantaneous delivery credit.
        Call only after sustained_rate has established sufficient observations.
        """
        samples = list(self.samples)
        start = max(i for i, sample in enumerate(samples) if sample[0] <= now-window)
        gaps = [b[0]-a[0] for a, b in zip(samples[start:], samples[start+1:]) if b[0] > a[0]]
        gap = max(gaps)
        elapsed = max(0., now-samples[-1][0])
        first = gap-elapsed if elapsed < gap else gap
        deliveries = []
        at = first
        while at <= horizon:
            deliveries.append((at, rate*gap))
            at += gap
        return deliveries, gap


class AdmissionSafety:
    def admission_safety_init(self):
        self.last_admission_decision = None
        self.delivery_history = {}

    def observe_client_delivery(self, feedback):
        r = self.w.requests[feedback.request_id]
        book = getattr(self.w, 'book', None)
        demand = book.demand if book else {self.modality: self.w.demand}
        if r.state == 'finished' or feedback.complete:
            self.delivery_history.pop(r.request_id, None)
            return
        self.delivery_history.setdefault(r.request_id, DeliveryHistory()).observe(
            feedback, demand, active=r.state == 'admitted')

    def _admission_forecast(self, candidate, now, *, proposed_pauses=()):
        coordinator = getattr(self, 'coordinator', None)
        gateways = list(coordinator.gateways.values()) if coordinator else [self]
        book = getattr(self.w, 'book', None)
        startup = number(self.config.get('admission_startup_s', .5), 'admission startup', positive=True)
        before = book.demand if book else {self.modality: self.w.demand}
        cost = book.costs(self.modality, candidate) if book else {self.modality: candidate.rate}
        after = {k: before[k] + cost.get(k, 0.) for k in before}
        for gateway in gateways:
            for r in gateway.w.members:
                if r.request_id in proposed_pauses:
                    released = book.costs(gateway.modality, r) if book else {self.modality: r.rate}
                    for k in after:
                        after[k] -= released.get(k, 0.)
        checks = []
        bounds = {}
        # The two resource ledgers remain in their own units. Do not add
        # Thinker tokens/s to audio media seconds/s, or compare W's count to 32.
        for resource, need in after.items():
            if need <= 0:
                continue
            estimate = book.estimates[resource] if book else self.capacity(now)
            modality = ('text' if resource == 'thinker' else 'audio') if book else self.modality
            if estimate is None or not estimate.valid(now, modality):
                return SafetyDecision(False, 'stage_capacity_unknown', resource), checks
            bounds[resource] = estimate.admission_bound
            checks.append(dict(resource=resource, demand_after=need, capacity=bounds[resource],
                               model='stage_rate_budget'))
            if need > bounds[resource]+1e-9:
                return SafetyDecision(False, 'stage_rate_capacity_exceeded', resource), checks
        # If all currently paused owners can return alongside the newcomer,
        # their deadlines do not require holding them until its first output.
        # Reserve collectively: checking each owner separately oversubscribes
        # capacity when several owners return together.
        restored = dict(after)
        for gateway in gateways:
            for owner in gateway.w.outstanding:
                if owner.state == 'paused' or owner.request_id in proposed_pauses:
                    owner_cost = book.costs(gateway.modality, owner) if book else {self.modality: owner.rate}
                    for resource, value in owner_cost.items():
                        restored[resource] = restored.get(resource, 0.) + value
        restoration_fits = True
        for resource, need in restored.items():
            if need <= 0:
                continue
            estimate = book.estimates[resource] if book else self.capacity(now)
            modality = ('text' if resource == 'thinker' else 'audio') if book else self.modality
            if estimate is None or not estimate.valid(now, modality) or need > estimate.admission_bound + 1e-9:
                restoration_fits = False
        for gateway in gateways:
            resume = number(gateway.config.get('admission_resume_s',
                2. if gateway.modality == 'audio' else .5), 'admission resume', positive=True)
            margin = number(gateway.config.get('admission_margin_s', .5), 'admission margin')
            for r in gateway.w.outstanding:
                paused = r.state == 'paused' or r.request_id in proposed_pauses
                live_cost = book.costs(gateway.modality, r) if book else {self.modality: r.rate}
                resources = [k for k in after if live_cost.get(k, 0.) > 0]
                if not paused and not any(cost.get(k, 0.) > 0 for k in resources):
                    continue
                feedback = gateway.feedback.get(r.request_id)
                if paused:
                    # No predicted delivery funds a pause: its buffer must
                    # cover the full startup and recovery interval.
                    coverage = feedback.coverage(now, .5) if feedback else None
                    if coverage is None:
                        return SafetyDecision(False, 'paused_client_feedback_unavailable', r.request_id), checks
                    accepted = gateway.accepted_pause_leases.get(r.request_id)
                    if accepted and restoration_fits:
                        horizon = max(0., accepted['resume_by']-now)+resume+margin
                        minimum = minimum_buffer(coverage*r.rate, r.rate, horizon)
                        checks.append(dict(request_id=r.request_id, paused=True,
                            model='reserved_early_restoration', horizon_s=horizon,
                            forecast_delivery_units=0., minimum_buffer=minimum,
                            coverage_s=coverage, restored_demand=restored))
                        # A due return already has capacity; admission is not
                        # allowed to defer it until the newcomer's first output.
                        if accepted['resume_by'] <= now or minimum >= -1e-9:
                            continue
                        return SafetyDecision(False, 'insufficient_pause_buffer', r.request_id, minimum), checks
                    if accepted and now+startup >= accepted['resume_by']:
                        return SafetyDecision(False, 'recovery_before_new_first_output', r.request_id), checks
                    horizon = startup+resume+margin
                    minimum = minimum_buffer(coverage*r.rate, r.rate, horizon)
                    checks.append(dict(request_id=r.request_id, paused=True, horizon_s=horizon,
                        model='paused_zero_delivery', forecast_delivery_units=0.,
                        minimum_buffer=minimum, coverage_s=coverage))
                    if minimum < -1e-9:
                        return SafetyDecision(False, 'insufficient_pause_buffer', r.request_id, minimum), checks
                    continue
                # Feasible demand-weighted allocation: every live phase must
                # be able to allocate this request its minimum required rate.
                # This is NOT Gamma/N as an asserted actual generation speed.
                factor = min((bounds[k]/after[k] for k in resources), default=1.)
                allocated = r.rate*factor
                history = gateway.delivery_history.get(r.request_id)
                fresh = feedback is not None and 0 <= now-feedback.observed_at <= .5
                window = gateway.config.get('capacity_window_s', 8. if gateway.modality=='audio' else 2.)
                observed = history.sustained_rate(now, window) if history and fresh else None
                check = dict(request_id=r.request_id, paused=False, required_rate=r.rate,
                    allocated_rate=allocated, observed_delivery_rate=observed,
                    model='stage_rate_budget_with_delivery_observation',
                    observation='measured' if observed is not None else 'unknown_not_zero')
                checks.append(check)
                # End-client audio throughput cannot identify a Thinker-only
                # bottleneck for a text newcomer. Apply this negative evidence
                # within the same output path; shared capacity is checked above.
                if gateway.modality == self.modality and observed is not None:
                    # Compare the same receipt history with and without the
                    # candidate. The old absolute check could hide every waiter
                    # because a stream was ALREADY under-served, even with idle
                    # slots and ample Gamma. That is not evidence the candidate
                    # caused an underflow. Preserve it as a health observation.
                    # Demand-share scaling is an explicit adaptation heuristic,
                    # not a measured post-admission rate or a service guarantee.
                    # Never predict a rate gain from a proposed donor pause.
                    share = min((min(1., before[k]/after[k]) for k in resources
                                 if after[k] > 0), default=1.)
                    projected = observed*share
                    check.update(projected_delivery_rate=projected,
                                 demand_share_factor=share)
                    if projected+1e-9 >= r.rate:
                        continue
                    coverage = feedback.coverage(now, .5)
                    if coverage is None:
                        return SafetyDecision(False, 'existing_stream_feedback_unavailable', r.request_id), checks
                    horizon = startup+resume+margin
                    baseline, gap = history.forecast_deliveries(now, window, horizon, observed)
                    deliveries = [(at, units*share) for at, units in baseline]
                    baseline_minimum = minimum_buffer(coverage*r.rate, r.rate, horizon, baseline)
                    minimum = minimum_buffer(coverage*r.rate, r.rate, horizon, deliveries)
                    check.update(model='incremental_continuing_stream_buffer_forecast',
                        horizon_s=horizon, coverage_s=coverage,
                        forecast_delivery_units=sum(units for _, units in deliveries),
                        forecast_chunk_gap_s=gap, minimum_buffer=minimum,
                        baseline_minimum_buffer=baseline_minimum,
                        forecast_basis='receipt_history_scaled_by_demand_share_not_service_guarantee')
                    if baseline_minimum < -1e-9:
                        check['observation'] = 'pre_existing_underflow'
                        continue
                    if minimum < -1e-9:
                        return SafetyDecision(False, 'admission_would_introduce_stream_underflow',
                                              r.request_id, minimum), checks
        return SafetyDecision(True, 'stage_rates_feasible'), checks

    def admission_safety(self, candidate, now):
        if self.config['arm'] == 'fcfs' or self.config.get('capacity_only', False):
            return True
        decision, forecasts = self._admission_forecast(candidate, now)
        self.last_admission_decision = decision
        signature = (candidate.request_id, decision.allowed, decision.reason, decision.request_id,
                     tuple((f.get('request_id', f.get('resource')), f['model']) for f in forecasts))
        if signature != getattr(self, '_admission_safety_signature', None):
            self.event('admission_safety', candidate=candidate.request_id,
                allowed=decision.allowed, reason=decision.reason,
                blocking_request=decision.request_id, minimum_buffer=decision.minimum_buffer,
                delivery_model='stage_rate_admission_client_safe_pauses', forecasts=forecasts,
                cold_start=not any(gateway.w.outstanding for gateway in
                    (self.coordinator.gateways.values() if hasattr(self, 'coordinator') else [self])))
            self._admission_safety_signature = signature
        return decision.allowed

    def safety_pause_waiter(self, now):
        """Expose a blocked newcomer only if a specific safe swap can admit it.

        This is a proposal, never a quota release. Actual pause telemetry and its
        acknowledgement must arrive before the gateway changes W or admits it.
        """
        coordinator = getattr(self, 'coordinator', None)
        book = getattr(self.w, 'book', None)
        pending = book.pending if book else [r for r in self.w.requests.values() if r.state == 'waiting']
        if not pending or self.config['arm'] != 'buffer' or self.config.get('capacity_only'):
            return None
        candidate = pending[0]
        owner = (next(g for g in coordinator.gateways.values() if candidate.request_id in g.w.requests)
                 if coordinator else self)
        if book:
            if book.reason(now) is not None:
                return None
            costs = book.costs(owner.modality, candidate)
            for key, value in costs.items():
                if not value:
                    continue
                e = book.estimates[key]
                if e is None or not e.valid(now, 'text' if key == 'thinker' else 'audio'):
                    return None
                if book.demand[key] + value > e.admission_bound + 1e-9:
                    return None  # existing capacity-deficit path handles this
        else:
            e = self.capacity(now)
            if e is None or not e.valid(now, self.modality) or self.w.demand+candidate.rate > e.admission_bound:
                return None
            costs = {self.modality: candidate.rate}
        current, _ = owner._admission_forecast(candidate, now)
        if current.allowed or current.reason != 'admission_would_introduce_stream_underflow':
            return None
        eligible = []
        for donor in self.w.members:
            decision, _ = owner._admission_forecast(candidate, now, proposed_pauses={donor.request_id})
            if decision.allowed:
                eligible.append(donor.request_id)
        if not eligible:
            return None
        return dict(request_id=candidate.request_id, modality=owner.modality, rate=candidate.rate,
            resource_cost=costs, resource_deficit={k: 0. for k in costs},
            eligible_pause_sources=eligible, reason='observed_rate_safe_swap')
