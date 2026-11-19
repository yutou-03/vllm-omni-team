"""Acknowledged transfer of active admission demand during client-safe pauses.

The scheduler proves a request is not executing. Only validated CLIENT credit
can fund the lease. Confirmed releases and returns share one admission transaction. A resume is
charged before execution is acknowledged. Lost telemetry freezes new admission
without revoking a safe lease; deadlines and explicit fallback still restore.
"""
import json
import math
import time
from pathlib import Path


class PauseAdmission:
    def init_pause_admission(self):
        self.registered_at = {}
        self.pause_acks = {}
        self.accepted_pause_leases = {}
        self.recovery_pending = False
        self.pause_clock = time.monotonic
        self.pause_telemetry_seen = False
        self.pause_telemetry_status = None

    @property
    def pause_admission_enabled(self):
        return (self.config.get('pause_releases_admission', False)
                and self.config.get('arm') == 'buffer'
                and not self.config.get('capacity_only', False))

    def pause_snapshot(self, now, *, mixed=False):
        if not self.pause_admission_enabled:
            return {}
        if mixed:
            book = self.coordinator.book
            head = book.admission_waiter(now)
            costs = dict(thinker=book.weights[self.modality],
                         audio=1. if self.modality == 'audio' else 0.)
            extra = dict(pause_release_cost_by_request={
                r.request_id: {key: value / r.rate for key, value in
                               book.costs(self.modality, r).items()}
                for r in self.w.outstanding})
        else:
            key = 'thinker' if self.modality == 'text' else 'audio'
            costs = {key: 1.}
            pending = next((r for r in self.w.requests.values() if r.state == 'waiting'), None)
            est = self.capacity(now)
            head = None
            if pending and est and est.valid(now, self.modality) and self.w.mode == 'buffer':
                deficit = self.w.demand + pending.rate - est.admission_bound
                if deficit > 1e-9:
                    head = dict(request_id=pending.request_id, modality=self.modality,
                                rate=pending.rate, resource_deficit={key: deficit})
            extra = dict(startup_members={r.request_id:dict(rate=r.rate)
                         for r in self.w.requests.values() if r.state != 'finished'})
        if head is None:
            head = self.safety_pause_waiter(now)
        startup_s = self.config.get('admission_startup_s', .5)
        if mixed and head:
            startup_s = max(startup_s, self.coordinator.profile.get(head['modality']+'_startup_bound_s', .5))
        if head:
            head = dict(head, startup_latency_estimate_s=startup_s)
        return dict(pause_admission_protocol=1, admission_waiter=head,
                    pause_release_min_s=.5,
                    pause_release_cost_per_rate=costs,
                    pause_acknowledgements=dict(self.pause_acks), **extra)

    def _restore_pause(self, rid, epoch, reason):
        before = self.w.demand
        restored = self.w.resume(rid)
        self.accepted_pause_leases.pop(rid, None)
        self.pause_acks[rid] = dict(epoch=epoch, state='active')
        if restored:
            self.delivery_history.pop(rid, None)
            self.event('pause_admission_restored', request_id=rid, epoch=epoch,
                       reason=reason, demand_before=before, demand_after=self.w.demand)

    def process_pause_leases(self, now):
        if not self.pause_admission_enabled:
            return
        # Sample AFTER the read: the scheduler may replace the file between
        # the caller's timestamp and this read. That is not a future clock.
        t, leases, age = {}, {}, None
        reason = None
        try:
            t = json.loads(Path(self.config['execution_telemetry_path']).read_text())
            now = max(now, self.pause_clock())
            age = now - float(t['written_at'])
            if t.get('session') != self.session:
                reason = 'session_mismatch'
            elif not math.isfinite(age) or age < 0:
                reason = 'invalid_timestamp'
            elif age > .5:
                reason = 'stale'
            elif not isinstance(t.get('pause_leases', {}), dict):
                reason = 'invalid_leases'
            else:
                leases = t.get('pause_leases', {})
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            now = max(now, self.pause_clock())
            reason = 'read_or_parse_error'
        fresh = reason is None
        if fresh:
            self.pause_telemetry_seen = True
        status = reason or 'fresh'
        if status != self.pause_telemetry_status:
            self.event('pause_telemetry_status', status=status,
                       age_s=age if age is not None and math.isfinite(age) else None,
                       accepted_leases=len(self.accepted_pause_leases))
            self.pause_telemetry_status = status
        # Restored leases leave accepted_pause_leases, so terminal ACKs need
        # their own cleanup. Keep live ACKs until the scheduler can observe them.
        for rid in list(self.pause_acks):
            request = self.w.requests.get(rid)
            if request is None or request.state == 'finished':
                self.pause_acks.pop(rid, None)
        # A terminal client stream has no outstanding restoration obligation,
        # even if the last execution-monitor update still contains its lease.
        for rid in list(self.accepted_pause_leases):
            request = self.w.requests.get(rid)
            if request is None or request.state == 'finished':
                self.accepted_pause_leases.pop(rid, None)
                self.pause_acks.pop(rid, None)
        # Bootstrap without execution telemetry must remain possible. Once
        # observed, freeze admission on lost visibility while work is live.
        self.recovery_pending = (bool(t.get('recovery_pending')) if fresh else
                                 bool(self.accepted_pause_leases or
                                      self.pause_telemetry_seen and self.w.outstanding))
        restorations = {}
        for rid, accepted in list(self.accepted_pause_leases.items()):
            incoming = leases.get(rid, {})
            same = isinstance(incoming, dict) and incoming.get('epoch') == accepted['epoch']
            if same and incoming.get('state') == 'paused':
                try:
                    deadline, guard = float(incoming['resume_by']), float(incoming['resume_guard_s'])
                    confirmed = float(incoming['not_scheduled_at'])
                    f = self.feedback.get(rid)
                    credit = f.coverage(now, .5) if f else None
                    if (all(math.isfinite(x) for x in (deadline, guard, confirmed))
                            and guard > 0 and 0 <= now-confirmed <= .5
                            and credit is not None and credit + 1e-6 >= deadline-now+guard):
                        accepted['resume_by'] = deadline
                except (ValueError, TypeError, KeyError):
                    pass
            elif same and incoming.get('state') == 'resume_requested':
                # A larger replenishment estimate can move the deadline earlier.
                # A return request cannot extend an accepted safe pause.
                try:
                    deadline = float(incoming['resume_by'])
                    if math.isfinite(deadline):
                        accepted['resume_by'] = min(accepted['resume_by'], deadline)
                except (ValueError, TypeError, KeyError):
                    pass
            if now >= accepted['resume_by']:
                restorations[rid] = (accepted['epoch'], 'lease_expired', True)
            elif not fresh:
                # Missing monitoring is not evidence of resumed execution.
                # Keep the last verified deadline, without renewing it, and
                # admit nobody until visibility returns. Never pause forever.
                continue
            elif same and incoming.get('state') == 'resume_requested':
                restorations[rid] = (accepted['epoch'], 'resume_requested',
                                     bool(t.get('recovery_pending')))
            elif not same or incoming.get('state') != 'paused':
                # The execution side no longer proves a pause: charge it now.
                restorations[rid] = (accepted['epoch'], 'pause_no_longer_observed', True)
        minimum_lease = self.pause_snapshot(now, mixed=hasattr(self, 'coordinator')).get('pause_release_min_s', .5)
        for rid, lease in leases.items():
            r = self.w.requests.get(rid)
            if r is None or r.state not in {'admitted', 'paused'} or not isinstance(lease, dict):
                continue
            try:
                epoch = float(lease['epoch'])
                if not math.isfinite(epoch) or epoch < self.registered_at.get(rid, 0.):
                    continue
                previous = self.pause_acks.get(rid, {})
                if epoch < previous.get('epoch', -1.):
                    continue
                if lease.get('state') == 'resume_requested':
                    restorations.setdefault(rid, (epoch, 'resume_requested',
                        rid not in self.accepted_pause_leases or bool(t.get('recovery_pending'))))
                    continue
                # Once restored, the same lease can never release demand again.
                if (lease.get('state') != 'paused'
                        or previous.get('epoch') == epoch and previous.get('state') == 'active'):
                    continue
                if rid in self.accepted_pause_leases:
                    continue
                resume_by, guard = float(lease['resume_by']), float(lease['resume_guard_s'])
                confirmed = float(lease['not_scheduled_at'])
                feedback = self.feedback.get(rid)
                coverage = feedback.coverage(now, .5) if feedback else None
                if (not all(math.isfinite(x) for x in (resume_by, guard, confirmed))
                        or guard <= 0 or not epoch <= confirmed <= now
                        or now - confirmed > .5 or now + minimum_lease + .5 >= resume_by
                        or coverage is None or coverage + 1e-6 < resume_by - now + guard):
                    continue
                before = self.w.demand
                if self.w.pause(rid):
                    self.delivery_history.pop(rid, None)
                    self.accepted_pause_leases[rid] = dict(epoch=epoch, resume_by=resume_by)
                    self.pause_acks[rid] = dict(epoch=epoch, state='paused')
                    self.event('pause_admission_released', request_id=rid, epoch=epoch,
                               resume_by=resume_by, coverage_s=coverage,
                               inventory_source=self.config.get('buffer_source', 'client_consumption_buffer'),
                               demand_before=before, demand_after=self.w.demand)
            except (ValueError, TypeError, KeyError):
                continue
        # Apply all confirmed client-safe releases before any return. No new
        # admission or snapshot publication can interleave this transaction.
        for rid, (epoch, reason, forced) in sorted(restorations.items(),
                key=lambda item: self.accepted_pause_leases.get(item[0], {}).get('resume_by', now)):
            fits = self._restoration_fits(rid, now)
            if not forced and not fits:
                ack = dict(epoch=epoch, state='resume_deferred')
                if self.pause_acks.get(rid) != ack:
                    self.event('pause_restore_deferred', request_id=rid, epoch=epoch,
                               resume_by=self.accepted_pause_leases[rid]['resume_by'],
                               reason='capacity_exchange_required')
                self.pause_acks[rid] = ack
                self.recovery_pending = True
                continue
            if forced and not fits:
                self.event('pause_restore_forced', request_id=rid, epoch=epoch,
                           reason=reason, demand_before=self.w.demand)
                self.recovery_pending = True
            self._restore_pause(rid, epoch, reason)
        self.w.update(self.capacity(now), now)

    def _restoration_fits(self, rid, now):
        request = self.w.requests.get(rid)
        if request is None or request.state != 'paused':
            return True
        if hasattr(self, 'coordinator'):
            book = self.coordinator.book
            demand = book.demand
            for resource, cost in book.costs(self.modality, request).items():
                if cost <= 0:
                    continue
                estimate = book.estimates[resource]
                modality = 'text' if resource == 'thinker' else 'audio'
                if (estimate is None or not estimate.valid(now, modality)
                        or demand[resource] + cost > estimate.gamma + 1e-9):
                    return False
            return True
        estimate = self.capacity(now)
        return (estimate is not None and estimate.valid(now, self.modality)
                and self.w.demand + request.rate <= estimate.gamma + 1e-9)
