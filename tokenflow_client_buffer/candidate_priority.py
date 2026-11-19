"""Score feasible actions on a single final-output path, after mandatory returns."""
import math

from .pause import PausedRequest


def candidate_value(coverage, debt_s, waiting_s, output_tokens, remaining_prompt, token_budget):
    # Time terms use seconds for both modalities. Bound the progress bonus so
    # long-running streams cannot permanently crowd out first outputs.
    urgency = 1.5 / (1. + coverage)
    progress = .6 * output_tokens / (1. + output_tokens)
    prefill = remaining_prompt / max(1, token_budget)
    return urgency + math.log1p(debt_s) + math.log1p(waiting_s) + progress - prefill


class CandidatePriority:
    def __init__(self, adapter, snapshot, now, cover):
        self.adapter = adapter
        self.snapshot = snapshot
        self.now = now
        self.cover = cover
        self.scores = {}

    def score(self, request):
        rid = request.request_id
        if rid not in self.scores:
            source = self.adapter.source_id(request)
            since = self.snapshot.get('registered_at', {}).get(source, self.now)
            pause = self.adapter.pauses.get(rid)
            if pause is not None:
                since = pause.epoch
            elif self.snapshot.get('startup_feedback', self.snapshot.get('feedback', {})).get(source, {}).get('started'):
                since = self.now
            if rid not in self.cover:
                self.cover[rid] = self.adapter.coverage(request, self.snapshot, self.now)
            debt = self.snapshot.get('signed_buffer_s', {}).get(source, 0.)
            self.scores[rid] = candidate_value(
                self.cover[rid] or 0., max(0., -debt), max(0., self.now - since),
                request.num_output_tokens, max(0, request.num_prompt_tokens - request.num_computed_tokens),
                getattr(self.adapter.scheduler, "max_num_scheduled_tokens", 1))
        return self.scores[rid]

    def fill(self, chosen, returns, beneficiaries, waiting, original, new_pauses,
             guard, safe_victims, budget, quota_protocol, eligible_return):
        adapter = self.adapter
        snapshot = self.snapshot
        selected = {r.request_id for r in chosen + returns}
        admitted = (snapshot.get('members', {}).keys() | snapshot.get('paused_members', {}).keys()
                    | set(snapshot.get('buffer_reclaimed_ids', ())))
        reclaimed = set(snapshot.get('buffer_reclaimed_ids', ()))
        waiting_ids = {r.request_id for r in waiting}
        candidates = []
        for r in original + waiting:
            rid = r.request_id
            if (rid in selected or rid in new_pauses or adapter.source_id(r) not in admitted
                    or adapter.metadata(r).get('sched_request_path') != adapter.config['modality']):
                continue
            if rid in adapter.pauses and not eligible_return(r):
                continue
            candidates.append(r)
        candidates.sort(key=lambda r: (-self.score(r), adapter.order(r)))
        donors = sorted(safe_victims(), key=lambda r: (self.score(r), adapter.order(r)), reverse=True)
        if self.now - adapter.last_opportunistic_pause < adapter.min_slice:
            donors.clear()
        bound = snapshot.get('gamma', float('inf'))
        for r in candidates:
            source = adapter.source_id(r)
            returning = r.request_id in adapter.pauses
            cost = (snapshot.get('paused_members', {}).get(source, {}).get('rate', 0.)
                    if returning and adapter.pauses.releases_quota(r.request_id) else 0.)
            if source in reclaimed and source not in snapshot.get('members', {}):
                cost = snapshot.get('startup_members', {}).get(source, {}).get('rate')
                if cost is None or budget + cost > bound:
                    continue  # An unconfirmed pause cannot fund a reclaimed request.
            full = len(chosen) + len(returns) + len(beneficiaries) >= adapter.cap
            if full or budget + cost > bound:
                if not donors:
                    continue
                victim = donors[-1]
                released = (snapshot.get('members', {}).get(adapter.source_id(victim), {}).get('rate', 0.)
                            if quota_protocol else 0.)
                if self.score(r) <= self.score(victim) or budget + cost - released > bound:
                    continue
                donors.pop()
                chosen.remove(victim)
                budget -= released
                progress = snapshot.get('startup_feedback', snapshot.get('feedback', {})).get(source, {})
                new_pauses[victim.request_id] = PausedRequest(
                    self.now, self.now + self.cover[victim.request_id] - guard, guard,
                    releases_quota=quota_protocol, beneficiary=source,
                    beneficiary_received=progress.get('received') if progress.get('started') else None)
                adapter.last_opportunistic_pause = self.now
                adapter.emit('client_buffer_priority_exchange', stage_id=adapter.stage,
                             request_id=victim.request_id, beneficiary=source,
                             action='resume' if returning else 'start' if not r.num_output_tokens else 'continue',
                             beneficiary_score=self.score(r), donor_score=self.score(victim),
                             signed_buffer_s=snapshot.get('signed_buffer_s', {}).get(source))
            if r.request_id in waiting_ids:
                (returns if returning else beneficiaries).append(r)
            else:
                chosen.append(r)
            budget += cost
        return bool(candidates)
