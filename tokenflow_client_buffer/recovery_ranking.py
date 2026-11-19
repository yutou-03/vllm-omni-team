"""One native step of TTFT/stall trade-off among already feasible requests."""


def exchange_values(step, replenish, coverage, remaining_prompt, token_budget):
    # One step of head start, discounted if the available tokens can only
    # process part of the prompt. This is a local proxy, not measured TTFT.
    gain = step * min(1., token_budget / max(1, remaining_prompt))
    added_stall = max(0., replenish + step - coverage) - max(0., replenish - coverage)
    return gain, min(step, max(0., added_stall))


def rank_recovery(adapter, snapshot, now, due, chosen, returns, beneficiaries,
                  blocked, waiting, original, cover, newly_paused):
    """Swap a deadline-first slot for a runnable newcomer at most once per lease."""
    step = min(adapter.max_slice, max(adapter.min_slice,
                   adapter.actual_slices[-1] if adapter.actual_slices else adapter.min_slice))
    replenish = adapter.resume_s + adapter.margin_s
    selected = {r.request_id for r in chosen + returns + beneficiaries}
    waiting_ids = {r.request_id for r in waiting}
    # Only compare requests that can still legally be executed in buffer mode.
    eligible_sources = snapshot.get('members', {}).keys() | set(snapshot.get('buffer_reclaimed_ids', ()))
    fresh = []
    for r in original + waiting:
        rid = r.request_id
        if (rid in selected or rid in adapter.pauses or rid in newly_paused or r.num_output_tokens
                or adapter.source_id(r) not in eligible_sources
                or adapter.metadata(r).get('sched_request_path') != adapter.config['modality']):
            continue
        fresh.append(r)
    # Use truly free slots first. Replacing a returning request when both fit
    # cannot buy a step of head start on a sequence-slot conflict.
    fresh.sort(key=adapter.order)
    spare = max(0, adapter.cap - len(selected))
    if len(fresh) <= spare:
        return []
    contenders = fresh[spare:]
    reserved = fresh[:spare]
    decisions = []
    use_new = []
    budget = max(0, getattr(adapter.scheduler, 'max_num_scheduled_tokens', 0)
                 - sum(max(1, r.num_prompt_tokens-r.num_computed_tokens) for r in chosen+returns+beneficiaries+reserved))
    # A swapped decode return gives back at least its one token in this step.
    eligible_returns = [r for r in chosen + returns if r.request_id in due
                        and cover.get(r.request_id) is not None
                        and not adapter.pauses[r.request_id].ranked_yield]
    eligible_returns.sort(key=lambda r: -cover[r.request_id])
    for old in eligible_returns:
        if not contenders:
            break
        new = contenders[0]
        remaining = max(1, new.num_prompt_tokens - new.num_computed_tokens)
        gain, cost = exchange_values(step, replenish, cover[old.request_id], remaining, budget + 1)
        wins = gain > cost + 1e-9
        decisions.append((old.request_id, new.request_id, wins))
        adapter.emit('client_buffer_recovery_rank', stage_id=adapter.stage, request_id=old.request_id,
                     newcomer=new.request_id, newcomer_source=adapter.source_id(new),
                     decision_monotonic=now, step_s=step, coverage_s=cover[old.request_id],
                     startup_gain_s=gain, incremental_stall_s=cost, chose_new= wins)
        if not wins:
            # Subsequent returns have even less coverage. This FIFO newcomer
            # cannot beat any of them at the current prompt budget.
            break
        contenders.pop(0)
        budget -= min(budget, remaining-1)
        adapter.pauses[old.request_id].ranked_yield = True
        (returns if old in returns else chosen).remove(old)
        if old.request_id in waiting_ids:
            blocked.add(old.request_id)
        if new.request_id in waiting_ids:
            blocked.discard(new.request_id)
            beneficiaries.append(new)
            use_new.append(new)
        else:
            chosen.append(new)
    if use_new:
        # Both native waiting queues must honour the decision. Keep the
        # original ready returns and all free-slot assignments serviceable.
        priority = {r.request_id for r in returns + beneficiaries + reserved}
        blocked.update(r.request_id for r in waiting if r.request_id not in priority)
        blocked.difference_update(r.request_id for r in reserved if r.request_id in waiting_ids)
    return decisions
