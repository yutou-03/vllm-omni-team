"""Hide quota-waiting target-stage requests for one native scheduling step.

Only source IDs explicitly registered by this output gateway are gated. Audio
requests passing through the text/Thinker adapter are unchanged. W and P stay
visible to the policy scheduler; this helper does not authorize P execution or
replace acknowledged restoration. No intermediate inventory supplies credit.
"""
from contextlib import contextmanager


def _replace_waiting(queue, requests):
    if isinstance(queue, list):
        queue[:] = requests
        return
    while queue:
        queue.pop_request()
    for request in requests:
        queue.add_request(request)


def _restore_hidden(current, original, hidden):
    """Linear anchor merge, preserving native order among visible requests.

    Old hidden entries go before their next surviving old anchor. If none
    survives, they follow their preceding old anchor, or all current entries.
    New native-preempted entries keep their existing relative precedence.
    """
    present = {id(r) for r in current}
    hidden_ids = {id(r) for r in hidden} - present
    if not hidden_ids:
        return list(current)
    following = {}
    anchor = None
    for request in reversed(original):
        rid = id(request)
        if rid in hidden_ids:
            following[rid] = anchor
        if rid in present:
            anchor = rid
    before, after, hidden_bucket = {}, {}, {}
    tail = []
    preceding = None
    for request in original:
        rid = id(request)
        if rid in present:
            preceding = rid
        elif rid in hidden_ids:
            next_anchor = following[rid]
            if next_anchor is not None:
                bucket = before.setdefault(next_anchor, [])
            elif preceding in present:
                bucket = after.setdefault(preceding, [])
            elif preceding in hidden_bucket:
                bucket = hidden_bucket[preceding]
            else:
                bucket = tail
            bucket.append(request)
            hidden_bucket[rid] = bucket
            preceding = rid
    result = []
    for request in current:
        result.extend(before.get(id(request), ()))
        result.append(request)
        result.extend(after.get(id(request), ()))
    result.extend(tail)
    return result


@contextmanager
def engine_queue_admission_context(scheduler, snapshot, source_id, *,
                                   hide_all_waiting=False, blocked_waiting_ids=()):
    """Mask pending requests in running, waiting and optional skipped_waiting.

    The caller supplies a validated fresh snapshot and applies normal P
    pause/restoration selection inside this scope. Explicit blocked_waiting_ids
    and hide_all_waiting work even when engine admission is disabled, to prevent
    native refill while a returning request awaits its quota acknowledgement.
    Mask IDs are physical request IDs for excluding policy-held work from
    independent capacity-pressure evidence. Disabled mode is a strict no-op.
    """
    empty = dict(hidden_running_ids=set(), hidden_waiting_ids=set(),
                 hidden_source_ids=set(), excluded_ids=set())
    enabled = snapshot.get('engine_queue_admission') is True
    blocked_waiting_ids = set(blocked_waiting_ids)
    if not enabled and not hide_all_waiting and not blocked_waiting_ids:
        yield empty
        return
    registered = set(snapshot.get('registered_ids', ())) if enabled else set()
    allowed = set(snapshot.get('members', ())) | set(snapshot.get('paused_members', ()))
    # Only explicit, fresh control-plane handoffs bypass the rate gate. Native
    # max_num_seqs and KV allocation still decide actual execution admission.
    if snapshot.get('native_fcfs_admission') and snapshot.get('mode') == 'fcfs':
        allowed.update(snapshot.get('native_fcfs_ids', ()))
    elif (snapshot.get('mode') == 'buffer'
          and snapshot.get('native_fcfs_admission') is False):
        # Explicit takeover of previously handed-off engine waiters. They stay
        # in native FIFO, selectable as safe-pause beneficiaries. Execution/KV
        # allocation and restoration barriers still apply below/inside scope.
        allowed.update(snapshot.get('buffer_reclaimed_ids', ()))

    def blocked(request):
        source = source_id(request)
        return source in registered and source not in allowed

    running = list(scheduler.running)
    hidden_running = [r for r in running if blocked(r)]
    hidden_running_ids = {r.request_id for r in hidden_running}
    queues = []
    seen_queues = set()
    for name in ('waiting', 'skipped_waiting'):
        queue = getattr(scheduler, name, None)
        if queue is None or id(queue) in seen_queues:
            continue
        seen_queues.add(id(queue))
        original = list(queue)
        hidden = [r for r in original if hide_all_waiting
                  or r.request_id in blocked_waiting_ids or blocked(r)]
        queues.append((name, original, hidden))
    hidden_waiting = [r for _, _, hidden in queues for r in hidden]
    hidden_waiting_ids = {r.request_id for r in hidden_waiting}
    mask = dict(hidden_running_ids=hidden_running_ids,
                hidden_waiting_ids=hidden_waiting_ids,
                hidden_source_ids={source_id(r) for r in hidden_running + hidden_waiting},
                excluded_ids=hidden_running_ids | hidden_waiting_ids)
    if not hidden_running and not hidden_waiting:
        yield mask
        return
    if hidden_running:
        scheduler.running[:] = [r for r in running if r.request_id not in hidden_running_ids]
    for name, original, hidden in queues:
        if hidden:
            hidden_ids = {id(r) for r in hidden}
            _replace_waiting(getattr(scheduler, name), [r for r in original if id(r) not in hidden_ids])
    try:
        yield mask
    finally:
        current_running = list(scheduler.running)
        current_queues = {name:list(getattr(scheduler, name)) for name, _, _ in queues}
        current_ids = {id(r) for r in current_running}
        current_ids.update(id(r) for requests in current_queues.values() for r in requests)

        def alive(request):
            if scheduler.requests.get(request.request_id) is not request:
                return False
            status = getattr(request.status, 'name', str(request.status))
            return not status.startswith('FINISHED')

        restore_running = [r for r in hidden_running if id(r) not in current_ids and alive(r)
                           and getattr(r.status, 'name', str(r.status)) == 'RUNNING']
        if restore_running:
            scheduler.running[:] = _restore_hidden(current_running, running, restore_running)
        for name, original, hidden in queues:
            restore = [r for r in hidden if id(r) not in current_ids and alive(r)
                       and getattr(r.status, 'name', str(r.status)) != 'RUNNING']
            if restore:
                _replace_waiting(getattr(scheduler, name),
                                 _restore_hidden(current_queues[name], original, restore))
