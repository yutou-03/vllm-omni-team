"""Opt-in bounded text delivery FIFO; receiver credit is not client receipt.

A separate producer drains model output while the consumer paces SSE delivery.
Original SSE records, token IDs, finish and usage records are preserved verbatim.
Canonical tokenizer counts are recomputed from visible prefixes (not summed
per-fragment encodings); exact character counts and SHA256 audit conservation.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

from vllm_omni.profiling.stage_queue_trace import emit_stage_event


def configuration():
    path = os.environ.get('VLLM_OMNI_TEXT_DELIVERY_CONFIG')
    if not path:
        return None
    c = json.loads(Path(path).read_text())
    rate = float(c.get('tokens_per_s', 4.8))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('text delivery rate must be finite and positive')
    if int(c.get('max_chars', 131072)) < 1 or int(c.get('max_events', 4096)) < 1:
        raise ValueError('text delivery FIFO must be bounded and nonempty')
    if int(c.get('boundary_reserve_tokens', 32)) < 1:
        raise ValueError('text inventory requires tokenizer boundary reserve')
    lead = float(c.get('delivery_lead_tokens', 1.))
    if not math.isfinite(lead) or not 0 <= lead <= 4:
        raise ValueError('delivery lead must be within zero to four tokens')
    return c


def content_of(packet):
    if not packet.startswith('data: ') or packet.strip() == 'data: [DONE]':
        return ''
    obj = json.loads(packet[6:])
    choices = obj.get('choices', [])
    if len(choices) > 1:
        raise ValueError('experimental text delivery supports n=1 only')
    return ''.join(c.get('delta', {}).get('content') or '' for c in choices)


async def pace_text_stream(source, *, tokenizer, request_id, config):
    queue = asyncio.Queue(maxsize=int(config.get('max_events', 4096)))
    rate = float(config.get('tokens_per_s', 4.8))
    max_chars = int(config.get('max_chars', 131072))
    reserve = int(config.get('boundary_reserve_tokens', 32))
    lead = float(config.get('delivery_lead_tokens', 1.))
    received = ''
    released = ''
    received_events = released_events = 0
    received_wire = hashlib.sha256()
    released_wire = hashlib.sha256()
    released_tokens = 0
    max_release = 1
    producer_done = False
    first = False
    sequence = 0
    sentinel = object()

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def snapshot(action, closed=False):
        nonlocal sequence
        sequence += 1
        emit_stage_event('stage_buffer_text_inventory', stage_id='frontend',
            request_id=request_id, logical_request_id=request_id,
            action=action, sequence=sequence, closed=closed,
            received_chars=len(received), released_chars=len(released),
            buffered_chars=len(received)-len(released),
            received_tokens=count(received), released_tokens=count(released),
            buffered_tokens=count(received[len(released):]),
            received_events=received_events, released_events=released_events,
            tokens_per_s=rate, max_release_tokens=max_release+lead,
            boundary_reserve_tokens=reserve, first_content_released=first,
            producer_done=producer_done,
            received_text_sha256=hashlib.sha256(received.encode()).hexdigest() if closed else None,
            released_text_sha256=hashlib.sha256(released.encode()).hexdigest() if closed else None,
            received_wire_sha256=received_wire.hexdigest() if closed else None,
            released_wire_sha256=released_wire.hexdigest() if closed else None,
            inventory_owner='frontend_host_memory', client_receipt_claim=False)

    async def produce():
        nonlocal received, received_events, producer_done, max_release
        try:
            async for packet in source:
                content = content_of(packet)
                if len(received)+len(content) > max_chars:
                    raise RuntimeError('text delivery character bound exceeded')
                # Reserve the next whole SSE burst before publishing its stock.
                max_release = max(max_release, count(content))
                await queue.put((packet, content))
                received += content
                received_events += 1
                received_wire.update(packet.encode())
                snapshot('received')
        except Exception as error:
            await queue.put(error)
        # Do not enqueue a sentinel from a cancellation finally block: a full
        # queue has no consumer after disconnect and would deadlock cleanup.
        producer_done = True
        await queue.put(sentinel)

    task = asyncio.create_task(produce())
    next_release = 0.
    completed = False
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                completed = True
                break
            if isinstance(item, Exception):
                raise item
            packet, content = item
            if content:
                await asyncio.sleep(max(0., next_release-time.monotonic()))
                old_count = released_tokens
                released += content
                released_tokens = count(released)
                mass = max(0, released_tokens-old_count)
                max_release = max(max_release, mass)
                # Absolute pacing avoids accumulating event-loop jitter. Limit
                # catch-up to the explicit lead; never drain a long-overdue FIFO
                # in an unbounded burst after a stalled event loop.
                next_release = max(time.monotonic()-lead/rate, next_release)+mass/rate
                first = True
            released_events += 1
            released_wire.update(packet.encode())
            snapshot('released')
            if packet.strip() == 'data: [DONE]':
                completed = True
            yield packet
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        close = getattr(source, 'aclose', None)
        if close is not None:
            await close()
        snapshot('completed' if completed else 'cancelled', closed=True)
