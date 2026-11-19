"""Benchmark HTTP sends are concurrent; the server queue enforces ordering."""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

class OrderedBackend:
    def __init__(self, trace_path, timeout=30., event=None):
        self.timeout, self.event = timeout, event
        self.condition = asyncio.Condition()
        self.next_register = 0
        self.group = 'tokenflow_benchmark:' + uuid.uuid4().hex

    async def register(self, rank):
        async with self.condition:
            await asyncio.wait_for(self.condition.wait_for(lambda: rank == self.next_register), self.timeout)
            self.next_register += 1
            self.condition.notify_all()

    @asynccontextmanager
    async def post(self, session, url, *, payload, rid, rank):
        payload = dict(payload)
        payload['omni_scheduling'] = dict(payload['omni_scheduling'], service_class=self.group, ingress_order=rank)
        if self.event:
            self.event('benchmark_engine_submit', request_id=rid, rank=rank,
                       submitted_at=time.monotonic(), protocol='server_queue_order_no_ack_wait_v4')
        async with session.post(url, json=payload) as response:
            yield response
