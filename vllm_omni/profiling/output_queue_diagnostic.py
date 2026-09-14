"""Opt-in Stage2 receive-queue diagnosis; never enabled by default.

observe preserves FIFO semantics. drop_stats omits ONLY pure telemetry after
socket reception/decoding, not data, completion, exception or control messages.
Both modes use identical instrumentation. Engine timestamp is construction
time, NOT an instrumented send time. No upstream vLLM source patch is needed.
"""
from __future__ import annotations

import asyncio
from collections import Counter, deque
import time


def is_pure_statistics(item) -> bool:
    fields = getattr(type(item), "__struct_fields__", ())
    if not fields or getattr(item, "outputs", None) or getattr(item, "scheduler_stats", None) is None:
        return False
    ordinary = {"engine_index", "outputs", "scheduler_stats", "timestamp"}
    for name in fields:
        if name in ordinary:
            continue
        value = getattr(item, name)
        # Fail closed for unknown nonempty fields, including wave/control 0.
        if value is not None and value != [] and value != {} and value != set():
            return False
    return True


class DiagnosticOutputQueue(asyncio.Queue):
    def __init__(self, *, stage_id, mode, emit, clock=time.monotonic):
        if mode not in ("observe", "drop_stats"):
            raise ValueError(f"unknown output queue diagnostic mode: {mode}")
        super().__init__()
        self.stage_id = stage_id
        self.mode = mode
        self.emit = emit
        self.clock = clock
        self.metadata = deque()
        self.counts = Counter()
        self.depths = Counter()
        self.sequence = 0
        self.peak_depth = 0
        self.last_summary = float("-inf")

    def _snapshot(self):
        return dict(mode=self.mode, queue_depth=self.qsize(),
                    queue_depths=dict(self.depths), counts=dict(self.counts),
                    peak_depth=self.peak_depth)

    def put_nowait(self, item):
        now = self.clock()
        kind = "stats" if is_pure_statistics(item) else "data" if getattr(item, "outputs", None) else "control"
        self.sequence += 1
        self.counts[f"received_{kind}"] += 1
        if kind == "stats" and self.mode == "drop_stats":
            self.counts["dropped_stats"] += 1
        else:
            before = self.qsize()
            ahead = dict(self.depths)
            super().put_nowait(item)
            self.metadata.append((self.sequence, now, kind))
            self.depths[kind] += 1
            self.peak_depth = max(self.peak_depth, self.qsize())
            if kind == "data":
                self.emit("output_queue_enqueued", stage_id=self.stage_id,
                          output_message_seq=self.sequence, received_monotonic=now,
                          engine_created_monotonic=getattr(item, "timestamp", None),
                          request_ids=[o.request_id for o in item.outputs],
                          num_outputs=len(item.outputs), depth_before=before,
                          messages_ahead=ahead, **self._snapshot())
        if now - self.last_summary >= 1.0:
            self.last_summary = now
            self.emit("output_queue_summary", stage_id=self.stage_id,
                      observed_monotonic=now, **self._snapshot())

    def get_nowait(self):
        item = super().get_nowait()
        now = self.clock()
        sequence, entered, kind = self.metadata.popleft()
        self.depths[kind] -= 1
        self.counts[f"dequeued_{kind}"] += 1
        if kind == "data":
            self.emit("output_queue_dequeued", stage_id=self.stage_id,
                      output_message_seq=sequence, received_monotonic=entered,
                      dequeued_monotonic=now, queue_wait_s=now-entered,
                      engine_created_monotonic=getattr(item, "timestamp", None),
                      request_ids=[o.request_id for o in item.outputs],
                      num_outputs=len(item.outputs), **self._snapshot())
        return item
