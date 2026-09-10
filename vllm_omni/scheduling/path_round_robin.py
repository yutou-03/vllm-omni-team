"""Admission-only path WRR; successful admissions, not scheduler ticks, count.

The native scheduler may inspect/pop a request without admitting it (blocked
input, LoRA limits, asynchronous KV load). Reconcile the previous selection
against the running set before choosing another head and after schedule().
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Mapping


class PathRoundRobin:
    def __init__(self, text: int, audio: int, phase: int = 0) -> None:
        if any(type(x) is not int or x <= 0 for x in (text, audio)):
            raise ValueError("path RR weights must be positive integers")
        if type(phase) is not int or not 0 <= phase < text + audio:
            raise ValueError("path RR phase must lie within the cycle")
        self.text = text
        self.audio = audio
        self.cursor = phase
        self.admissions = 0
        self.pending: tuple[Any, dict] | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None):
        source = os.environ if environ is None else environ
        raw = source.get("VLLM_OMNI_PATH_RR")
        if raw is None:
            if "VLLM_OMNI_PATH_RR_PHASE" in source:
                raise ValueError("path RR phase requires path RR weights")
            return None
        if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", raw) is None:
            raise ValueError("VLLM_OMNI_PATH_RR must be N:M")
        text, audio = map(int, raw.split(":"))
        phase = source.get("VLLM_OMNI_PATH_RR_PHASE", "0")
        if re.fullmatch(r"0|[1-9][0-9]*", phase) is None:
            raise ValueError("VLLM_OMNI_PATH_RR_PHASE must be nonnegative")
        return cls(text, audio, int(phase))

    @property
    def target(self) -> str:
        return "text" if self.cursor < self.text else "audio"

    def key(self, request, *, path: str, ready_order: int):
        if path not in {"text", "audio"}:
            raise ValueError(f"unknown path RR request class: {path!r}")
        return (path != self.target, ready_order, str(request.request_id))

    def reconcile(self, running, emit: Callable[[dict], None]) -> None:
        if self.pending is None:
            return
        request, record = self.pending
        self.pending = None
        admitted = any(item is request for item in running)
        if admitted:
            self.admissions += 1
            self.cursor = (self.cursor + 1) % (self.text + self.audio)
        emit({**record, "admitted": admitted,
              "cursor_after": self.cursor, "admissions": self.admissions})

    def select(self, queues, *, running, key, path, emit):
        self.reconcile(running, emit)
        nonempty = [queue for queue in queues if queue]
        if not nonempty:
            return None
        queue = min(nonempty, key=lambda q: key(q.peek_request()))
        request = queue.peek_request()
        counts = {"text": 0, "audio": 0}
        # Counts describe visible candidates, not proven KV-admissible requests.
        for candidate_queue in nonempty:
            for candidate in candidate_queue:
                counts[path(candidate)] += 1
        actual = path(request)
        self.pending = (request, {
            "request_id": str(request.request_id),
            "weights": [self.text, self.audio], "cursor_before": self.cursor,
            "target_path": self.target, "actual_path": actual,
            "borrowed": actual != self.target,
            "visible_candidate_counts": counts,
            "running_count_before": len(running),
        })
        return queue
