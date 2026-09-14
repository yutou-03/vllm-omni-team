"""Preserve output FIFO while compacting consecutive replaceable telemetry.

Only Stage2 installs this queue by default. No producer, data, control or delta
statistics are discarded. Unknown schemas fail closed to ordinary FIFO.
"""
from __future__ import annotations

import asyncio
from dataclasses import fields
import math

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats

from vllm_omni.engine import OmniEngineCoreOutputs

_GAUGES = frozenset({"num_running_reqs", "num_waiting_reqs", "num_skipped_waiting_reqs", "kv_cache_usage"})
_OTHER_STATS = frozenset({
    "step_counter", "current_wave", "prefix_cache_stats", "connector_prefix_cache_stats",
    "kv_cache_eviction_events", "spec_decoding_stats", "kv_connector_stats",
    "waiting_lora_adapters", "running_lora_adapters", "cudagraph_stats", "perf_stats",
})
_ENVELOPE = frozenset({
    "engine_index", "outputs", "scheduler_stats", "timestamp", "utility_output",
    "finished_requests", "wave_complete", "start_wave", "stage_preempted_request_ids",
    "stage_preempted_connector_req_ids", "stage_preemption_causes", "stage_preemption_progress",
    "stage_preemption_rule_ids", "stage_preemption_recovery_modes",
})
_ORDINARY = frozenset({"engine_index", "outputs", "scheduler_stats", "timestamp"})
_EMPTY_STATS = SchedulerStats()
_SCHEMA_KNOWN = {f.name for f in fields(SchedulerStats)} == _GAUGES | _OTHER_STATS


def snapshot_key(item):
    """Return a stream/time key only for known, delta-free status snapshots."""
    if type(item) not in (EngineCoreOutputs, OmniEngineCoreOutputs) or item.outputs:
        return None
    if not set(type(item).__struct_fields__) <= _ENVELOPE:
        return None
    for name in type(item).__struct_fields__:
        if name in _ORDINARY:
            continue
        value = getattr(item, name)
        if value is not None and not (type(value) in (list, dict, set) and not value):
            return None  # Includes meaningful wave/control zero and False.
    stats = item.scheduler_stats
    if not _SCHEMA_KNOWN or type(stats) is not SchedulerStats:
        return None
    if set(vars(stats)) != _GAUGES | _OTHER_STATS:
        return None
    # Cache hit/reset counters and eviction/speculation/performance events
    # are NOT last-value gauges. Never overwrite non-default instances.
    try:
        if any(getattr(stats, name) != getattr(_EMPTY_STATS, name) for name in _OTHER_STATS):
            return None
    except (TypeError, ValueError, RuntimeError):
        return None
    if (type(item.engine_index) is not int or type(item.timestamp) not in (int, float)
            or not math.isfinite(item.timestamp)):
        return None
    return item.engine_index, item.timestamp


class SnapshotOutputQueue(asyncio.Queue):
    """Unbounded asyncio FIFO with O(1) adjacent-snapshot replacement.

    Replacing a queued item does not add an unfinished task or wake another
    waiter. Data/control/delta messages form barriers; engine changes and
    backwards timestamps also form barriers. Async get/put and join retain
    their usual Queue semantics. No diagnostic tracing in the default path.
    """

    def __init__(self):
        super().__init__()
        self.coalesced_snapshots = 0

    def put_nowait(self, item):
        key = snapshot_key(item)
        if key is not None and self._queue:
            previous = snapshot_key(self._queue[-1])
            if previous is not None and previous[0] == key[0] and previous[1] <= key[1]:
                self._queue[-1] = item
                self.coalesced_snapshots += 1
                return
        super().put_nowait(item)
