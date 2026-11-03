"""Opt-in receiver-owned lookahead; payloads are held by reference on the CPU.

The receive cursor is independent of the existing model-visible release cursor.
One receive thread owns polling; adapter cleanup and scheduler registration use
its drain lock. Configuration is experimental and disabled by default.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
import sys
import time

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.profiling import stage_study
from vllm_omni.profiling.small_inventory import inventory_fields, emit_inventory_event


def resident_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return size + value.numel() * value.element_size()
    if isinstance(value, dict):
        return size + sum(resident_bytes(k, seen) + resident_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(resident_bytes(v, seen) for v in value)
    return size


@dataclass
class Stream:
    request: object
    external_id: str
    chunks: deque = field(default_factory=deque)
    received: int = 0
    released: int = 0
    bytes: int = 0
    units: int = 0
    ended: bool = False
    gate_until: float | None = None
    gate_done: bool = False
    next_release: float = 0.
    release_rate: float = 0.


class ReceiverBuffer:
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.stage = int(adapter.connector.stage_id)
        self.config = config
        self.streams = {}
        self.total_bytes = 0
        self.max_chunks = int(config.get("max_chunks", 128))
        self.max_units = int(config.get("max_units", 1024))
        self.max_bytes = int(config.get("max_bytes_per_request", 67108864))
        self.max_total_bytes = int(config.get("max_total_bytes", 536870912))
        if min(self.max_chunks, self.max_units, self.max_bytes, self.max_total_bytes) <= 0:
            raise ValueError("receiver buffer limits must be positive")

    @classmethod
    def configured(cls, adapter):
        path = os.environ.get("VLLM_OMNI_RECEIVER_BUFFER_CONFIG")
        if not path:
            return None
        with open(path) as f:
            config = json.load(f).get("stages", {}).get(str(adapter.connector.stage_id))
        return cls(adapter, config) if config is not None else None

    def register(self, request):
        rid = request.request_id
        external = getattr(request, "connector_req_id", None) or request.external_req_id
        existing = self.streams.get(rid)
        if existing is not None:
            if existing.request is not request or existing.external_id != external:
                raise RuntimeError("receiver request identity reused before cleanup")
            return
        stream = Stream(request, external)
        info = deserialize_additional_information(getattr(request, "additional_information", None))
        slo = float(info.get("meta", {}).get("sched_slo_ms", 0))
        if slo >= float(self.config.get("minimum_slo_ms", 60000)):
            stream.release_rate = float(self.config.get("release_units_per_s", 0.))
        self.streams[rid] = stream

    def cleanup(self, rid):
        stream = self.streams.pop(rid, None)
        if stream is not None:
            self.total_bytes -= stream.bytes
            self.event("receiver_buffer_cleanup", stream, discarded_chunks=len(stream.chunks),
                       discarded_units=stream.units, discarded_bytes=stream.bytes)

    def event(self, name, stream, **fields):
        stage_study.emit(name, stage_id=self.stage, request_id=stream.external_id,
                         info=deserialize_additional_information(getattr(stream.request, "additional_information", None)),
                         received_cursor=stream.received, released_cursor=stream.released,
                         buffered_chunks=len(stream.chunks), buffered_units=stream.units,
                         buffered_bytes=stream.bytes, total_buffered_bytes=self.total_bytes, **fields)

    def prefetch_tick(self):
        a = self.adapter
        progress = False
        with a._drain_state_lock:
            streams = list(self.streams.values())
        for stream in streams:
            rid = stream.request.request_id
            # Bound per-pass work so one fast producer cannot starve peers.
            for _ in range(8):
                with a._drain_state_lock:
                    if (self.streams.get(rid) is not stream or stream.ended
                            or len(stream.chunks) >= self.max_chunks or stream.units >= self.max_units
                            or stream.bytes >= self.max_bytes or self.total_bytes >= self.max_total_bytes):
                        break
                    cid = stream.received
                    a._recv_inflight += 1
                try:
                    result = a.connector.get(str(self.stage-1), str(self.stage),
                                             f"{stream.external_id}_{self.stage-1}_{cid}")
                    if result is None:
                        break
                    payload, size = result
                    if not isinstance(payload, dict) or not payload:
                        raise ValueError("received empty/invalid chunk")
                    fields = stage_study.text_entity_fields(payload, prefix="raw_") if self.stage == 1 else inventory_fields(payload)
                    if self.stage == 1:
                        rows = fields.get("raw_entity_embedding_rows_total") or 0
                        prompt = fields.get("raw_ids_prompt_count") or 0
                        units = max(0, rows - prompt) if cid == 0 else rows
                    else:
                        units = int(fields.get("new_frames", 0))
                    used = resident_bytes(payload)
                    with a._drain_state_lock:
                        if self.streams.get(rid) is not stream:
                            self.event("receiver_buffer_late_drop", stream, chunk_id=cid)
                            break
                        # One packet is read before its decoded size is known. On
                        # oversize, fail explicitly; never silently lose payload.
                        if (stream.bytes + used > self.max_bytes or self.total_bytes + used > self.max_total_bytes
                                or stream.units + units > self.max_units):
                            raise MemoryError("receiver buffer capacity exceeded by incoming packet")
                        stream.chunks.append((cid, payload, size, used, units))
                        stream.received += 1
                        stream.bytes += used
                        stream.units += units
                        self.total_bytes += used
                        stream.ended = bool(payload.get("meta", {}).get("finished", False))
                        if self.stage == 1:
                            stage_study.emit("text_payload_received", stage_id=1, request_id=stream.external_id,
                                             info=payload, chunk_id=cid, receiver_state="buffer_owned",
                                             size_bytes=size, **fields)
                        else:
                            emit_inventory_event("small_hold_chunk_received", stage_id=2,
                                request_id=stream.external_id, fields=fields,
                                received_chunk_id=cid, receiver_state="buffer_owned")
                        self.event("receiver_buffer_enqueued", stream, chunk_id=cid, units=units, payload_bytes=used)
                    progress = True
                except Exception as exc:
                    a._record_background_fault("prefetch", exc)
                    raise
                finally:
                    with a._drain_state_lock:
                        a._recv_inflight -= 1
        return bool(streams), progress

    def pop_for_release(self, request):
        # Caller holds the adapter drain lock.
        stream = self.streams.get(request.request_id)
        if stream is None or not stream.chunks:
            return None
        now = time.monotonic()
        if not stream.gate_done:
            after = int(self.config.get("build_after_chunks", 32 if self.stage == 1 else 1))
            delay = float(self.config.get("build_delay_s", 0.0))
            info = deserialize_additional_information(getattr(request, "additional_information", None))
            meta = info.get("meta", {})
            slo = float(meta.get("sched_slo_ms", 0))
            deadline = float(meta.get("sched_deadline_monotonic_s", 0))
            eligible = (slo >= float(self.config.get("minimum_slo_ms", 60000))
                        and deadline - now > delay + float(self.config.get("remaining_guard_s", 5.0)))
            if stream.released >= after and delay > 0 and eligible:
                target = int(self.config.get("build_target_units", 0))
                target_met = target > 0 and stream.units >= target
                if stream.gate_until is None and not target_met:
                    stream.gate_until = now + delay
                    self.event("receiver_buffer_build_started", stream, build_until=stream.gate_until)
                if not target_met and now < stream.gate_until and not stream.ended:
                    return None
                stream.gate_done = True
                self.event("receiver_buffer_build_finished", stream, upstream_finished=stream.ended,
                           inventory_target_met=target_met, build_target_units=target)
        # A deterministic release clock also bounds downstream burst intake.
        # EOS does not bypass pacing; doing so would invalidate pause credit.
        if stream.release_rate > 0 and now < stream.next_release:
            return None
        effective_rate = stream.release_rate
        deadline_guard = float(self.config.get("deadline_drain_guard_s", 0.))
        if effective_rate > 0 and deadline_guard > 0:
            info = deserialize_additional_information(getattr(request, "additional_information", None))
            deadline = float(info.get("meta", {}).get("sched_deadline_monotonic_s", 0))
            if deadline > 0:
                # Drain receiver-owned backlog before the final deadline. This
                # is not a fixed rate bound that can fund a Stage1 pause.
                remaining = max(.001, deadline - now - deadline_guard)
                effective_rate = max(effective_rate, stream.units / remaining)
        cid, payload, size, used, units = stream.chunks.popleft()
        if cid != stream.released:
            raise RuntimeError("receiver release sequence discontinuity")
        stream.released += 1
        stream.bytes -= used
        stream.units -= units
        self.total_bytes -= used
        if stream.release_rate > 0:
            stream.next_release = now + units / effective_rate
        self.event("receiver_buffer_released", stream, chunk_id=cid, units=units,
                   release_units_per_s=effective_rate,
                   configured_release_units_per_s=stream.release_rate,
                   fixed_release_bound=deadline_guard <= 0,
                   release_burst_units=units, next_release_monotonic_s=stream.next_release)
        return payload, size
