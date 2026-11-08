"""Pure control model. There is deliberately no inter-stage inventory input.

All times are seconds on a common host monotonic clock. Remote clients must
use a different timestamp protocol; the pilot rejects clock-domain mismatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
from typing import Any


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid {name}: {value}")
    return value


def rate_field(modality: str) -> str:
    if modality not in {"text", "audio"}:
        raise ValueError("one output modality required: text or audio")
    return "text_consume_rate_tps" if modality == "text" else "audio_consume_rate"


def stream_contract(payload: dict, modality: str) -> tuple[str, float]:
    metadata = payload.get("omni_scheduling", {})
    if metadata.get("schema_version") != 2 or metadata.get("request_path") != modality:
        raise ValueError("stream speed must travel in schema-v2 omni_scheduling")
    rid = metadata.get("source_request_id")
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError("source_request_id required")
    if payload.get("stream") is not True:
        raise ValueError("stream=true required")
    return rid, number(metadata[rate_field(modality)], "stream rate", positive=True)


@dataclass
class ConsumerBuffer:
    """Fluid consumer clock shared by text and decoded PCM.

    Receiving and ticking are separate operations. Empty intervals after
    startup are stalls, but an EOS tail never becomes a stall. Units are
    canonical visible tokens or decoded media seconds, never codec tokens.
    """
    rate: float
    startup_units: float = 0.0
    received: float = 0.0
    consumed: float = 0.0
    last_time: float | None = None
    first_ready: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    eos: bool = False
    stalled_s: float = 0.0
    max_stall_s: float = 0.0
    stall_count: int = 0
    current_stall_s: float = 0.0
    sequence: int = 0

    def __post_init__(self):
        self.rate = number(self.rate, "rate", positive=True)
        self.startup_units = number(self.startup_units, "startup units")

    @property
    def buffered(self):
        return max(0.0, self.received - self.consumed)

    def tick(self, now: float):
        now = number(now, "clock")
        if self.last_time is not None and now < self.last_time:
            raise ValueError("consumer clock went backwards")
        if self.started_at is not None and self.completed_at is None:
            dt = now - self.last_time
            before = self.buffered
            used = min(before, dt * self.rate)
            self.consumed += used
            empty_s = max(0.0, dt - used / self.rate)
            if self.eos and self.buffered <= 1e-9:
                self.consumed = self.received
                self.completed_at = self.last_time + used / self.rate
            elif empty_s > 1e-9:
                if not self.current_stall_s:
                    self.stall_count += 1
                self.current_stall_s += empty_s
                self.stalled_s += empty_s
                self.max_stall_s = max(self.max_stall_s, self.current_stall_s)
        self.last_time = now

    def receive(self, units: float, now: float):
        units = number(units, "received units", positive=True)
        if self.eos:
            raise ValueError("content received after EOS")
        self.tick(now)
        self.received += units
        self.current_stall_s = 0.0
        if self.first_ready is None:
            self.first_ready = now
        if self.started_at is None and self.buffered >= self.startup_units:
            self.started_at = now

    def finish(self, now: float):
        self.tick(now)
        self.eos = True
        if self.started_at is None:
            if self.buffered:
                self.started_at = now  # short stream below initial threshold
            else:
                self.completed_at = now
        if not self.buffered and self.completed_at is None:
            self.completed_at = now

    def feedback(self, now: float, *, request_id: str, session: str, modality: str,
                 clock_domain: str) -> dict:
        self.tick(now)
        self.sequence += 1
        return dict(schema_version=1, source="client_consumption_buffer",
                    request_id=request_id, session=session, modality=modality,
                    rate=self.rate, received=self.received, consumed=self.consumed,
                    observed_at=now, sequence=self.sequence, clock_domain=clock_domain,
                    started=self.started_at is not None, eos=self.eos,
                    complete=self.completed_at is not None)


@dataclass(frozen=True)
class ClientFeedback:
    request_id: str
    session: str
    modality: str
    rate: float
    received: float
    consumed: float
    observed_at: float
    sequence: int
    started: bool
    complete: bool

    @classmethod
    def parse(cls, data: dict, *, now: float, session: str, modality: str,
              clock_domain: str, rate: float, previous: ClientFeedback | None = None):
        if data.get("schema_version") != 1 or data.get("source") != "client_consumption_buffer":
            raise ValueError("only client consumption feedback can fund a pause")
        for key, expected in (("session", session), ("modality", modality),
                              ("clock_domain", clock_domain)):
            if data.get(key) != expected:
                raise ValueError(f"feedback {key} mismatch")
        received = number(data["received"], "received")
        consumed = number(data["consumed"], "consumed")
        observed = number(data["observed_at"], "observation time")
        actual_rate = number(data["rate"], "feedback rate", positive=True)
        seq = data["sequence"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("invalid feedback sequence")
        if actual_rate != rate or consumed > received + 1e-8 or observed > now:
            raise ValueError("feedback rate, conservation, or clock invalid")
        if not isinstance(data.get("started"), bool) or not isinstance(data.get("complete"), bool):
            raise ValueError("feedback state must be boolean")
        if previous and (data["request_id"] != previous.request_id or seq <= previous.sequence
                         or observed < previous.observed_at
                         or received < previous.received or consumed < previous.consumed
                         or (previous.started and not data["started"])
                         or (previous.complete and not data["complete"])):
            raise ValueError("stale, reordered, or nonconserving feedback")
        return cls(data["request_id"], session, modality, actual_rate, received, consumed,
                   observed, seq, data["started"], data["complete"])

    def coverage(self, now: float, max_age_s: float, reserve_units: float = 0.0):
        age = now - self.observed_at
        if not self.started or self.complete or age < 0 or age > max_age_s:
            return None
        return max(0.0, (self.received - self.consumed - reserve_units) / self.rate - age)


@dataclass(frozen=True)
class CapacityEstimate:
    gamma: float
    observed_at: float
    valid_until: float
    modality: str
    profile_id: str
    source: str

    def valid(self, now, modality):
        return (self.modality == modality and math.isfinite(self.gamma) and self.gamma > 0
                and self.observed_at <= now <= self.valid_until and bool(self.profile_id))


class CapacityEstimator:
    """Conservative sustained-window estimate, never fitted to client pacing.

    A pre-calibrated profile can seed this estimator. Saturated measurement
    windows update it downward immediately and upward by EWMA. Demand-limited
    windows cannot lower or refresh its evidence. The caller must measure
    final generated output and label saturation from execution, not arrivals.
    """
    def __init__(self, modality: str, *, safety: float = .9, ttl_s: float = 30.0,
                 min_window_s: float = 2.0, alpha: float = .2):
        rate_field(modality)
        if not 0 < safety <= 1 or not 0 < alpha <= 1:
            raise ValueError("invalid estimator factors")
        self.modality, self.safety, self.alpha = modality, safety, alpha
        self.ttl_s = number(ttl_s, "capacity ttl", positive=True)
        self.min_window_s = number(min_window_s, "window", positive=True)
        self.estimate: CapacityEstimate | None = None

    def seed(self, raw_capacity: float, now: float, profile_id: str):
        raw_capacity = number(raw_capacity, "capacity", positive=True)
        if not profile_id:
            raise ValueError("calibration provenance required")
        self.estimate = CapacityEstimate(raw_capacity * self.safety, now, now + self.ttl_s,
                                        self.modality, profile_id, "calibration")
        return self.estimate

    def observe(self, *, generated_units: float, duration_s: float, now: float,
                saturated: bool, profile_id: str):
        units = number(generated_units, "generated units")
        duration = number(duration_s, "capacity duration", positive=True)
        if not saturated or duration < self.min_window_s:
            return self.estimate
        if not profile_id:
            raise ValueError("execution profile required")
        raw = self.safety * units / duration
        if self.estimate and self.estimate.profile_id == profile_id:
            old = self.estimate.gamma
            raw = min(raw, (1 - self.alpha) * old + self.alpha * raw)
        self.estimate = CapacityEstimate(raw, now, now + self.ttl_s, self.modality,
                                        profile_id, "saturated_execution_window")
        return self.estimate


@dataclass
class StreamRequest:
    request_id: str
    rate: float
    arrival_order: int
    state: str = "waiting"


class WorkingSet:
    """Single authoritative, atomic admission ledger for logical requests.

    W persists across stage waits and scheduler pauses. Only terminal
    generation/cancellation releases demand. A fallback does not delete W.
    """
    def __init__(self, modality: str, *, max_inflight: int = 64, recovery_samples: int = 2):
        rate_field(modality)
        if max_inflight < 1 or recovery_samples < 1:
            raise ValueError("invalid working set bounds")
        self.modality = modality
        self.max_inflight, self.recovery_samples = max_inflight, recovery_samples
        self.requests: dict[str, StreamRequest] = {}
        self.mode, self.reason = "fcfs", "capacity_unknown"
        self._safe_samples = 0
        self._last_estimate = None
        self._lock = threading.RLock()

    @property
    def members(self):
        return [r for r in self.requests.values() if r.state == "admitted"]

    @property
    def demand(self):
        return math.fsum(r.rate for r in self.members)

    def add(self, rid: str, rate: float):
        with self._lock:
            if rid in self.requests:
                raise ValueError("duplicate logical request")
            self.requests[rid] = StreamRequest(rid, number(rate, "rate", positive=True), len(self.requests))

    def complete(self, rid: str):
        with self._lock:
            self.requests[rid].state = "finished"

    def update(self, estimate: CapacityEstimate | None, now: float):
        with self._lock:
            if estimate is None or not estimate.valid(now, self.modality):
                self.mode, self.reason, self._safe_samples = "fcfs", "capacity_unknown", 0
                self._last_estimate = None
            elif self.demand > estimate.gamma + 1e-9:
                self.mode, self.reason, self._safe_samples = "fcfs", "overload", 0
                self._last_estimate = estimate
            else:
                if estimate != self._last_estimate:
                    self._safe_samples += 1
                    self._last_estimate = estimate
                if self.mode == "buffer" or self._safe_samples >= self.recovery_samples:
                    self.mode, self.reason = "buffer", "capacity_feasible"
            return self.mode

    def admit_next(self, estimate: CapacityEstimate | None, now: float, *, native=False):
        with self._lock:
            self.update(estimate, now)
            pending = [r for r in self.requests.values() if r.state == "waiting"]
            if not pending or len(self.members) >= self.max_inflight:
                return None
            candidate = pending[0]
            # Overload drains the existing working set under native FCFS;
            # waiting requests cannot expand it while D still exceeds Gamma.
            if not native and self.reason == "overload" and self.members:
                return None
            if not native and self.mode == "buffer" and self.demand + candidate.rate > estimate.gamma + 1e-9:
                if self.members:
                    return None  # preserve FIFO, no skipping smaller requests
                # A single demand exceeds capacity: service via FCFS, not deadlock.
                self.mode, self.reason = "fcfs", "single_request_overload"
            candidate.state = "admitted"
            self.update(estimate, now)
            return candidate.request_id

    def snapshot(self, estimate: CapacityEstimate | None, now: float, *, native=False):
        with self._lock:
            self.update(estimate, now)
            return dict(mode="fcfs" if native else self.mode,
                        reason="baseline" if native else self.reason,
                        demand=self.demand, gamma=estimate.gamma if estimate and estimate.valid(now, self.modality) else None,
                        members={r.request_id: asdict(r) for r in self.members},
                        waiting=sum(r.state == "waiting" for r in self.requests.values()))
