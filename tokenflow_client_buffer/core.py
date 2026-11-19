"""Pure control model. There is deliberately no inter-stage inventory input.

All times are seconds on a common host monotonic clock. Remote clients must
use a different timestamp protocol; the pilot rejects clock-domain mismatch.
"""
from __future__ import annotations

from collections import OrderedDict
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


def buffer_conservativeness(value):
    factor = number(value, "buffer conservativeness", positive=True)
    if factor < 1:
        raise ValueError("buffer conservativeness must be at least 1")
    return factor


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

    Consumption starts at the first usable receipt, with no startup threshold.
    Receiving and ticking are separate operations. Empty intervals after
    startup are stalls, but an EOS tail never becomes a stall. Units are
    canonical visible tokens or decoded media seconds, never codec tokens.
    """
    rate: float
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
        if self.started_at is None:
            self.started_at = now

    def finish(self, now: float):
        self.tick(now)
        self.eos = True
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
    admission_gamma: float | None = None

    @property
    def admission_bound(self):
        return min(self.gamma, self.admission_gamma if self.admission_gamma is not None else self.gamma)

    def valid(self, now, modality):
        return (self.modality == modality and math.isfinite(self.gamma) and self.gamma > 0
                and self.observed_at <= now <= self.valid_until and bool(self.profile_id)
                and (self.admission_gamma is None or
                     math.isfinite(self.admission_gamma) and self.admission_gamma > 0))


class CapacityEstimator:
    """Current delivered service estimate from complete execution windows.

    Every sustained output window supplies a demonstrated rate lower bound,
    regardless of FCFS/buffer policy or sequence count. A lower observation
    updates a still-valid capacity only with independent backend pressure evidence.
    After expiry, fresh output establishes a new lower bound without inheriting
    the stale calibration. Policy
    and request-length context are logged by callers; this is an online
    measurement, not a fitted physical peak or an extrapolation from GPU busy.
    """
    def __init__(self, modality: str, *, safety: float = 1.0, ttl_s: float = 30.0,
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
        self.estimate = CapacityEstimate(raw_capacity, now, now + self.ttl_s,
                                        self.modality, profile_id, "calibration",
                                        raw_capacity * self.safety)
        return self.estimate

    def observe(self, *, generated_units: float, duration_s: float, now: float,
                saturated: bool, profile_id: str, policy: str = "buffer"):
        units = number(generated_units, "generated units")
        duration = number(duration_s, "capacity duration", positive=True)
        # A chunk-free interval is not evidence of zero service capacity. The
        # caller retains its elapsed time and joins it with the next output.
        if units == 0 or duration < self.min_window_s:
            return self.estimate
        if not profile_id:
            raise ValueError("execution profile required")
        raw = units / duration
        old = (self.estimate if self.estimate and self.estimate.profile_id == profile_id
               and self.estimate.valid(now, self.modality) else None)
        if not saturated and old and raw < old.gamma:
            # Self-limited/idle windows cannot shrink the proven service bound.
            return self.estimate
        if not saturated and old is None:
            # An uncalibrated sustained observation still establishes a lower
            # bound; it does not establish a physical maximum.
            source = "observed_output_lower_bound"
        else:
            source = "saturated_execution_window" if saturated else "observed_output_lower_bound"
        if old:
            raw = (1 - self.alpha) * old.gamma + self.alpha * raw
        self.estimate = CapacityEstimate(raw, now, now + self.ttl_s, self.modality,
                                        profile_id, source, raw * self.safety)
        return self.estimate


@dataclass
class StreamRequest:
    request_id: str
    rate: float
    arrival_order: int
    state: str = "waiting"


class WorkingSet:
    """Single authoritative, atomic admission ledger for logical requests.

    Active W excludes confirmed buffer-safe pauses. Paused requests retain
    their identity and memory accounting; resumption restores their demand
    before execution. A capacity overload freezes NEW admissions, not restores.
    """
    def __init__(self, modality: str, *, max_inflight: int = 64, recovery_samples: int = 2,
                 native_execution_limit: int = 1, terminal_history_limit: int = 4096):
        rate_field(modality)
        if max_inflight < 1 or recovery_samples < 1 or native_execution_limit < 1:
            raise ValueError("invalid working set bounds")
        if type(terminal_history_limit) is not int or terminal_history_limit < 1:
            raise ValueError("positive terminal history limit required")
        self.modality = modality
        self.max_inflight, self.recovery_samples = max_inflight, recovery_samples
        self.requests: dict[str, StreamRequest] = {}
        self.completed_ids = OrderedDict()
        self.terminal_history_limit = terminal_history_limit
        self._arrival_sequence = 0
        self.mode, self.reason = "fcfs", "capacity_unknown"
        self._safe_samples = 0
        self._last_estimate = None
        self._last_safe_demand = None
        self._lock = threading.RLock()
        self.admission_guard = None
        # Only unknown-capacity fallback uses this bound, replenishing native
        # slots as requests finish. Buffer mode admits by rate, never by |W|.
        self.native_execution_limit = native_execution_limit

    @property
    def members(self):
        return [r for r in self.requests.values() if r.state == "admitted"]


    @property
    def outstanding(self):
        return [r for r in self.requests.values() if r.state in {"admitted", "paused"}]

    @property
    def demand(self):
        return math.fsum(r.rate for r in self.members)



    def pause(self, rid: str):
        with self._lock:
            r = self.requests[rid]
            if r.state == "admitted":
                r.state = "paused"
                return True
            return False

    def resume(self, rid: str):
        with self._lock:
            r = self.requests[rid]
            if r.state == "paused":
                r.state = "admitted"
                return True
            return False

    def add(self, rid: str, rate: float):
        with self._lock:
            if self.has_request_id(rid):
                raise ValueError("duplicate logical request")
            self.requests[rid] = StreamRequest(rid, number(rate, "rate", positive=True), self._arrival_sequence)
            self._arrival_sequence += 1

    def has_request_id(self, rid: str):
        return rid in self.requests or rid in self.completed_ids

    def complete(self, rid: str):
        with self._lock:
            request = self.requests.pop(rid, None)
            if request is None:
                return
            request.state = "finished"
            # Full history belongs in the event journal, not the hot ledger.
            self.completed_ids[rid] = None
            while len(self.completed_ids) > self.terminal_history_limit:
                self.completed_ids.popitem(last=False)

    def update(self, estimate: CapacityEstimate | None, now: float):
        with self._lock:
            if estimate is None or not estimate.valid(now, self.modality):
                self.mode, self.reason, self._safe_samples = "fcfs", "capacity_unknown", 0
                self._last_estimate = None
                self._last_safe_demand = None
            elif self.demand > estimate.gamma + 1e-9:
                self.mode, self.reason, self._safe_samples = "fcfs", "overload", 0
                self._last_estimate = estimate
                self._last_safe_demand = None
            else:
                # Completion and acknowledged quota exchange can restore
                # feasibility without a new throughput sample. Do not retain
                # a stale overload reason merely because Gamma is unchanged.
                if estimate != self._last_estimate or self.demand != self._last_safe_demand:
                    self._safe_samples += 1
                    self._last_estimate = estimate
                    self._last_safe_demand = self.demand
                if self.mode == "buffer" or self._safe_samples >= self.recovery_samples:
                    self.mode, self.reason = "buffer", "capacity_feasible"
            return self.mode

    def admit_next(self, estimate: CapacityEstimate | None, now: float, *, native=False):
        with self._lock:
            self.update(estimate, now)
            pending = [r for r in self.requests.values() if r.state == "waiting"]
            if not pending or len(self.outstanding) >= self.max_inflight:
                return None
            candidate = pending[0]
            # Overload drains the existing working set under native FCFS;
            # waiting requests cannot expand it while D still exceeds Gamma.
            if not native and self.reason == "overload" and self.members:
                return None
            valid = estimate is not None and estimate.valid(now, self.modality)
            if not native and not valid:
                # Keep a rolling native FCFS window while capacity is unknown.
                # Expiry neither flushes the entire queue nor requires the last
                # member of a batch to finish before replenishing a free slot.
                # Paused owners must recover first; they still retain KV/state.
                if any(r.state == "paused" for r in self.outstanding):
                    return None
                if len(self.outstanding) >= self.native_execution_limit:
                    return None
            if not native and valid and self.demand + candidate.rate > estimate.admission_bound + 1e-9:
                if self.members:
                    return None  # preserve FIFO, no skipping smaller requests
                # A single demand exceeds capacity: service via FCFS, not deadlock.
                self.mode, self.reason = "fcfs", "single_request_overload"
            if (not native and self.mode == "buffer" and self.admission_guard is not None
                    and not self.admission_guard(candidate, now)):
                return None
            candidate.state = "admitted"
            self.update(estimate, now)
            return candidate.request_id

    def snapshot(self, estimate: CapacityEstimate | None, now: float, *, native=False):
        with self._lock:
            self.update(estimate, now)
            return dict(mode="fcfs" if native else self.mode,
                        reason="baseline" if native else self.reason,
                        demand=self.demand, gamma=estimate.gamma if estimate and estimate.valid(now, self.modality) else None,
                        admission_gamma=estimate.admission_bound if estimate and estimate.valid(now, self.modality) else None,
                        members={r.request_id: asdict(r) for r in self.members},
                        paused_members={r.request_id: asdict(r) for r in self.outstanding if r.state == "paused"},
                        waiting=sum(r.state == "waiting" for r in self.requests.values()))
