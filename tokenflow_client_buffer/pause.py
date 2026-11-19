"""Explicit lifecycle for a compute pause which retains the request's KV.

Gateway quota acknowledgement permits a return; only a native scheduled-token
result completes it. The wire epoch/state fields remain compatible with leases.py.
"""
from dataclasses import dataclass
from enum import Enum


class PausePhase(str, Enum):
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    RESUME_READY = "resume_ready"


@dataclass
class PausedRequest:
    epoch: float
    resume_by: float
    resume_guard_s: float
    releases_quota: bool = False
    beneficiary: str | None = None
    beneficiary_received: float | None = None
    phase: PausePhase = PausePhase.PAUSED
    ack_wait_since: float | None = None
    ranked_yield: bool = False

    @property
    def resume_requested(self):
        return self.phase != PausePhase.PAUSED

    def request_resume(self, now):
        if self.phase == PausePhase.PAUSED:
            self.phase = PausePhase.RESUME_REQUESTED
        if self.ack_wait_since is None:
            self.ack_wait_since = now

    def acknowledge(self, ack, *, charged):
        if ack.get("epoch") != self.epoch or ack.get("state") != "active" or not charged:
            return False
        self.phase = PausePhase.RESUME_READY
        self.ack_wait_since = None
        return True

    def lease(self, now):
        return dict(epoch=self.epoch,
                    state="resume_requested" if self.resume_requested else "paused",
                    resume_by=self.resume_by, resume_guard_s=self.resume_guard_s,
                    not_scheduled_at=now)


class PauseRegistry(dict[str, PausedRequest]):
    """One record per paused backend request, independent of its native queue."""

    def is_resume_requested(self, rid):
        pause = self.get(rid)
        return pause is not None and pause.resume_requested

    def releases_quota(self, rid):
        pause = self.get(rid)
        return pause is not None and pause.releases_quota

    @property
    def requested_resumes(self):
        return {rid for rid, pause in self.items() if pause.resume_requested}

    @property
    def quota_pauses(self):
        return {rid for rid, pause in self.items() if pause.releases_quota}

    def discard_terminal(self, requests):
        for rid in list(self):
            request = requests.get(rid)
            if request is None or getattr(request.status, "name", "").startswith("FINISHED"):
                self.pop(rid)
