"""Server-side unread output. Only final text tokens or playable PCM fund pauses."""
from dataclasses import dataclass

from .core import ClientFeedback


@dataclass
class VirtualOutput:
    source_id: str
    rate: float
    first_at: float
    produced_at: float
    received: float = 0.
    consumed: float = 0.
    sequence: int = 0

    def produce(self, units, now):
        if units <= 0 or now < self.produced_at:
            raise ValueError("invalid final output event")
        self.consumed = min(self.received, self.consumed + (now - self.produced_at) * self.rate)
        self.received += units
        self.produced_at = now
        self.sequence += 1

    def feedback(self, session, modality, now):
        consumed = min(self.received, self.consumed + (now - self.produced_at) * self.rate)
        return ClientFeedback(self.source_id, session, modality, self.rate, self.received,
                              consumed, now, self.sequence, True, False)
