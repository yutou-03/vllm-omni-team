# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit execution versions for safe multi-stage recovery.

Physical request IDs are transport and resource identities.  They are kept
opaque here: recovery correctness comes from explicit pipeline and stage
attempts rather than parsing naming conventions such as ``__executionN``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class ExecutionStateError(ValueError):
    """Raised when an execution-state transition violates the contract."""


class PreemptionRecoveryMode(str, Enum):
    """Supported recovery actions after a scheduler resource preemption."""

    FULL_PIPELINE_RESTART = "full_pipeline_restart"
    STAGE_LOCAL_RECOMPUTE = "stage_local_recompute"


class MessageDropReason(str, Enum):
    """Reason an asynchronous message does not belong to the current state."""

    UNKNOWN_LOGICAL_REQUEST = "unknown_logical_request"
    INVALID_STAGE = "invalid_stage"
    UNREGISTERED_PHYSICAL_REQUEST = "unregistered_physical_request"
    RETIRED_PIPELINE_ATTEMPT = "retired_pipeline_attempt"
    RETIRED_STAGE_ATTEMPT = "retired_stage_attempt"
    RETIRED_PHYSICAL_REQUEST = "retired_physical_request"
    EVENT_BEFORE_ATTEMPT_START = "event_before_attempt_start"


class OutputDecision(str, Enum):
    """Result of checking and, when valid, committing one output chunk."""

    COMMIT = "commit"
    COMMIT_TERMINAL_ONLY = "commit_terminal_only"
    DROP_REPLAY_DUPLICATE = "drop_replay_duplicate"
    DROP_DUPLICATE = "drop_duplicate"
    DROP_AFTER_TERMINAL = "drop_after_terminal"
    GAP = "gap"


def _validate_non_negative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionStateError(f"{field_name} must be a non-negative integer")
    return value


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionStateError(f"{field_name} must be a non-empty string")
    return value


def _validate_timestamp(value: float, *, field_name: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as error:
        raise ExecutionStateError(f"{field_name} must be finite") from error
    if not math.isfinite(timestamp):
        raise ExecutionStateError(f"{field_name} must be finite")
    return timestamp


def _validate_physical_request_ids(
    physical_request_ids: Mapping[int, str],
    *,
    expected_stage_ids: set[int],
) -> dict[int, str]:
    if not isinstance(physical_request_ids, Mapping):
        raise ExecutionStateError("physical_request_ids must be a stage mapping")

    normalized = dict(physical_request_ids)
    if set(normalized) != expected_stage_ids:
        raise ExecutionStateError(
            "physical_request_ids must contain exactly stages "
            f"{sorted(expected_stage_ids)}"
        )
    for stage_id, physical_request_id in normalized.items():
        _validate_non_negative_int(stage_id, field_name="stage_id")
        _validate_identifier(
            physical_request_id,
            field_name=f"physical_request_ids[{stage_id}]",
        )
    return normalized


@dataclass(frozen=True, slots=True)
class StageExecutionRef:
    """The one execution version currently valid for a pipeline stage."""

    logical_request_id: str
    pipeline_attempt_id: int
    stage_id: int
    stage_attempt_id: int
    physical_request_id: str
    started_monotonic_s: float

    def __post_init__(self) -> None:
        _validate_identifier(
            self.logical_request_id,
            field_name="logical_request_id",
        )
        _validate_non_negative_int(
            self.pipeline_attempt_id,
            field_name="pipeline_attempt_id",
        )
        _validate_non_negative_int(self.stage_id, field_name="stage_id")
        _validate_non_negative_int(
            self.stage_attempt_id,
            field_name="stage_attempt_id",
        )
        _validate_identifier(
            self.physical_request_id,
            field_name="physical_request_id",
        )
        _validate_timestamp(
            self.started_monotonic_s,
            field_name="started_monotonic_s",
        )


@dataclass(slots=True)
class ExecutionLedger:
    """Current and historical physical identities for one logical request."""

    logical_request_id: str
    final_stage_id: int
    pipeline_attempt_id: int
    current_stages: dict[int, StageExecutionRef]
    physical_history: set[str] = field(default_factory=set)

    @classmethod
    def initialize(
        cls,
        *,
        logical_request_id: str,
        final_stage_id: int,
        physical_request_ids: Mapping[int, str],
        started_monotonic_s: float,
    ) -> ExecutionLedger:
        logical_request_id = _validate_identifier(
            logical_request_id,
            field_name="logical_request_id",
        )
        final_stage_id = _validate_non_negative_int(
            final_stage_id,
            field_name="final_stage_id",
        )
        timestamp = _validate_timestamp(
            started_monotonic_s,
            field_name="started_monotonic_s",
        )
        stage_ids = set(range(final_stage_id + 1))
        physical_ids = _validate_physical_request_ids(
            physical_request_ids,
            expected_stage_ids=stage_ids,
        )
        current_stages = {
            stage_id: StageExecutionRef(
                logical_request_id=logical_request_id,
                pipeline_attempt_id=0,
                stage_id=stage_id,
                stage_attempt_id=0,
                physical_request_id=physical_ids[stage_id],
                started_monotonic_s=timestamp,
            )
            for stage_id in range(final_stage_id + 1)
        }
        return cls(
            logical_request_id=logical_request_id,
            final_stage_id=final_stage_id,
            pipeline_attempt_id=0,
            current_stages=current_stages,
            physical_history=set(physical_ids.values()),
        )

    def begin_full_restart(
        self,
        *,
        physical_request_ids: Mapping[int, str],
        started_monotonic_s: float,
    ) -> dict[int, StageExecutionRef]:
        """Atomically replace every stage with a new pipeline attempt."""

        timestamp = self._validate_transition_timestamp(started_monotonic_s)
        stage_ids = set(range(self.final_stage_id + 1))
        physical_ids = _validate_physical_request_ids(
            physical_request_ids,
            expected_stage_ids=stage_ids,
        )
        next_pipeline_attempt = self.pipeline_attempt_id + 1
        next_stages = {
            stage_id: StageExecutionRef(
                logical_request_id=self.logical_request_id,
                pipeline_attempt_id=next_pipeline_attempt,
                stage_id=stage_id,
                stage_attempt_id=0,
                physical_request_id=physical_ids[stage_id],
                started_monotonic_s=timestamp,
            )
            for stage_id in range(self.final_stage_id + 1)
        }
        retired = dict(self.current_stages)

        # All validation and object construction happens before this commit.
        self.pipeline_attempt_id = next_pipeline_attempt
        self.current_stages = next_stages
        self.physical_history.update(physical_ids.values())
        return retired

    def begin_stage_recompute(
        self,
        *,
        stage_id: int,
        physical_request_ids: Mapping[int, str],
        started_monotonic_s: float,
    ) -> dict[int, StageExecutionRef]:
        """Atomically replace one stage attempt and all downstream attempts."""

        stage_id = _validate_non_negative_int(stage_id, field_name="stage_id")
        if stage_id > self.final_stage_id:
            raise ExecutionStateError(
                f"stage_id must be between 0 and {self.final_stage_id}"
            )
        timestamp = self._validate_transition_timestamp(started_monotonic_s)
        affected_stage_ids = set(range(stage_id, self.final_stage_id + 1))
        physical_ids = _validate_physical_request_ids(
            physical_request_ids,
            expected_stage_ids=affected_stage_ids,
        )

        for downstream_stage_id in range(stage_id + 1, self.final_stage_id + 1):
            old_physical_id = self.current_stages[
                downstream_stage_id
            ].physical_request_id
            if physical_ids[downstream_stage_id] == old_physical_id:
                raise ExecutionStateError(
                    "downstream stage recompute requires a new physical request id "
                    f"for stage {downstream_stage_id}"
                )

        replacements = {
            affected_stage_id: StageExecutionRef(
                logical_request_id=self.logical_request_id,
                pipeline_attempt_id=self.pipeline_attempt_id,
                stage_id=affected_stage_id,
                stage_attempt_id=(
                    self.current_stages[affected_stage_id].stage_attempt_id + 1
                ),
                physical_request_id=physical_ids[affected_stage_id],
                started_monotonic_s=timestamp,
            )
            for affected_stage_id in range(stage_id, self.final_stage_id + 1)
        }
        retired = {
            affected_stage_id: self.current_stages[affected_stage_id]
            for affected_stage_id in range(stage_id, self.final_stage_id + 1)
        }
        next_stages = dict(self.current_stages)
        next_stages.update(replacements)

        # All validation and object construction happens before this commit.
        self.current_stages = next_stages
        self.physical_history.update(physical_ids.values())
        return retired

    def classify_message(
        self,
        *,
        logical_request_id: str,
        pipeline_attempt_id: int,
        stage_id: int,
        stage_attempt_id: int,
        physical_request_id: str,
        event_monotonic_s: float,
    ) -> MessageDropReason | None:
        """Return why a message is stale, or ``None`` when it is current."""

        if logical_request_id != self.logical_request_id:
            return MessageDropReason.UNKNOWN_LOGICAL_REQUEST
        if (
            isinstance(stage_id, bool)
            or not isinstance(stage_id, int)
            or stage_id not in self.current_stages
        ):
            return MessageDropReason.INVALID_STAGE
        if physical_request_id not in self.physical_history:
            return MessageDropReason.UNREGISTERED_PHYSICAL_REQUEST
        if pipeline_attempt_id != self.pipeline_attempt_id:
            return MessageDropReason.RETIRED_PIPELINE_ATTEMPT

        current = self.current_stages[stage_id]
        if stage_attempt_id != current.stage_attempt_id:
            return MessageDropReason.RETIRED_STAGE_ATTEMPT
        if physical_request_id != current.physical_request_id:
            return MessageDropReason.RETIRED_PHYSICAL_REQUEST
        try:
            event_time = float(event_monotonic_s)
        except (TypeError, ValueError):
            return MessageDropReason.EVENT_BEFORE_ATTEMPT_START
        if (
            not math.isfinite(event_time)
            or event_time < current.started_monotonic_s
        ):
            return MessageDropReason.EVENT_BEFORE_ATTEMPT_START
        return None

    def cleanup_physical_request_ids(self) -> set[str]:
        """Return every physical ID that must be aborted/released on cleanup."""

        return set(self.physical_history)

    def _validate_transition_timestamp(self, value: float) -> float:
        timestamp = _validate_timestamp(value, field_name="started_monotonic_s")
        latest_start = max(
            ref.started_monotonic_s for ref in self.current_stages.values()
        )
        if timestamp < latest_start:
            raise ExecutionStateError(
                "started_monotonic_s cannot precede the current execution"
            )
        return timestamp


@dataclass(slots=True)
class OutputCommitState:
    """Commit and replay watermark for one frontend output stream."""

    stream_id: str
    committed_sequence: int = -1
    replay_watermark: int = -1
    terminal_committed: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.stream_id, field_name="stream_id")


@dataclass(slots=True)
class OutputCommitLedger:
    """Keep output ordering and replay suppression independent per stream."""

    states: dict[str, OutputCommitState]

    @classmethod
    def create(cls, stream_ids: Iterable[str]) -> OutputCommitLedger:
        normalized: list[str] = []
        for stream_id in stream_ids:
            normalized.append(
                _validate_identifier(stream_id, field_name="stream_id")
            )
        if not normalized:
            raise ExecutionStateError("at least one output stream is required")
        if len(normalized) != len(set(normalized)):
            raise ExecutionStateError("output stream ids must be unique")
        return cls(
            states={
                stream_id: OutputCommitState(stream_id=stream_id)
                for stream_id in normalized
            }
        )

    def begin_replay(self, stream_ids: Iterable[str]) -> None:
        """Atomically snapshot committed sequence numbers for affected streams."""

        normalized = list(stream_ids)
        if not normalized:
            raise ExecutionStateError("at least one replay stream is required")
        if len(normalized) != len(set(normalized)):
            raise ExecutionStateError("replay stream ids must be unique")
        missing = [stream_id for stream_id in normalized if stream_id not in self.states]
        if missing:
            raise ExecutionStateError(f"unknown output streams: {sorted(missing)}")
        terminal = [
            stream_id
            for stream_id in normalized
            if self.states[stream_id].terminal_committed
        ]
        if terminal:
            raise ExecutionStateError(
                f"cannot replay terminal output streams: {sorted(terminal)}"
            )

        for stream_id in normalized:
            state = self.states[stream_id]
            state.replay_watermark = state.committed_sequence

    def classify_and_commit(
        self,
        stream_id: str,
        attempt_sequence: int,
        *,
        finished: bool = False,
    ) -> OutputDecision:
        """Classify one chunk and update state only when it is committable."""

        state = self._state(stream_id)
        sequence = _validate_non_negative_int(
            attempt_sequence,
            field_name="attempt_sequence",
        )
        if not isinstance(finished, bool):
            raise ExecutionStateError("finished must be a boolean")

        if state.terminal_committed:
            return OutputDecision.DROP_AFTER_TERMINAL
        if sequence <= state.replay_watermark:
            if finished:
                state.terminal_committed = True
                return OutputDecision.COMMIT_TERMINAL_ONLY
            return OutputDecision.DROP_REPLAY_DUPLICATE
        if sequence <= state.committed_sequence:
            return OutputDecision.DROP_DUPLICATE
        if sequence != state.committed_sequence + 1:
            return OutputDecision.GAP

        state.committed_sequence = sequence
        if finished:
            state.terminal_committed = True
        return OutputDecision.COMMIT

    def _state(self, stream_id: str) -> OutputCommitState:
        if stream_id not in self.states:
            raise ExecutionStateError(f"unknown output stream: {stream_id!r}")
        return self.states[stream_id]
