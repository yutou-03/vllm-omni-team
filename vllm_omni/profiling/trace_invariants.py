# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline invariants for versioned multi-stage request traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FindingSeverity(str, Enum):
    INFORMATION = "information"
    VIOLATION = "violation"


@dataclass(frozen=True, slots=True)
class TraceInvariantFinding:
    severity: FindingSeverity
    code: str
    event_index: int
    event: str
    message: str


@dataclass(frozen=True, slots=True)
class TraceInvariantReport:
    findings: tuple[TraceInvariantFinding, ...]

    @property
    def valid(self) -> bool:
        return not any(
            finding.severity is FindingSeverity.VIOLATION
            for finding in self.findings
        )

    def _counts(self, severity: FindingSeverity) -> dict[str, int]:
        counts = Counter(
            finding.code
            for finding in self.findings
            if finding.severity is severity
        )
        return dict(sorted(counts.items()))

    @property
    def violation_counts(self) -> dict[str, int]:
        return self._counts(FindingSeverity.VIOLATION)

    @property
    def information_counts(self) -> dict[str, int]:
        return self._counts(FindingSeverity.INFORMATION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violation_counts": self.violation_counts,
            "information_counts": self.information_counts,
            "findings": [
                {
                    "severity": finding.severity.value,
                    "code": finding.code,
                    "event_index": finding.event_index,
                    "event": finding.event,
                    "message": finding.message,
                }
                for finding in self.findings
            ],
        }


_REQUEST_IDENTITY_FIELDS = (
    "physical_request_id",
    "logical_request_id",
    "execution_id",
    "pipeline_attempt_id",
    "stage_attempt_id",
)


def _request_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("logical_request_id"),
        event.get("execution_id"),
        event.get("stage_id"),
        event.get("stage_attempt_id"),
    )


def check_trace_invariants(
    events: Iterable[Mapping[str, Any]],
) -> TraceInvariantReport:
    """Check request identity, timing, commit, and stage lifecycle invariants."""

    findings: list[TraceInvariantFinding] = []
    lifecycle: dict[tuple[Any, ...], str] = {}
    committed_outputs: set[tuple[Any, ...]] = set()

    def add(
        severity: FindingSeverity,
        code: str,
        index: int,
        event_name: str,
        message: str,
    ) -> None:
        findings.append(
            TraceInvariantFinding(
                severity=severity,
                code=code,
                event_index=index,
                event=event_name,
                message=message,
            )
        )

    for index, raw_event in enumerate(events):
        event = dict(raw_event)
        event_name = str(event.get("event", ""))
        request_scoped = event.get("request_id") is not None
        identity_complete = True
        if request_scoped and event.get("trace_schema_version") == 2:
            missing = [
                field
                for field in _REQUEST_IDENTITY_FIELDS
                if event.get(field) is None
            ]
            for attempt_field in ("pipeline_attempt_id", "stage_attempt_id"):
                attempt = event.get(attempt_field)
                if (
                    attempt is not None
                    and (
                        isinstance(attempt, bool)
                        or not isinstance(attempt, int)
                        or attempt < 0
                    )
                ):
                    missing.append(attempt_field)
            if missing:
                identity_complete = False
                add(
                    FindingSeverity.VIOLATION,
                    "missing_request_identity",
                    index,
                    event_name,
                    f"invalid or missing identity fields: {sorted(set(missing))}",
                )

        duration = event.get("duration_ms")
        monotonic_start = event.get("monotonic_start")
        monotonic_end = event.get("monotonic_end")
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration < 0
        ) or (
            isinstance(monotonic_start, (int, float))
            and isinstance(monotonic_end, (int, float))
            and monotonic_end < monotonic_start
        ):
            add(
                FindingSeverity.VIOLATION,
                "negative_duration",
                index,
                event_name,
                "duration or monotonic interval is negative",
            )

        if event_name == "stage_output_version_rejected":
            add(
                FindingSeverity.INFORMATION,
                "late_output_rejected",
                index,
                event_name,
                str(event.get("drop_reason", "unspecified")),
            )

        if event_name == "stage_output_committed" and identity_complete:
            commit_key = (
                event.get("logical_request_id"),
                event.get("stage_id"),
                event.get("output_stream_id", f"stage:{event.get('stage_id')}"),
                event.get("output_sequence"),
            )
            if commit_key in committed_outputs:
                add(
                    FindingSeverity.VIOLATION,
                    "duplicate_output_commit",
                    index,
                    event_name,
                    f"duplicate output commit {commit_key}",
                )
            committed_outputs.add(commit_key)

        if not identity_complete or not isinstance(event.get("stage_id"), int):
            continue
        key = _request_key(event)
        previous = lifecycle.get(key)
        if event_name == "stage_enqueue":
            lifecycle[key] = "enqueued"
        elif event_name == "stage_schedule":
            if previous not in {"enqueued", "scheduled"}:
                add(
                    FindingSeverity.VIOLATION,
                    "illegal_stage_transition",
                    index,
                    event_name,
                    f"stage_schedule follows {previous!r}",
                )
            lifecycle[key] = "scheduled"
        elif event_name == "stage_done":
            if previous not in {"enqueued", "scheduled"}:
                add(
                    FindingSeverity.VIOLATION,
                    "illegal_stage_transition",
                    index,
                    event_name,
                    f"stage_done follows {previous!r}",
                )
            lifecycle[key] = "done"

    return TraceInvariantReport(findings=tuple(findings))
