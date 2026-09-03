from __future__ import annotations

from copy import deepcopy

import pytest

from vllm_omni.recovery.execution import (
    ExecutionLedger,
    ExecutionStateError,
    MessageDropReason,
    OutputCommitLedger,
    OutputDecision,
)


def _three_stage_ledger() -> ExecutionLedger:
    return ExecutionLedger.initialize(
        logical_request_id="req-1",
        final_stage_id=2,
        physical_request_ids={0: "req-1", 1: "req-1", 2: "req-1"},
        started_monotonic_s=10.0,
    )


def test_initial_ledger_has_pipeline_and_stage_attempt_zero():
    ledger = _three_stage_ledger()

    assert ledger.logical_request_id == "req-1"
    assert ledger.pipeline_attempt_id == 0
    assert set(ledger.current_stages) == {0, 1, 2}
    assert {
        stage_id: ref.stage_attempt_id
        for stage_id, ref in ledger.current_stages.items()
    } == {0: 0, 1: 0, 2: 0}
    assert ledger.physical_history == {"req-1"}


def test_full_restart_advances_pipeline_and_replaces_every_stage():
    ledger = _three_stage_ledger()

    retired = ledger.begin_full_restart(
        physical_request_ids={
            0: "req-1__execution1",
            1: "req-1__execution1",
            2: "req-1__execution1",
        },
        started_monotonic_s=20.0,
    )

    assert ledger.pipeline_attempt_id == 1
    assert {ref.pipeline_attempt_id for ref in ledger.current_stages.values()} == {
        1
    }
    assert {ref.stage_attempt_id for ref in ledger.current_stages.values()} == {0}
    assert {ref.physical_request_id for ref in retired.values()} == {"req-1"}
    assert ledger.physical_history == {"req-1", "req-1__execution1"}


def test_stage_one_recompute_updates_only_stage_one_and_downstream():
    ledger = _three_stage_ledger()
    stage_zero_before = ledger.current_stages[0]

    retired = ledger.begin_stage_recompute(
        stage_id=1,
        physical_request_ids={
            1: "req-1",
            2: "req-1__recovery1__stage2",
        },
        started_monotonic_s=30.0,
    )

    assert ledger.pipeline_attempt_id == 0
    assert ledger.current_stages[0] is stage_zero_before
    assert ledger.current_stages[1].stage_attempt_id == 1
    assert ledger.current_stages[1].physical_request_id == "req-1"
    assert ledger.current_stages[2].stage_attempt_id == 1
    assert (
        ledger.current_stages[2].physical_request_id
        == "req-1__recovery1__stage2"
    )
    assert set(retired) == {1, 2}


def test_final_stage_recompute_does_not_change_upstream_stages():
    ledger = _three_stage_ledger()
    upstream_before = dict(ledger.current_stages)

    ledger.begin_stage_recompute(
        stage_id=2,
        physical_request_ids={2: "req-1"},
        started_monotonic_s=40.0,
    )

    assert ledger.current_stages[0] is upstream_before[0]
    assert ledger.current_stages[1] is upstream_before[1]
    assert ledger.current_stages[2].stage_attempt_id == 1


def test_same_physical_id_from_retired_stage_attempt_is_rejected():
    ledger = _three_stage_ledger()
    ledger.begin_stage_recompute(
        stage_id=1,
        physical_request_ids={
            1: "req-1",
            2: "req-1__recovery1__stage2",
        },
        started_monotonic_s=30.0,
    )

    assert (
        ledger.classify_message(
            logical_request_id="req-1",
            pipeline_attempt_id=0,
            stage_id=1,
            stage_attempt_id=0,
            physical_request_id="req-1",
            event_monotonic_s=31.0,
        )
        is MessageDropReason.RETIRED_STAGE_ATTEMPT
    )
    assert (
        ledger.classify_message(
            logical_request_id="req-1",
            pipeline_attempt_id=0,
            stage_id=1,
            stage_attempt_id=1,
            physical_request_id="req-1",
            event_monotonic_s=31.0,
        )
        is None
    )


def test_message_from_retired_pipeline_is_rejected_even_with_known_physical_id():
    ledger = _three_stage_ledger()
    ledger.begin_full_restart(
        physical_request_ids={
            0: "req-1__execution1",
            1: "req-1__execution1",
            2: "req-1__execution1",
        },
        started_monotonic_s=20.0,
    )

    assert (
        ledger.classify_message(
            logical_request_id="req-1",
            pipeline_attempt_id=0,
            stage_id=2,
            stage_attempt_id=0,
            physical_request_id="req-1",
            event_monotonic_s=21.0,
        )
        is MessageDropReason.RETIRED_PIPELINE_ATTEMPT
    )


def test_text_and_audio_replay_watermarks_are_independent():
    outputs = OutputCommitLedger.create(["text", "audio"])

    assert outputs.classify_and_commit("text", 0) is OutputDecision.COMMIT
    assert outputs.classify_and_commit("text", 1) is OutputDecision.COMMIT
    assert outputs.classify_and_commit("audio", 0) is OutputDecision.COMMIT

    outputs.begin_replay(["audio"])

    assert (
        outputs.classify_and_commit("audio", 0)
        is OutputDecision.DROP_REPLAY_DUPLICATE
    )
    assert outputs.classify_and_commit("text", 2) is OutputDecision.COMMIT
    assert outputs.states["text"].replay_watermark == -1
    assert outputs.states["audio"].replay_watermark == 0


def test_duplicate_gap_and_output_after_terminal_do_not_mutate_sequence():
    outputs = OutputCommitLedger.create(["audio"])

    assert outputs.classify_and_commit("audio", 0) is OutputDecision.COMMIT
    assert (
        outputs.classify_and_commit("audio", 0)
        is OutputDecision.DROP_DUPLICATE
    )
    assert outputs.classify_and_commit("audio", 2) is OutputDecision.GAP
    assert outputs.states["audio"].committed_sequence == 0

    assert (
        outputs.classify_and_commit("audio", 1, finished=True)
        is OutputDecision.COMMIT
    )
    assert (
        outputs.classify_and_commit("audio", 2)
        is OutputDecision.DROP_AFTER_TERMINAL
    )
    assert outputs.states["audio"].committed_sequence == 1


def test_invalid_transition_is_atomic():
    ledger = _three_stage_ledger()
    stages_before = deepcopy(ledger.current_stages)
    history_before = set(ledger.physical_history)

    with pytest.raises(ExecutionStateError, match="exactly stages"):
        ledger.begin_stage_recompute(
            stage_id=1,
            physical_request_ids={1: "req-1"},
            started_monotonic_s=30.0,
        )

    assert ledger.pipeline_attempt_id == 0
    assert ledger.current_stages == stages_before
    assert ledger.physical_history == history_before


def test_cleanup_ids_include_every_historical_physical_request():
    ledger = _three_stage_ledger()
    ledger.begin_stage_recompute(
        stage_id=1,
        physical_request_ids={
            1: "req-1",
            2: "req-1__recovery1__stage2",
        },
        started_monotonic_s=20.0,
    )
    ledger.begin_full_restart(
        physical_request_ids={
            0: "req-1__execution1",
            1: "req-1__execution1",
            2: "req-1__execution1",
        },
        started_monotonic_s=30.0,
    )

    assert ledger.cleanup_physical_request_ids() == {
        "req-1",
        "req-1__recovery1__stage2",
        "req-1__execution1",
    }
