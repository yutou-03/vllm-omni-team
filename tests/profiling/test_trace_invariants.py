from vllm_omni.profiling.trace_invariants import check_trace_invariants


def _event(event, **fields):
    return {
        "trace_schema_version": 2,
        "event": event,
        "stage_id": 0,
        "request_id": "physical-1",
        "physical_request_id": "physical-1",
        "logical_request_id": "logical-1",
        "execution_id": "physical-1",
        "pipeline_attempt_id": 0,
        "stage_attempt_id": 0,
        **fields,
    }


def test_valid_stage_lifecycle_and_commits_pass():
    report = check_trace_invariants(
        [
            _event("stage_enqueue"),
            _event("stage_schedule"),
            _event("stage_output_committed", output_sequence=0),
            _event("stage_output_committed", output_sequence=1),
            _event("stage_done"),
        ]
    )

    assert report.valid is True
    assert report.violation_counts == {}


def test_local_reentry_starts_a_new_stage_attempt():
    report = check_trace_invariants(
        [
            _event("stage_enqueue", stage_attempt_id=0),
            _event("stage_schedule", stage_attempt_id=0),
            _event("stage_reentered", stage_attempt_id=1),
            _event("stage_schedule", stage_attempt_id=1),
            _event("stage_done", stage_attempt_id=1),
        ]
    )

    assert report.valid is True
    assert report.violation_counts == {}


def test_checker_detects_negative_duration_duplicate_and_illegal_transition():
    report = check_trace_invariants(
        [
            _event("stage_schedule", duration_ms=-1),
            _event("stage_output_committed", output_sequence=0),
            _event("stage_output_committed", output_sequence=0),
        ]
    )

    assert report.valid is False
    assert report.violation_counts == {
        "duplicate_output_commit": 1,
        "illegal_stage_transition": 1,
        "negative_duration": 1,
    }


def test_checker_reports_safely_rejected_late_output_as_information():
    report = check_trace_invariants(
        [
            _event("stage_enqueue"),
            _event(
                "stage_output_version_rejected",
                drop_reason="retired_stage_attempt",
            ),
        ]
    )

    assert report.valid is True
    assert report.information_counts == {"late_output_rejected": 1}


def test_schema_v2_request_event_missing_identity_fails():
    report = check_trace_invariants(
        [
            {
                "trace_schema_version": 2,
                "event": "stage_enqueue",
                "stage_id": 0,
                "request_id": "physical-1",
            }
        ]
    )

    assert report.violation_counts == {"missing_request_identity": 1}
