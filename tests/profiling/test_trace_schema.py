from vllm_omni.profiling import stage_queue_trace


def test_stage_event_adds_canonical_request_identity(monkeypatch):
    captured = []
    monkeypatch.setattr(
        stage_queue_trace,
        "_write_jsonl",
        lambda filename, payload: captured.append((filename, payload)),
    )

    stage_queue_trace.emit_stage_event(
        "stage_schedule",
        stage_id=1,
        request_id="logical-1__execution2",
        logical_request_id="logical-1",
        pipeline_attempt_id=2,
        stage_attempt_id=1,
    )

    filename, event = captured[0]
    assert filename == "stage_events.jsonl"
    assert event["trace_schema_version"] == 2
    assert event["request_id"] == "logical-1__execution2"
    assert event["physical_request_id"] == "logical-1__execution2"
    assert event["logical_request_id"] == "logical-1"
    assert event["execution_id"] == "logical-1__execution2"
    assert event["pipeline_attempt_id"] == 2
    assert event["stage_attempt_id"] == 1


def test_full_restart_event_uses_new_execution_identity(monkeypatch):
    captured = []
    monkeypatch.setattr(
        stage_queue_trace,
        "_write_jsonl",
        lambda filename, payload: captured.append(payload),
    )

    stage_queue_trace.emit_stage_event(
        "full_pipeline_restarted",
        stage_id=1,
        request_id="logical-1",
        old_execution_id="logical-1",
        new_execution_id="logical-1__execution1",
        pipeline_attempt_id=1,
    )

    event = captured[0]
    assert event["logical_request_id"] == "logical-1"
    assert event["execution_id"] == "logical-1__execution1"
    assert event["pipeline_attempt_id"] == 1
    assert event["stage_attempt_id"] == 0


def test_non_request_iteration_still_has_trace_schema(monkeypatch):
    captured = []
    monkeypatch.setattr(
        stage_queue_trace,
        "_write_jsonl",
        lambda filename, payload: captured.append(payload),
    )

    stage_queue_trace.emit_iteration_event(
        stage_id=0,
        iteration_id=1,
        timestamp_start=1.0,
        timestamp_end=2.0,
    )

    assert captured[0]["trace_schema_version"] == 2
    assert "logical_request_id" not in captured[0]


def test_disabled_trace_skips_clock_identity_and_writer_work(monkeypatch):
    monkeypatch.setattr(stage_queue_trace, "_enabled", lambda: False)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled trace performed hot-path work")

    monkeypatch.setattr(stage_queue_trace.time, "time", unexpected)
    monkeypatch.setattr(stage_queue_trace.time, "time_ns", unexpected)
    monkeypatch.setattr(stage_queue_trace.time, "monotonic", unexpected)
    monkeypatch.setattr(stage_queue_trace, "_request_identity_fields", unexpected)
    monkeypatch.setattr(stage_queue_trace, "_write_jsonl", unexpected)

    stage_queue_trace.emit_stage_event(
        "stage_schedule",
        stage_id=0,
        request_id="request-1",
    )
    stage_queue_trace.emit_stage_duration_event(
        "stage_forward",
        stage_id=0,
        request_id="request-1",
        timestamp_start=1.0,
        timestamp_end=2.0,
        monotonic_start=1.0,
        monotonic_end=2.0,
    )
    stage_queue_trace.emit_iteration_event(
        stage_id=0,
        iteration_id=1,
        timestamp_start=1.0,
        timestamp_end=2.0,
    )
    stage_queue_trace.emit_connector_event(
        "transfer_rx",
        from_stage=0,
        to_stage=1,
        request_id="request-1",
    )
