"""Replay the paper's text metric from Thinker production events on the host clock."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def configured_trace(manifest):
    dirs = [arg.split("=", 1)[1] for arg in manifest.get("command", [])
            if arg.startswith("STAGE_QUEUE_TRACE_DIR=")]
    if len(dirs) == 1:
        return Path(dirs[0]) / "stage_events.jsonl"
    return None


def trace_offset(path):
    return path.stat().st_size if path and path.exists() else 0


def score_request(events, length, rate, arrival):
    """A token leaves the unread count when its reading interval completes."""
    if length <= 0 or len(events) != length or not math.isfinite(rate) or rate <= 0:
        raise ValueError("incomplete generation sequence or invalid reading rate")
    first = events[0][1]
    if first < arrival or not math.isfinite(arrival):
        raise ValueError("generation precedes arrival")
    # Work in time relative to the first token to avoid accumulation of clock rounding.
    busy_start = 0.
    busy_tokens = 0
    utility = stall = 0.
    previous = 0.
    for index, (counter, stamp) in enumerate(events, start=1):
        elapsed = stamp - first
        if counter != index or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError("missing, repeated or unordered generation event")
        previous = elapsed
        ticks = (elapsed - busy_start) * rate
        # Timestamps are seconds from a large monotonic clock; ties favour consumption.
        completed = math.floor(ticks + 2 * math.ulp(stamp) * rate)
        unread = max(0, busy_tokens - completed)
        if unread == 0:
            stall += max(0., elapsed - (busy_start + busy_tokens / rate))
            busy_start, busy_tokens = elapsed, 0
        busy_tokens += 1
        utility += min(1., max(0., (.2 * length - unread - 1) / (.1 * length)))
    return dict(utility=utility, units=length, ttft_s=first-arrival, stall_s=stall)


def wire_length(path):
    # The benchmark requests stream_options.include_usage. Most streams have
    # one short terminal usage event. Read the whole stream only if necessary.
    with path.open("rb") as f:
        size = f.seek(0, 2)
        for start in (max(0, size-8192), 0):
            f.seek(start)
            if start:
                f.readline()
            for line in f:
                if line.startswith(b"data:") and b'"completion_tokens"' in line:
                    usage = json.loads(line[5:]).get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        return int(usage["completion_tokens"])
    raise ValueError(f"missing final completion_tokens: {path}")


def replay_text(root, rows, window, connector_proxy=False):
    trace = Path(window["trace"]) / "stage_events.jsonl"
    start, end = window["start_offset"], window["end_offset"]
    if not 0 <= start < end <= trace.stat().st_size:
        raise ValueError("invalid generation trace byte range")
    source_ids = {row["request_id"] for row in rows}
    if len(source_ids) != len(rows):
        raise ValueError("duplicate workload request ID")
    backend = {}
    events = defaultdict(list)
    emitted = "stage_study_thinker_output_produced" if connector_proxy else "stage_token_output"
    event_marker = json.dumps(emitted).encode()
    group_end = {}
    with trace.open("rb") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if b'"request_ingress"' not in line and event_marker not in line:
                continue
            event = json.loads(line)
            if event["event"] == "request_ingress" and event.get("sched_source_request_id") in source_ids:
                source = event["sched_source_request_id"]
                rid = event["request_id"]
                if backend.setdefault(source, rid) != rid:
                    raise ValueError(f"multiple backend streams for {source}")
            elif event["event"] == emitted and str(event["stage_id"]) == "0":
                rid = event["request_id"]
                if connector_proxy:
                    index = event["generated_output_tokens_total"]
                else:
                    # The engine emits one record per accepted ID. Speculative
                    # tokens from one step share an output_len and generation time.
                    index = len(events[rid]) + 1
                    end_index = event["output_len"]
                    if index > group_end.get(rid, 0):
                        if index + event["num_new_tokens"] - 1 != end_index:
                            raise ValueError(f"{rid}: invalid engine token batch")
                        group_end[rid] = end_index
                    elif end_index != group_end[rid]:
                        raise ValueError(f"{rid}: incomplete engine token batch")
                stamp = event["monotonic"]
                if not connector_proxy and index > end_index-event["num_new_tokens"]+1:
                    stamp = events[rid][-1][1]
                events[rid].append((index, stamp))
    scores = []
    for row in rows:
        source = row["request_id"]
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        sequence = sorted(events.get(backend.get(source), ()))
        try:
            length = wire_length(root/key/"stream.sse")
            if not connector_proxy and group_end.get(backend.get(source)) != length:
                raise ValueError("missing final engine token batch or usage mismatch")
            score = score_request(sequence, length, row["rate"], row["scheduled_at"])
            # Trace emission and client DONE run in different processes.
            # Preserve the measured probe skew for the timing audit.
            score["probe_after_done_s"] = max(0., sequence[-1][1] - row["generation_done_at"])
        except ValueError as exc:
            raise ValueError(f"{source}: {exc}") from exc
        scores.append(dict(request_id=source, **score))
    return scores
