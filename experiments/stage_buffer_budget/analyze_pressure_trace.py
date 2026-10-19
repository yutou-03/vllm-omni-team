#!/usr/bin/env python3
"""Summarize native KV-pool pressure and recovery events from one real replay.

This reports the scheduler's native ``kv_cache_usage`` (which can include
evictable cache), not request-held B_live or physical GPU bytes. Each stage has
its own denominator; block counts from different stages are never pooled.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


EVENTS = (
    "stage_study_profile_capacity_check_failed",
    "stage_study_profile_allocation_result",
    "stage_study_profile_allocation_exception",
    "stage_study_profile_victim_selected",
    "full_pipeline_restarted",
    "stage_local_recompute_reset",
    "first_recomputed_progress",
    "full_pipeline_recovery_completed",
    "stage_study_before_preempt",
    "full_pipeline_restart_requested",
    "full_pipeline_progress_caught_up",
)


def summarize_snapshots(rows: list[dict], origin: float, window_s: float | None,
                        thresholds: tuple[float, ...]) -> dict:
    rows.sort(key=lambda row: row["monotonic"])
    if not rows:
        return {"available": False, "reason": "no before_schedule KV snapshots"}
    pools = {row["kv_total_blocks"] for row in rows}
    if len(pools) != 1:
        raise ValueError(f"KV pool block count changed within stage: {pools}")
    pool = pools.pop()
    times = [row["monotonic"] for row in rows]
    start = max(times[0], origin)
    end = times[-1] if window_s is None else min(times[-1], origin + window_s)
    if end <= start:
        return {"available": False, "reason": "no observed interval in window"}

    # A snapshot reports the state before a scheduling round. Hold that state
    # until the next snapshot, clip both ends, and never extrapolate past the
    # last observed snapshot. Include the last state at a clipped endpoint.
    first = max(0, bisect_right(times, start) - 1)
    samples = [rows[first]]
    samples.extend(row for row in rows[first + 1:] if row["monotonic"] <= end)
    area = 0.0
    above = {threshold: 0.0 for threshold in thresholds}
    for left, right in zip(samples, samples[1:] + [{"monotonic": end}]):
        left_time = max(start, left["monotonic"])
        right_time = min(end, right["monotonic"])
        dt = max(0.0, right_time - left_time)
        usage = left["kv_cache_usage"]
        area += usage * dt
        for threshold in thresholds:
            if usage > threshold:
                above[threshold] += dt
    duration = end - start
    peak = max(samples, key=lambda row: row["kv_cache_usage"])
    return {
        "available": True,
        "sample_count": len(samples),
        "pool_blocks": pool,
        "observed_start_s": round(start - origin, 6),
        "observed_end_s": round(end - origin, 6),
        "observed_duration_s": round(duration, 6),
        "requested_window_s": window_s,
        "window_coverage_fraction": round(duration / window_s, 6) if window_s else None,
        "peak_native_usage_pct": round(100 * peak["kv_cache_usage"], 6),
        "peak_time_s": round(peak["monotonic"] - origin, 6),
        "min_snapshot_free_blocks": min(row["kv_free_blocks"] for row in samples),
        "time_weighted_native_usage_pct": round(100 * area / duration, 6),
        "threshold_time_s": {
            f"gt_{100 * threshold:g}pct": round(above[threshold], 6)
            for threshold in thresholds
        },
        "threshold_fraction_of_observed": {
            f"gt_{100 * threshold:g}pct": round(above[threshold] / duration, 6)
            for threshold in thresholds
        },
    }


def nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(quantile * len(ordered)) - 1)], 6)


def client_summary(result: dict) -> dict:
    rows = result["results"]
    origin = result["origin_monotonic_s"]
    injection_end = origin + result["arrival_window_s"]
    completed = [row["client_completion_monotonic_s"] for row in rows]
    return {
        "planned": result["planned_requests"],
        "attempted": result["attempted_requests"],
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "send_lateness_p50_s": nearest_rank([row["send_lateness_s"] for row in rows], 0.5),
        "send_lateness_p95_s": nearest_rank([row["send_lateness_s"] for row in rows], 0.95),
        "send_lateness_max_s": round(max(row["send_lateness_s"] for row in rows), 6),
        "ttft_p50_s": nearest_rank([row["ttft_s"] for row in rows if row["ttft_s"] is not None], 0.5),
        "ttft_p95_s": nearest_rank([row["ttft_s"] for row in rows if row["ttft_s"] is not None], 0.95),
        "ttfp_p50_s": nearest_rank([row["ttfp_s"] for row in rows if row["ttfp_s"] is not None], 0.5),
        "ttfp_p95_s": nearest_rank([row["ttfp_s"] for row in rows if row["ttfp_s"] is not None], 0.95),
        "e2e_p50_s": nearest_rank([row["e2e_s"] for row in rows], 0.5),
        "e2e_p95_s": nearest_rank([row["e2e_s"] for row in rows], 0.95),
        "completed_by_injection_end": sum(t <= injection_end for t in completed),
        "last_completion_s": round(max(completed) - origin, 6),
        "drain_after_injection_s": round(max(completed) - injection_end, 6),
        "visible_text_chunks_total": sum(len(row["text_chunk_arrival_monotonic_s"]) for row in rows),
        "visible_audio_chunks_total": sum(len(row["audio_chunks"]) for row in rows),
        "visible_audio_frames_total": sum(row["audio_total_frames"] for row in rows),
        "slo_available": False,
        "slo_reason": "workload has no preregistered SLO contract; 240 s timeout is a safety limit",
    }


def analyze(run_dir: Path, window_s: float | None,
            thresholds: tuple[float, ...]) -> dict:
    result = json.loads((run_dir / "results.json").read_text())
    origin = result.get("origin_monotonic_s")
    if not isinstance(origin, (int, float)):
        raise ValueError("results.json lacks origin_monotonic_s")
    snapshots: dict[int, list[dict]] = {0: [], 1: []}
    counts: dict[int, Counter] = defaultdict(Counter)
    capacity_roots: dict[int, set[str]] = defaultdict(set)
    global_events: Counter = Counter()
    causes: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    trace = run_dir / "traces" / "stage_events.jsonl"
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            stage = row.get("stage_id")
            event = row.get("event")
            in_window = (window_s is None or
                         0 <= row.get("monotonic", float("-inf")) - origin < window_s)
            if event in EVENTS and in_window:
                global_events[event] += 1
            if stage not in snapshots:
                continue
            if event == "stage_study_kv_snapshot":
                if row.get("observation_point") == "before_schedule":
                    snapshots[stage].append(row)
            elif event in EVENTS and in_window:
                if event == "stage_study_profile_capacity_check_failed":
                    request_id = row.get("logical_request_id") or row.get("request_id")
                    if request_id:
                        capacity_roots[stage].add(str(request_id).split("__execution", 1)[0])
                if event == "stage_study_profile_allocation_result":
                    key = "allocation_succeeded" if row["allocation_succeeded"] else "allocation_failed"
                else:
                    key = event
                counts[stage][key] += 1
                if event == "stage_study_profile_victim_selected":
                    causes[stage]["victim_cause"][row.get("explicit_preemption_cause", "unknown")] += 1
                elif event in ("full_pipeline_restarted", "first_recomputed_progress"):
                    causes[stage][event + "_cause"][row.get("preemption_cause", "unknown")] += 1

    return {
        "run_dir": str(run_dir.resolve()),
        "origin_monotonic_s": origin,
        "metric_semantics": "native scheduler KV pool usage; not request-held B_live or GB",
        "time_weighting": "left-held snapshots over observed before_schedule intervals only; uncovered endpoints are excluded",
        "event_count_window": "full_trace" if window_s is None else f"[0, {window_s}) seconds",
        "global_events": dict(global_events),
        "client": client_summary(result),
        "stages": {
            str(stage): {
                "kv": summarize_snapshots(snapshots[stage], origin, window_s, thresholds),
                "events": {name: counts[stage][name] for name in
                           ("allocation_succeeded", "allocation_failed", *EVENTS)
                           if name != "stage_study_profile_allocation_result"},
                "unique_root_requests_with_failed_capacity_checks": len(capacity_roots[stage]),
                "causes": {key: dict(value) for key, value in causes[stage].items()},
            }
            for stage in (0, 1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="replay directory containing results.json and traces/stage_events.jsonl")
    parser.add_argument("--window-s", type=float, help="clip to this many seconds after the shared client origin")
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds",
                        help="native KV usage fraction; repeatable (default: 0.5, 0.7, 0.9)")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    if args.window_s is not None and args.window_s <= 0:
        parser.error("--window-s must be positive")
    thresholds = tuple(args.thresholds if args.thresholds is not None else (0.5, 0.7, 0.9))
    if any(not 0 <= x <= 1 for x in thresholds):
        parser.error("each --threshold must be between 0 and 1")
    data = analyze(args.run_dir, args.window_s, thresholds)
    encoded = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
