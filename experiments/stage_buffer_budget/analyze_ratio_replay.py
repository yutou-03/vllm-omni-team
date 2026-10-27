#!/usr/bin/env python3
"""Export measured, per-engine KV-pool trajectories for a stage-ratio replay.

The source is the native allocator's before-schedule snapshot.  A Thinker TP
engine has one scheduler trace for all its synchronized ranks, so its plotted
capacity is *per rank*.  Talker replicas have separate engine PIDs and pools.
This is allocator occupancy, not request-held B_live.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values.sort()
    return values[min(len(values) - 1, math.ceil(quantile * len(values)) - 1)]


def client_metrics(results: dict, timeout_s: float) -> dict:
    rows = results["results"]
    origin = results["origin_monotonic_s"]
    window = results["arrival_window_s"]
    observed_end = max(row["client_completion_monotonic_s"] for row in rows) - origin
    fixed_horizon = window + timeout_s
    completed = [row for row in rows if row["status"] == "success"]
    audio_chunks = [chunk for row in rows for chunk in row.get("audio_chunks", ())]
    in_window = [chunk for chunk in audio_chunks
                 if 0 <= chunk["arrival_monotonic_s"] - origin < window]
    in_fixed_horizon = [chunk for chunk in audio_chunks
                        if 0 <= chunk["arrival_monotonic_s"] - origin <= fixed_horizon]
    return {
        "planned": results["planned_requests"],
        "attempted": results["attempted_requests"],
        "statuses": dict(Counter(row["status"] for row in rows)),
        "injection_window_s": window,
        "observed_end_s": observed_end,
        "successes_by_injection_end": sum(
            row["client_completion_monotonic_s"] - origin < window for row in completed
        ),
        "observed_successes_per_s": len(completed) / observed_end,
        "accepted_audio_content_seconds_per_wall_s_observed_span": sum(
            chunk["duration_s"] for chunk in audio_chunks) / observed_end,
        "all_requests_completed_without_timeout": len(completed) == len(rows),
        "fixed_horizon_s": fixed_horizon,
        "successes_by_fixed_horizon": sum(row["client_completion_monotonic_s"] - origin <= fixed_horizon for row in completed),
        "successes_per_s_fixed_horizon": sum(row["client_completion_monotonic_s"] - origin <= fixed_horizon for row in completed) / fixed_horizon,
        "successful_request_ttft_p50_s": percentile(
            [row["ttft_s"] for row in completed if row.get("ttft_s") is not None], .5
        ),
        "successful_request_ttft_p95_s": percentile(
            [row["ttft_s"] for row in completed if row.get("ttft_s") is not None], .95
        ),
        "successful_request_ttfp_p50_s": percentile(
            [row["ttfp_s"] for row in completed if row.get("ttfp_s") is not None], .5
        ),
        "successful_request_ttfp_p95_s": percentile(
            [row["ttfp_s"] for row in completed if row.get("ttfp_s") is not None], .95
        ),
        "successful_request_e2e_p50_s": percentile(
            [row["e2e_s"] for row in completed if row.get("e2e_s") is not None], .5
        ),
        "successful_request_e2e_p95_s": percentile(
            [row["e2e_s"] for row in completed if row.get("e2e_s") is not None], .95
        ),
        "accepted_audio_frames_in_injection": sum(chunk["frames"] for chunk in in_window),
        "accepted_audio_frames_total": sum(chunk["frames"] for chunk in audio_chunks),
        "accepted_audio_seconds_total": sum(chunk["duration_s"] for chunk in audio_chunks),
        "accepted_audio_frames_per_s_fixed_horizon": sum(
            chunk["frames"] for chunk in in_fixed_horizon) / fixed_horizon,
        "accepted_audio_content_seconds_per_wall_s_fixed_horizon": sum(
            chunk["duration_s"] for chunk in in_fixed_horizon) / fixed_horizon,
        "slo_goodput_per_s": None,
        "slo_missing_reason": "User deferred the SLO contract; safety timeout is not an SLO",
        "censoring_note": "Finite-replay completion rate; not sustainable throughput",
    }


def load_pools(run_dir: Path, origin: float) -> tuple[dict[tuple[int, int], list[tuple]], dict]:
    pools: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    last_release: dict[tuple[int, int], tuple] = {}
    events: dict[tuple[int, int], Counter] = defaultdict(Counter)
    with (run_dir / "traces" / "stage_events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            stage = row.get("stage_id")
            pid = row.get("pid")
            if stage not in (0, 1) or not isinstance(pid, int):
                continue
            key = stage, pid
            event = row.get("event")
            if event == "stage_study_kv_snapshot" and row.get("observation_point") == "before_schedule":
                pools[key].append((row["monotonic"] - origin, row["kv_used_blocks"],
                                   row["kv_total_blocks"], row["kv_free_blocks"], "snapshot",
                                   row["kv_cache_usage"]))
            elif event == "stage_study_profile_kv_free":
                last_release[key] = (row["monotonic"] - origin, row["free_blocks_after"])
            elif event == "stage_study_profile_allocation_result" and row.get("allocation_succeeded") is False:
                events[key]["allocation_failure"] += 1
            elif event == "stage_study_profile_victim_selected":
                events[key]["passive_preemption"] += row.get("explicit_preemption_cause") == "allocation_failure"
    for key, samples in pools.items():
        samples.sort()
        if not samples:
            continue
        total = samples[0][2]
        if any(row[2] != total or row[1] + row[3] != total for row in samples):
            raise ValueError(f"native block accounting changed for pool {key}")
        release = last_release.get(key)
        if release and release[0] > samples[-1][0]:
            t, free = release
            samples.append((t, total - free, total, free, "final_kv_free", None))
    return dict(pools), {f"{stage}:{pid}": dict(counts)
                         for (stage, pid), counts in events.items()}


def summarize_pool(samples: list[tuple], block_bytes: int, origin: float,
                   observation_end: float) -> dict:
    rows = [row for row in samples if 0 <= row[0] <= observation_end]
    if not rows:
        return {"available": False, "reason": "no snapshots inside observation window"}
    duration = rows[-1][0] - rows[0][0]
    area = 0.0
    high = 0.0
    for left, right in zip(rows, rows[1:]):
        dt = right[0] - left[0]
        usage = left[5] if left[5] is not None else left[1] / left[2]
        area += usage * dt
        if usage > .9:
            high += dt
    return {
        "available": True,
        "sample_count": len(rows),
        "first_observed_s": rows[0][0],
        "last_observed_s": rows[-1][0],
        "observed_duration_s": duration,
        "pool_blocks_per_rank": rows[0][2],
        "block_bytes": block_bytes,
        "pool_bytes_per_rank": rows[0][2] * block_bytes,
        "peak_used_bytes_per_rank": max(row[1] for row in rows) * block_bytes,
        "peak_native_usage_pct": max(row[5] for row in rows if row[5] is not None) * 100,
        "peak_allocated_block_pct": max(row[1] / row[2] for row in rows) * 100,
        "mean_native_usage_pct_observed": area / duration * 100 if duration > 0 else None,
        "gt90_time_s_observed": high,
        "minimum_free_blocks": min(row[3] for row in rows),
        "unobserved_tail_s": observation_end - rows[-1][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thinker-block-bytes", type=int, required=True)
    parser.add_argument("--talker-block-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, help="write derived files here (default: run directory)")
    parser.add_argument("--pid-gpu-map", type=Path, help="nvidia-smi observation containing container_pid and gpu_index")
    args = parser.parse_args()
    if min(args.thinker_block_bytes, args.talker_block_bytes) <= 0:
        parser.error("block bytes must be positive")
    results = json.loads((args.run_dir / "results.json").read_text())
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_by_pid = {}
    if args.pid_gpu_map:
        process_rows = json.loads(args.pid_gpu_map.read_text())["processes"]
        gpu_by_pid = {row["container_pid"]: row["gpu_index"] for row in process_rows if row.get("container_pid") is not None}
    origin = results["origin_monotonic_s"]
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    metrics = client_metrics(results, float(protocol["timeout_s"]))
    pools, events = load_pools(args.run_dir, origin)
    if not pools:
        raise ValueError("no measured KV-pool snapshots")
    block_sizes = {0: args.thinker_block_bytes, 1: args.talker_block_bytes}
    timeseries = output_dir / "kv_pools_by_engine.csv"
    with timeseries.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stage_id", "engine_pid", "relative_time_s", "used_blocks",
                         "free_blocks", "total_blocks", "block_bytes", "used_bytes_per_rank",
                         "pool_bytes_per_rank", "native_usage_pct", "allocated_block_usage_pct",
                         "measurement_source"))
        for (stage, pid), rows in sorted(pools.items()):
            size = block_sizes[stage]
            for t, used, total, free, source, native_usage in rows:
                if t >= 0:
                    writer.writerow((stage, pid, t, used, free, total, size,
                                     used * size, total * size,
                                     native_usage * 100 if native_usage is not None else "",
                                     used / total * 100, source))
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "metric": "native allocator KV occupancy per independent engine pool, not B_live",
        "drained_after": bool(results.get("drained_after") and results["drained_after"].get("drained")),
        "run_error": results.get("run_error"),
        "thinker_tp_note": "one scheduler snapshot represents each synchronized TP rank; bytes are per rank",
        "client": metrics,
        "pools": {f"{stage}:{pid}": summarize_pool(rows, block_sizes[stage], origin,
                                                   metrics["observed_end_s"])
                  for (stage, pid), rows in sorted(pools.items())},
        "gpu_by_engine_pid": {str(pid): gpu_by_pid[pid] for _, pid in pools if pid in gpu_by_pid},
        "events_by_pool": events,
    }
    (output_dir / "ratio_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "request_summary.csv").open("w", newline="") as stream:
        fields = ("source_request_id", "status", "error", "scheduled_arrival_s",
                  "send_lateness_s", "completion_s", "ttft_s", "ttfp_s", "e2e_s",
                  "audio_chunks", "audio_frames", "audio_duration_s", "text_token_ids")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results["results"], key=lambda item: item["source_request_id"]):
            writer.writerow({
                "source_request_id": row["source_request_id"],
                "status": row["status"],
                "error": row.get("error"),
                "scheduled_arrival_s": row["scheduled_arrival_monotonic_s"] - origin,
                "send_lateness_s": row.get("send_lateness_s"),
                "completion_s": row["client_completion_monotonic_s"] - origin,
                "ttft_s": row.get("ttft_s"),
                "ttfp_s": row.get("ttfp_s"),
                "e2e_s": row.get("e2e_s"),
                "audio_chunks": len(row.get("audio_chunks", ())),
                "audio_frames": row.get("audio_total_frames"),
                "audio_duration_s": row.get("audio_duration_s"),
                "text_token_ids": len(row.get("text_token_ids") or ()),
            })
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout="constrained")
    for (stage, pid), rows in sorted(pools.items()):
        visible = [row for row in rows if 0 <= row[0] <= metrics["observed_end_s"]]
        if visible:
            pool_label = f"GPU {gpu_by_pid[pid]}" if pid in gpu_by_pid else f"engine PID {pid}"
            axes[stage].step([r[0] for r in visible],
                             [r[1] * block_sizes[stage] / 1e9 for r in visible],
                             where="post", label=pool_label, linewidth=1.5)
    for stage, ax in enumerate(axes):
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Native KV used (GB per rank)")
        ax.set_title("Thinker TP pool (per rank)" if stage == 0 else "Talker independent replica pools")
        ax.axvline(metrics["injection_window_s"], color="0.5", linestyle=":", linewidth=1)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlim(0, metrics["observed_end_s"])
    axes[-1].set_xlabel("Time since replay start (s)")
    fig.suptitle("Measured KV-pool occupancy by stage and engine")
    fig.savefig(output_dir / "kv_pools_by_engine.png", dpi=320)
    fig.savefig(output_dir / "kv_pools_by_engine.pdf")
    plt.close(fig)
    print(json.dumps({"timeseries": str(timeseries), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
