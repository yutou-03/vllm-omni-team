#!/usr/bin/env python3
"""Plot measured native KV-pool occupancy and a passive preemption detail.

These plots are allocator pool observations, not physical-block B_live ownership.
No smoothing, interpolation, or cross-GPU capacity pooling is applied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir: Path) -> tuple[float, dict[int, list[tuple[float, float, int]]], dict]:
    result = json.loads((run_dir / "results.json").read_text())
    origin = result["origin_monotonic_s"]
    snapshots = {0: [], 1: []}
    pool_blocks = {}
    failures = []
    releases = []
    last_free = {0: None, 1: None}
    with (run_dir / "traces" / "stage_events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            stage = row.get("stage_id")
            event = row.get("event")
            t = row.get("monotonic", 0) - origin
            if event == "stage_study_kv_snapshot" and stage in snapshots:
                if row.get("observation_point") == "before_schedule":
                    pool_blocks[stage] = row["kv_total_blocks"]
                    snapshots[stage].append((t, 100 * row["kv_cache_usage"], row["kv_free_blocks"]))
            elif stage == 1 and event == "stage_study_profile_allocation_result":
                if row.get("allocation_succeeded") is False:
                    failures.append(row)
            elif stage in last_free and event == "stage_study_profile_kv_free":
                last_free[stage] = row
                if stage == 1 and row.get("free_blocks_after", 0) > row.get("free_blocks_before", 0):
                    releases.append(row)
    failure = min(failures, key=lambda row: row["monotonic"]) if failures else None
    released = (min((row for row in releases
                     if failure is not None and
                     0 <= row["monotonic"] - failure["monotonic"] < 1),
                    key=lambda row: row["monotonic"], default=None))
    for stage in snapshots:
        snapshots[stage].sort(key=lambda sample: sample[0])
        if not snapshots[stage]:
            raise ValueError(f"no stage {stage} KV snapshots")
        tail = last_free[stage]
        if tail is not None and tail["monotonic"] - origin > snapshots[stage][-1][0]:
            total = pool_blocks[stage]
            snapshots[stage].append((tail["monotonic"] - origin,
                                     100 * (total - tail["free_blocks_after"]) / total,
                                     tail["free_blocks_after"]))
    return origin, snapshots, {"failure": failure, "release": released, "results": result}


def save(fig, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_full(snapshots: dict, events: dict, prefix: Path) -> None:
    results = events["results"]
    last_completion = max(row["client_completion_monotonic_s"] for row in results["results"])
    xmax = math.ceil((last_completion - results["origin_monotonic_s"]) / 5) * 5
    fig, ax = plt.subplots(figsize=(10.5, 4.3), layout="constrained")
    for stage, label, style in ((0, "Thinker (stage 0)", "--"),
                                (1, "Talker (stage 1)", "-")):
        xs = [sample[0] for sample in snapshots[stage]]
        ys = [sample[1] for sample in snapshots[stage]]
        ax.step(xs, ys, where="post", label=label, linewidth=1.7, linestyle=style)
    injection = results["arrival_window_s"]
    ax.axvline(injection, color="0.45", linewidth=1, linestyle=":")
    ax.text(injection + 1, 6, "Injection ends", fontsize=9, color="0.35")
    failure = events["failure"]
    if failure is not None:
        ft = failure["monotonic"] - results["origin_monotonic_s"]
        ax.axvline(ft, color="C3", linewidth=1.2, linestyle=":")
        ax.text(ft + 1, 91, "KV allocation fails; passive preemption", fontsize=9,
                color="C3", va="top")
    ax.set(xlim=(0, xmax), ylim=(0, 105), xlabel="Time since replay start (s)",
           ylabel="Native KV pool occupancy (% of each stage's pool)",
           title="Measured stage KV-pool pressure | 3× original arrivals")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.10, -0.025,
             "Before-schedule snapshots plus the final KV-free event; stage 0 trace ends when idle. "
             "Pool usage is not request-held KV.", fontsize=8, color="0.35")
    save(fig, prefix)


def plot_detail(origin: float, snapshots: dict, events: dict, prefix: Path) -> bool:
    failure, release = events["failure"], events["release"]
    if failure is None or release is None:
        return False
    ft = failure["monotonic"] - origin
    rt = release["monotonic"] - origin
    if not 0 <= rt - ft < 1:
        raise ValueError("nearest KV release does not match the actual allocation failure")
    xs = [sample[0] for sample in snapshots[1]]
    free = [sample[2] for sample in snapshots[1]]
    lo, hi = ft - 0.35, ft + 0.35
    selected = [(x, y) for x, y in zip(xs, free) if lo <= x <= hi]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.step([x for x, _ in selected], [y for _, y in selected], where="post",
            label="Scheduler snapshots", color="C0", linewidth=1.7)
    before, after = release["free_blocks_before"], release["free_blocks_after"]
    ax.plot([rt, rt], [before, after], marker="o", color="C1", linewidth=2.2,
            label=f"Actual KV release: {before} → {after} blocks")
    ax.axvline(ft, color="C3", linestyle=":", linewidth=1.3,
               label="Actual allocation failure / passive preemption")
    ax.set(xlim=(lo, hi), ylim=(-4, max(100, after + 10)),
           xlabel="Time since replay start (s)", ylabel="Talker free KV blocks",
           title="Talker pool around the first measured passive preemption")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, prefix)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    origin, snapshots, events = load(args.run_dir)
    plot_full(snapshots, events, args.run_dir / "pool_occupancy_full_3x")
    plot_detail(origin, snapshots, events, args.run_dir / "talker_free_blocks_preemption_detail_3x")


if __name__ == "__main__":
    main()
