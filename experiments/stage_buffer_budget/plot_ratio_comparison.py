#!/usr/bin/env python3
"""Compare measured native KV-pool pressure for two frozen replays.

Each line is one independent pool, except the Thinker TP line represents the
same scheduler block state on each synchronized TP rank.  Percentages always
use that pool's own capacity; no GPUs are treated as sharing free blocks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(directory: Path) -> tuple[dict, dict[tuple[int, int], list[tuple[float, float]]]]:
    summary = json.loads((directory / "ratio_summary.json").read_text())
    pools = defaultdict(list)
    with (directory / "kv_pools_by_engine.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            stage, pid = int(row["stage_id"]), int(row["engine_pid"])
            if row["native_usage_pct"]:
                usage = float(row["native_usage_pct"])
            else:
                # The trailing KV-free event has no scheduler snapshot.  The
                # single null block remains reserved in these measured pools.
                used, total = int(row["used_blocks"]), int(row["total_blocks"])
                usage = max(0.0, (used - 1) / (total - 1) * 100)
            pools[stage, pid].append((float(row["relative_time_s"]), usage))
    for values in pools.values():
        values.sort()
    return summary, pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="derived analysis of the first replay")
    parser.add_argument("second", type=Path, help="derived analysis of the second replay")
    parser.add_argument("--first-label", default="4:1")
    parser.add_argument("--second-label", default="2:3")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runs = [(args.first_label, *load(args.first)), (args.second_label, *load(args.second))]
    windows = {summary["client"]["injection_window_s"] for _, summary, _ in runs}
    if len(windows) != 1:
        raise ValueError("injection windows differ; plots cannot share an arrival marker")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True, sharey=True,
                             layout="constrained")
    for row_index, (label, summary, pools) in enumerate(runs):
        gpu_map = {int(pid): gpu for pid, gpu in summary.get("gpu_by_engine_pid", {}).items()}
        for stage in (0, 1):
            ax = axes[row_index, stage]
            keys = sorted(key for key in pools if key[0] == stage)
            capacities = {summary["pools"][f"{sid}:{pid}"]["pool_bytes_per_rank"] for sid, pid in keys}
            if len(capacities) != 1:
                raise ValueError(f"pool capacities differ within {label} stage {stage}")
            capacity_gb = next(iter(capacities)) / 1e9
            for _, pid in keys:
                samples = pools[stage, pid]
                gpu = gpu_map.get(pid)
                curve_label = (f"GPU {gpu}" if gpu is not None else
                               ("single pool" if len(keys) == 1 else f"engine PID {pid}"))
                ax.step([t for t, _ in samples], [usage for _, usage in samples],
                        where="post", linewidth=1.45, label=curve_label)
            failures = sum(summary["events_by_pool"].get(f"{sid}:{pid}", {}).get("passive_preemption", 0)
                           for sid, pid in keys)
            stage_name = "Thinker (per TP rank)" if stage == 0 else "Talker (independent pools)"
            ax.set_title(f"{label}  ·  {stage_name}\n{capacity_gb:.3f} GB/pool, {failures} passive preemptions")
            ax.axvline(next(iter(windows)), color="0.45", linestyle=":", linewidth=1)
            ax.grid(axis="y", alpha=.2)
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Native KV pool used (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since replay start (s)")
    for ax in axes.flat:
        ax.set_xlim(0, 270)
        ax.set_ylim(0, 105)
    fig.suptitle("Measured KV-pool pressure | same 280 requests in 30 s")
    fig.text(.01, -.017,
             "Dotted line: injection ends. One run per ratio. Pool sizes and Thinker model length differ; "
             "2:3 has two residual Talker input-wait states after client completion. Native allocator use is not B_live.",
             fontsize=8, color="0.35")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
