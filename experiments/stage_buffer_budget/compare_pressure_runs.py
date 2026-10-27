#!/usr/bin/env python3
"""Compare two measured Talker native KV-pool traces on one replay time axis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_stage_pool_pressure import load, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(10.5, 4.3), layout="constrained")
    last = 0.0
    request_counts = []
    injection_windows = []
    for run, label, color, style in ((args.first_run, args.first_label, "C0", "--"),
                                     (args.second_run, args.second_label, "C1", "-")):
        origin, snapshots, events = load(run)
        xs = [sample[0] for sample in snapshots[1]]
        ys = [sample[1] for sample in snapshots[1]]
        ax.step(xs, ys, where="post", label=label, linewidth=1.7,
                linestyle=style, color=color)
        injection = events["results"]["arrival_window_s"]
        injection_windows.append(injection)
        request_counts.append(events["results"]["planned_requests"])
        ax.axvline(injection, color=color, linewidth=1, linestyle=":", alpha=0.8)
        for failure, _ in events["failure_releases"]:
            ft = failure["monotonic"] - origin
            ax.plot(ft, 100, marker="v", color=color, markersize=6)
        last = max(last, max(row["client_completion_monotonic_s"] - origin
                             for row in events["results"]["results"]))
    ax.set(xlim=(0, math.ceil(last / 5) * 5), ylim=(0, 105),
           xlabel="Time since each replay start (s)",
           ylabel="Talker native KV pool occupancy (% of 1.894 GB pool)",
           title="Measured Talker KV-pool pressure under frozen replays")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.10, -0.025,
             "Dotted lines: injection ends; triangles: actual passive preemptions. "
             f"{request_counts[0]} vs {request_counts[1]} request instances; "
             f"injection windows {injection_windows[0]:g}s / {injection_windows[1]:g}s. "
             "Single run per load; not request-held KV.",
             fontsize=8, color="0.35")
    save(fig, args.output_prefix)


if __name__ == "__main__":
    main()
