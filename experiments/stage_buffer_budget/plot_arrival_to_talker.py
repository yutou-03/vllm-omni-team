#!/usr/bin/env python3
"""Plot measured client sends, Talker input readiness, and native pool pressure.

This is a first-100-s diagnostic view; it does not replace the full drain
figure, and native pool usage is not request-held B_live.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir: Path) -> dict:
    result = json.loads((run_dir / "results.json").read_text())
    origin = result["origin_monotonic_s"]
    sent = sorted(row["send_monotonic_s"] - origin for row in result["results"])
    ready_by_id: dict[str, float] = {}
    occupancy = []
    preemptions = []
    with (run_dir / "traces" / "stage_events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("stage_id") != 1:
                continue
            event = row.get("event")
            t = row.get("monotonic", 0.0) - origin
            if event == "stage_data_ready" and row.get("pipeline_attempt_id") == 0:
                request_id = row.get("logical_request_id") or row.get("request_id")
                ready_by_id.setdefault(str(request_id), t)
            elif event == "stage_study_kv_snapshot" and row.get("observation_point") == "before_schedule":
                occupancy.append((t, 100.0 * row["kv_cache_usage"]))
            elif event == "stage_study_profile_victim_selected":
                preemptions.append(t)
    if len(sent) != result["planned_requests"] or len(ready_by_id) != result["planned_requests"]:
        raise ValueError(f"incomplete send or Talker-ready trace in {run_dir}")
    return {
        "sent": sent,
        "ready": sorted(ready_by_id.values()),
        "occupancy": sorted(occupancy),
        "preemptions": sorted(preemptions),
        "injection_window_s": result["arrival_window_s"],
    }


def cumulative(times: list[float]) -> tuple[list[float], list[int]]:
    return [0.0, *times], [0, *range(1, len(times) + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("second_run", type=Path)
    parser.add_argument("--first-label", default="3×")
    parser.add_argument("--second-label", default="4×")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--view-end-s", type=float, default=100.0)
    args = parser.parse_args()
    if args.view_end_s <= 0:
        parser.error("--view-end-s must be positive")
    runs = [(load(args.first_run), args.first_label, "C0"),
            (load(args.second_run), args.second_label, "C1")]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.6, 6.9), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.1]},
                                     layout="constrained")
    for run, label, color in runs:
        x, y = cumulative(run["sent"])
        top.step(x, y, where="post", color=color, linewidth=1.8,
                 label=f"{label}: actual client send")
        x, y = cumulative(run["ready"])
        top.step(x, y, where="post", color=color, linestyle="--", linewidth=1.8,
                 label=f"{label}: Talker input ready")
        bottom.step([x for x, _ in run["occupancy"]],
                    [y for _, y in run["occupancy"]], where="post", color=color,
                    linewidth=1.5, label=label)
        for t in run["preemptions"]:
            bottom.plot(t, 100, marker="v", color=color, markersize=7)
        for ax in (top, bottom):
            ax.axvline(run["injection_window_s"], color=color, linestyle=":",
                       linewidth=1.0, alpha=0.65)
    top.set(ylabel="Cumulative requests", ylim=(0, 72),
            title="Client arrivals reach Talker at a different pace")
    top.legend(loc="lower right", ncol=2, fontsize=9, frameon=False)
    bottom.axhline(90, color="0.5", linestyle=":", linewidth=1)
    bottom.set(xlabel="Time since each replay start (s)",
               ylabel="Talker native KV pool occupancy (%)", ylim=(0, 105),
               xlim=(0, args.view_end_s))
    bottom.legend(loc="upper left", frameon=False)
    for ax in (top, bottom):
        ax.grid(axis="y", alpha=0.2)
    fig.text(0.02, -0.015,
             "Dotted vertical lines: injection ends; triangles: actual passive preemptions. "
             "First 100 s diagnostic; each run has 70 requests. Native pool usage is not B_live.",
             fontsize=8, color="0.35")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
