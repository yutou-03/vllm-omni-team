#!/usr/bin/env python3
"""Plot compact batch-aware timelines for three scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


@dataclass(frozen=True)
class Request:
    label: str
    kind: str
    deadline_ms: int
    color: str


@dataclass(frozen=True)
class Segment:
    req: str
    stage: int
    start_ms: int
    end_ms: int


REQUESTS = {
    "T_long": Request("Tl", "text", 1600, "#F2C49B"),
    "T_short": Request("Ts", "text", 900, "#8ECAE6"),
    "A1": Request("A1", "audio", 1650, "#95D5B2"),
    "A2": Request("A2", "audio", 1650, "#DDB6E6"),
    "A3": Request("A3", "audio", 1650, "#B8C0FF"),
}

# One full batch takes 200/400/100 ms at S0/S1/S2, respectively.
BATCH_MS = {0: 200, 1: 400, 2: 100}

POLICIES = {
    "FCFS": [
        Segment("T_long", 0, 0, 600),
        Segment("T_short", 0, 600, 1000),
        Segment("A1", 0, 1000, 1200),
        Segment("A2", 0, 1200, 1400),
        Segment("A3", 0, 1400, 1600),
        Segment("A1", 1, 1200, 1600),
        Segment("A2", 1, 1600, 2000),
        Segment("A3", 1, 2000, 2400),
        Segment("A1", 2, 1600, 1700),
        Segment("A2", 2, 2000, 2100),
        Segment("A3", 2, 2400, 2500),
    ],
    "Final-deadline EDF": [
        Segment("T_short", 0, 0, 400),
        Segment("T_long", 0, 400, 1000),
        Segment("A1", 0, 1000, 1200),
        Segment("A2", 0, 1200, 1400),
        Segment("A3", 0, 1400, 1600),
        Segment("A1", 1, 1200, 1600),
        Segment("A2", 1, 1600, 2000),
        Segment("A3", 1, 2000, 2400),
        Segment("A1", 2, 1600, 1700),
        Segment("A2", 2, 2000, 2100),
        Segment("A3", 2, 2400, 2500),
    ],
    "DAG checkpoint": [
        Segment("A1", 0, 0, 200),
        Segment("A2", 0, 200, 400),
        Segment("T_short", 0, 400, 800),
        Segment("A3", 0, 800, 1000),
        Segment("T_long", 0, 1000, 1600),
        Segment("A1", 1, 200, 600),
        Segment("A2", 1, 600, 1000),
        Segment("A3", 1, 1000, 1400),
        Segment("A1", 2, 600, 700),
        Segment("A2", 2, 1000, 1100),
        Segment("A3", 2, 1400, 1500),
    ],
}

PANEL_TITLES = {
    "FCFS": "(a) FCFS",
    "Final-deadline EDF": "(b) Final-deadline EDF",
    "DAG checkpoint": "(c) DAG checkpoint",
}

STAGE_HATCH = {0: "", 1: "///", 2: "\\\\\\"}


def final_finish_times(segments: list[Segment]) -> dict[str, int]:
    finishes: dict[str, int] = {}
    for seg in segments:
        req = REQUESTS[seg.req]
        if (req.kind == "text" and seg.stage == 0) or (
            req.kind == "audio" and seg.stage == 2
        ):
            finishes[seg.req] = seg.end_ms
    return finishes


def draw_segment(ax: plt.Axes, seg: Segment, lane_y: dict[int, float]) -> None:
    req = REQUESTS[seg.req]
    y = lane_y[seg.stage]
    height = 0.48
    ax.add_patch(
        Rectangle(
            (seg.start_ms, y - height / 2),
            seg.end_ms - seg.start_ms,
            height,
            facecolor=req.color,
            edgecolor="black",
            linewidth=0.9,
            hatch=STAGE_HATCH[seg.stage],
            zorder=3,
        )
    )

    # Thin cuts expose the fixed max-token batch boundaries.
    cut = seg.start_ms + BATCH_MS[seg.stage]
    while cut < seg.end_ms:
        ax.vlines(
            cut,
            y - height / 2,
            y + height / 2,
            color="black",
            linewidth=0.75,
            zorder=4,
        )
        cut += BATCH_MS[seg.stage]

    ax.text(
        (seg.start_ms + seg.end_ms) / 2,
        y,
        req.label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
        zorder=5,
    )


def draw_policy(
    ax: plt.Axes,
    policy: str,
    segments: list[Segment],
    show_deadline_labels: bool,
) -> None:
    lane_y = {0: 2.0, 1: 1.0, 2: 0.0}
    finishes = final_finish_times(segments)

    for seg in segments:
        draw_segment(ax, seg, lane_y)

    deadlines = [
        (900, "D_Ts", REQUESTS["T_short"].color),
        (1600, "D_Tl", REQUESTS["T_long"].color),
        (1650, "D_A", "#4B5563"),
    ]
    for x, label, color in deadlines:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
        if show_deadline_labels:
            align = "right" if x == 1600 else "left"
            offset = -12 if x == 1600 else 12
            ax.text(
                x + offset,
                2.48,
                label,
                ha=align,
                va="center",
                fontsize=8,
                color=color,
                fontweight="semibold",
            )

    misses = 0
    for req_name, finish in finishes.items():
        req = REQUESTS[req_name]
        final_stage = 0 if req.kind == "text" else 2
        missed = finish > req.deadline_ms
        misses += int(missed)
        ax.plot(
            finish,
            lane_y[final_stage] + 0.34,
            marker="x" if missed else "o",
            markersize=7 if missed else 4,
            markeredgewidth=1.6 if missed else 0.8,
            color="#D62728" if missed else "#168A68",
            zorder=6,
        )

    meets = len(finishes) - misses
    ax.text(
        2480,
        2.48,
        f"{meets}/5",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#168A68" if misses == 0 else "#D62728",
    )

    ax.set_title(PANEL_TITLES[policy], loc="left", fontsize=11, fontweight="bold", pad=5)
    ax.set_xlim(0, 2520)
    ax.set_ylim(-0.55, 2.7)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["S0", "S1", "S2"], fontsize=9)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    out_dir = Path("output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "hatch.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 6.7), sharex=True)
    for index, (ax, (policy, segments)) in enumerate(zip(axes, POLICIES.items())):
        draw_policy(ax, policy, segments, show_deadline_labels=index == 0)

    axes[-1].set_xlabel("Time (ms)", fontsize=9.5)

    request_handles = [
        Patch(facecolor=req.color, edgecolor="black", label=req.label)
        for req in REQUESTS.values()
    ]
    request_handles += [
        Line2D([0], [0], marker="x", color="#D62728", lw=0, label="miss"),
    ]
    fig.legend(
        handles=request_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.text(
        0.5,
        0.952,
        "Batch budgets  B0/B1/B2 = 20/20/10 tokens",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.07, right=0.985, hspace=0.42)

    png_path = out_dir / "stage_slo_policy_timeline.png"
    pdf_path = out_dir / "stage_slo_policy_timeline.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
