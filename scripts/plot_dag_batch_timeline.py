#!/usr/bin/env python3
"""Compare fixed max-fill and dynamic batching for DAG-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


@dataclass(frozen=True)
class Batch:
    start_ms: int
    end_ms: int
    allocations: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Segment:
    req: str
    stage: int
    start_ms: int
    end_ms: int


COLORS = {
    "Tl": "#F2C49B",
    "Ts": "#8ECAE6",
    "A1": "#95D5B2",
    "A2": "#DDB6E6",
    "A3": "#B8C0FF",
}

DEADLINES = {"Ts": 900, "Tl": 1600, "A1": 1650, "A2": 1650, "A3": 1650}

POLICIES = {
    "fixed": {
        "title": "(a) DAG priority + fixed max-fill",
        "batches": [
            Batch(0, 400, (("A1", 10), ("A2", 10), ("Ts", 20))),
            Batch(400, 800, (("Ts", 20), ("A3", 10), ("Tl", 10))),
            Batch(800, 1200, (("Tl", 40),)),
            Batch(1200, 1300, (("Tl", 10),)),
        ],
        "segments": [
            Segment("A1", 1, 400, 800),
            Segment("A2", 1, 800, 1200),
            Segment("A3", 1, 1200, 1600),
            Segment("A1", 2, 800, 900),
            Segment("A2", 2, 1200, 1300),
            Segment("A3", 2, 1600, 1700),
        ],
        "finishes": {"Ts": 800, "Tl": 1300, "A1": 900, "A2": 1300, "A3": 1700},
        "idle_ms": 400,
    },
    "dynamic": {
        "title": "(b) DAG priority + dynamic batch",
        "batches": [
            Batch(0, 100, (("A1", 10),)),
            Batch(100, 500, (("A2", 10), ("Ts", 30))),
            Batch(500, 900, (("Ts", 10), ("A3", 10), ("Tl", 20))),
            Batch(900, 1300, (("Tl", 40),)),
        ],
        "segments": [
            Segment("A1", 1, 100, 500),
            Segment("A2", 1, 500, 900),
            Segment("A3", 1, 900, 1300),
            Segment("A1", 2, 500, 600),
            Segment("A2", 2, 900, 1000),
            Segment("A3", 2, 1300, 1400),
        ],
        "finishes": {"Ts": 900, "Tl": 1300, "A1": 600, "A2": 1000, "A3": 1400},
        "idle_ms": 100,
    },
}


def draw_s0_batch(ax: plt.Axes, batch: Batch, y: float = 2.0) -> None:
    height = 0.62
    total = sum(tokens for _, tokens in batch.allocations)
    bottom = y - height / 2

    for req, tokens in batch.allocations:
        part_height = height * tokens / total
        ax.add_patch(
            Rectangle(
                (batch.start_ms, bottom),
                batch.end_ms - batch.start_ms,
                part_height,
                facecolor=COLORS[req],
                edgecolor="black",
                linewidth=0.55,
                zorder=3,
            )
        )
        ax.text(
            (batch.start_ms + batch.end_ms) / 2,
            bottom + part_height / 2,
            f"{req}:{tokens}",
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="semibold",
            zorder=4,
        )
        bottom += part_height

    ax.add_patch(
        Rectangle(
            (batch.start_ms, y - height / 2),
            batch.end_ms - batch.start_ms,
            height,
            facecolor="none",
            edgecolor="black",
            linewidth=1.05,
            zorder=5,
        )
    )
    ax.text(
        (batch.start_ms + batch.end_ms) / 2,
        y + 0.43,
        f"b={total}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#374151",
    )


def draw_downstream_segment(ax: plt.Axes, segment: Segment) -> None:
    y = 1.0 if segment.stage == 1 else 0.0
    height = 0.48
    hatch = "///" if segment.stage == 1 else "\\\\\\"
    ax.add_patch(
        Rectangle(
            (segment.start_ms, y - height / 2),
            segment.end_ms - segment.start_ms,
            height,
            facecolor=COLORS[segment.req],
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch,
            zorder=3,
        )
    )
    ax.text(
        (segment.start_ms + segment.end_ms) / 2,
        y,
        segment.req,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        zorder=4,
    )


def draw_panel(ax: plt.Axes, policy: dict, show_deadline_labels: bool) -> None:
    for batch in policy["batches"]:
        draw_s0_batch(ax, batch)
    for segment in policy["segments"]:
        draw_downstream_segment(ax, segment)

    idle_ms = policy["idle_ms"]
    ax.add_patch(
        Rectangle(
            (0, 1 - 0.24),
            idle_ms,
            0.48,
            facecolor="#F3F4F6",
            edgecolor="#9CA3AF",
            linewidth=0.8,
            hatch="..",
            zorder=2,
        )
    )
    ax.text(
        idle_ms / 2,
        1,
        f"idle {idle_ms}",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#6B7280",
        zorder=4,
    )

    deadline_lines = [
        (900, "D_Ts", COLORS["Ts"]),
        (1600, "D_Tl", COLORS["Tl"]),
        (1650, "D_A", "#4B5563"),
    ]
    for x, label, color in deadline_lines:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.0, zorder=1)
        if show_deadline_labels:
            ha = "right" if x == 1600 else "left"
            offset = -10 if x == 1600 else 10
            ax.text(x + offset, -0.45, label, ha=ha, va="center", fontsize=7.5, color=color)

    misses = 0
    for req, finish in policy["finishes"].items():
        final_y = 2.0 if req.startswith("T") else 0.0
        missed = finish > DEADLINES[req]
        misses += int(missed)
        ax.plot(
            finish,
            final_y + 0.39,
            marker="x" if missed else "o",
            markersize=7 if missed else 4,
            markeredgewidth=1.6 if missed else 0.8,
            color="#D62728" if missed else "#168A68",
            zorder=6,
        )

    meets = 5 - misses
    ax.text(
        1730,
        2.55,
        f"{meets}/5",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#168A68" if misses == 0 else "#D62728",
    )
    ax.set_title(policy["title"], loc="left", fontsize=11, fontweight="bold", pad=5)
    ax.set_xlim(0, 1750)
    ax.set_ylim(-0.58, 2.82)
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
            "hatch.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.7), sharex=True)
    for index, (ax, policy) in enumerate(zip(axes, POLICIES.values())):
        draw_panel(ax, policy, show_deadline_labels=index == 1)

    axes[-1].set_xlabel("Time (ms)", fontsize=9.5)

    handles = [
        Patch(facecolor=COLORS[name], edgecolor="black", label=name)
        for name in ("Tl", "Ts", "A1", "A2", "A3")
    ]
    handles += [
        Patch(facecolor="#F3F4F6", edgecolor="#9CA3AF", hatch="..", label="idle"),
        Line2D([0], [0], marker="x", color="#D62728", lw=0, label="miss"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=8.3,
        handlelength=1.6,
        columnspacing=1.05,
    )
    fig.text(
        0.5,
        0.922,
        "B0max/B1max/B2max = 40/20/10 tokens",
        ha="center",
        fontsize=8.8,
        color="#374151",
    )
    fig.subplots_adjust(top=0.84, bottom=0.085, left=0.07, right=0.985, hspace=0.42)

    png_path = out_dir / "dag_fixed_vs_dynamic_batch_timeline.png"
    pdf_path = out_dir / "dag_fixed_vs_dynamic_batch_timeline.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
