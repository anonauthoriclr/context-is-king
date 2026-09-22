#!/usr/bin/env python3
"""Generate the position-sweep handoff figure from the cached activation results.

Reproduce:
  CIK_DATA=/path/to/context-is-king/data python paper/figures/fig_possweep.py

Input:
  $CIK_DATA/geometry/possweep_gemma-4-31B-it_all.npz

Output:
  paper/figures/fig_possweep.{pdf,png}
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CIK_DATA", HERE / "../data"))
OUTPUT_DIR = Path(os.environ.get("CIK_OUT", HERE / "output")); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT = DATA / "geometry/possweep_gemma-4-31B-it_all.npz"
OUTPUT = OUTPUT_DIR / "fig_possweep"

NATURAL = "#2B6CB0"
IMPOSED = "#C23B4A"
NEUTRAL = "#4A5568"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def mds2(centroids):
    n = len(centroids)
    x = centroids - centroids.mean(axis=0)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    distances = 1 - x @ x.T
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ (distances**2) @ center
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    points = vectors[:, order[:2]] * np.sqrt(np.clip(values[order[:2]], 0, None))
    return points / (np.max(np.linalg.norm(points, axis=1)) + 1e-12)


def crossing_count(points, sequence):
    closed = sequence + [sequence[0]]
    edges = [(closed[i], closed[i + 1]) for i in range(len(sequence))]

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    count = 0
    for i, edge_a in enumerate(edges):
        for edge_b in edges[i + 1 :]:
            if set(edge_a) & set(edge_b):
                continue
            p1, p2 = points[edge_a[0]], points[edge_a[1]]
            p3, p4 = points[edge_b[0]], points[edge_b[1]]
            if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                count += 1
    return count


def imposed_sequence(positions):
    return [int(np.where(positions == position)[0][0]) for position in range(len(positions))]


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    count = values.shape[0]
    mean = values.mean(axis=0)
    half_width = student_t.ppf(0.975, count - 1) * values.std(axis=0, ddof=1) / np.sqrt(count)
    return mean, half_width


def ring_panel(axis, centroids, days, imposed, title, subtitle):
    points = mds2(centroids)
    natural = list(range(len(days))) + [0]
    imposed_closed = imposed + [imposed[0]]

    axis.plot(
        points[natural, 0], points[natural, 1], linestyle=(0, (4, 2.4)),
        color=NATURAL, linewidth=1.55, alpha=0.88, zorder=1,
    )
    axis.plot(
        points[imposed_closed, 0], points[imposed_closed, 1],
        color=IMPOSED, linewidth=1.85, alpha=0.95, zorder=2,
    )

    node_colors = plt.get_cmap("twilight")(np.linspace(0, 1, len(days), endpoint=False))
    axis.scatter(
        points[:, 0], points[:, 1], c=node_colors, s=72,
        edgecolor="white", linewidth=1.0, zorder=3,
    )
    for index, day in enumerate(days):
        label = axis.text(
            points[index, 0], points[index, 1], day[:3], ha="center", va="center",
            fontsize=6.2, color="white", zorder=4,
        )
        label.set_path_effects([path_effects.withStroke(linewidth=0.7, foreground="#1A202C")])

    axis.set_title(title, fontweight="bold", pad=13)
    axis.text(
        0.5, 1.02, subtitle, transform=axis.transAxes, ha="center", va="bottom",
        fontsize=7.4, color=NEUTRAL,
    )
    axis.set_aspect("equal")
    x_center = 0.5 * (points[:, 0].min() + points[:, 0].max())
    y_center = 0.5 * (points[:, 1].min() + points[:, 1].max())
    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) * 1.14
    axis.set_xlim(x_center - span / 2, x_center + span / 2)
    axis.set_ylim(y_center - span / 2, y_center + span / 2)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main():
    values = np.load(INPUT, allow_pickle=True)
    fractions = values["fracs"]
    days = list(values["tok"])
    start_centroids = values["cents0"]
    end_centroids = values["cents1"]
    imposed_positions = values["ipos_all"]
    imposed_rsa = values["rimp_all"]
    natural_rsa = values["rnat_all"]

    candidates = []
    for index in range(len(start_centroids)):
        sequence = imposed_sequence(imposed_positions[index])
        crossings = crossing_count(mds2(end_centroids[index]), sequence)
        contrast = imposed_rsa[index, -1] - imposed_rsa[index, 0]
        candidates.append((crossings, -contrast, index))
    candidates.sort()
    # Fixed illustration selected from the original eight-scramble analysis.
    chosen = 1
    imposed = imposed_sequence(imposed_positions[chosen])
    natural_mean, natural_ci = mean_ci(natural_rsa)
    imposed_mean, imposed_ci = mean_ci(imposed_rsa)

    figure = plt.figure(figsize=(7.25, 2.70), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.16, 1.05, 1.05), wspace=0.09)

    axis = figure.add_subplot(grid[0, 0])
    axis.fill_between(
        fractions, natural_mean - natural_ci, natural_mean + natural_ci,
        color=NATURAL, alpha=0.16, linewidth=0,
    )
    axis.fill_between(
        fractions, imposed_mean - imposed_ci, imposed_mean + imposed_ci,
        color=IMPOSED, alpha=0.16, linewidth=0,
    )
    axis.plot(
        fractions, natural_mean, color=NATURAL, marker="o", markersize=3.6,
        linewidth=1.9, label="Pretrained order",
    )
    axis.plot(
        fractions, imposed_mean, color=IMPOSED, marker="s", markersize=3.4,
        linewidth=1.9, label="Context-imposed order",
    )
    axis.axhline(0, color="#A0AEC0", linewidth=0.65, zorder=0)
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(-0.18, 0.86)
    axis.set_xticks(fractions)
    axis.set_xticklabels(["Entity\ntoken", ".2", ".4", ".6", ".8", "Sentence\nend"])
    axis.set_ylabel("Cyclic-distance RSA")
    axis.set_xlabel("Normalized readout position", labelpad=1.5)
    axis.set_title("A   Geometry shifts downstream", loc="left", fontweight="bold", pad=13)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.52, 0.01), frameon=False,
        handlelength=2.3, borderaxespad=0.2,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    ring_panel(
        figure.add_subplot(grid[0, 1]), start_centroids[chosen], days, imposed,
        "B   Entity token", "pretrained order retained",
    )
    ring_panel(
        figure.add_subplot(grid[0, 2]), end_centroids[chosen], days, imposed,
        "C   Sentence end", "context-imposed order dominates",
    )
    figure.align_titles()

    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight", pad_inches=0.025)

    print(
        f"chosen scramble={chosen}; natural RSA {natural_rsa[chosen, 0]:+.2f}->{natural_rsa[chosen, -1]:+.2f}; "
        f"imposed RSA {imposed_rsa[chosen, 0]:+.2f}->{imposed_rsa[chosen, -1]:+.2f}; "
        f"aggregate natural RSA {natural_mean[0]:+.2f}->{natural_mean[-1]:+.2f}; "
        f"aggregate imposed RSA {imposed_mean[0]:+.2f}->{imposed_mean[-1]:+.2f}; "
        f"wrote {OUTPUT}.{{pdf,png}}"
    )


if __name__ == "__main__":
    main()
