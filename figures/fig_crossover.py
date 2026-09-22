#!/usr/bin/env python3
"""Generate the causal-binding depth sweep from the Gemma-31B results.

Reproduce:
  CIK_DATA=/path/to/context-is-king/data python paper/figures/fig_crossover.py

Input:
  $CIK_DATA/causal/causaluse_gemma-4-31B-it.json

Output:
  paper/figures/fig_crossover.{pdf,png}
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CIK_DATA", HERE / "../data"))
OUTPUT_DIR = Path(os.environ.get("CIK_OUT", HERE / "output")); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT = DATA / "causal/causaluse_gemma-4-31B-it.json"
OUTPUT = OUTPUT_DIR / "fig_crossover"

IMPOSED = "#2B6CB0"
NATURAL = "#1B7F3A"
NEUTRAL = "#4A5568"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def mean_ci(values):
    """Return the mean and pointwise Student-t 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    half_width = (
        student_t.ppf(0.975, values.shape[0] - 1)
        * values.std(axis=0, ddof=1)
        / np.sqrt(values.shape[0])
    )
    return mean, np.clip(mean - half_width, 0, 1), np.clip(mean + half_width, 0, 1)


def main():
    results = json.loads(INPUT.read_text())
    imposed = results["contexts"]["imposed"]["per_scr"]
    natural = results["contexts"]["natural"]["per_scr"]
    layers = results["sweep"]
    layer_count = results["nl"]

    assert len(imposed) == 20
    assert len({tuple(scramble["order"]) for scramble in imposed}) == 20

    natural_curves = [
        [scramble["layers"][str(layer)]["patch_success"] for layer in layers]
        for scramble in natural
    ]
    assert all(curve == natural_curves[0] for curve in natural_curves[1:])

    patch_values = [
        [scramble["layers"][str(layer)]["patch_success"] for layer in layers]
        for scramble in imposed
    ]
    patch_mean, patch_low, patch_high = mean_ci(patch_values)

    depth = np.asarray(layers) / layer_count
    natural_curve = np.asarray(natural_curves[0])
    clean_layer = 27

    figure, axis = plt.subplots(figsize=(9.6, 3.35))
    axis.fill_between(depth, patch_low, patch_high, color=IMPOSED, alpha=0.16, linewidth=0)
    axis.plot(
        depth,
        patch_mean,
        "-o",
        color=IMPOSED,
        linewidth=2.2,
        markersize=5.5,
        label="imposed-order patch (mean, 95% CI)",
    )
    axis.plot(
        depth,
        natural_curve,
        "-^",
        color=NATURAL,
        linewidth=2.2,
        markersize=5.5,
        label="natural-order control (fixed)",
    )
    axis.axhline(1 / 7, color=NEUTRAL, linestyle=":", linewidth=1.4, label="random-answer rate (1/7)")
    axis.axvline(clean_layer / layer_count, color="#A0AEC0", linewidth=1.1)
    axis.text(
        clean_layer / layer_count + 0.012,
        0.82,
        "clean causal locus",
        color=NEUTRAL,
        rotation=90,
        verticalalignment="center",
        fontsize=8.5,
    )

    axis.set_xlabel(r"network depth (layer / $n_{\mathrm{layers}}$)")
    axis.set_ylabel("patched-answer rate")
    axis.set_ylim(-0.035, 1.04)
    axis.set_xlim(depth.min() - 0.035, depth.max() + 0.035)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), framealpha=0.94)

    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(
            OUTPUT.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )

    print(f"wrote {OUTPUT}.{{pdf,png}} from {len(imposed)} imposed orders")


if __name__ == "__main__":
    main()
