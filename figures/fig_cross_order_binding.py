#!/usr/bin/env python3
"""Validate and plot the nine-layer cross-order intervention sweep.

The original and extension runs remain separate raw artifacts. This script first
checks that their model, prompt, seed, order pairs, and scoring design match, then
combines their disjoint layer summaries for visualization.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CIK_DATA", HERE / "../data"))
OUTPUT_DIR = Path(os.environ.get("CIK_OUT", HERE / "output"))


MATCH_FIELDS = (
    "experiment",
    "model",
    "concept",
    "seed",
    "npairs",
    "number_of_layers",
    "min_successor_differences",
    "order_pair_design",
    "prompt_template",
    "decisions",
)


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def validate_and_merge(original, extension):
    mismatches = [field for field in MATCH_FIELDS if original[field] != extension[field]]
    if mismatches:
        raise ValueError(f"run metadata differ for: {', '.join(mismatches)}")

    original_layers = set(original["layers"])
    extension_layers = set(extension["layers"])
    overlap = original_layers & extension_layers
    if overlap:
        raise ValueError(f"runs contain overlapping layers: {sorted(overlap)}")

    aggregate = {}
    for run in (original, extension):
        for layer, values in run["summary"]["aggregate_by_layer"].items():
            if values["positive_success"]["n"] != original["npairs"]:
                raise ValueError(f"layer {layer} does not contain all order pairs")
            aggregate[int(layer)] = values

    expected = sorted(original_layers | extension_layers)
    if sorted(aggregate) != expected:
        raise ValueError("aggregate layer set does not match declared layer set")
    return aggregate


def series(aggregate, metric):
    layers = np.array(sorted(aggregate), dtype=float)
    means = np.array([aggregate[int(layer)][metric]["mean"] for layer in layers])
    cis = np.array([aggregate[int(layer)][metric]["ci95"] for layer in layers])
    errors = np.vstack((means - cis[:, 0], cis[:, 1] - means))
    return layers, means, errors


def draw_metric(ax, aggregate, metric, label, color, marker, linestyle="-"):
    layers, means, errors = series(aggregate, metric)
    ax.errorbar(
        layers,
        means,
        yerr=errors,
        color=color,
        marker=marker,
        markersize=4.8,
        markeredgewidth=0.9,
        linewidth=1.65,
        elinewidth=1.0,
        capsize=2.2,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


def write_table(path, aggregate):
    metrics = (
        "recipient_map_donor_rate",
        "donor_map_donor_rate",
        "stuck_source_rate",
        "other_rate",
        "positive_success",
        "same_entity_recipient_map_rate",
        "sham_success",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "metric", "n_pairs", "mean", "ci95_low", "ci95_high"])
        for layer in sorted(aggregate):
            for metric in metrics:
                item = aggregate[layer][metric]
                writer.writerow([layer, metric, item["n"], item["mean"], *item["ci95"]])


def make_figure(aggregate, output_prefix):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.75,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharey=True)
    ax_outcomes, ax_controls = axes

    draw_metric(
        ax_outcomes,
        aggregate,
        "recipient_map_donor_rate",
        "Recipient-context successor",
        "#0072B2",
        "o",
    )
    draw_metric(
        ax_outcomes,
        aggregate,
        "donor_map_donor_rate",
        "Donor-context successor",
        "#D55E00",
        "s",
    )
    draw_metric(
        ax_outcomes,
        aggregate,
        "stuck_source_rate",
        "Original source answer",
        "#666666",
        "^",
    )
    draw_metric(
        ax_outcomes,
        aggregate,
        "other_rate",
        "Other answer",
        "#CC79A7",
        "D",
        linestyle=":",
    )
    ax_outcomes.set_title("A  Cross-order patch outcome", loc="left", fontweight="bold")
    ax_outcomes.set_ylabel("Fraction of trials")
    ax_outcomes.legend(loc="center left", bbox_to_anchor=(0.015, 0.56), frameon=False)

    draw_metric(
        ax_controls,
        aggregate,
        "positive_success",
        "Different entity, same order",
        "#111111",
        "o",
    )
    draw_metric(
        ax_controls,
        aggregate,
        "same_entity_recipient_map_rate",
        "Same entity, different order",
        "#009E73",
        "s",
        linestyle="--",
    )
    ax_controls.axhline(
        1.0,
        color="#A0A0A0",
        linewidth=1.0,
        linestyle=":",
        label="Sham control",
        zorder=1,
    )
    ax_controls.annotate(
        "identity transfer\ncollapses",
        xy=(36, aggregate[36]["positive_success"]["mean"]),
        xytext=(41, 0.34),
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
        color="#444444",
        fontsize=7.4,
        ha="left",
    )
    ax_controls.set_title("B  Intervention controls", loc="left", fontweight="bold")
    ax_controls.legend(loc="center left", bbox_to_anchor=(0.015, 0.56), frameon=False)

    layers = sorted(aggregate)
    for ax in axes:
        ax.set_xlim(min(layers) - 2, max(layers) + 2)
        ax.set_ylim(-0.04, 1.045)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer (of 60)")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.005,
        "Means and Student-t 95% CIs across 20 independently sampled order pairs; 1,290 primary trials per layer.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.22, wspace=0.18)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)
    write_table(prefix.with_suffix(".csv"), aggregate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", nargs="?", type=Path, default=DATA / "causal/crossorder_gemma-4-31B-it_n20.json")
    parser.add_argument("extension", nargs="?", type=Path, default=DATA / "causal/crossorder_gemma-4-31B-it_n20_extension.json")
    parser.add_argument("output_prefix", nargs="?", type=Path, default=OUTPUT_DIR / "fig_cross_order_binding")
    args = parser.parse_args()

    original = load_json(args.original)
    extension = load_json(args.extension)
    aggregate = validate_and_merge(original, extension)
    make_figure(aggregate, args.output_prefix)
    print(f"Validated matching metadata and order pairs across {len(aggregate)} layers.")
    print(f"Wrote {args.output_prefix}.pdf, .png, and .csv")


if __name__ == "__main__":
    main()
