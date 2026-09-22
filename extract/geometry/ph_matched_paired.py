#!/usr/bin/env python3
"""Paired cycle-versus-tree follow-up for the matched PH audit.

Each cycle/tree pair shares model, N, token-to-node assignment, query templates,
layer, and readout.  We report an exact paired sign-flip test across scrambles
and a crossed paired bootstrap that resamples scrambles and query templates
together in the two conditions.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from ripser import ripser


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--boot-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--outdir", default="results/geometry/ph_matched")
    return p.parse_args()


def h1(centroids):
    x = centroids.astype(np.float64)
    x -= x.mean(axis=0)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    d = np.clip(1.0 - x @ x.T, 0.0, None)
    np.fill_diagonal(d, 0.0)
    scale = float(np.median(d[np.triu_indices(len(d), 1)]))
    bars = ripser(d, maxdim=1, distance_matrix=True)["dgms"][1]
    bars = bars[np.isfinite(bars[:, 1])] if len(bars) else bars
    persistence = np.sort(bars[:, 1] - bars[:, 0])[::-1] if len(bars) else np.array([])
    top = float(persistence[0]) if len(persistence) else 0.0
    second = float(persistence[1]) if len(persistence) > 1 else 0.0
    return {
        "top": top / (scale + 1e-12),
        "second": second / (scale + 1e-12),
        "margin": (top - second) / (scale + 1e-12),
        "n_bars": int(len(persistence)),
    }


def exact_sign_flip(differences):
    observed = float(np.mean(differences))
    null = np.array([
        np.mean(np.asarray(signs) * differences)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ])
    return {
        "one_sided_p": float(np.mean(null >= observed - 1e-15)),
        "two_sided_p": float(np.mean(np.abs(null) >= abs(observed) - 1e-15)),
    }


def analyze(path, boot_reps, seed):
    z = np.load(path, allow_pickle=False)
    raw = z["raw"].astype(np.float32)
    conditions = [str(x) for x in z["conditions"]]
    cycle_i, tree_i = conditions.index("cycle"), conditions.index("tree")
    nscr, n_templates = raw.shape[1], raw.shape[3]

    per = []
    for si in range(nscr):
        cyc = h1(raw[cycle_i, si].mean(axis=1))
        tree = h1(raw[tree_i, si].mean(axis=1))
        per.append({
            "scramble": si + 1,
            "cycle": cyc,
            "tree": tree,
            "top_difference": cyc["top"] - tree["top"],
            "margin_difference": cyc["margin"] - tree["margin"],
        })

    top_diff = np.array([x["top_difference"] for x in per])
    margin_diff = np.array([x["margin_difference"] for x in per])
    rng = np.random.default_rng(seed + int(raw.shape[2]))
    boot_top = np.empty(boot_reps)
    boot_margin = np.empty(boot_reps)
    for bi in range(boot_reps):
        scramble_idx = rng.integers(0, nscr, nscr)
        template_idx = rng.integers(0, n_templates, n_templates)
        td, md = [], []
        for si in scramble_idx:
            cyc = h1(raw[cycle_i, si][:, template_idx].mean(axis=1))
            tree = h1(raw[tree_i, si][:, template_idx].mean(axis=1))
            td.append(cyc["top"] - tree["top"])
            md.append(cyc["margin"] - tree["margin"])
        boot_top[bi] = np.mean(td)
        boot_margin[bi] = np.mean(md)

    return {
        "input": Path(path).name,
        "N": int(raw.shape[2]),
        "nscr": nscr,
        "n_templates": n_templates,
        "per_scramble": per,
        "top_difference": {
            "mean": float(top_diff.mean()),
            "positive_scrambles": int(np.sum(top_diff > 0)),
            "ci95_crossed_bootstrap": [float(x) for x in np.percentile(boot_top, [2.5, 97.5])],
            **exact_sign_flip(top_diff),
        },
        "dominance_margin_difference": {
            "mean": float(margin_diff.mean()),
            "positive_scrambles": int(np.sum(margin_diff > 0)),
            "ci95_crossed_bootstrap": [float(x) for x in np.percentile(boot_margin, [2.5, 97.5])],
            **exact_sign_flip(margin_diff),
        },
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = [analyze(path, args.boot_reps, args.seed) for path in args.inputs]
    out = {
        "primary_contrast": "normalized top-H1 persistence: cycle minus matched tree",
        "secondary_contrast": "normalized (top-H1 minus second-H1) persistence margin: cycle minus matched tree",
        "exact_test": "all 2^10 paired sign flips across matched scrambles",
        "bootstrap": "paired crossed resampling of scrambles and query templates",
        "bootstrap_reps": args.boot_reps,
        "results": results,
    }
    path = outdir / "ph_matched_paired_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    for r in results:
        t = r["top_difference"]
        m = r["dominance_margin_difference"]
        print(
            f"{Path(r['input']).stem:34s} N={r['N']:2d} "
            f"top diff={t['mean']:+.3f} CI[{t['ci95_crossed_bootstrap'][0]:+.3f},"
            f"{t['ci95_crossed_bootstrap'][1]:+.3f}] exact p2={t['two_sided_p']:.4f}; "
            f"margin diff={m['mean']:+.3f} CI[{m['ci95_crossed_bootstrap'][0]:+.3f},"
            f"{m['ci95_crossed_bootstrap'][1]:+.3f}] p2={m['two_sided_p']:.4f}",
            flush=True,
        )
    print(f"WROTE {path.resolve()}")


if __name__ == "__main__":
    main()
