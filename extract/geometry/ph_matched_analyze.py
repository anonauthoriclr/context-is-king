#!/usr/bin/env python3
"""Analyze matched cycle/tree activations with persistent homology.

Primary quantity: longest H1 lifetime divided by the median off-diagonal
distance.  The query-aware null independently permutes entity identities within
each query template before centroiding, preserving query effects and activation
anisotropy while removing entity geometry shared across templates.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ripser import ripser


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--null-reps", type=int, default=1000)
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--outdir", default="results/geometry/ph_matched")
    return p.parse_args()


def cosine_rdm(centroids):
    x = centroids.astype(np.float64) - centroids.astype(np.float64).mean(axis=0)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    d = np.clip(1.0 - x @ x.T, 0.0, None)
    np.fill_diagonal(d, 0.0)
    return d


def h1_stats(centroids):
    d = cosine_rdm(centroids)
    bars = ripser(d, maxdim=1, distance_matrix=True)["dgms"][1]
    scale = float(np.median(d[np.triu_indices(len(d), 1)]))
    if not len(bars):
        return {"top": 0.0, "second": 0.0, "normalized_top": 0.0, "dominance": 0.0, "bars": []}
    finite = bars[np.isfinite(bars[:, 1])]
    persistence = finite[:, 1] - finite[:, 0]
    order = np.argsort(persistence)[::-1]
    finite = finite[order]
    persistence = persistence[order]
    top = float(persistence[0]) if len(persistence) else 0.0
    second = float(persistence[1]) if len(persistence) > 1 else 0.0
    return {
        "top": top,
        "second": second,
        "normalized_top": top / (scale + 1e-12),
        "dominance": top / (second + 1e-12),
        "scale": scale,
        "bars": [[float(b), float(dth), float(p)] for (b, dth), p in zip(finite, persistence)],
    }


def metric_from_raw(raw):
    return h1_stats(raw.astype(np.float32).mean(axis=1))["normalized_top"]


def summarize_file(path, null_reps, boot_reps, seed):
    z = np.load(path, allow_pickle=False)
    raw = z["raw"].astype(np.float32)  # condition, scramble, entity, template, hidden
    conditions = [str(x) for x in z["conditions"]]
    rng = np.random.default_rng(seed + int(raw.shape[2]))
    result = {
        "input": Path(path).name,
        "N": int(raw.shape[2]),
        "nscr": int(raw.shape[1]),
        "n_templates": int(raw.shape[3]),
        "conditions": {},
    }
    for ci, condition in enumerate(conditions):
        observed_full = [h1_stats(raw[ci, si].mean(axis=1)) for si in range(raw.shape[1])]
        observed = np.array([x["normalized_top"] for x in observed_full])

        null_means = np.empty(null_reps)
        for bi in range(null_reps):
            vals = []
            for si in range(raw.shape[1]):
                shuffled = raw[ci, si].copy()
                for ti in range(raw.shape[3]):
                    shuffled[:, ti] = shuffled[rng.permutation(raw.shape[2]), ti]
                vals.append(metric_from_raw(shuffled))
            null_means[bi] = np.mean(vals)

        boot_means = np.empty(boot_reps)
        for bi in range(boot_reps):
            scramble_idx = rng.integers(0, raw.shape[1], raw.shape[1])
            template_idx = rng.integers(0, raw.shape[3], raw.shape[3])
            vals = [metric_from_raw(raw[ci, si][:, template_idx]) for si in scramble_idx]
            boot_means[bi] = np.mean(vals)

        obs_mean = float(observed.mean())
        result["conditions"][condition] = {
            "normalized_top_h1_mean": obs_mean,
            "normalized_top_h1_std": float(observed.std(ddof=1)),
            "bootstrap_ci95": [float(x) for x in np.percentile(boot_means, [2.5, 97.5])],
            "query_null_mean": float(null_means.mean()),
            "query_null_p95": float(np.percentile(null_means, 95)),
            "empirical_p": float((1 + np.sum(null_means >= obs_mean)) / (null_reps + 1)),
            "per_scramble": observed_full,
        }
    return result


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = [summarize_file(p, args.null_reps, args.boot_reps, args.seed) for p in args.inputs]
    output = {
        "metric": "longest H1 lifetime / median off-diagonal mean-centered cosine distance",
        "null": "independent entity-label permutation within each query template before centroiding",
        "bootstrap": "crossed resampling of scrambles and query templates",
        "null_reps": args.null_reps,
        "bootstrap_reps": args.boot_reps,
        "results": results,
    }
    json_path = outdir / "ph_matched_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    csv_path = outdir / "ph_matched_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "N", "condition", "mean", "ci_low", "ci_high", "null_p95", "p_empirical"])
        for result in results:
            for condition, stats in result["conditions"].items():
                writer.writerow([
                    result["input"], result["N"], condition,
                    stats["normalized_top_h1_mean"], *stats["bootstrap_ci95"],
                    stats["query_null_p95"], stats["empirical_p"],
                ])
                print(
                    f"N={result['N']:2d} {condition:5s}: {stats['normalized_top_h1_mean']:.3f} "
                    f"CI [{stats['bootstrap_ci95'][0]:.3f}, {stats['bootstrap_ci95'][1]:.3f}] "
                    f"null95={stats['query_null_p95']:.3f} p={stats['empirical_p']:.4f}",
                    flush=True,
                )
    print(f"WROTE {json_path.resolve()}\nWROTE {csv_path.resolve()}")


if __name__ == "__main__":
    main()
