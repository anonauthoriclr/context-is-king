#!/usr/bin/env python3
"""Regenerate all paper figures from the cached data in ../data/.

Runs each figure script in turn and reports which succeeded. Outputs land in
figures/output/ (override with CIK_OUT). Data is read from ../data/ (override
with CIK_DATA). No GPU or model weights required; all cached data is committed,
so this works straight from a clone.

    python make_all.py
    python make_all.py --list        # show the figure -> script mapping
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output"))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(HERE, "..", "data")))

# (paper label, script) — the canonical figure producers.
FIGURES = [
    ("fig:flip",       "fig1_flip.py"),
    ("fig:defs",       "fig_defs.py"),
    ("fig:dominance",  "fig_dominance.py"),
    ("fig:hierarchy",  "fig_hierarchy.py"),
    ("fig:possweep",   "geom_possweep_plot.py"),
    ("fig:layersweep", "fig_layersweep.py"),
    ("fig:kmarg",      "k_marg_plot.py"),
    ("fig:behavior",   "fig_behavior.py"),
    ("fig:regimes",    "fig_regimes.py"),
    ("fig:topology",   "fig_topology.py"),
    ("fig:shapes",     "build_paper_fig.py"),
    ("fig:queryreloc", "demo_tree_queryfig.py"),
]


def main() -> int:
    if "--list" in sys.argv:
        for label, script in FIGURES:
            print(f"{label:16} {script}")
        return 0

    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, CIK_DATA=DATA, CIK_OUT=OUT)
    print(f"data:   {DATA}")
    print(f"output: {OUT}\n")

    ok, fail = [], []
    for label, script in FIGURES:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            fail.append((label, script, "script missing"))
            continue
        r = subprocess.run([sys.executable, path], cwd=HERE, env=env,
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok.append((label, script))
            print(f"  OK   {label:16} {script}")
        else:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            fail.append((label, script, " | ".join(tail)))
            print(f"  FAIL {label:16} {script}")

    print(f"\n{len(ok)}/{len(FIGURES)} figures regenerated into {OUT}")
    if fail:
        print("\nfailures:")
        for label, script, why in fail:
            print(f"  {label} ({script}): {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
