#!/usr/bin/env python3
"""Emit the LaTeX rows for the behavioral-validation table (tab:behavior).
Per model: natural-order accuracy (un-redefined task, the ceiling) and imposed-order accuracy at k=1,2,3,
plus the rate of reverting to the pretrained order. Establishes that imposed-order failures are not base-task
incapacity, that capable models follow the rule one-shot, and that weak models' errors are pretrained-order
reversions. Source: results/behavior/acc_<model>_days.json (list format, days, 2 scrambles).

Reproduce:  python paper/figures/behavior_table.py
"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
B = os.path.join(DATA, "behavior")
ROWS = [("Gemma E2B", "gemma-4-E2B-it"), ("Gemma E4B", "gemma-4-E4B-it"),
        ("Gemma 12B", "gemma-4-12B-it"), ("Gemma 31B", "gemma-4-31B-it"),
        ("Qwen 4B", "Qwen3.5-4B"), ("Qwen 9B", "Qwen3.5-9B"), ("Qwen 27B", "Qwen3.5-27B"),
        ("Llama 8B", "Llama-3.1-8B-Instruct")]
def f2(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "--"
for i, (lab, m) in enumerate(ROWS):
    d = json.load(open(os.path.join(B, f"acc_{m}_days.json")))["res"]
    nk = d["natural_ceiling"]["kcurve_imp"]; ik = d["list"]["kcurve_imp"]
    rev = d["list"]["acc_nat_revert"]
    cells = [f2(nk[str(k)]) for k in (1, 2, 3)] + [f2(ik[str(k)]) for k in (1, 2, 3)] + [f2(rev)]
    print(f"{lab} & " + " & ".join(cells) + "\\\\")
    if i in (3,):  # blank line between families
        print("\\addlinespace")
