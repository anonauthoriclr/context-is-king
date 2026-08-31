#!/usr/bin/env python3
"""Figure: behavioral validation across hop count. One panel per model; in each, accuracy on the natural
(un-redefined) order (dashed, the per-k traversal ceiling) vs the imposed order (solid), for k=1..6, with the
1/7 chance line. Capable models hold the natural order near ceiling through k=6 yet fall on the imposed order,
so the imposed falloff is override-traversal difficulty, not base incapacity; weak models also lose the natural
order with k. Days, list format, bare <=3-token answers, two scrambles.

Reproduce:  python paper/figures/fig_behavior.py
Data: results/behavior/acc_<model>_days.json (res.natural_ceiling.kcurve_imp, res.list.kcurve_imp, acc_nat_revert)
Out:  paper/figures/fig_behavior.pdf (+ .png)"""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
B = os.path.join(DATA, "behavior")
GRID = [("Gemma E2B", "gemma-4-E2B-it", "#1b7837"), ("Gemma E4B", "gemma-4-E4B-it", "#1b7837"),
        ("Gemma 12B", "gemma-4-12B-it", "#1b7837"), ("Gemma 31B", "gemma-4-31B-it", "#1b7837"),
        ("Qwen 4B", "Qwen3.5-4B", "#762a83"), ("Qwen 9B", "Qwen3.5-9B", "#762a83"),
        ("Qwen 27B", "Qwen3.5-27B", "#762a83"), ("Llama 8B", "Llama-3.1-8B-Instruct", "#d95f02")]
KS = list(range(1, 7)); INK = "#1d2430"; NATC = "#8a93a3"
fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.8), sharex=True, sharey=True)
fig.patch.set_facecolor("white")
for ax, (lab, m, col) in zip(axes.ravel(), GRID):
    d = json.load(open(os.path.join(B, f"acc_{m}_days.json")))["res"]
    nk = [d["natural_ceiling"]["kcurve_imp"][str(k)] for k in KS]
    ik = [d["list"]["kcurve_imp"][str(k)] for k in KS]
    rev = d["list"]["acc_nat_revert"]
    ax.axhline(1/7, color="#c2c7d0", lw=0.9, ls=":", zorder=1)
    ax.plot(KS, nk, "--", color=NATC, lw=1.6, marker="o", ms=3.5, zorder=2)
    ax.plot(KS, ik, "-", color=col, lw=2.1, marker="o", ms=4, zorder=3)
    ax.set_title(lab, fontsize=10, color=INK, pad=3)
    ax.text(0.96, 0.93, f"revert {rev:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=NATC)
    ax.set_ylim(-0.03, 1.05); ax.set_xticks(KS)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#c2c7d0")
    ax.tick_params(labelsize=8, length=0)
for ax in axes[:, 0]: ax.set_ylabel("accuracy", fontsize=9, color=INK)
for ax in axes[1, :]: ax.set_xlabel("hop count $k$", fontsize=9, color=INK)
handles = [Line2D([0], [0], color=NATC, ls="--", marker="o", ms=4, label="natural order (ceiling)"),
           Line2D([0], [0], color=INK, ls="-", marker="o", ms=4, label="imposed order"),
           Line2D([0], [0], color="#c2c7d0", ls=":", label="chance (1/7)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.16, wspace=0.12, hspace=0.32)
out = os.path.join(_OUT, "fig_behavior")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
