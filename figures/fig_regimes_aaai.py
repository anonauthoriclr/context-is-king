#!/usr/bin/env python3
"""Figure: imposed-order accuracy under the three prompt regimes (direct / scripted / natural), per model.
One panel per model; in each, the imposed-order k-curve (k=1..6) under the three instruction regimes, with the
1/7 chance line. One-shot direct reading degrades with hop count, while reasoning (scripted, and natural where it
is not truncation-limited) recovers traversal. Days, list format, bare answers, si=0.
NOTE: Qwen natural hits the 2048-token cap on 81% of generations (over-thinking), so its natural curve is a
truncation-limited lower bound (annotated); Gemma natural has zero truncation.

Reproduce:  python paper/figures/fig_regimes.py
Data: results/behavior/behavior3_<model>_days.json (res.regimes[regime].kcurve_imp, trunc_rate)
Out:  paper/figures/fig_regimes.pdf (+ .png)"""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
B = os.path.join(DATA, "behavior")
MODELS = [("Gemma-4-31B", "gemma-4-31B-it"), ("Qwen-3.5-27B", "Qwen3.5-27B")]
REGIMES = [("direct", "#d95f02", "-"), ("scripted", "#1b7837", "-"), ("natural", "#762a83", "-")]
DISPLAY = {"direct": "direct", "scripted": "scripted", "natural": "free-form"}  # data key 'natural' -> label
KS = list(range(1, 7)); INK = "#1d2430"
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True); fig.patch.set_facecolor("white")
for ax, (lab, m) in zip(axes, MODELS):
    d = json.load(open(os.path.join(B, f"behavior3_{m}_days.json")))["regimes"]
    ax.axhline(1/7, color="#c2c7d0", lw=1.5, ls=":", zorder=1)
    # distinct marker + small x-offset per regime so coincident curves (e.g. Gemma scripted==natural==1.0) stay visible
    STYLE = {"direct": ("o", -0.10, 4.5), "scripted": ("s", 0.0, 4.5), "natural": ("D", 0.10, 4.0)}
    for ri, (reg, col, ls) in enumerate(REGIMES):
        kc = [d[reg]["kcurve_imp"][str(k)] for k in KS]
        trunc = d[reg].get("trunc_rate", 0.0); dashed = trunc > 0.2
        mk, dx, ms = STYLE[reg]; xs = [k + dx for k in KS]
        ax.plot(xs, kc, ls=("--" if dashed else "-"), color=col, lw=3.4, marker=mk, ms=ms,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3 + ri, alpha=(0.8 if dashed else 1.0))
        if dashed:
            ax.text(xs[-1], kc[-1], f"  {int(round(trunc*100))}% over budget", fontsize=18.2, color=col, va="center")
    ax.set_title(lab, fontsize=27.3, color=INK, pad=3)
    ax.set_xlabel("hop count $k$", fontsize=24.7, color=INK); ax.set_xticks(KS)
    ax.set_ylim(-0.03, 1.06)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color("#c2c7d0")
    ax.tick_params(labelsize=22.1, length=0)
axes[0].set_ylabel("imposed-order accuracy", fontsize=24.7, color=INK)
handles = [Line2D([0], [0], color=c, marker=STYLE[r][0], ms=4.5, label=DISPLAY[r]) for r, c, _ in REGIMES] + \
          [Line2D([0], [0], color="#c2c7d0", ls=":", label="chance (1/7)"),
           Line2D([0], [0], color="#762a83", ls="--", alpha=0.75, label="exceeds 2048-tok budget")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=22.1, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.22, wspace=0.08)
out = os.path.join(_OUT, "fig_regimes_aaai")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
for lab, m in MODELS:
    d = json.load(open(os.path.join(B, f"behavior3_{m}_days.json")))["regimes"]
    print(f"  {lab}:", {r: (round(d[r]['acc_imp'], 2), f"trunc{int(d[r]['trunc_rate']*100)}%") for r in ['direct','scripted','natural']})
