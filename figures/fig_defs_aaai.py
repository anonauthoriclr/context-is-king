#!/usr/bin/env python3
"""Schematic (Experimental Setup): what a specification looks like.
Three panels, no data. (1) The SAME order stated in two surface forms -- a numbered list
vs. adjacency edges. (2) The wrap edge: an open chain reads as a line, closing it with a
wrap edge reads as a cycle. (3) A declarative hierarchy is a different structure type (a
tree, read as depth bands). Illustrates the taxonomy in Sec. Framework / Experimental Setup.

Reproduce: python paper/figures/fig_defs.py   (pure matplotlib, no data)
Out: paper/figures/fig_defs.pdf (+ .png)"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
IMP, INK, MUT, WRAP = "#1b7837", "#1d2430", "#8a93a3", "#c0392b"

fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.75)); fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for s in ax.spines.values(): s.set_visible(False)

def node(ax, xy, c=IMP, r=0.045):
    ax.add_patch(Circle(xy, r, facecolor=c, edgecolor="white", lw=2.2, zorder=3))
def arrow(ax, a, b, c=INK, ls="-", lw=2.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=c,
                 lw=lw, ls=ls, shrinkA=7, shrinkB=7,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))

# ---- Panel 1: two surface forms of the same order ----
ax = axes[0]
ax.set_title("Two surface forms, one order", fontsize=27.3, color=INK, pad=6)
ax.text(0.02, 0.86, "list (positions)", fontsize=23.4, color=MUT, style="italic")
ax.text(0.05, 0.70, "1. Wednesday\n2. Monday\n3. Friday\n     $\\vdots$",
        fontsize=24.7, color=INK, va="top", family="monospace", linespacing=1.35)
ax.axhline(0.44, 0.03, 0.97, color="#e2e5ea", lw=1.7)
ax.text(0.02, 0.37, "adjacency (edges)", fontsize=23.4, color=MUT, style="italic")
ax.text(0.05, 0.24, "Wednesday $\\to$ Monday\nMonday $\\to$ Friday\n     $\\vdots$",
        fontsize=24.7, color=INK, va="top", family="monospace", linespacing=1.35)

# ---- Panel 2: wrap edge => line vs cycle ----
ax = axes[1]
ax.set_title("Wrap edge sets the topology", fontsize=27.3, color=INK, pad=6)
th = np.linspace(np.pi*0.92, np.pi*0.08, 5)          # 5 nodes on an arc
R, cx, cy = 0.34, 0.5, 0.30
pts = np.c_[cx + R*np.cos(th), cy + R*np.sin(th)]
for i in range(4):
    arrow(ax, pts[i], pts[i+1], c=IMP, lw=3.1,
          rad=-0.18)                                  # chain along the arc
for p in pts: node(ax, p, c=IMP)
# wrap edge: dashed, closes last->first
arrow(ax, pts[-1], pts[0], c=WRAP, ls=(0, (4, 2)), lw=2.7, rad=-0.9)
ax.text(0.5, 0.02, "wrap edge (optional)", fontsize=22.1, color=WRAP, ha="center", style="italic")
ax.text(0.5, 0.90, "open chain", fontsize=22.1, color=MUT, ha="center")
ax.text(0.5, 0.79, "$+$ wrap edge $\\Rightarrow$ cycle", fontsize=24.7, color=WRAP, ha="center")

# ---- Panel 3: hierarchy is a different structure type ----
ax = axes[2]
ax.set_title("Hierarchy $\\Rightarrow$ a different type", fontsize=27.3, color=INK, pad=6)
root = (0.5, 0.74)
mid  = [(0.28, 0.46), (0.72, 0.46)]
leaf = [(0.15, 0.18), (0.41, 0.18), (0.59, 0.18), (0.85, 0.18)]
arrow(ax, root, mid[0], c=IMP, lw=2.9); arrow(ax, root, mid[1], c=IMP, lw=2.9)
arrow(ax, mid[0], leaf[0], c=IMP, lw=2.5); arrow(ax, mid[0], leaf[1], c=IMP, lw=2.5)
arrow(ax, mid[1], leaf[2], c=IMP, lw=2.5); arrow(ax, mid[1], leaf[3], c=IMP, lw=2.5)
for p in [root]+mid+leaf: node(ax, p, c=IMP)
ax.text(0.5, 0.02, "read as depth bands", fontsize=22.1, color=MUT, ha="center", style="italic")

fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06, wspace=0.10)
out = os.path.join(_OUT, "fig_defs_aaai")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
