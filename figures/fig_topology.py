#!/usr/bin/env python3
"""Figure: topology follows the specification (the 2x2 decider).
LEFT: a literal 2x2 for the cleanest model (Gemma-4-31B). Rows = surface form (numbered list vs edge list),
columns = wrap factor (no wrap vs + wrap). Reading DOWN a column changes the format and the shape is preserved;
reading ACROSS a row adds the wrap relation and a line closes into a cycle. Points are colored by imposed
sequence position; the wrap edge (last->first) is drawn in red; each panel is annotated with its wrap-pair
closure ratio (~1 closed, >>1 open) and dRSA.
RIGHT: the dissociation is systematic and cross-family -- wrap-pair closure ratio for three capable models
(Gemma-31B, Gemma-12B, Qwen-27B) across all four specs; +wrap specs collapse to ~1 (cycle) regardless of form.

Reproduce:  python paper/figures/fig_topology.py
Data: results/behavior/wrap2x2_<tag>.npz (R,COND,SI,ENT,imp) + results/behavior/wrap2x2_summary.json (closure,dRSA)
Out:  paper/figures/fig_topology.pdf (+ .png)"""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.cm import viridis

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
B = os.path.join(DATA, "behavior")
N = 7
CONDS = ["list", "list_wrap", "adj_nowrap", "adj"]            # (form, wrap): list/no, list/yes, edge/no, edge/yes
HERO = "gemma-4-31B-it"
# 2x2 layout: rows = form, cols = wrap.  grid[row][col] -> cond key
GRID = [["list", "list_wrap"], ["adj_nowrap", "adj"]]
ROW_LAB = ["numbered list", "edge list"]
COL_LAB = ["no wrap edge", "with wrap edge"]
SUMMARY = [("Gemma 31B", "gemma-4-31B-it", "#1b7837"),
           ("Gemma 12B", "gemma-4-12B-it", "#74c476"),
           ("Qwen 27B",  "Qwen3.5-27B",    "#762a83")]
IMP_COL, WRAP_COL, INK, FRAME = "#1b7837", "#d62728", "#1d2430", "#c2c7d0"
S = json.load(open(os.path.join(B, "wrap2x2_summary.json")))

def panel_geom(tag, cond):
    d = np.load(os.path.join(B, f"wrap2x2_{tag}.npz"), allow_pickle=True)
    R = d["R"].astype(np.float64); C = d["COND"]; SI = d["SI"]; E = d["ENT"]; imp = d["imp"]
    c = CONDS.index(cond); m = (C == c) & (SI == 0)
    cents = np.stack([R[m & (E == e)].mean(0) for e in range(N)])
    ip = list(imp[0]); seq = list(np.argsort(ip))                 # entity indices in imposed sequence order
    mu = cents.mean(0); _, _, Vt = np.linalg.svd(cents - mu, full_matrices=False)
    P = (cents - mu) @ Vt[:2].T; P = P / (np.abs(P).max() + 1e-9)
    pos = np.argsort(np.argsort(ip))                              # sequence position per entity (for color)
    clos = float(np.mean([x["closure"] for x in S[tag][cond]]))
    drsa = float(np.mean([x["dRSA"] for x in S[tag][cond]]))
    return P, seq, pos, clos, drsa

fig = plt.figure(figsize=(10.2, 4.6)); fig.patch.set_facecolor("white")
outer = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30, figure=fig)
gg = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], wspace=0.10, hspace=0.10)

for r in range(2):
    for c in range(2):
        cond = GRID[r][c]; a = fig.add_subplot(gg[r, c])
        P, seq, pos, clos, drsa = panel_geom(HERO, cond)
        loop = seq + [seq[0]]
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            a.plot([P[u, 0], P[v, 0]], [P[u, 1], P[v, 1]], "-", color="#c7ccd4", lw=1.3, zorder=1)
        u, v = seq[-1], seq[0]                                    # wrap edge
        a.plot([P[u, 0], P[v, 0]], [P[u, 1], P[v, 1]], "-", color=WRAP_COL, lw=2.0, zorder=2)
        a.scatter(P[:, 0], P[:, 1], c=pos, cmap=viridis, s=80, edgecolor="white", linewidth=1.0, zorder=3)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.margins(0.22)
        for sname in a.spines: a.spines[sname].set_color(FRAME)
        cyc = clos < 1.3
        a.text(0.5, 0.025, f"{'cycle' if cyc else 'line'}   closure {clos:.2f}   dRSA {drsa:+.2f}",
               transform=a.transAxes, ha="center", va="bottom", fontsize=8,
               color=(IMP_COL if cyc else WRAP_COL), fontweight="bold")
        if r == 0:
            a.set_title(COL_LAB[c], fontsize=10.5, color=INK, pad=6)
        if c == 0:
            a.text(-0.07, 0.5, ROW_LAB[r], transform=a.transAxes, rotation=90,
                   ha="center", va="center", fontsize=10.5, color=INK)

# ---- right: cross-family closure-ratio dissociation ----
axc = fig.add_subplot(outer[1])
ys = {"list": 3, "list_wrap": 2, "adj_nowrap": 1, "adj": 0}
SPEC_LAB = {"list": "numbered, no wrap", "list_wrap": "numbered, + wrap",
            "adj_nowrap": "edges, no wrap", "adj": "edges, + wrap"}
axc.axvspan(1.3, 3.2, color="#f6e9e8", zorder=0)                  # "line" region
axc.axvline(1.0, color="#9aa1ac", lw=1.0, ls="--", zorder=1)
for lab, tag, col in SUMMARY:
    for cond in CONDS:
        clos = float(np.mean([x["closure"] for x in S[tag][cond]]))
        jit = (SUMMARY.index((lab, tag, col)) - 1) * 0.16
        axc.scatter(clos, ys[cond] + jit, s=55, color=col, edgecolor="white", linewidth=0.6, zorder=3)
axc.set_yticks(list(ys.values())); axc.set_yticklabels([SPEC_LAB[c] for c in ys], fontsize=9, color=INK)
axc.set_ylim(-0.6, 3.6); axc.set_xlim(0.7, 3.2)
axc.set_xlabel("wrap-pair closure ratio   (≈1 cycle, $\\gg$1 line)", fontsize=9.5, color=INK)
axc.text(2.25, 3.45, "open (line)", fontsize=8, color="#b0473d", ha="center")
axc.text(1.0, -0.5, "closed", fontsize=8, color="#5b636f", ha="center")
for sname in ["top", "right"]: axc.spines[sname].set_visible(False)
for sname in ["left", "bottom"]: axc.spines[sname].set_color(FRAME)
axc.tick_params(length=0)
axc.legend(handles=[Line2D([0], [0], marker="o", ls="", color=col, label=lab) for lab, _, col in SUMMARY],
           fontsize=8.5, loc="lower right", frameon=False)

fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.12)
out = os.path.join(_OUT, "fig_topology")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
for cond in CONDS:
    print(f"  {HERO} {cond}: closure {np.mean([x['closure'] for x in S[HERO][cond]]):.2f}  "
          f"dRSA {np.mean([x['dRSA'] for x in S[HERO][cond]]):+.2f}")
