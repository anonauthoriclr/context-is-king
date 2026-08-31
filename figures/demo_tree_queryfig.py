"""Static 3D-PCA figure of an imposed depth-4 tree read under MANY queries (cached npz, no GPU).
Combines the 24 neutral paraphrases (demo_tree) with 12 diverse query TYPES (demo_tree_diverse) = 36
queries; PCA recomputed on the pooled activations. Panel A: 4 DIVERSE query types each relocate the
same tree (query vectors from the centroid of all 36 queries). Panel B: per-node mean over ALL 36
queries -> one consensus tree; variance decomposition (relocation vs shared shape vs residual) printed."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PAPER = "--paper" in sys.argv   # paper mode: drop in-figure titles (they go in the LaTeX caption)
import matplotlib.gridspec as gridspec
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

N = 31; PARENT = {i: (i - 1) // 2 for i in range(1, N)}
depth = np.floor(np.log2(np.arange(N) + 1)).astype(int)
OLD_TPLS = ["Consider the item {e}.","Take note of {e}.","The item is {e}.","Focus on {e}.","Here is an item: {e}.",
    "Item of interest: {e}.","Note this item: {e}.","Regarding the item {e}.","We turn to {e}.","Look at {e}.",
    "The item named {e}.","Observe the item {e}.","About the item {e}.","This concerns {e}.","Attend to {e}.",
    "The relevant item is {e}.","Item under review: {e}.","Consider {e} now.","The item {e}, specifically.",
    "Examine {e}.","Recall the item {e}.","The chosen item: {e}.","Point to the item {e}.","Item: {e}."]

import os
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
old = np.load(os.path.join(DATA,"geometry","demo_tree_gemma-4-31B-it_d4.npz")); nq_old = len(OLD_TPLS)
_dp = os.path.join(DATA,"geometry","demo_tree_diverse_gemma-4-31B-it_d4.npz")
if os.path.exists(_dp):                                        # full: 24 neutral + 12 diverse
    new = np.load(_dp); NEW_TPLS = [str(t) for t in new["tpls"]]; nq_new = len(NEW_TPLS)
    Rall = np.vstack([old["R"].astype(np.float64), new["R"].astype(np.float64)])
    NODEall = np.concatenate([old["NODE"], new["NODE"]])
    qid = np.concatenate([np.arange(len(old["R"])) % nq_old, nq_old + (np.arange(len(new["R"])) % nq_new)])
    QI = [nq_old + 1, nq_old + 2, nq_old + 3, nq_old + 4]      # diverse TYPES for Panel A
else:                                                          # PREVIEW: neutral only (diverse not ready)
    NEW_TPLS = []; nq_new = 0
    Rall = old["R"].astype(np.float64); NODEall = old["NODE"]; qid = np.arange(len(old["R"])) % nq_old
    QI = [0, 9, 16, 23]
LABELS = OLD_TPLS + NEW_TPLS; NQ = nq_old + nq_new
mu = Rall.mean(0); U, S, Vt = np.linalg.svd(Rall - mu, full_matrices=False); Pall = (Rall - mu) @ Vt[:3].T
c0 = Pall.mean(0)

def coords(q):
    C = np.zeros((N, 3))
    for n in range(N):
        C[n] = Pall[(qid == q) & (NODEall == n)][0]
    return C
CQ_all = [coords(q) for q in range(nq_old + nq_new)]          # 36 per-query trees

# ---- variance decomposition over ALL 36 queries ----
Xs = np.stack(CQ_all); grand = Xs.reshape(-1, 3).mean(0); qcm = Xs.mean(1); Xc_ = Xs - qcm[:, None, :]; Mc = Xc_.mean(0)
_ss = lambda a: float((a ** 2).sum())
tot = _ss(Xs - grand); trans = N * _ss(qcm - grand); struct = len(CQ_all) * _ss(Mc); resid = _ss(Xc_ - Mc[None])
Rrep = struct / (struct + resid)
rdms = [pdist(C) for C in CQ_all]
rho = np.mean([spearmanr(rdms[i], rdms[j]).correlation for i, j in combinations(range(len(CQ_all)), 2)])
jit = np.sqrt(((Xc_ - Mc[None]) ** 2).sum(-1).mean(0)); edge = np.mean([np.linalg.norm(Mc[n] - Mc[PARENT[n]]) for n in range(1, N)])
jratio = float(np.median(jit) / edge)

# 4 DIVERSE query TYPES to showcase in Panel A (indices into the new/diverse block)
qcol = [plt.cm.tab10(k) for k in (0, 1, 2, 3)]
def short(t): return t.replace(" {e}", "").replace("{e}", "").strip().rstrip(".")

fig = plt.figure(figsize=(15.5, 7.2)); G = gridspec.GridSpec(1, 2, figure=fig, wspace=.08)
elevA, azimA = 28, 81; elevB, azimB = 28, 81   # lower camera (was elev 39)

axA = fig.add_subplot(G[0, 0], projection="3d")
for j, q in enumerate(QI):
    C = CQ_all[q]
    for n in range(1, N):
        axA.plot(*zip(C[n], C[PARENT[n]]), color=qcol[j], alpha=0.20, lw=0.7)
    axA.scatter(C[:, 0], C[:, 1], C[:, 2], c=depth, cmap="viridis", s=28, edgecolor=qcol[j],
                linewidths=1.2, depthshade=False, zorder=3)
    axA.quiver(*c0, *(C.mean(0) - c0), color=qcol[j], lw=2.6, arrow_length_ratio=0.14, zorder=4)
axA.scatter(*c0, color="k", s=160, marker="*", zorder=6)
axA.set_title("" if PAPER else f"A. Queries relocate the same tree\n(arrows = query vectors from ★, the centroid of all {NQ} queries)", fontsize=11)
if PAPER: axA.text2D(0.04, 0.93, "(a)", transform=axA.transAxes, fontsize=13, weight="bold")
axA.set_xlabel("PC1"); axA.set_ylabel("PC2"); axA.set_zlabel("PC3"); axA.view_init(elevA, azimA)
_allp = np.vstack([CQ_all[q] for q in QI] + [c0[None]]); _lo, _hi = _allp.min(0), _allp.max(0); _m = (_hi - _lo) * 0.04
axA.set_xlim(_lo[0]-_m[0], _hi[0]+_m[0]); axA.set_ylim(_lo[1]-_m[1], _hi[1]+_m[1]); axA.set_zlim(_lo[2]-_m[2], _hi[2]+_m[2])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 4)); cb = fig.colorbar(sm, ax=axA, shrink=.5, pad=.11); cb.set_label("tree depth")

axB = fig.add_subplot(G[0, 1], projection="3d")
M = Mc                                                        # consensus over all 36 queries
for n in range(1, N):
    axB.plot(*zip(M[n], M[PARENT[n]]), color="0.35", lw=1.4, zorder=3)
axB.scatter(M[:, 0], M[:, 1], M[:, 2], c=depth, cmap="viridis", s=70, edgecolor="k", linewidths=.8, depthshade=False, zorder=4)
axB.set_title("" if PAPER else f"B. Mean over all {NQ} queries: one consensus tree\ntopology shared across queries (RDM agreement ρ = {rho:.2f})", fontsize=11)
if PAPER: axB.text2D(0.04, 0.93, "(b)", transform=axB.transAxes, fontsize=13, weight="bold")
axB.set_xlabel("PC1"); axB.set_ylabel("PC2"); axB.set_zlabel("PC3"); axB.view_init(elevB, azimB)
axB.text2D(0.01, 0.99,
           f"query relocation = {100*trans/tot:.0f}% of positional variance\n"
           f"shared shape  R = {Rrep:.2f}   ·   residual jitter = {100*resid/tot:.1f}%\n"
           f"per-node jitter = {jratio:.2f}× parent-child spacing   ({NQ} queries)",
           transform=axB.transAxes, fontsize=8.3, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.92))

leg = [Line2D([0], [0], color=qcol[j], lw=2.4, marker="o", markerfacecolor="none", markeredgecolor=qcol[j],
              label=f'"{short(LABELS[QI[j]])}"') for j in range(4)]
leg.append(Line2D([0], [0], color="k", marker="*", lw=0, label=f"centroid of all {NQ} queries"))
plt.tight_layout(rect=[0, 0, 1, 1])
# legend: vertical, larger, tucked under the LEFT panel
fig.legend(handles=leg, loc="lower left", ncol=1, fontsize=10.5, framealpha=.92,
           bbox_to_anchor=(0.05, 0.02))
_OUT = os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(_OUT, exist_ok=True)
plt.savefig(os.path.join(_OUT, "fig_queryreloc.png"), dpi=150, bbox_inches="tight")
if PAPER:
    plt.savefig(os.path.join(_OUT, "fig_queryreloc.pdf"), bbox_inches="tight")
print(f"saved | {nq_old}+{nq_new} queries | rho={rho:.3f} reloc={100*trans/tot:.0f}% R={Rrep:.2f} | PAPER={PAPER}")
