#!/usr/bin/env python3
"""Figure: an imposed hierarchy is built as a depth-stratified manifold (3D hero) plus quantification.
(A) The load-bearing 3D view: a declarative depth-4 binary tree on arbitrary tokens, neutral queries; per-sample
    residuals projected into the entity-centroid PCA basis so the depth axis (PC1, r=+0.91) is a single visible
    direction. The cloud funnels from the root out to the leaves by depth; centroids joined by tree edges.
(B) Depth-RSA vs tree depth (d2/d3/d4), both families: the depth axis holds (declines gently) as the tree grows.
(C) Branch is a metric: among same-depth leaves cosine distance rises with common-ancestor distance; the
    co-occurrence control (edges in separate sentences, siblings shuffled apart) leaves depth-RSA unchanged.
CPU-only from saved demo_tree_*.npz (A) and tree_*.npz (B,C).

Reproduce:  python paper/figures/fig_hierarchy.py
Out: paper/figures/fig_hierarchy.pdf (+ .png)"""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
G = os.path.join(DATA, "geometry") + "/"
def tree(N):
    P = {i: (i - 1) // 2 for i in range(1, N)}
    def anc(k):
        a = {k}
        while k != 0: k = P[k]; a.add(k)
        return a
    A = {k: anc(k) for k in range(N)}; DEP = {k: len(A[k]) - 1 for k in range(N)}
    return P, A, DEP
def load(f):
    d = np.load(f, allow_pickle=True); cs = [d[k] for k in d.files if k.startswith("c") and k[1:].isdigit()]
    return cs, np.array(d["nodes"])
def depth_rsa(f):
    cs, nodes = load(f); N = cs[0].shape[0]; _, _, DEP = tree(N); r = []
    for si in range(len(cs)):
        ndi = nodes[si] if nodes.ndim == 2 else nodes; pos = {int(ndi[k]): k for k in range(len(ndi))}
        C = cs[si].astype(float); Cc = C - C.mean(0); Cn = Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-12); COS = 1 - Cn @ Cn.T
        idx = [pos[g] for g in range(N)]; iu = np.triu_indices(N, 1)
        dep = np.array([[abs(DEP[i] - DEP[j]) for j in range(N)] for i in range(N)], float)
        r.append(spearmanr(COS[np.ix_(idx, idx)][iu], dep[iu]).correlation)
    return np.mean(r), np.std(r)
def genealogy(f):
    cs, nodes = load(f); N = cs[0].shape[0]; P, A, DEP = tree(N); D = max(DEP.values())
    leaves = [g for g in range(N) if DEP[g] == D]
    def coph(i, j): c = A[i] & A[j]; return DEP[i] + DEP[j] - 2 * max(DEP[x] for x in c)
    from collections import defaultdict; by = defaultdict(list)
    for si in range(len(cs)):
        ndi = nodes[si] if nodes.ndim == 2 else nodes; pos = {int(ndi[k]): k for k in range(len(ndi))}
        C = cs[si].astype(float); Cc = C - C.mean(0); Cn = Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-12); COS = 1 - Cn @ Cn.T
        for a in range(len(leaves)):
            for b in range(a + 1, len(leaves)):
                by[coph(leaves[a], leaves[b])].append(COS[pos[leaves[a]], pos[leaves[b]]])
    return {k: (np.mean(v), np.std(v)) for k, v in by.items()}

GREEN, PURPLE = "#1b7837", "#762a83"
fig = plt.figure(figsize=(15.2, 4.7)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.05], wspace=0.26)

# ---- A: 3D depth-stratified manifold (load-bearing view) ----
axA = fig.add_subplot(gs[0, 0], projection="3d"); axA.set_facecolor("white")
d = np.load(G + "demo_tree_gemma-4-31B-it_d4.npz", allow_pickle=True)
R = d["R"].astype(np.float64); NODE = d["NODE"]; Nn = int(NODE.max()) + 1
PAR, _, DEPd = tree(Nn); DEP = np.array([DEPd[n] for n in range(Nn)]); maxd = DEP.max()
cf = np.stack([R[NODE == n].mean(0) for n in range(Nn)]); mu = cf.mean(0)
_, _, Vt = np.linalg.svd(cf - mu, full_matrices=False)
if spearmanr((cf - mu) @ Vt[0], DEP).correlation < 0: Vt[0] = -Vt[0]
P3 = (R - mu) @ Vt[:3].T; cent = np.stack([P3[NODE == n].mean(0) for n in range(Nn)])
# tree skeleton: centroids + parent/child edges, colored by depth. (cloud omitted for legibility)
for c, p in PAR.items():
    axA.plot([cent[c, 0], cent[p, 0]], [cent[c, 1], cent[p, 1]], [cent[c, 2], cent[p, 2]],
             "-", color="#aeb4bd", lw=1.4, alpha=0.85, zorder=2)
scA = axA.scatter(cent[:, 0], cent[:, 1], cent[:, 2], c=DEP, cmap=cm.viridis, vmin=0, vmax=maxd,
                  s=150, edgecolor="white", linewidth=1.3, zorder=3, depthshade=False)
axA.scatter([cent[0, 0]], [cent[0, 1]], [cent[0, 2]], s=300, facecolor="none",
            edgecolor="#1d2430", linewidth=2.0, zorder=4)
axA.text(cent[0, 0], cent[0, 1], cent[0, 2], "  root", fontsize=9, color="#1d2430", zorder=5)
# tight zoom to the centroid extent (small symmetric margin)
def lims(i):
    lo, hi = cent[:, i].min(), cent[:, i].max(); m = 0.08 * (hi - lo)
    return lo - m, hi + m
axA.set_xlim(*lims(0)); axA.set_ylim(*lims(1)); axA.set_zlim(*lims(2))
try: axA.set_box_aspect((np.ptp(cent[:, 0]), np.ptp(cent[:, 1]), np.ptp(cent[:, 2])))
except Exception: pass
axA.set_xlabel("PC 1  (depth)", fontsize=9, labelpad=2)
axA.set_ylabel("PC 2", fontsize=9, labelpad=2); axA.set_zlabel("PC 3", fontsize=9, labelpad=2)
for axis in (axA.xaxis, axA.yaxis, axA.zaxis):
    axis.set_ticklabels([]); axis.pane.set_alpha(0.0); axis._axinfo["grid"].update(color="#e7e9ee", linewidth=0.5)
axA.view_init(elev=14, azim=-78); axA.set_title("A", fontsize=11, color="#1d2430", loc="left", fontweight="bold")
cb = fig.colorbar(scA, ax=axA, shrink=0.5, pad=0.14, ticks=range(maxd + 1)); cb.set_label("tree depth", fontsize=8); cb.ax.tick_params(labelsize=7)

# ---- B: depth-RSA holds as the tree grows ----
axB = fig.add_subplot(gs[0, 1]); depths = [2, 3, 4]
gem = [depth_rsa(G + f) for f in ["tree_gemma-4-31B-it_neutral.npz", "tree_gemma-4-31B-it_neutral_d3.npz", "tree_gemma-4-31B-it_neutral_d4.npz"]]
qwn = [depth_rsa(G + f) for f in ["tree_Qwen3.5-27B_neutral.npz", "tree_Qwen3.5-27B_neutral_d3.npz", "tree_Qwen3.5-27B_neutral_d4.npz"]]
axB.errorbar(depths, [m for m, _ in gem], yerr=[s for _, s in gem], marker="o", capsize=3, label="Gemma-4-31B", color=GREEN)
axB.errorbar(depths, [m for m, _ in qwn], yerr=[s for _, s in qwn], marker="s", capsize=3, label="Qwen-3.5-27B", color=PURPLE)
axB.set_xticks(depths); axB.set_xticklabels(["d2 (7n)", "d3 (15n)", "d4 (31n)"]); axB.set_ylim(0, 1)
axB.set_ylabel("depth-RSA (Spearman)", fontsize=9); axB.axhline(0, color="0.7", lw=0.6)
axB.set_title("B", fontsize=11, color="#1d2430", loc="left", fontweight="bold"); axB.legend(fontsize=8)
for s in ["top", "right"]: axB.spines[s].set_visible(False)

# ---- C: branch metric + co-occurrence control ----
axC = fig.add_subplot(gs[0, 2])
gb = genealogy(G + "tree_gemma-4-31B-it_neutral_d3.npz"); gsep = genealogy(G + "tree_gemma-4-31B-it_neutral_d3_sep.npz")
qb = genealogy(G + "tree_Qwen3.5-27B_neutral_d3.npz"); qsep = genealogy(G + "tree_Qwen3.5-27B_neutral_d3_sep.npz")
ks = [2, 4, 6]; labs = ["sibling", "cousin", "2nd cousin"]; x = np.arange(3); w = 0.2
axC.bar(x - 1.5 * w, [gb[k][0] for k in ks], w, label="Gemma co-listed", color=GREEN)
axC.bar(x - 0.5 * w, [gsep[k][0] for k in ks], w, label="Gemma separated", color="#7fbf7b")
axC.bar(x + 0.5 * w, [qb[k][0] for k in ks], w, label="Qwen co-listed", color=PURPLE)
axC.bar(x + 1.5 * w, [qsep[k][0] for k in ks], w, label="Qwen separated", color="#c2a5cf")
axC.set_xticks(x); axC.set_xticklabels(labs, fontsize=9); axC.set_ylabel("cosine distance", fontsize=9)
axC.set_title("C", fontsize=11, color="#1d2430", loc="left", fontweight="bold")
axC.legend(fontsize=6.8, ncol=2)
for s in ["top", "right"]: axC.spines[s].set_visible(False)

out = os.path.join(_OUT, "fig_hierarchy")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=160, bbox_inches="tight")
print("WROTE", out)
print("depth-RSA Gemma", [round(m, 3) for m, _ in gem], "Qwen", [round(m, 3) for m, _ in qwn])
