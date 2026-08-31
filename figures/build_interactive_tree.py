#!/usr/bin/env python3
"""Interactive 3-D query-relocation explorer for the imposed tree.

The same depth-4 tree on 31 arbitrary tokens, read under many queries. Each query
relocates the whole tree bodily, yet the LOCAL structure (depth bands, parent-child
spacing) is preserved -- the paper's fig:queryreloc, made interactive.

Shows 8 example queries (4 neutral + 4 diverse). For each, a colored vector runs
from the grand centroid (star) to that query's tree-centroid -- the relocation
direction. The consensus tree (per-node mean over ALL 36 queries) is drawn bold,
nodes colored by tree depth.

Controls: buttons switch Trees+vectors / Vectors only / Consensus only; click a
query in the legend to add/remove it.

Data (committed, no GPU): data/geometry/demo_tree_gemma-4-31B-it_d4.npz (24 neutral)
+ demo_tree_diverse_gemma-4-31B-it_d4.npz (12 diverse query types).

    python figures/build_interactive_tree.py
Out: assets/tree_queries_3d.html
"""
import os
import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(HERE, "..", "data")))
ASSETS = os.environ.get("CIK_ASSETS", os.path.normpath(os.path.join(HERE, "..", "assets")))
os.makedirs(ASSETS, exist_ok=True)

N = 31
PARENT = {i: (i - 1) // 2 for i in range(1, N)}
depth = np.floor(np.log2(np.arange(N) + 1)).astype(int)
EDGES = [(i, PARENT[i]) for i in range(1, N)]

OLD_TPLS = ["Consider the item {e}.", "Take note of {e}.", "The item is {e}.", "Focus on {e}.",
            "Here is an item: {e}.", "Item of interest: {e}.", "Note this item: {e}.",
            "Regarding the item {e}.", "We turn to {e}.", "Look at {e}.", "The item named {e}.",
            "Observe the item {e}.", "About the item {e}.", "This concerns {e}.", "Attend to {e}.",
            "The relevant item is {e}.", "Item under review: {e}.", "Consider {e} now.",
            "The item {e}, specifically.", "Examine {e}.", "Recall the item {e}.",
            "The chosen item: {e}.", "Point to the item {e}.", "Item: {e}."]

old = np.load(os.path.join(DATA, "geometry", "demo_tree_gemma-4-31B-it_d4.npz"))
new = np.load(os.path.join(DATA, "geometry", "demo_tree_diverse_gemma-4-31B-it_d4.npz"))
NEW_TPLS = [str(t) for t in new["tpls"]]
nq_old, nq_new = len(OLD_TPLS), len(NEW_TPLS)
Rall = np.vstack([old["R"].astype(np.float64), new["R"].astype(np.float64)])
NODEall = np.concatenate([old["NODE"], new["NODE"]])
qid = np.concatenate([np.arange(len(old["R"])) % nq_old, nq_old + (np.arange(len(new["R"])) % nq_new)])
LABELS = [t.replace("{e}", "X") for t in OLD_TPLS] + [t.replace("{e}", "X") for t in NEW_TPLS]  # readable query text
NQ = nq_old + nq_new

mu = Rall.mean(0)
_, _, Vt = np.linalg.svd(Rall - mu, full_matrices=False)
Pall = (Rall - mu) @ Vt[:3].T


def coords(q):
    return np.stack([Pall[(qid == q) & (NODEall == n)][0] for n in range(N)])


CQ = [coords(q) for q in range(NQ)]
M = np.stack(CQ).mean(0)      # consensus tree (per-node mean over all 36 queries)
STAR = Pall.mean(0)           # grand centroid

# relocation stat -- same decomposition as the paper's fig:queryreloc
Xs = np.stack(CQ); qcm = Xs.mean(1)
_ss = lambda a: float((a ** 2).sum())
reloc = 100 * (N * _ss(qcm - STAR)) / _ss(Xs - STAR)

# 8 example queries: 4 neutral + 4 diverse (spread across each family)
SEL = [0, 8, 15, 23] + [nq_old + i for i in (1, 3, 6, 9)]   # 4 plain paraphrases + 4 distinct query types
QCOL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def edge_xyz(C):
    xs, ys, zs = [], [], []
    for a, b in EDGES:
        xs += [C[a, 0], C[b, 0], None]; ys += [C[a, 1], C[b, 1], None]; zs += [C[a, 2], C[b, 2], None]
    return xs, ys, zs


fig = go.Figure()
always, qgroups = [], []   # trace-index bookkeeping for the toggle buttons
ti = 0

# --- consensus tree (bold) + star ---
ex, ey, ez = edge_xyz(M)
fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", name="consensus tree",
                           legendgroup="consensus", line=dict(color="rgba(70,70,70,0.85)", width=5),
                           hoverinfo="skip")); always.append(ti); ti += 1
fig.add_trace(go.Scatter3d(x=M[:, 0], y=M[:, 1], z=M[:, 2], mode="markers",
                           legendgroup="consensus", showlegend=False,
                           marker=dict(size=5.5, color=depth, colorscale="Viridis",
                                       line=dict(color="black", width=1),
                                       colorbar=dict(orientation="h", x=0.5, xanchor="center",
                                                     y=-0.02, len=0.35, thickness=12,
                                                     title=dict(text="tree depth", side="top"),
                                                     tickvals=[0, 1, 2, 3, 4])),
                           hovertext=[f"depth {d}" for d in depth], hoverinfo="text")); always.append(ti); ti += 1
fig.add_trace(go.Scatter3d(x=[STAR[0]], y=[STAR[1]], z=[STAR[2]], mode="markers",
                           name="grand centroid", legendgroup="star",
                           marker=dict(size=7, color="black", symbol="diamond"),
                           hovertext=["grand centroid (star)"], hoverinfo="text")); always.append(ti); ti += 1

# --- 8 example queries: tree (depth-colored) + relocation vector (query-colored) ---
vec_idx, tree_idx = [], []
for j, q in enumerate(SEL):
    C = CQ[q]; qc = C.mean(0); col = QCOL[j]; grp = f"q{q}"; lab = LABELS[q][:32]
    # relocation vector: star -> query centroid (this is the legend entry for the group)
    fig.add_trace(go.Scatter3d(x=[STAR[0], qc[0]], y=[STAR[1], qc[1]], z=[STAR[2], qc[2]],
                               mode="lines", name=lab, legendgroup=grp,
                               line=dict(color=col, width=6), hoverinfo="skip")); vec_idx.append(ti); ti += 1
    fig.add_trace(go.Scatter3d(x=[qc[0]], y=[qc[1]], z=[qc[2]], mode="markers",
                               legendgroup=grp, showlegend=False,
                               marker=dict(size=6, color=col, symbol="diamond", line=dict(color="white", width=1)),
                               hovertext=[lab], hoverinfo="text")); vec_idx.append(ti); ti += 1
    # the query's tree (local structure): faint edges + depth-colored nodes
    ex, ey, ez = edge_xyz(C)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", legendgroup=grp, showlegend=False,
                               line=dict(color=col, width=1.5), opacity=0.35, hoverinfo="skip")); tree_idx.append(ti); ti += 1
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers", legendgroup=grp, showlegend=False,
                               marker=dict(size=3, color=depth, colorscale="Viridis", opacity=0.75),
                               hovertext=[lab] * N, hoverinfo="text")); tree_idx.append(ti); ti += 1

n = ti
def _vis(keep):
    return [i in keep for i in range(n)]
A = set(always); V = set(vec_idx); T = set(tree_idx)

scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
             aspectmode="cube", camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)))
fig.update_layout(
    title=dict(text=f"One imposed tree, {NQ} queries (8 shown): each relocates it bodily, "
                    f"local structure preserved<br><sub>vectors run from the grand centroid to each "
                    f"query's tree-centroid &nbsp;|&nbsp; relocation = {reloc:.0f}% of positional variance "
                    f"&nbsp;|&nbsp; drag to rotate, click the legend to add/remove queries</sub>",
               x=0.5, font=dict(size=15)),
    scene=scene, margin=dict(l=0, r=200, t=70, b=40), width=1280, height=720, paper_bgcolor="white",
    legend=dict(title="queries (click to toggle)", font=dict(size=10), itemsizing="constant"),
    updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center", y=1.13, showactive=True,
                      buttons=[
                          dict(label="Trees + vectors", method="update", args=[{"visible": _vis(A | V | T)}]),
                          dict(label="Vectors only", method="update", args=[{"visible": _vis(A | V)}]),
                          dict(label="Consensus only", method="update", args=[{"visible": _vis(A)}]),
                      ])],
)
out = os.path.join(ASSETS, "tree_queries_3d.html")
fig.write_html(out, include_plotlyjs=True, full_html=True)
print("WROTE", out, f"({len(SEL)} of {NQ} queries shown, relocation {reloc:.0f}%)")
