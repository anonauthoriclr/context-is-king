#!/usr/bin/env python3
"""Interactive 3-D 'structure from nothing' explorer -- the RING version.

Seven MEANING-FREE tokens (Apple, River, Chair, Cloud, Tiger, Bridge, Lamp) with
no pretrained order are given only a declared cyclic order and asked step queries.
A plain numbered list arranges them along a LINE; adding the single fact "the order
wraps around: <last> is immediately followed by <first>" snaps the same tokens into
a closed RING. Topology follows the in-context specification, on arbitrary tokens.

Button toggles the two conditions so you can watch the wrap edge close the ring.
A line and a ring agree on almost every pair, so raw cycle-RSA barely moves; the
metrics that separate them are CLOSURE (last->first gap / neighbour gap: ~1 when
the ring closes, >>1 when the line stays open) and dRSA (cycle - line):
  * Plain list (no wrap)     -> line   (closure ~2.8, dRSA < 0)
  * Cyclic 'wraps around'    -> ring   (closure ~1.1, dRSA > 0)

Nodes are the actual arbitrary tokens (labeled), colored by imposed cyclic
position; the grey path follows the declared order.

Data (committed, no GPU): data/behavior/wrap2x2_gemma-4-31B-it_arb7.npz
(R = per-(cond,scramble,entity,query) residual, COND/SI/ENT indices,
imp[si] = imposed cyclic position per entity, conds = the 4 condition names).
Layout uses scramble 0; consensus position per token = mean over its queries.

    python figures/build_interactive_rings.py
Out: assets/rings_3d.html
"""
import os
import numpy as np
import plotly.graph_objects as go
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(HERE, "..", "data")))
ASSETS = os.environ.get("CIK_ASSETS", os.path.normpath(os.path.join(HERE, "..", "assets")))
os.makedirs(ASSETS, exist_ok=True)
TAG = "gemma-4-31B-it"
CONCEPT = "arb7"
TOKENS = ["Apple", "River", "Chair", "Cloud", "Tiger", "Bridge", "Lamp"]
N = len(TOKENS)
SI = 0                                            # layout scramble

z = np.load(os.path.join(DATA, "behavior", f"wrap2x2_{TAG}_{CONCEPT}.npz"), allow_pickle=True)
R = z["R"].astype(np.float64)
COND, SIv, ENT = z["COND"], z["SI"], z["ENT"]
conds = [str(c) for c in z["conds"]]              # ['list','list_wrap','adj_nowrap','adj']
ip = z["imp"][SI]                                 # imposed cyclic position per entity (token order)


def cyc_rdm(p):
    P = np.array(p)
    D = np.abs(P[:, None] - P[None, :])
    return np.minimum(D, N - D).astype(float)


def lin_rdm(p):
    P = np.array(p)
    return np.abs(P[:, None] - P[None, :]).astype(float)


def cos_rdm(c):
    X = c - c.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return 1 - X @ X.T


ut = lambda M: M[np.triu_indices(N, 1)]


def ring_stats(C):
    """dRSA(cycle-line) and closure (wrap gap / mean neighbour gap) for centroids C."""
    D = cos_rdm(C)
    drsa = (spearmanr(ut(D), ut(cyc_rdm(ip))).correlation
            - spearmanr(ut(D), ut(lin_rdm(ip))).correlation)
    p2e = {ip[e]: e for e in range(N)}
    wrap = D[p2e[0], p2e[N - 1]]
    interior = np.mean([D[p2e[i], p2e[i + 1]] for i in range(N - 1)])
    return float(drsa), float(wrap / (interior + 1e-9))


def centroids(cond):
    """Per-token consensus residual for one condition, scramble SI (token order)."""
    ci = conds.index(cond)
    m = (COND == ci) & (SIv == SI)
    return np.stack([R[m & (ENT == e)].mean(0) for e in range(N)])


# Joint 3-D PCA over the two conditions we show, so their layouts are comparable.
PANELS = [("list_wrap", "Cyclic order (“wraps around”) → ring"),
          ("list", "Plain numbered list → line")]
Cs = {c: centroids(c) for c, _ in PANELS}
stack = np.vstack([Cs[c] for c, _ in PANELS])
mu = stack.mean(0)
_, _, Vt = np.linalg.svd(stack - mu, full_matrices=False)
proj = {c: (Cs[c] - mu) @ Vt[:3].T for c, _ in PANELS}

order = list(np.argsort(ip))                      # tokens in imposed cyclic order
loop = order + [order[0]]                          # close the path

fig = go.Figure()
cond_traces = {}
ti = 0
for cond, _lab in PANELS:
    M = proj[cond]
    drsa, closure = ring_stats(Cs[cond])
    vis = (cond == "list_wrap")
    # declared-order path
    fig.add_trace(go.Scatter3d(x=M[loop, 0], y=M[loop, 1], z=M[loop, 2], mode="lines",
                               line=dict(color="rgba(120,120,120,0.7)", width=4),
                               hoverinfo="skip", showlegend=False, visible=vis))
    e_i = ti; ti += 1
    # nodes: arbitrary tokens, colored by imposed cyclic position
    fig.add_trace(go.Scatter3d(
        x=M[:, 0], y=M[:, 1], z=M[:, 2], mode="markers+text",
        text=TOKENS, textposition="top center", textfont=dict(size=10),
        marker=dict(size=7, color=ip, colorscale="HSV", cmin=0, cmax=N - 1,
                    line=dict(color="white", width=1),
                    colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, len=0.35,
                                  thickness=12, title=dict(text="imposed cyclic position", side="top"))),
        hovertext=[f"{TOKENS[k]} — position {ip[k]}" for k in range(N)], hoverinfo="text",
        showlegend=False, visible=vis))
    n_i = ti; ti += 1
    cond_traces[cond] = ([e_i, n_i], closure, drsa)

n = ti


def _vis(cond):
    keep = set(cond_traces[cond][0])
    return [i in keep for i in range(n)]


scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
             aspectmode="cube", camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)))
fig.update_layout(
    title=dict(text="Structure from nothing: 12 meaning-free tokens under a declared cyclic order "
                    "snap into a ring<br><sub>the single “wraps around” fact closes the line into a "
                    "ring &nbsp;|&nbsp; nodes are arbitrary tokens, colored by imposed position &nbsp;|&nbsp; "
                    "drag to rotate, use the buttons to add/remove the wrap</sub>", x=0.5, font=dict(size=15)),
    scene=scene, margin=dict(l=0, r=0, t=70, b=40), width=1050, height=720, paper_bgcolor="white",
    updatemenus=[dict(type="buttons", direction="down", x=0.0, xanchor="left", y=0.98, yanchor="top", showactive=True,
                      buttons=[dict(label=f"{lab}   (closure {cond_traces[cond][1]:.2f}, dRSA {cond_traces[cond][2]:+.2f})",
                                    method="update", args=[{"visible": _vis(cond)}]) for cond, lab in PANELS])],
)
out = os.path.join(ASSETS, "rings_3d.html")
fig.write_html(out, include_plotlyjs=True, full_html=True)
print("WROTE", out, f"({N} arbitrary tokens; ring closure {cond_traces['list_wrap'][1]:.2f}/dRSA {cond_traces['list_wrap'][2]:+.2f}, "
      f"line closure {cond_traces['list'][1]:.2f}/dRSA {cond_traces['list'][2]:+.2f})")
