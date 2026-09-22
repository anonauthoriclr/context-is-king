#!/usr/bin/env python3
"""Interactive 3-D 'topology on command' hero: the same seven weekdays rendered
as a ring, a cycle, and a tree by the in-context specification alone.

Emits two things into assets/ (committed, so the repo landing page is alive):
  * shapes_3d.html      -- self-contained interactive plotly figure (rotate it)
  * shapes_rotating.gif -- auto-rotating orbit for embedding in the README

Data: data/shapes/shape_v4cent_<tag>.npz  (Gemma-4-31B-it entity centroids per
condition + tree edges/depths). No GPU needed.

    python figures/build_interactive_shapes.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(HERE, "..", "data")))
ASSETS = os.environ.get("CIK_ASSETS", os.path.normpath(os.path.join(HERE, "..", "assets")))
os.makedirs(ASSETS, exist_ok=True)
TAG = "gemma-4-31B-it"

z = np.load(os.path.join(DATA, "shapes", f"shape_v4cent_{TAG}.npz"), allow_pickle=True)
days = [str(d) for d in z["days"]]
N = len(days)
edges = z["tree_edges"].tolist()
depth = z["tree_depth"]

# (key, title, mode, positions-for-ordering/coloring)
PANELS = [
    ("NAT_day",   "Natural (no rule) - a ring",   "ring", list(range(N))),
    ("CYCLE_end", "Imposed cycle - a ring",       "ring", z["CYCLE_ipos"].tolist()),
    ("TREE_end",  "Imposed tree - depth bands",   "tree", None),
]


def pca3(C):
    X = C - C.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:3].T


def ring_path(ip):
    o = list(np.argsort(ip))
    return o + [o[0]]


# ---------------------------------------------------------------- interactive HTML
fig = make_subplots(
    rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
    subplot_titles=[t for _, t, _, _ in PANELS], horizontal_spacing=0.02,
)
for c, (key, _title, mode, ip) in enumerate(PANELS, start=1):
    P = pca3(z[key])
    if mode == "ring":
        col = np.array(ip, dtype=float)
        path = ring_path(ip)
        fig.add_trace(go.Scatter3d(
            x=P[path, 0], y=P[path, 1], z=P[path, 2], mode="lines",
            line=dict(color="rgba(120,120,120,0.6)", width=4),
            hoverinfo="skip", showlegend=False), row=1, col=c)
        cmap = "Twilight"
    else:
        col = depth.astype(float)
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs += [P[a, 0], P[b, 0], None]
            ys += [P[a, 1], P[b, 1], None]
            zs += [P[a, 2], P[b, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(120,120,120,0.6)", width=4),
            hoverinfo="skip", showlegend=False), row=1, col=c)
        cmap = "Viridis"
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
        text=[d[:3] for d in days], textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=7, color=col, colorscale=cmap,
                    line=dict(color="white", width=1)),
        hovertext=days, hoverinfo="text", showlegend=False), row=1, col=c)

scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
             zaxis=dict(visible=False), aspectmode="data",
             camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)))
fig.update_layout(
    title=dict(text="Context sets the topology: the same seven weekdays, "
                    "three specifications  (drag to rotate)", x=0.5, font=dict(size=16)),
    scene=scene, scene2=scene, scene3=scene,
    margin=dict(l=0, r=0, t=60, b=0), width=1200, height=480,
    paper_bgcolor="white")
html_path = os.path.join(ASSETS, "shapes_3d.html")
fig.write_html(html_path, include_plotlyjs=True, full_html=True)
print("WROTE", html_path)

# ---------------------------------------------------------------- rotating GIF
mfig = plt.figure(figsize=(12, 4.2))
axes, data3d = [], []
for c, (key, title, mode, ip) in enumerate(PANELS, start=1):
    ax = mfig.add_subplot(1, 3, c, projection="3d")
    P = pca3(z[key])
    if mode == "ring":
        col = np.array(ip, dtype=float); cm = "twilight"; path = ring_path(ip)
        ax.plot(P[path, 0], P[path, 1], P[path, 2], "-", color="0.6", lw=1.6)
    else:
        col = depth.astype(float); cm = "viridis"
        for a, b in edges:
            ax.plot(*zip(P[a], P[b]), color="0.6", lw=1.6)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, cmap=cm, s=55,
               edgecolor="k", lw=.4, depthshade=False)
    for i in range(N):
        ax.text(P[i, 0], P[i, 1], P[i, 2], "  " + days[i][:3], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    axes.append(ax)

mfig.tight_layout()


def _spin(frame):
    for ax in axes:
        ax.view_init(elev=18, azim=frame)
    return []


anim = FuncAnimation(mfig, _spin, frames=range(0, 360, 6), interval=60, blit=False)
gif_path = os.path.join(ASSETS, "shapes_rotating.gif")
anim.save(gif_path, writer=PillowWriter(fps=18), dpi=80)
print("WROTE", gif_path)
