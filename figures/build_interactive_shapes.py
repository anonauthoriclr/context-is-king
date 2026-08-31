#!/usr/bin/env python3
"""Interactive 3-D 'topology on command' hero: the same seven weekdays rendered
as a ring, a cycle, and a tree by the in-context specification alone.

Emits into assets/ (committed, so the repo landing page is alive):
  * shapes_3d.html      -- self-contained interactive plotly figure. Buttons toggle
                           Centroids / Point cloud / Both (the cloud is the
                           per-prompt residuals behind each entity centroid).
  * shapes_rotating.gif -- auto-rotating orbit for embedding in the README.

Data: data/shapes/shape_v4cent_<tag>.npz  (entity centroids per condition, tree
edges/depths, and -- if present -- the per-prompt point clouds *_pts). No GPU.

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
day_idx = {d: i for i, d in enumerate(days)}

# (centroid key, title, mode, positions-for-ordering/coloring)
PANELS = [
    ("NAT_day",   "Natural (no rule)",   "ring", list(range(N))),
    ("CYCLE_end", "Imposed cycle",       "ring", z["CYCLE_ipos"].tolist()),
    ("TREE_end",  "Imposed tree",        "tree", None),
]


def basis_of(M):
    """3-D PCA basis (mean, top-3 components) fit on matrix M."""
    mu = M.mean(0)
    _, _, Vt = np.linalg.svd(M - mu, full_matrices=False)
    return mu, Vt[:3]


def project(X, b):
    mu, V = b
    return (X - mu) @ V.T


def ring_path(ip):
    o = list(np.argsort(ip))
    return o + [o[0]]


def panel_cloud(key, mode, ip):
    """Return (pts3d, colors) for the point cloud of a panel, or (None, None)."""
    pk = key + "_pts"
    if pk not in z.files:
        return None, None
    pts, ent = np.asarray(z[pk]), np.asarray(z[pk + "_ent"])
    di = np.array([day_idx[str(e)] for e in ent])
    colors = depth[di].astype(float) if mode == "tree" else np.asarray(ip, float)[di]
    return pts, colors


# ---------------------------------------------------------------- interactive HTML
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                    subplot_titles=[t for _, t, _, _ in PANELS], horizontal_spacing=0.02)
cen_idx, cloud_idx = [], []   # trace indices by kind, for the toggle buttons
ti = 0
have_clouds = False
for c, (key, _title, mode, ip) in enumerate(PANELS, start=1):
    cmap = "Viridis" if mode == "tree" else "Twilight"
    ccol = depth.astype(float) if mode == "tree" else np.array(ip, dtype=float)
    pts, pcol = panel_cloud(key, mode, ip)
    b = basis_of(pts if pts is not None else z[key])   # joint frame (fit on cloud if we have it)
    P = project(z[key], b)
    # centroid skeleton (edges) + nodes
    xs, ys, zs = [], [], []
    seq = (ring_path(ip) and [(ring_path(ip)[i], ring_path(ip)[i + 1]) for i in range(len(ring_path(ip)) - 1)]) if mode == "ring" else edges
    for a, bb in seq:
        xs += [P[a, 0], P[bb, 0], None]; ys += [P[a, 1], P[bb, 1], None]; zs += [P[a, 2], P[bb, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color="rgba(120,120,120,0.6)", width=4),
                               hoverinfo="skip", showlegend=False), row=1, col=c); cen_idx.append(ti); ti += 1
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                               text=[d[:3] for d in days], textposition="top center", textfont=dict(size=11),
                               marker=dict(size=7, color=ccol, colorscale=cmap, line=dict(color="white", width=1)),
                               hovertext=days, hoverinfo="text", showlegend=False), row=1, col=c); cen_idx.append(ti); ti += 1
    if pts is not None:
        have_clouds = True
        Pc = project(pts, b)
        fig.add_trace(go.Scatter3d(x=Pc[:, 0], y=Pc[:, 1], z=Pc[:, 2], mode="markers",
                                   marker=dict(size=2.6, color=pcol, colorscale=cmap, opacity=0.45),
                                   hoverinfo="skip", showlegend=False), row=1, col=c); cloud_idx.append(ti); ti += 1

n = ti
def _vis(on):
    return [i in on for i in range(n)]
both = set(range(n)); cen = set(cen_idx); cloud = set(cloud_idx)

scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
             aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)))
layout = dict(title=dict(text="Context sets the topology: the same seven weekdays, three "
                              "specifications  (drag to rotate)", x=0.5, font=dict(size=16)),
              scene=scene, scene2=scene, scene3=scene,
              margin=dict(l=0, r=0, t=70, b=0), width=1200, height=500, paper_bgcolor="white")
if have_clouds:
    layout["updatemenus"] = [dict(type="buttons", direction="right", x=0.5, xanchor="center", y=1.12,
                                  buttons=[
                                      dict(label="Both", method="update", args=[{"visible": _vis(both)}]),
                                      dict(label="Centroids", method="update", args=[{"visible": _vis(cen)}]),
                                      dict(label="Point cloud", method="update", args=[{"visible": _vis(cloud | cen - set(m for m in cen_idx if False))}]),
                                  ])]
    # "Point cloud" = clouds + centroid nodes (keep the labelled anchors), hide skeleton lines
    node_idx = set(cen_idx[1::2])
    layout["updatemenus"][0]["buttons"][2]["args"] = [{"visible": _vis(cloud | node_idx)}]
fig.update_layout(**layout)
html_path = os.path.join(ASSETS, "shapes_3d.html")
fig.write_html(html_path, include_plotlyjs=True, full_html=True)
print("WROTE", html_path, "(clouds)" if have_clouds else "(centroids only)")

# ---------------------------------------------------------------- rotating GIF
mfig = plt.figure(figsize=(12, 4.4))
axes = []
for c, (key, title, mode, ip) in enumerate(PANELS, start=1):
    ax = mfig.add_subplot(1, 3, c, projection="3d")
    cm = "viridis" if mode == "tree" else "twilight"
    ccol = depth.astype(float) if mode == "tree" else np.array(ip, dtype=float)
    b = basis_of(z[key])            # centroids-only for the GIF (the point cloud lives in the HTML toggle)
    P = project(z[key], b)
    if mode == "ring":
        pth = ring_path(ip); ax.plot(P[pth, 0], P[pth, 1], P[pth, 2], "-", color="0.6", lw=1.6)
    else:
        for a, bb in edges: ax.plot(*zip(P[a], P[bb]), color="0.6", lw=1.6)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=ccol, cmap=cm, s=55, edgecolor="k", lw=.4, depthshade=False)
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


anim = FuncAnimation(mfig, _spin, frames=range(0, 360, 4), interval=83, blit=False)
gif_path = os.path.join(ASSETS, "shapes_rotating.gif")
anim.save(gif_path, writer=PillowWriter(fps=12), dpi=80)
print("WROTE", gif_path)
