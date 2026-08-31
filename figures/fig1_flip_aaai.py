#!/usr/bin/env python3
"""Figure 1 (hero) -- AAAI two-column restyle of fig1_flip.py.
Identical data and logic; only the visual scale differs (larger fonts, thicker
lines/markers) so the figure stays legible when shrunk to AAAI column width.

Reproduce:  python figures/fig1_flip_aaai.py
Out:  output/fig1_flip_aaai.pdf (+ .png)"""
import os, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import twilight_shifted
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
CACHE = os.path.join(DATA, "geometry", "cache")
TAG = "gemma-4-31B-it"
ENT = {
 "days":   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
 "months": ["January","February","March","April","May","June","July","August","September","October","November","December"],
 "zodiac": ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],
 "clock":  ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"],
}
CONCEPTS = ["days", "months", "clock"]
IMP_COL, NAT_COL, INK, FRAME = "#1b7837", "#8a93a3", "#1d2430", "#c2c7d0"

def cyc_rsa(cents, pos):
    X = cents - cents.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); COS = 1 - X @ X.T
    N = len(pos); iu = np.triu_indices(N, 1)
    P = np.array(pos); D = np.abs(P[:, None] - P[None, :]); tmpl = np.minimum(D, N - D).astype(float)
    return spearmanr(COS[iu], tmpl[iu]).correlation

fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(11.4, 4.0)); fig.patch.set_facecolor("white")
for ax, C in zip(axes, CONCEPTS):
    base = ENT[C]; N = len(base)
    f = sorted(glob.glob(os.path.join(CACHE, f"defladder_{TAG}_{C}_list_s0_L*.npz")))
    d = np.load(f[0], allow_pickle=True); A = d["A"].astype(np.float64); ent = d["ent"]
    cents = np.stack([A[ent == e].mean(0) for e in base])
    order = list(np.random.default_rng(0).permutation(base))           # imposed order (as in extraction)
    ip = np.array([order.index(e) for e in base])                       # imposed position per base entity
    nat = np.arange(N)                                                  # canonical order
    mu = cents.mean(0); _, _, Vt = np.linalg.svd(cents - mu, full_matrices=False); P = (cents - mu) @ Vt[:2].T
    P = P / (np.abs(P).max() + 1e-9)
    colors = twilight_shifted(np.linspace(0.06, 0.94, N))
    def loop(pos): o = list(np.argsort(pos)); return o + [o[0]]
    ax.plot(P[loop(nat), 0], P[loop(nat), 1], "--", color=NAT_COL, lw=2.6, dashes=(5, 3), zorder=1)
    ax.plot(P[loop(ip), 0], P[loop(ip), 1], "-", color=IMP_COL, lw=3.8, zorder=2)
    ax.scatter(P[:, 0], P[:, 1], s=260, c=colors[ip], edgecolor="white", linewidth=2.4, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22)   # identical box on every panel
    for s in ax.spines.values(): s.set_color(FRAME); s.set_linewidth(1.4)
    ax.set_xlabel("PC 1", fontsize=23, color=INK); ax.set_ylabel("PC 2", fontsize=23, color=INK)
    ax.set_title(f"{C} (N$=${N})", fontsize=26, color=INK)
handles = [Line2D([0], [0], color=IMP_COL, lw=3.8, label="in-context (imposed) order"),
           Line2D([0], [0], color=NAT_COL, lw=2.6, ls="--", dashes=(5, 3), label="pretrained (canonical) order")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=24, bbox_to_anchor=(0.5, -0.04))
fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.18, wspace=0.12)
out = os.path.join(_OUT, "fig1_flip_aaai")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
