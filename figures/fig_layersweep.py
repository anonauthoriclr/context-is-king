#!/usr/bin/env python3
"""Appendix figure: the imposed-order geometry is a broad late-network plateau, not a single layer.
Full all-layer sweep of the entity-layout probe (days) for the two flagship models. We plot imposed-order RSA
and residual natural-order RSA at every layer vs network depth. The probe layer used throughout, 0.75*depth, is
marked; it sits on the plateau and below each model's own peak (so it is not tuned to the best layer).

Reproduce:  python paper/figures/fig_layersweep.py
Data: results/geometry/layer_sweep_gemma-4-31B-it.json, results/geometry/qwen_layer_sweep.json (scr_imp, scr_nat, nL)
Out:  paper/figures/fig_layersweep.pdf (+ .png)"""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
G = os.path.join(DATA, "geometry") + "/"
MODELS = [("Gemma-4-31B", "layer_sweep_gemma-4-31B-it.json", "#1b7837"),
          ("Qwen-3.5-27B", "qwen_layer_sweep.json", "#762a83")]
INK = "#1d2430"
fig, ax = plt.subplots(figsize=(6.4, 4.0)); fig.patch.set_facecolor("white")
ax.axhline(0, color="0.8", lw=0.6)
ax.axvline(0.75, color="#c0392b", lw=1.4, ls="--", zorder=1)
ax.text(0.75, 1.02, "probe layer (0.75)", color="#c0392b", fontsize=8.5, ha="center", va="bottom")
for name, fn, col in MODELS:
    d = json.load(open(G + fn)); nl = d["nL"]
    x = np.arange(nl) / (nl - 1)
    imp = np.array(d["scr_imp"], float); nat = np.array(d["scr_nat"], float)
    ax.plot(x, imp, "-", color=col, lw=2.0, label=f"{name}  (imposed)", zorder=3)
    ax.plot(x, nat, ":", color=col, lw=1.4, alpha=0.7, label=f"{name}  (natural)", zorder=2)
ax.set_xlabel("network depth  (layer / n_layers)", fontsize=10, color=INK)
ax.set_ylabel("RSA to the imposed / natural order", fontsize=10, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(-0.55, 1.05)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#c2c7d0")
ax.legend(fontsize=8, loc="lower right", frameon=False)
fig.tight_layout()
out = os.path.join(_OUT, "fig_layersweep")
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
print("WROTE", out)
for name, fn, _ in MODELS:
    d = json.load(open(G + fn)); nl = d["nL"]; imp = np.array(d["scr_imp"], float)
    L = int(round(0.75 * nl))
    print(f"  {name}: nL={nl} RSA@0.75={imp[L]:.2f} peak={np.nanmax(imp):.2f}@L{int(np.nanargmax(imp))}")
