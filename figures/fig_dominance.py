#!/usr/bin/env python3
"""Section 4 figure: dominance on conflict, across family and scale, with a geometry calibration.
LEFT: per model, imposed-order RSA (green) vs residual natural-order RSA (gray), two families across scale (days, direct
prompt, per-scramble 95% CIs). RIGHT: imposed-order centroid geometry for three models, showing that the RSA value tracks
what is visible (high RSA = clean ring, low/0 = tangle).

Reproduce:  python paper/figures/fig_dominance.py
Data: results/geometry/multiscr_<model>.json (res['days']: per, nat_per); results/geometry/cache/defladder_<model>_days_list_s0_L*.npz
Out:  paper/figures/fig_dominance.pdf (+ .png)"""
import os, json, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
from scipy.stats import t as tdist, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CIK_OUT", os.path.join(HERE, "output")); os.makedirs(_OUT, exist_ok=True)
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
G = os.path.join(DATA, "geometry")
CACHE = os.path.join(G, "cache")
IMP_COL, NAT_COL, INK, REV = "#1b7837", "#8a93a3", "#1d2430", "#c0392b"
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N = 7
FAMILIES = [("Gemma", [("31B","gemma-4-31B-it"), ("E4B","gemma-4-E4B-it"), ("E2B","gemma-4-E2B-it")]),
            ("Qwen",  [("27B","Qwen3.5-27B"), ("9B","Qwen3.5-9B"), ("4B","Qwen3.5-4B")])]
THUMBS = [("Gemma 31B","gemma-4-31B-it"), ("Gemma E2B","gemma-4-E2B-it"), ("Qwen 9B","Qwen3.5-9B")]  # high -> low
def ci(a): a=np.asarray(a,float); n=len(a); return a.mean(), tdist.ppf(0.975,n-1)*a.std(ddof=1)/np.sqrt(n)
def cyc(p): P=np.array(p); D=np.abs(P[:,None]-P[None,:]); return np.minimum(D,N-D).astype(float)
def imp_geom(mf):
    f=sorted(glob.glob(os.path.join(CACHE, f"defladder_{mf}_days_list_s0_L*.npz")))[0]
    d=np.load(f,allow_pickle=True); A=d["A"].astype(np.float64); ent=d["ent"]
    cents=np.stack([A[ent==e].mean(0) for e in DAYS])
    order=list(np.random.default_rng(0).permutation(DAYS)); ip=[order.index(e) for e in DAYS]
    X=cents-cents.mean(0); Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); COS=1-Xn@Xn.T
    iu=np.triu_indices(N,1); rsa=spearmanr(COS[iu],cyc(ip)[iu]).correlation
    _,_,Vt=np.linalg.svd(X,full_matrices=False); P=(X)@Vt[:2].T; P=P/(np.abs(P).max()+1e-9)
    return P, np.array(ip), rsa

fig = plt.figure(figsize=(9.6, 4.3)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1.0], wspace=0.06, hspace=0.35)
axd = fig.add_subplot(gs[:, 0])

# ---- left: dumbbell ----
yticks=[]; ylabs=[]; spans=[]; y=0.0; rows=[]
for fam, models in FAMILIES:
    y0=y
    for lab, mf in models:
        d=json.load(open(os.path.join(G, f"multiscr_{mf}.json")))["res"]["days"]
        im, ih = ci(d["per"]); nm, nh = ci(d["nat_per"]); rows.append((y, im, ih, nm, nh, im-nm))
        yticks.append(y); ylabs.append(lab); y-=1
    spans.append((fam, y0, y+1)); y-=0.8
axd.axvline(0, color="#cfd3da", lw=1, zorder=0)
for (yy, im, ih, nm, nh, gap) in rows:
    # connector drawn only for reversals (semantically meaningful); the neutral-grey
    # connector was dropped -- it read as part of the grey CI (reviewer #75).
    if gap < 0: axd.plot([nm, im], [yy, yy], "-", color=REV, lw=2.0, zorder=1)
    axd.errorbar(nm, yy, xerr=nh, fmt="o", color=NAT_COL, ms=8, capsize=3, zorder=3)
    axd.errorbar(im, yy, xerr=ih, fmt="o", color=IMP_COL, ms=8, capsize=3, zorder=3)
    if gap < 0: axd.annotate("reverses", (max(im,nm)+0.04, yy), va="center", fontsize=8, color=REV, fontstyle="italic")
blend = mtrans.blended_transform_factory(axd.transAxes, axd.transData)
for fam, ytop, ybot in spans:
    axd.text(-0.165, (ytop+ybot)/2, fam, rotation=90, va="center", ha="center", fontsize=11,
             color=INK, fontweight="bold", transform=blend, clip_on=False)
axd.set_yticks(yticks); axd.set_yticklabels(ylabs, fontsize=9.5, color=INK)
axd.set_xlabel("RSA to the order  (days, direct prompt)", fontsize=10, color=INK)
axd.set_xlim(-0.5, 1.0)
for s in ["top","right"]: axd.spines[s].set_visible(False)
for s in ["left","bottom"]: axd.spines[s].set_color("#c2c7d0")
axd.tick_params(length=0)
axd.legend(handles=[Line2D([0],[0],marker="o",ls="",color=IMP_COL,label="imposed order"),
                    Line2D([0],[0],marker="o",ls="",color=NAT_COL,label="pretrained order")],
           fontsize=8.5, loc="lower right", frameon=False)

# ---- right: centroid geometry calibration ----
for i,(lab, mf) in enumerate(THUMBS):
    a=fig.add_subplot(gs[i,1]); P, ip, rsa = imp_geom(mf)
    o=list(np.argsort(ip)); o=o+[o[0]]
    a.plot(P[o,0], P[o,1], "-", color=IMP_COL, lw=1.8)
    a.scatter(P[:,0], P[:,1], s=34, c=np.argsort(np.argsort(ip)), cmap="twilight_shifted", edgecolor="white", linewidth=0.8, zorder=3)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.margins(0.2)
    for s in a.spines.values(): s.set_color("#d6d9e0")
    a.set_title(f"{lab}:  RSA {rsa:+.2f}", fontsize=8.5, color=INK, pad=2)
fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.12)
out=os.path.join(_OUT,"fig_dominance")
fig.savefig(out+".pdf",bbox_inches="tight"); fig.savefig(out+".png",dpi=170,bbox_inches="tight")
print("WROTE",out)
for lab,mf in THUMBS: _,_,r=imp_geom(mf); print(f"  thumb {lab}: RSA {r:+.2f}")
