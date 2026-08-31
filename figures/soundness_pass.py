#!/usr/bin/env python3
"""CPU soundness pass: (A) per-scramble CIs on every headline number; (B) k>=2 RSA-vs-accuracy (real dynamic range).
Saved data only. Writes results/soundness_ci.md + FIG_rsa_vs_kacc.png."""
import os, json, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist, spearmanr
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
G=os.path.join(DATA,"geometry")+"/"; B=os.path.join(DATA,"behavior")+"/"; C=os.path.join(DATA,"causal")+"/"
_OUT = os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(_OUT, exist_ok=True)
def ci(a):  # mean, half-width 95% CI (t)
    a=np.asarray(a,float); n=len(a)
    if n<2: return (float(a.mean()) if n else float('nan')), float('nan'), n
    h=tdist.ppf(0.975,n-1)*a.std(ddof=1)/np.sqrt(n); return float(a.mean()),float(h),n
def fmt(m,h,n): return f"{m:+.3f} ± {h:.3f} (n={n})" if h==h else f"{m:+.3f} (n={n}, no CI)"
PARAM={"gemma-4-E2B":2,"gemma-4-E4B":4,"gemma-4-12B":12,"gemma-4-31B":31,"Qwen3.5-4B":4,"Qwen3.5-9B":9,"Qwen3.5-27B":27,"Llama-3.1-8B":8}
FAM=lambda m:("Gemma" if "gemma" in m.lower() else "Qwen" if "qwen" in m.lower() else "Llama")
def key(s):
    for k in PARAM:
        if k.lower() in s.lower(): return k
    return None
out=["# Soundness pass — per-scramble CIs + k>=2 predictive (2026-06-25, CPU)\n"]

# ---------- A1: dominance gap (snap) per model/concept, paired CI ----------
out.append("## A. Headline numbers with 95% CIs\n\n### Dominance: imposed-RSA vs residual-natural-RSA (snap, paired by scramble)\n")
out.append("| model | concept | imposed-RSA | natural-RSA | gap (imp−nat) |\n|---|---|---|---|---|")
for f in sorted(glob.glob(G+"multiscr_*.json")):
    d=json.load(open(f)); k=key(d["model"]) or key(f)
    for c in ("days","months"):
        if c not in d["res"]: continue
        r=d["res"][c]; per=r.get("per"); nat=r.get("nat_per")
        if not per or not nat: continue
        gi=ci(per); gn=ci(nat); gg=ci(np.array(per)-np.array(nat))
        out.append(f"| {k} | {c} | {fmt(*gi)} | {fmt(*gn)} | **{fmt(*gg)}** |")

# ---------- A2: dominance gap natural-prompt (Gemma-31B days), recompute per-scramble ----------
def cyc_rsa(C, order, N):
    Cc=C-C.mean(0); Cn=Cc/(np.linalg.norm(Cc,axis=1,keepdims=True)+1e-12); COS=1-Cn@Cn.T
    iu=np.triu_indices(N,1)
    tmpl=np.array([[min(abs(order[i]-order[j]),N-abs(order[i]-order[j])) for j in range(N)] for i in range(N)],float)
    return spearmanr(COS[iu],tmpl[iu]).correlation
ng=B+"naturalgeom_gemma-4-31B-it.npz"
try:
    d=np.load(ng,allow_pickle=True)
    cs=[d[k] for k in sorted(d.files,key=lambda x:(len(x),x)) if k.startswith("c") and k[1:].isdigit()]
    imp=d["imp"]; N=cs[0].shape[0]; nat_order=list(range(N))
    ii=[cyc_rsa(cs[si].astype(float), list(imp[si]), N) for si in range(len(cs))]
    nn=[cyc_rsa(cs[si].astype(float), nat_order, N) for si in range(len(cs))]
    gi=ci(ii); gn=ci(nn); gg=ci(np.array(ii)-np.array(nn))
    out.append(f"| gemma-4-31B | days **(natural prompt)** | {fmt(*gi)} | {fmt(*gn)} | **{fmt(*gg)}** |")
except Exception as e: out.append(f"| gemma-4-31B | days (natural prompt) | ERR {e} | | |")

# ---------- A3: depth-RSA (tree) per-scramble CI ----------
def tree_depth_rsa_perscr(f):
    d=np.load(f,allow_pickle=True); cs=[d[k] for k in d.files if k.startswith("c") and k[1:].isdigit()]; nodes=np.array(d["nodes"])
    N=cs[0].shape[0]; D=0
    while 2**(D+1)-1<N: D+=1
    PAR={i:(i-1)//2 for i in range(1,N)}
    def dep(n):
        x=n;dd=0
        while x:x=PAR[x];dd+=1
        return dd
    DEP=np.array([dep(n) for n in range(N)]); r=[]
    for si in range(len(cs)):
        ndi=nodes[si] if nodes.ndim==2 else nodes; pos={int(ndi[k]):k for k in range(len(ndi))}
        Cm=cs[si].astype(float); Cc=Cm-Cm.mean(0); Cn=Cc/(np.linalg.norm(Cc,axis=1,keepdims=True)+1e-12); COS=1-Cn@Cn.T
        idx=[pos[g] for g in range(N)]; iu=np.triu_indices(N,1)
        dep_t=np.abs(DEP[:,None]-DEP[None,:])[iu]
        r.append(spearmanr(COS[np.ix_(idx,idx)][iu],dep_t).correlation)
    return r
out.append("\n### Depth-RSA (imposed tree, neutral, per-scramble)\n| tree | model | depth-RSA |\n|---|---|---|")
for tag,lab in [("gemma-4-31B-it_neutral_d3","Gemma d3"),("Qwen3.5-27B_neutral_d3","Qwen d3"),
                ("gemma-4-31B-it_neutral_d4","Gemma d4"),("Qwen3.5-27B_neutral_d4","Qwen d4")]:
    f=G+f"tree_{tag}.npz"
    try: out.append(f"| {lab.split()[1]} | {lab.split()[0]} | {fmt(*ci(tree_depth_rsa_perscr(f)))} |")
    except Exception as e: out.append(f"| {lab} | | ERR {e} |")

# ---------- A4: causal flip (n small — report honestly) ----------
out.append("\n### Causal flip (entity-substitution, per-scramble) — thin base, flagged\n| model | layer | patch_success | cross-map |\n|---|---|---|---|")
try:
    d=json.load(open(C+"causaluse_gemma-4-31B-it.json")); ps=d["contexts"]["imposed"]["per_scr"]; sw=d["sweep"]
    # flip locus = layer with highest mean patch_success (documented clean locus L27, depth ~0.45)
    means={L:np.mean([p["layers"][str(L)]["patch_success"] for p in ps]) for L in sw}
    Lf=max(means,key=means.get)
    s=[p["layers"][str(Lf)]["patch_success"] for p in ps]; cm=[p["layers"][str(Lf)]["cross_map"] for p in ps]
    out.append(f"| gemma-4-31B | L{Lf} (flip locus, d={Lf/d['nl']:.2f}) | {fmt(*ci(s))} | {fmt(*ci(cm))} |")
    out.append(f"\n*Causal n is only {len(ps)} scrambles (snap) — CI degenerate; per-scramble values are "
               f"{[round(x,2) for x in s]} (own-map flip) vs {[round(x,2) for x in cm]} (other-map=chance). "
               f"Flagged as a limitation; +scrambles queued (GPU).*")
except Exception as e: out.append(f"| gemma-4-31B | | ERR {e} | |")

# ---------- B: k>=2 RSA vs accuracy (dynamic range) ----------
out.append("\n## B. Does construction quality predict MULTI-HOP (k>=2) reading, not just k=1?\n")
xs={}  # model -> imposed RSA (k1summary, regime-matched)
for f in glob.glob(B+"k1summary_*_list.json"):
    d=json.load(open(f)); k=key(d["model"]) or key(f)
    if k and d.get("rsa_imp") is not None: xs[k]=d["rsa_imp"]
rows=[]  # k -> list of (rsa, acc) across models
KMAX=4
for f in glob.glob(B+"acc_*_days.json"):
    d=json.load(open(f)); k=key(d["model"]) or key(f)
    kc=d.get("res",{}).get("list",{}).get("kcurve_imp",{})
    if k in xs and kc:
        for kk in range(1,KMAX+1):
            if str(kk) in kc: rows.append((kk,xs[k],kc[str(kk)],k))
out.append("| k (hops) | n models | Spearman ρ | p | sig@.05 | acc range (min–max) |\n|---|---|---|---|---|---|")
fig,ax=plt.subplots(figsize=(5.6,4.2)); cols=plt.cm.viridis(np.linspace(0.1,0.85,KMAX))
for kk in range(1,KMAX+1):
    pts=[(r[1],r[2],r[3]) for r in rows if r[0]==kk]
    if len(pts)<3: continue
    rr,pp=spearmanr([p[0] for p in pts],[p[1] for p in pts])
    accs=[p[1] for p in pts]
    out.append(f"| k={kk} | {len(pts)} | {rr:+.2f} | {pp:.3f} | {'yes' if pp<0.05 else 'no'} | {min(accs):.2f}–{max(accs):.2f} |")
    ax.scatter([p[0] for p in pts],[p[1] for p in pts],color=cols[kk-1],s=45,label=f"k={kk} (ρ={rr:+.2f})",zorder=3)
ax.set_xlabel("construction quality (imposed-order RSA)"); ax.set_ylabel("imposed k-step accuracy (snap)")
ax.set_title("Construction quality vs MULTI-HOP reading\n(k=1 is ceiling-saturated; dynamic range opens at k>=2)",fontsize=10.5)
ax.legend(fontsize=8); ax.axvline(0,color="0.8",ls=":",lw=0.6); ax.set_ylim(0,1.05)
fig.tight_layout(); fig.savefig(os.path.join(_OUT,"FIG_rsa_vs_kacc.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
out.append("\n*Note: regime-matched snap (list). Only k=1 is significant at .05 (n=8); k>=2 point estimates stay positive "
           "but underpowered. CONFOUND: the 8 models span only 3 families (Gemma high-RSA/high-acc, Qwen mid, Llama low) — "
           "the rank correlation is closer to a 3-cluster separation than 8 independent points. Report as directional, not a law.*")

open(os.path.join(_OUT,"soundness_ci.md"),"w").write("\n".join(out))
print("\n".join(out)); print(f"\nWROTE {_OUT}/soundness_ci.md + {_OUT}/FIG_rsa_vs_kacc.png")
