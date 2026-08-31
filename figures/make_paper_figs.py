#!/usr/bin/env python3
"""Three paper figures for the two-pillar framing (CPU, saved data):
  FIG_scale_gate.png  — Pillar 2: imposed-order RSA vs model scale, WITHIN-FAMILY ladders (disentangles scale from family).
  FIG_rsa_vs_acc.png  — predictive so-what: construction quality (imposed-RSA) vs one-shot k=1 accuracy, 8 models/3 families.
  FIG_boundary.png    — F7: identity->binding boundary. wrong-slot control leaks early then cleans; answer commits ~0.7."""
import os, json, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
G=os.path.join(DATA,"geometry")+"/"; B=os.path.join(DATA,"behavior")+"/"; C=os.path.join(DATA,"causal")+"/"
PARAM={"gemma-4-E2B":2,"gemma-4-E4B":4,"gemma-4-12B":12,"gemma-4-31B":31,
       "Qwen3.5-4B":4,"Qwen3.5-9B":9,"Qwen3.5-27B":27,"Llama-3.1-8B":8}
FAM=lambda m:("Gemma" if "gemma" in m.lower() else "Qwen" if "qwen" in m.lower() else "Llama")
FCOL={"Gemma":"#1b7837","Qwen":"#762a83","Llama":"#d95f02"}
def key(s):  # normalize a filename/model id to a PARAM key
    for k in PARAM:
        if k.lower() in s.lower(): return k
    return None

# ---------- FIG 1: scale is the gate ----------
def scale_fig():
    pts={}  # fam -> list of (param, [per-concept rsa], k)
    for f in glob.glob(G+"multiscr_*.json"):
        d=json.load(open(f)); m=d["model"]; k=key(m) or key(f)
        if not k: continue
        vals=[d["res"][c]["imp_mean"] for c in ("days","months") if c in d["res"] and d["res"][c].get("imp_mean") is not None]
        pts.setdefault(FAM(k),[]).append((PARAM[k],vals,k))
    fig,ax=plt.subplots(figsize=(5.6,4.2))
    for fam,rows in pts.items():
        rows=sorted(rows)
        xs=[r[0] for r in rows]; mean=[np.mean(r[1]) for r in rows]; lo=[min(r[1]) for r in rows]; hi=[max(r[1]) for r in rows]
        ax.plot(xs,mean,"-o",color=FCOL[fam],label=fam,lw=2.2,ms=7,zorder=3)
        ax.fill_between(xs,lo,hi,color=FCOL[fam],alpha=0.13,zorder=1)  # days–months spread
        for x,r in zip(xs,rows):  # faint per-concept points
            for v in r[1]: ax.scatter(x,v,color=FCOL[fam],s=14,alpha=0.4,zorder=2)
    ax.scatter([8],[0.08],marker="X",s=110,color=FCOL["Llama"],zorder=5,label="Llama-8B (nat-prompt)")
    ax.axhline(0,color="0.7",lw=0.6); ax.set_xscale("log")
    ax.set_xticks([2,4,8,12,27,31]); ax.set_xticklabels(["2","4","8","12","27","31"])
    ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("imposed-order RSA (direct prompt, days–months mean)")
    ax.legend(fontsize=8,loc="lower right"); ax.set_ylim(-0.1,1.0)
    fig.tight_layout(); fig.savefig(G+"FIG_scale_gate.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("WROTE",G+"FIG_scale_gate.png")
    for fam,rows in pts.items(): print(" ",fam,sorted([(r[2],round(float(np.mean(r[1])),2)) for r in rows]))

# ---------- FIG 2: RSA vs one-shot accuracy ----------
def rsa_acc_fig():
    xs=[];ys=[];cs=[];labs=[]
    for f in glob.glob(B+"k1summary_*_list.json"):
        d=json.load(open(f)); m=d["model"]; k=key(m) or key(f)
        if k is None or d.get("rsa_imp") is None or d.get("acc_imp") is None: continue
        xs.append(d["rsa_imp"]); ys.append(d["acc_imp"]); cs.append(FCOL[FAM(k)]); labs.append(k)
    xs=np.array(xs);ys=np.array(ys)
    rho=spearmanr(xs,ys).correlation
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    ax.axvline(0,color="0.8",lw=0.6,ls=":")
    for x,y,c,l in zip(xs,ys,cs,labs):
        ax.scatter(x,y,color=c,s=80,zorder=3,edgecolor="k",lw=0.4)
        ax.annotate(l.replace("gemma-4-","G").replace("Qwen3.5-","Q").replace("Llama-3.1-","L"),
                    (x,y),fontsize=7,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("construction quality (imposed-order RSA)"); ax.set_ylabel("one-shot k=1 adjacency accuracy")
    ax.set_title(f"Construction quality tracks one-shot behavior\nSpearman ρ={rho:.2f} (8 models, 3 families)",fontsize=11)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=FCOL[f],label=f) for f in FCOL],fontsize=8,loc="lower right")
    ax.set_ylim(0,1.05)
    fig.tight_layout(); fig.savefig(B+"FIG_rsa_vs_acc.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("WROTE",B+"FIG_rsa_vs_acc.png","| rho",round(rho,3),"| n",len(xs))

# ---------- FIG 3 (F7): identity->binding boundary ----------
def boundary_fig():
    d=json.load(open(C+"causaluse_gemma-4-31B-it.json")); nl=d["nl"]; sw=d["sweep"]; ps=d["contexts"]["imposed"]["per_scr"]
    patch={};cross={};
    for L in sw:
        patch[L]=np.mean([p["layers"][str(L)]["patch_success"] for p in ps])
        cross[L]=np.mean([p["layers"][str(L)]["cross_map"] for p in ps])
    # wrong-slot control: controls/{layer}/position_token_success
    ctrl_layers=sorted({int(k) for p in ps for k in p.get("controls",{})})
    wrong={L:np.mean([p["controls"][str(L)]["position_token_success"] for p in ps if str(L) in p.get("controls",{})]) for L in ctrl_layers}
    fig,ax=plt.subplots(figsize=(5.8,4.2))
    xs=[L/nl for L in sw]
    ax.plot(xs,[patch[L] for L in sw],"-o",color="#1b7837",lw=2,ms=6,label="entity-slot patch (imposed flip)")
    ax.plot([L/nl for L in ctrl_layers],[wrong[L] for L in ctrl_layers],"-s",color="#d95f02",lw=2,ms=6,
            label="wrong-slot control (position token)")
    ax.plot(xs,[cross[L] for L in sw],":",color="0.45",lw=1.6,label="other-map control (chance)")
    ax.axvspan(0.20,0.45,color="#fde725",alpha=0.18)
    ax.annotate("identity→binding\nboundary",(0.325,0.55),ha="center",fontsize=8.5,color="#5a5a00")
    ax.axvline(0.70,color="0.6",ls="--",lw=1); ax.annotate("answer commits",(0.71,0.85),fontsize=8,rotation=90,va="top")
    ax.set_xlabel("network depth (layer / n_layers)"); ax.set_ylabel("flip success")
    ax.legend(fontsize=8,loc="upper right"); ax.set_ylim(-0.05,1.05)
    fig.tight_layout(); fig.savefig(C+"FIG_boundary_identity_binding.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print("WROTE",C+"FIG_boundary_identity_binding.png")
    print("  patch:",{L:round(patch[L],2) for L in sw}); print("  wrong-slot:",{L:round(wrong[L],2) for L in ctrl_layers})

scale_fig(); rsa_acc_fig(); boundary_fig()
