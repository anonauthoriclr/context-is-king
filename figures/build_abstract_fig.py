"""BlackboxNLP abstract Figure 1: same 7 weekdays -> natural ring / imposed cycle / imposed tree.
3-panel 2D-PCA, larger nodes + readable labels + PC axis labels. Gemma-4-31B.
Reads persisted data from results/; writes fig_shapes.pdf into the abstract's figures/ dir."""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
RES=os.path.join(DATA,"shapes")
OUT=os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(OUT, exist_ok=True)
tag="gemma-4-31B-it"
z=np.load(f"{RES}/shape_v4cent_{tag}.npz",allow_pickle=True)
days=[str(d) for d in z["days"]]; N=len(days)
J=json.load(open(f"{RES}/shape_v4_{tag}.json"))
def stat(cond,pos,key):
    v=[r[key] for r in J if r["cond"]==cond and r["pos"]==pos and r[key] is not None]
    return (np.mean(v),np.std(v)) if v else (float("nan"),0)
def pr(cond,pos): m,s=stat(cond,pos,"partR"); return f"eff-dim {m:.1f}"
cyc_ip=z["CYCLE_ipos"].tolist(); depth=z["tree_depth"]; edges=z["tree_edges"].tolist()
# 3 panels: drop the LINE case (it lives in the full paper's appendix)
panels=[("NAT_day","Natural (no rule)\npretrained ring","ring",list(range(N)),pr("NAT-neutral","day")),
        ("CYCLE_end","Imposed cycle\nring","ring",cyc_ip,pr("CYCLE","end")),
        ("TREE_end","Imposed tree\ndepth-stratified","tree",None,pr("TREE","end"))]
def pca(C,d): X=C-C.mean(0); U,S,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:d].T
def order_of(ip): return list(np.argsort(ip))
fig,axs=plt.subplots(1,3,figsize=(11.5,4.1))
for ax,(key,title,mode,ip,ann) in zip(axs,panels):
    P=pca(z[key],2)
    if mode=="ring":
        s=order_of(ip); path=s+[s[0]]
        ax.plot(P[path,0],P[path,1],"-",color="0.55",lw=1.9,zorder=1); col=np.array(ip); cm="twilight"
    else:
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.55",lw=1.9,zorder=1)
        col=depth; cm="viridis"
    # bigger nodes so labels are legible at column/figure width
    ax.scatter(P[:,0],P[:,1],c=col,cmap=cm,s=460,edgecolor="k",lw=.9,zorder=3)
    for i in range(N):
        ax.annotate(days[i][:3],(P[i,0],P[i,1]),fontsize=9.5,fontweight="bold",
                    ha="center",va="center",zorder=4,color="black",
                    path_effects=[pe.withStroke(linewidth=2.4,foreground="white")])
    ax.set_title(title,fontsize=12)
    ax.set_aspect("equal","datalim")
    ax.set_xticks([]); ax.set_yticks([])
    # axes ARE the two leading PCs -> label them
    ax.set_xlabel("PC 1",fontsize=9,labelpad=2); ax.set_ylabel("PC 2",fontsize=9,labelpad=2)
    for sp in ax.spines.values(): sp.set_edgecolor("0.7")
    ax.text(0.5,-0.10,ann,transform=ax.transAxes,ha="center",va="top",fontsize=9.5,color="#333")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_shapes.pdf",bbox_inches="tight")
plt.savefig(f"{OUT}/fig_shapes.png",dpi=160,bbox_inches="tight")
print("WROTE",OUT+"/fig_shapes.{pdf,png}")
