"""Same seven weekdays under no rule, an imposed cycle, and an imposed tree.
All panels use the final pre-generation readout and orthographic views of the
first three principal components. Gemma-4-31B."""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import spearmanr
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
RES=os.path.join(DATA,"shapes")
OUT=os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(OUT, exist_ok=True)
tag="gemma-4-31B-it"
SHAPE_PATH=os.environ.get("CIK_SHAPE_CENTROIDS",f"{RES}/shape_v4cent_{tag}.npz")
z=np.load(SHAPE_PATH,allow_pickle=True)
NAT_PATH=os.environ.get("CIK_NAT_SENTEND",os.path.join(DATA,"geometry",f"natsentend_{tag}_query.npz"))
nat_end=np.load(NAT_PATH)["cent_sentend"].astype(np.float64)
days=[str(d) for d in z["days"]]; N=len(days)
J=json.load(open(f"{RES}/shape_v4_{tag}.json"))
def stat(cond,pos,key):
    v=[r[key] for r in J if r["cond"]==cond and r["pos"]==pos and r[key] is not None]
    return (np.mean(v),np.std(v)) if v else (float("nan"),0)
def pr(cond,pos): m,s=stat(cond,pos,"partR"); return f"eff-dim {m:.1f}"
def natural_stats(C):
    X=C-C.mean(0); sv=np.linalg.svd(X,compute_uv=False)**2
    ed=(sv.sum()**2)/(np.sum(sv**2)+1e-12)
    X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); D=1-X@X.T
    P=np.arange(N); T=np.abs(P[:,None]-P[None,:]); T=np.minimum(T,N-T)
    ii=np.triu_indices(N,1); rsa=spearmanr(D[ii],T[ii]).correlation
    return f"natural RSA {rsa:+.2f} · eff-dim {ed:.1f}"
cyc_ip=z["CYCLE_ipos"].tolist(); depth=z["tree_depth"]; edges=z["tree_edges"].tolist()
# 3 panels: drop the LINE case (it lives in the full paper's appendix)
panels=[(nat_end,"No rule\nnatural organization","ring",list(range(N)),natural_stats(nat_end)),
        (z["CYCLE_end"],"Imposed cycle\nring","ring",cyc_ip,pr("CYCLE","end")),
        (z["TREE_end"],"Imposed tree\ndepth-stratified","tree",None,pr("TREE","end"))]
def pca3(C):
    X=C-C.mean(0); U,S,Vt=np.linalg.svd(X,full_matrices=False)
    P=X@Vt[:3].T
    explained=float(np.sum(S[:3]**2)/np.sum(S**2))
    return P,explained
def selected_view(P):
    # Operator-selected Plotly orthographic camera for the no-rule panel.
    normal=np.array([-1.5031443083843847,1.7796085167477549,0.31788475158535817])
    normal=normal/np.linalg.norm(normal); up0=np.array([0.,0.,1.])
    right=np.cross(up0,normal); right=right/np.linalg.norm(right)
    up=np.cross(normal,right)
    Q=np.column_stack([P@right,P@up])
    return Q-Q.mean(0)
def order_of(ip): return list(np.argsort(ip))
fig,axs=plt.subplots(1,3,figsize=(11.5,4.1))
for panel_idx,(ax,(C,title,mode,ip,ann)) in enumerate(zip(axs,panels)):
    P3,explained=pca3(C)
    if panel_idx==0:
        P=selected_view(P3)
        footer=f"{ann} · PC1–3 {100*explained:.0f}% var"
    else:
        # Looking along PC3 recovers the clean PC1–PC2 orthographic view.
        P=P3[:,:2]
        footer=ann
    if mode=="ring":
        s=order_of(ip); path=s+[s[0]]
        ax.plot(P[path,0],P[path,1],"-",color="0.55",lw=1.9,zorder=1); col=np.array(ip); cm="twilight"
    else:
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.55",lw=1.9,zorder=1)
        col=depth; cm="viridis"
    # bigger nodes so labels are legible at column/figure width
    ax.scatter(P[:,0],P[:,1],c=col,cmap=cm,s=460,edgecolor="k",lw=.9,zorder=3)
    for i in range(N):
        ax.text(P[i,0],P[i,1],days[i][:3],fontsize=11,fontweight="bold",
                ha="center",va="center",zorder=4,color="white",
                path_effects=[pe.withStroke(linewidth=2.4,foreground="black")])
    ax.set_title(title,fontsize=14,pad=6)
    ax.set_aspect("equal","datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("0.7")
    ax.text(0.5,-0.10,footer,transform=ax.transAxes,ha="center",va="top",fontsize=10,color="#333")
plt.tight_layout(w_pad=1.0)
plt.savefig(f"{OUT}/fig_shapes.pdf",bbox_inches="tight")
plt.savefig(f"{OUT}/fig_shapes.png",dpi=220,bbox_inches="tight")
print("WROTE",OUT+"/fig_shapes.{pdf,png}")
