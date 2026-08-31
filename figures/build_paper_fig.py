"""Paper figure candidates for 'shapes, not orders': same 7 weekdays -> ring / line / tree.
Renders BOTH a 2D-PCA and a 3D-PCA version, each panel annotated with effective-dim (participation
ratio, mean+/-std over 10 scrambles) so line-vs-ring is quantitative on the page. Gemma-4-31B."""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
SP=os.path.join(DATA,"shapes")
_OUT = os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(_OUT, exist_ok=True)
tag="gemma-4-31B-it"; z=np.load(f"{SP}/shape_v4cent_{tag}.npz",allow_pickle=True)
days=[str(d) for d in z["days"]]; N=len(days)
J=json.load(open(f"{SP}/shape_v4_{tag}.json"))
def stat(cond,pos,key):
    v=[r[key] for r in J if r["cond"]==cond and r["pos"]==pos and r[key] is not None]
    return (np.mean(v),np.std(v)) if v else (float("nan"),0)
# panels: (npz-key, title, mode, order-source, annotation)
def pr(cond,pos): m,s=stat(cond,pos,"partR"); return f"eff-dim {m:.1f}±{s:.1f}"
def rsa(cond,pos,k): m,s=stat(cond,pos,k); return m
cyc_ip=z["CYCLE_ipos"].tolist(); lin_ip=z["LINE_ipos"].tolist(); depth=z["tree_depth"]; edges=z["tree_edges"].tolist()
panels=[("NAT_day","Natural (no rule)\npretrained ring","ring",list(range(N)),pr("NAT-neutral","day")),
        ("CYCLE_end","Imposed cycle\nring","ring",cyc_ip,pr("CYCLE","end")),
        ("LINE_end","Imposed line\nopen 1-D path","line",lin_ip,pr("LINE","end")),
        ("TREE_end","Imposed tree\ndepth funnel","tree",None,pr("TREE","end"))]
def pca(C,d): X=C-C.mean(0); U,S,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:d].T
def order_of(ip): return list(np.argsort(ip))
# ---- 2D ----
fig,axs=plt.subplots(1,4,figsize=(15,4));
for ax,(key,title,mode,ip,ann) in zip(axs,panels):
    P=pca(z[key],2)
    if mode in("ring","line"):
        s=order_of(ip); path=s+[s[0]] if mode=="ring" else s
        ax.plot(P[path,0],P[path,1],"-",color="0.55",lw=1.7,zorder=1); col=np.array(ip); cm="twilight"
    else:
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.55",lw=1.7,zorder=1)
        col=depth; cm="viridis"
    ax.scatter(P[:,0],P[:,1],c=col,cmap=cm,s=120,edgecolor="k",lw=.6,zorder=3)
    for i in range(N): ax.annotate(days[i][:3],(P[i,0],P[i,1]),fontsize=8,ha="center",va="center")
    ax.set_title(title,fontsize=11); ax.set_aspect("equal","datalim"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5,-0.06,ann,transform=ax.transAxes,ha="center",va="top",fontsize=9,color="#333")
plt.tight_layout(); plt.savefig(os.path.join(_OUT,"fig_shapes_2d.png"),dpi=160,bbox_inches="tight"); plt.savefig(os.path.join(_OUT,"fig_shapes_2d.pdf"),bbox_inches="tight"); plt.close()
# ---- 3D ----
fig=plt.figure(figsize=(15,4.4))
for k,(key,title,mode,ip,ann) in enumerate(panels,1):
    P=pca(z[key],3); ax=fig.add_subplot(1,4,k,projection="3d")
    if mode in("ring","line"):
        s=order_of(ip); path=s+[s[0]] if mode=="ring" else s
        ax.plot(P[path,0],P[path,1],P[path,2],"-",color="0.55",lw=1.7); col=np.array(ip); cm="twilight"
    else:
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.55",lw=1.7)
        col=depth; cm="viridis"
    ax.scatter(P[:,0],P[:,1],P[:,2],c=col,cmap=cm,s=70,edgecolor="k",lw=.5,depthshade=False)
    for i in range(N): ax.text(P[i,0],P[i,1],P[i,2],"  "+days[i][:3],fontsize=7)
    ax.set_title(title+"\n"+ann,fontsize=10); ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]); ax.view_init(20,58)
plt.tight_layout(); plt.savefig(os.path.join(_OUT,"fig_shapes_3d.png"),dpi=160,bbox_inches="tight"); plt.savefig(os.path.join(_OUT,"fig_shapes_3d.pdf"),bbox_inches="tight")
print("WROTE fig_shapes_2d.{png,pdf} and fig_shapes_3d.{png,pdf}")
