"""Is the LINE actually a line, or a ring in disguise? Intrinsic-dimensionality check on the
saved day-centroids: eigen-spectrum, lambda2/lambda1 (ring~1, line~0), participation ratio.
Also renders 2D PCA (PC1-PC2) so LINE's shape is visible. CPU only."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
SP=os.path.join(DATA,"shapes")
tag="gemma-4-31B-it"; z=np.load(f"{SP}/shape_plot_{tag}.npz",allow_pickle=True)
days=[str(d) for d in z["days"]]; ipos=z["ipos"].tolist(); depth=z["depth"]; edges=z["tree_edges"].tolist(); N=len(days)
conds=[("NAT_day","NAT @ day",list(range(N)),"ring"),("CYCLE_end","CYCLE @ end",ipos,"ring"),
       ("LINE_end","LINE @ end",ipos,"line"),("TREE_end","TREE @ end",None,"tree")]
def spec(C):
    X=C-C.mean(0); s=np.linalg.svd(X,compute_uv=False); lam=s**2; lam=lam/lam.sum()
    pr=(lam.sum()**2)/(lam**2).sum()  # participation ratio (effective dim)
    return lam,pr
print(f"{'cond':12s} {'PC1%':>5s} {'PC2%':>5s} {'PC3%':>5s}  {'l2/l1':>6s}  {'partR':>5s}")
specs={}
for key,name,_,_ in conds:
    lam,pr=spec(z[key]); specs[key]=lam
    print(f"{name:12s} {lam[0]*100:5.1f} {lam[1]*100:5.1f} {lam[2]*100:5.1f}  {lam[1]/lam[0]:6.2f}  {pr:5.2f}")
# 2D PCA panels
fig,axs=plt.subplots(1,4,figsize=(18,4.6));
def pca2(C): X=C-C.mean(0); U,S,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:2].T
for ax,(key,name,ip,mode) in zip(axs,conds):
    P=pca2(z[key])
    if mode in ("ring","line"):
        s=list(np.argsort(ip)); path=s+[s[0]] if mode=="ring" else s
        ax.plot(P[path,0],P[path,1],"-",color="0.5",lw=1.6,zorder=1); col=np.array(ip); cm="twilight"
    else:
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.5",lw=1.6,zorder=1)
        col=depth; cm="viridis"
    ax.scatter(P[:,0],P[:,1],c=col,cmap=cm,s=110,edgecolor="k",lw=.6,zorder=3)
    for i in range(N): ax.annotate(days[i][:3],(P[i,0],P[i,1]),fontsize=8,ha="center",va="center")
    lam=specs[key]; ax.set_title(f"{name}\nl2/l1={lam[1]/lam[0]:.2f}  PC1={lam[0]*100:.0f}%",fontsize=11)
    ax.set_aspect("equal","datalim"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f"2D PCA (PC1-PC2) of the 7 day-centroids — {tag}, one scramble",fontsize=13)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{SP}/shape_2dpca_{tag}.png",dpi=150,bbox_inches="tight")
print("WROTE shape_2dpca_"+tag+".png")
