"""3D-PCA panels for eyeballing the shapes-not-orders result. Reads shape_plot_<tag>.npz.
NAT@day-token (pretrained ring), CYCLE@end (imposed ring), LINE@end (open path), TREE@end (depth funnel).
Usage: python shape_plot3d.py <tag>"""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag=sys.argv[1] if len(sys.argv)>1 else "gemma-4-31B-it"
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
SP=os.path.join(DATA,"shapes")
z=np.load(f"{SP}/shape_plot_{tag}.npz", allow_pickle=True)
days=[str(d) for d in z["days"]]; ipos=z["ipos"]; depth=z["depth"]; edges=z["tree_edges"]; N=len(days)
def pca3(C): mu=C.mean(0); U,S,Vt=np.linalg.svd(C-mu,full_matrices=False); return (C-mu)@Vt[:3].T
def seq_from_ipos(ip): return list(np.argsort(ip))  # day-indices ordered by (imposed) position
fig=plt.figure(figsize=(13,11))
def panel(idx,C,title,mode,ip=None,close=False):
    P=pca3(C); ax=fig.add_subplot(2,2,idx,projection="3d")
    if mode=="ring" or mode=="line":
        s=seq_from_ipos(ip); path=s+[s[0]] if close else s
        ax.plot(P[path,0],P[path,1],P[path,2],"-",color="0.4",lw=1.8,zorder=1)
        col=np.array(ip); cmap="twilight"
    else:  # tree
        for a,b in edges: ax.plot(*zip(P[a],P[b]),color="0.4",lw=1.8,zorder=1)
        col=depth; cmap="viridis"
    ax.scatter(P[:,0],P[:,1],P[:,2],c=col,cmap=cmap,s=90,edgecolor="k",lw=.6,depthshade=False,zorder=3)
    for i in range(N): ax.text(P[i,0],P[i,1],P[i,2],"  "+days[i][:3],fontsize=8)
    ax.set_title(title,fontsize=12); ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.set_zlabel("PC3")
    ax.view_init(elev=18,azim=60)
panel(1,z["NAT_day"],  "NAT (no rule) @ day-token\nexpect: pretrained weekday ring","ring",ip=z["nat_natorder"],close=True)
panel(2,z["CYCLE_end"],"CYCLE imposed @ sentence-end\nexpect: ring in the imposed order","ring",ip=ipos,close=True)
panel(3,z["LINE_end"], "LINE imposed @ sentence-end\nexpect: open path (less circular)","line",ip=ipos,close=False)
panel(4,z["TREE_end"], "TREE imposed @ sentence-end\nexpect: depth-stratified funnel","tree")
plt.suptitle(f"Same 7 weekdays, four specifications  ({tag}, per-condition 3D PCA, one scramble)",fontsize=13)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(f"{SP}/shape_plot3d_{tag}.png",dpi=145,bbox_inches="tight")
plt.savefig(f"{SP}/shape_plot3d_{tag}.pdf",bbox_inches="tight")
print(f"WROTE shape_plot3d_{tag}.png/pdf")
