"""Per-k geometry: MDS of the day centroids computed from EACH single k value (k=1..6), traced in the IMPOSED order.
Shows the imposed ring is present at every individual hop value (k-stable = context-fixed), not built by multi-k sampling.
Gemma-31B, defladder list-mode s0 cache. Out: FIG_kmarg_<tag>.png"""
import os,sys,glob,numpy as np,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
tag=sys.argv[1] if len(sys.argv)>1 else "gemma-4-31B-it"
base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];N=7;NK=6;NT=6
order=list(np.random.default_rng(0).permutation(base))           # defladder s0 imposed order (index=imposed position)
ipos=[ {x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base]
seq=[base.index(order[p]) for p in range(N)]                     # visiting sequence in imposed order
def cyc(p):P=np.array(p);D=np.abs(P[:,None]-P[None,:]);D=np.minimum(D,N-D);return D[np.triu_indices(N,1)].astype(float)
def cos_rdm_full(C):X=C-C.mean(0);X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9);return 1-X@X.T
from scipy.stats import spearmanr
imp_rdm=cyc(ipos)
def mds2(C):
    D=cos_rdm_full(C);n=len(C);J=np.eye(n)-np.ones((n,n))/n;B=-0.5*J@(D**2)@J;w,V=np.linalg.eigh(B);i=np.argsort(w)[::-1]
    return V[:,i[:2]]*np.sqrt(np.clip(w[i[:2]],0,None))
def crossings(P,sq):
    s=list(sq)+[sq[0]];E=[(s[i],s[i+1]) for i in range(len(s)-1)];c=0
    ccw=lambda a,b,d:(d[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(d[0]-a[0])
    for a in range(len(E)):
        for b in range(a+1,len(E)):
            if set(E[a])&set(E[b]):continue
            p1,p2,p3,p4=P[E[a][0]],P[E[a][1]],P[E[b][0]],P[E[b][1]]
            if (ccw(p1,p3,p4)!=ccw(p2,p3,p4)) and (ccw(p1,p2,p3)!=ccw(p1,p2,p4)):c+=1
    return c
f=glob.glob(os.path.join(DATA,"geometry","cache",f"defladder_{tag}_days_list_s0_L*.npz"))[0]
z=np.load(f,allow_pickle=True);A=z["A"].astype(np.float64);ent=z["ent"]
Ar=np.stack([A[np.where(ent==e)[0]].reshape(NK,NT,-1) for e in base])   # (7,NK,NT,d)
fig,axes=plt.subplots(2,3,figsize=(12,7.6));fig.patch.set_facecolor("white")
for ki in range(NK):
    ax=axes[ki//3][ki%3];C=Ar[:,ki,:,:].mean(1);P=mds2(C)
    rsa=spearmanr(cos_rdm_full(C)[np.triu_indices(N,1)],imp_rdm).correlation
    pth=seq+[seq[0]]
    ax.plot(P[pth,0],P[pth,1],"-",color="#b2182b",lw=1.9,zorder=1,alpha=.85)
    ax.scatter(P[:,0],P[:,1],c=[ipos[i] for i in range(N)],cmap="twilight",s=170,zorder=3,edgecolor="k",lw=.5)
    for i in range(N):ax.annotate(base[i][:3],(P[i,0],P[i,1]),fontsize=8,ha="center",va="center")
    ax.set_title(f"$k = {ki+1}$    imposed-order RSA $= {rsa:+.2f}$",fontsize=10.5)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect("equal","datalim")
fig.suptitle(f"{tag}: imposed-order ring recovered from EACH single hop $k$ (traced in imposed order)",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
_OUT = os.environ.get("CIK_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")); os.makedirs(_OUT, exist_ok=True)
for ext in ("pdf","png"): plt.savefig(os.path.join(_OUT, f"fig_kmarg.{ext}"),dpi=200,bbox_inches="tight")
print(f"WROTE {_OUT}/fig_kmarg.{{pdf,png}}")
