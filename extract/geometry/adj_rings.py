import os,glob,numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];N=7
rng=np.random.default_rng(0); orders=[list(rng.permutation(base)) for _ in range(3)]
ORDER=["gemma-4-E2B-it","gemma-4-E4B-it","gemma-4-12B-it","gemma-4-31B-it","Qwen3.5-4B","Qwen3.5-9B","Qwen3.5-27B","Llama-3.1-8B-Instruct"]
def cosM(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)
def cyc(p): P=np.array(p);D=np.abs(P[:,None]-P[None,:]);return np.minimum(D,N-D).astype(float)
def _ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0])
def _in(p1,p2,p3,p4): return _ccw(p1,p3,p4)!=_ccw(p2,p3,p4) and _ccw(p1,p2,p3)!=_ccw(p1,p2,p4)
def crs(pts,seq):
    s=list(seq)+[seq[0]];E=[(s[i],s[i+1]) for i in range(len(s)-1)];c=0
    for a in range(len(E)):
        for b in range(a+1,len(E)):
            if set(E[a])&set(E[b]):continue
            if _in(pts[E[a][0]],pts[E[a][1]],pts[E[b][0]],pts[E[b][1]]):c+=1
    return c
MODE="adj"; avail=[t for t in ORDER if glob.glob(os.path.join(data_dir("geometry","cache"), f"defladder_{t}_days_{MODE}_s0_L*.npz"))]
n=len(avail);fig,ax=plt.subplots(1,n,figsize=(3.3*n,3.7));ax=np.atleast_1d(ax)
for j,t in enumerate(avail):
    f=glob.glob(os.path.join(data_dir("geometry","cache"), f"defladder_{t}_days_{MODE}_s0_L*.npz"))[0]
    d=np.load(f,allow_pickle=True);A=d["A"].astype(np.float32);ent=d["ent"]
    cents=np.stack([A[ent==e].mean(0) for e in base]);ip=[{x.lower():i for i,x in enumerate(orders[0])}[e.lower()] for e in base];iseq=list(np.argsort(ip))
    rsa=spearmanr(cosM(cents)[np.triu_indices(N,1)],cyc(ip)[np.triu_indices(N,1)]).correlation
    mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T;X=crs(P[:,[0,1]],iseq)
    a=ax[j];path=iseq+[iseq[0]];a.plot(P[path,0],P[path,1],"-",color="#888",lw=1.3);a.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=130,edgecolors="k",linewidths=.5)
    for i,e in enumerate(base): a.annotate(e[:3],(P[i,0],P[i,1]),fontsize=8,ha="center",va="center")
    a.set_xticks([]);a.set_yticks([]);a.set_title(f"{t.replace('gemma-4-','G').replace('Qwen3.5-','Q').replace('Llama-3.1-8B-Instruct','L8')}\nimpRSA={rsa:+.2f} X={X}",fontsize=10)
fig.suptitle("ADJACENCY format — k-mixed one-shot imposed ring (si=0), connected in imposed order (does it CLOSE vs the list-lines?)",fontsize=11)
fig.tight_layout();o=os.path.join(data_dir("behavior"), "FIG_adj_rings_days.png");fig.savefig(o,dpi=130);print("SAVED",os.path.abspath(o))
