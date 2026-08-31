import sys,glob,numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "months":["January","February","March","April","May","June","July","August","September","October","November","December"],
     "zodiac":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],
     "clock":["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"],
     "notes":["Do","Re","Mi","Fa","Sol","La","Ti"]}
TAG=sys.argv[1] if len(sys.argv)>1 else "gemma-4-31B-it"; CC=["days","months","zodiac","clock","notes"]
def cosM(c,N): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)
def cyc(p,N): P=np.array(p);D=np.abs(P[:,None]-P[None,:]);return np.minimum(D,N-D).astype(float)
def lin(p,N): P=np.array(p);return np.abs(P[:,None]-P[None,:]).astype(float)
ut=lambda M,N:M[np.triu_indices(N,1)]
fig,ax=plt.subplots(len(CC),2,figsize=(7.2,3.0*len(CC))); ax=np.atleast_2d(ax)
for r,C in enumerate(CC):
    base=ENT[C];N=len(base);order=list(np.random.default_rng(0).permutation(base))
    ip=[{x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base];iseq=list(np.argsort(ip))
    for cc,mode in enumerate(["list","adj"]):
        a=ax[r,cc]; fs=glob.glob(os.path.join(data_dir("geometry","cache"), f"defladder_{TAG}_{C}_{mode}_s0_L*.npz"))
        if not fs: a.set_axis_off(); a.set_title(f"{C}/{mode}: missing",fontsize=8); continue
        d=np.load(fs[0],allow_pickle=True);A=d["A"].astype(np.float32);ent=d["ent"]
        cents=np.stack([A[ent==e].mean(0) for e in base]);D=cosM(cents,N)
        drv=spearmanr(ut(D,N),ut(cyc(ip,N),N)).correlation-spearmanr(ut(D,N),ut(lin(ip,N),N)).correlation
        p2e={ip[i]:i for i in range(N)};clos=D[p2e[0],p2e[N-1]]/(np.mean([D[p2e[i],p2e[i+1]] for i in range(N-1)])+1e-9)
        mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T
        for i in range(N-1): u,v=iseq[i],iseq[i+1]; a.plot([P[u,0],P[v,0]],[P[u,1],P[v,1]],"-",color="#bbb",lw=1.1)
        u,v=iseq[-1],iseq[0]; a.plot([P[u,0],P[v,0]],[P[u,1],P[v,1]],"-",color="#e00",lw=2.2)
        a.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=70,edgecolors="k",linewidths=.4)
        a.set_xticks([]);a.set_yticks([])
        ver="CYCLE" if (drv>0 and clos<1.3) else "LINE"
        a.set_title(f"{C} · {'numbered list' if mode=='list' else 'adjacency+wrap'}\ndRSA={drv:+.2f} clos={clos:.2f} → {ver}",fontsize=8.5,color="#070" if ver=="CYCLE" else "#900")
fig.suptitle(f"{TAG}: centroid geometry per concept — LIST (left) vs ADJ+wrap (right). red=wrap edge.",fontsize=11)
fig.tight_layout();o=os.path.join(data_dir("behavior"), f"FIG_xconcept_rings_{TAG}.png");fig.savefig(o,dpi=130);print("SAVED",o)
