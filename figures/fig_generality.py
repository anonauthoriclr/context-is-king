import os,glob,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
CC={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "months":["January","February","March","April","May","June","July","August","September","October","November","December"]}
ORDER=[("gemma-4-E2B-it","Gemma",2),("gemma-4-E4B-it","Gemma",4),("gemma-4-12B-it","Gemma",12),("gemma-4-31B-it","Gemma",31),
       ("Qwen3.5-4B","Qwen",4),("Qwen3.5-9B","Qwen",9),("Qwen3.5-27B","Qwen",27),("Llama-3.1-8B-Instruct","Llama",8)]
def cosM(c,N): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)
def closure(tag,mode,concept):
    base=CC[concept];N=len(base);rng=np.random.default_rng(0);orders=[list(rng.permutation(base)) for _ in range(3)];rr=[]
    for si in range(3):
        fs=glob.glob(os.path.join(DATA,"geometry","cache",f"defladder_{tag}_{concept}_{mode}_s{si}_L*.npz"))
        if not fs: continue
        d=np.load(fs[0],allow_pickle=True);A=d["A"].astype(np.float32);ent=d["ent"]
        if not all((ent==e).any() for e in base): continue
        cents=np.stack([A[ent==e].mean(0) for e in base]);D=cosM(cents,N)
        p2e={ {x.lower():i for i,x in enumerate(orders[si])}[e.lower()]:bi for bi,e in enumerate(base)}
        w=D[p2e[0],p2e[N-1]];inter=np.mean([D[p2e[i],p2e[i+1]] for i in range(N-1)]);rr.append(w/(inter+1e-9))
    return (np.mean(rr) if rr else np.nan)
fig,axes=plt.subplots(1,2,figsize=(15,5.2))
for ax,concept in zip(axes,["days","months"]):
    rows=[(t,f,s) for t,f,s in ORDER]; xs=range(len(rows)); labs=[t.replace("gemma-4-","G").replace("Qwen3.5-","Q").replace("Llama-3.1-8B-Instruct","L8") for t,_,_ in rows]
    cl_list=[closure(t,"list",concept) for t,_,_ in rows]; cl_adj=[closure(t,"adj",concept) for t,_,_ in rows]
    ax.axhspan(0,1.3,color="#e8f5e9",alpha=.6,zorder=0); ax.axhspan(1.3,3,color="#fdeaea",alpha=.6,zorder=0)
    ax.plot(xs,cl_list,"s-",color="#900",label="list (numbered, no wrap)",ms=8)
    ax.plot(xs,cl_adj,"o-",color="#070",label="adj (edges + wrap)",ms=8)
    ax.axhline(1.0,color="#888",ls=":",lw=1); ax.text(len(rows)-.5,1.32,"line ↑ / cycle ↓",fontsize=8,ha="right",va="bottom",color="#555")
    ax.set_xticks(list(xs));ax.set_xticklabels(labs,rotation=45,ha="right");ax.set_ylabel("wrap-pair closure ratio (≈1=cycle, ≫1=line)")
    ax.set_title(f"{concept}: list→LINE (high) vs adj→CYCLE (≈1), across scale & family"); ax.legend(fontsize=9); ax.set_ylim(0.6,None); ax.grid(alpha=.2)
fig.suptitle("Topology generality: the imposed structure is a LINE under a numbered list, a CYCLE under wrap-edges — across scale, family, and concept",fontsize=12)
fig.tight_layout(); o="results/behavior/FIG2_generality.png"; fig.savefig(o,dpi=135); print("SAVED",o)
