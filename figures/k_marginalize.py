"""k-marginalization (from cache, no GPU): does the imposed geometry depend on WHICH k-hop values we probe, or is it
fixed by the context? Recompute the day centroids from k-subsets ({1},{1,2},{1,2,3},full,{6},{3}) of the cached
defladder list-mode (imposed-order) activations and report imposed-cyclic RSA per subset. If imposed-RSA is ~constant
across subsets (k=1 only ~ full ~ k=6 only), the geometry is context-fixed, not elicited by the traversal question or by
multi-k sampling. Also reports natural-RSA (should stay low) and self-consistency vs the full-k RDM (order-free).
Cache layout: A[252,d] in rows (entity x k=1..6 x template=0..5); ent[252]. Orders are default_rng(0) (defladder seed)."""
import os,glob,re,numpy as np
from scipy.stats import spearmanr
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N=7; NK=6; NT=6
rng=np.random.default_rng(0); ORDERS=[list(rng.permutation(base)) for _ in range(3)]   # defladder shared scramble orders
SUBSETS={"k1":[1],"k12":[1,2],"k123":[1,2,3],"full":[1,2,3,4,5,6],"k6":[6],"k3":[3]}
def cyc(pos): P=np.array(pos);D=np.abs(P[:,None]-P[None,:]);D=np.minimum(D,N-D);return D[np.triu_indices(N,1)].astype(float)
def cos_rdm(C): X=C-C.mean(0);X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9);return (1-X@X.T)[np.triu_indices(N,1)]
nat_rdm=cyc(list(range(N)))
def cents_for(A, ks):  # A reshaped (7, NK, NT, d); ks 1-indexed
    sel=[k-1 for k in ks]; return A[:,sel,:,:].mean(axis=(1,2))   # (7,d) centroid per entity over chosen k & templates
def process(tag):
    rows=[]
    for si in range(3):
        fs=glob.glob(os.path.join(DATA,"geometry","cache",f"defladder_{tag}_days_list_s{si}_L*.npz"))
        if not fs: continue
        z=np.load(fs[0],allow_pickle=True); A=z["A"].astype(np.float64); ent=z["ent"]
        # rows are entity-major (Monday x36, ...) then k then ti -> reshape by the canonical base order
        idxmap={e:np.where(ent==e)[0] for e in base}
        if any(len(idxmap[e])!=NK*NT for e in base): return None
        Ar=np.stack([A[idxmap[e]].reshape(NK,NT,-1) for e in base])  # (7,NK,NT,d) in base order
        o=ORDERS[si]; ipos=[ {x.lower():i for i,x in enumerate(o)}[e.lower()] for e in base]; imp_rdm=cyc(ipos)
        full=cos_rdm(cents_for(Ar,SUBSETS["full"]))
        r={}
        for name,ks in SUBSETS.items():
            obs=cos_rdm(cents_for(Ar,ks))
            r[name]=(float(spearmanr(obs,imp_rdm).correlation), float(spearmanr(obs,nat_rdm).correlation), float(spearmanr(obs,full).correlation))
        rows.append(r)
    if not rows: return None
    agg={name:tuple(float(np.mean([rw[name][i] for rw in rows])) for i in range(3)) for name in SUBSETS}
    return agg
def main():
    tags=sorted({re.search(r"defladder_(.+?)_days_list_s",f).group(1) for f in glob.glob(os.path.join(DATA,"geometry","cache","defladder_*_days_list_s*.npz"))})
    order=["gemma-4-E2B-it","gemma-4-E4B-it","gemma-4-12B-it","gemma-4-31B-it","Qwen3.5-4B","Qwen3.5-9B","Qwen3.5-27B","Llama-3.1-8B-Instruct"]
    tags=[t for t in order if t in tags]+[t for t in tags if t not in order]
    print("IMPOSED-order RSA per k-subset (is the geometry k-stable = context-fixed?)")
    print(f"{'model':24s}"+"".join(f"{s:>7s}" for s in SUBSETS)+"   | nat(full)  selfcons(k1)")
    for t in tags:
        a=process(t)
        if not a: continue
        line=f"{t:24s}"+"".join(f"{a[s][0]:>+7.2f}" for s in SUBSETS)
        print(line+f"   |   {a['full'][1]:+.2f}      {a['k1'][2]:+.2f}")
if __name__=="__main__": main()
