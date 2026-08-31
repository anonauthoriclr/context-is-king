#!/usr/bin/env python3
"""T2.3 — FORMAL topology via persistent homology (addresses 'pragmatic topology, not formal'). On the cached imposed
geometry, compute H0/H1 persistence (Vietoris-Rips on the SAME mean-centered cosine-distance RDM that RSA uses). A genuine
imposed CYCLE should carry one dominant, persistent H1 bar (b1=1, a 1-D hole); a TREE should carry none (b1=0); random
points of the same N/dim give only short-lived spurious H1. Reports the top-H1 persistence (death-birth) per structure +
a random baseline, so the loop is certified by a topological invariant independent of RSA. Uses the arbitrary-token ring
and tree (meaning-free tokens) = the formal version of the construction-on-arbitrary-tokens result.
Run with the phvenv (ripser). Out: prints + data_dir("geometry")/ph_topology.json"""
import os, json, glob, numpy as np
from ripser import ripser
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

def cos_rdm(c):
    X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1-X@X.T
def top_h1(D):
    d=ripser(D, maxdim=1, distance_matrix=True)['dgms'][1]
    if len(d)==0: return 0.0, 0, []
    pers=d[:,1]-d[:,0]; order=np.argsort(pers)[::-1]
    bars=[(float(d[i,0]),float(d[i,1]),float(pers[i])) for i in order]
    return float(pers.max()), len(d), bars[:3]
def rng_baseline(n, dim, reps=200, seed=0):
    r=np.random.default_rng(seed); tops=[]
    for _ in range(reps):
        c=r.normal(size=(n,dim)); tops.append(top_h1(cos_rdm(c))[0])
    return float(np.mean(tops)), float(np.percentile(tops,95))

def load_centroids(path):
    z=np.load(path, allow_pickle=True)
    cs=[z[k] for k in z.files if k.startswith('c') and z[k].ndim==2]
    return cs  # list of (N, hidden) per scramble

def main():
    G=data_dir("geometry"); out={}
    targets=[
        ("CYCLE ring (Gemma-31B, arbitrary tokens)", f"{G}/graph_ring_gemma-4-31B-it.npz"),
        ("CYCLE ring (Qwen-27B, arbitrary tokens)",  f"{G}/graph_ring_Qwen3.5-27B.npz"),
        ("CYCLE ring (Gemma-E2B, arbitrary tokens)", f"{G}/graph_ring_gemma-4-E2B-it.npz"),
        ("TREE (Gemma-31B, neutral queries)",        f"{G}/tree_gemma-4-31B-it_neutral.npz"),
        ("TREE (Qwen-27B, neutral queries)",         f"{G}/tree_Qwen3.5-27B_neutral.npz"),
    ]
    for name, path in targets:
        if not os.path.exists(path):
            # tree files may use a different key layout; try generic
            print(f"  [skip missing] {name}: {os.path.basename(path)}"); continue
        cs=load_centroids(path)
        if not cs:
            # tree npz stores per-scramble under different names; try 'c{si}' fallback already covered; else skip
            print(f"  [no 2-D centroid arrays in] {os.path.basename(path)} keys={list(np.load(path,allow_pickle=True).files)[:6]}"); continue
        tops=[]; nbars=[]
        for c in cs:
            t,nb,bars=top_h1(cos_rdm(c)); tops.append(t); nbars.append(nb)
        N=cs[0].shape[0]; dim=cs[0].shape[1]
        bmean,b95=rng_baseline(N,dim)
        out[name]=dict(N=N, n_scr=len(cs), top_h1_mean=float(np.mean(tops)), top_h1_std=float(np.std(tops)),
                       rand_mean=bmean, rand_p95=b95, ratio=float(np.mean(tops)/(b95+1e-9)))
        print(f"{name:44s} N={N} | top-H1 persist {np.mean(tops):.3f}±{np.std(tops):.3f} "
              f"| random b95 {b95:.3f} | ratio {np.mean(tops)/(b95+1e-9):.2f}x", flush=True)
    json.dump(out, open(f"{G}/ph_topology.json","w"), indent=2)
    print("\nWROTE", os.path.abspath(f"{G}/ph_topology.json"))
if __name__=="__main__": main()
