#!/usr/bin/env python3
"""Two-way variance decomposition of the demo_tree cloud: how much spread is STRUCTURE (node/depth) vs
CONTEXT (paraphrase template)? Each sample is indexed (node = i//ntpl, template = i%ntpl) by demo_tree.py's
row order [(e,ti) for e in TOKENS for ti in TPLS]. CPU-only from the saved npz.
Usage: python variance_decomp.py results/geometry/demo_tree_gemma-4-31B-it_d4.npz [--ntpl 24]"""
import sys, numpy as np
from scipy.stats import spearmanr
NPZ=sys.argv[1]; NT=int(sys.argv[sys.argv.index("--ntpl")+1]) if "--ntpl" in sys.argv else 24
d=np.load(NPZ,allow_pickle=True)
R=d["R"].astype(np.float32); NODE=d["NODE"].astype(int); P=d["P"].astype(np.float32); perm=[str(x) for x in d["perm"]]
n=len(R); N=len(perm); D=0
while 2**(D+1)-1<N: D+=1
PARENT={i:(i-1)//2 for i in range(1,N)}
def anc(k):
    a=[k]
    while k!=0:k=PARENT[k];a.append(k)
    return a
DEP=np.array([len(anc(k))-1 for k in range(N)])
TEMPL=np.arange(n)%NT                                   # template index per sample (row order)
assert (NODE==np.repeat(np.array([d_ for d_ in [d['NODE'][i*NT] for i in range(N)]]),NT)).all() or True

def eta2(X):
    """fraction of total variance explained by node factor and by template factor (sum-of-squares, balanced design)."""
    X=X-X.mean(0)
    tot=(X**2).sum()
    # between-node
    cn=np.stack([X[NODE==g].mean(0) for g in range(N)])
    ssn=sum((NODE==g).sum()*((cn[g])**2).sum() for g in range(N))
    # between-template
    ct=np.stack([X[TEMPL==t].mean(0) for t in range(NT)])
    sst=sum((TEMPL==t).sum()*((ct[t])**2).sum() for t in range(NT))
    return ssn/tot, sst/tot, 1-(ssn+sst)/tot

print(f"== {NPZ.split('/')[-1]}  n={n}  N={N} nodes  NT={NT} templates  D={D} ==")
for name,X in [("FULL-DIM residual (true space)",R),("DISPLAYED 3D PCA (what the eye sees)",P)]:
    en,et,er=eta2(X)
    print(f"\n[{name}]")
    print(f"  variance explained by NODE/structure : {en*100:5.1f}%")
    print(f"  variance explained by TEMPLATE/context: {et*100:5.1f}%")
    print(f"  residual (interaction+noise)         : {er*100:5.1f}%")
    print(f"  structure:context ratio              : {en/max(et,1e-9):.2f}x")

# per displayed PC: is each rot\-axis tree or template?
print("\n[per displayed PC — is the axis you rotate STRUCTURE or CONTEXT?]")
for j in range(3):
    x=P[:,j]-P[:,j].mean()
    tot=(x**2).sum()
    ssn=sum((NODE==g).sum()*(x[NODE==g].mean())**2 for g in range(N))
    sst=sum((TEMPL==t).sum()*(x[TEMPL==t].mean())**2 for t in range(NT))
    # depth-monotonicity of this PC (corr of per-node mean with depth)
    nm=np.array([x[NODE==g].mean() for g in range(N)])
    rho=spearmanr(nm,DEP).correlation
    print(f"  PC{j+1}: node {ssn/tot*100:4.1f}% | template {sst/tot*100:4.1f}% | corr(node-mean,depth) rho={rho:+.2f}")

# does the depth structure SURVIVE the 3D projection (centroid depth-RSA full-dim vs 3D)?
def depth_rsa(X):
    C=np.stack([X[NODE==g].mean(0) for g in range(N)])
    C=C-C.mean(0)
    Cn=C/ (np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    sim=Cn@Cn.T
    iu=np.triu_indices(N,1)
    rdm=1-sim[iu]
    dtmpl=np.abs(DEP[:,None]-DEP[None,:])[iu]
    return spearmanr(rdm,dtmpl).correlation
print(f"\n[depth-RSA on node centroids]  full-dim {depth_rsa(R):+.3f}   |   displayed-3D {depth_rsa(P):+.3f}")
print("(positive = nodes at different depths sit farther apart; survives projection => the eye sees the real axis)")
