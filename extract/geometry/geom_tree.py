#!/usr/bin/env python3
"""E4 (LEAD / renderer killer): does the renderer draw a NON-CYCLIC relational structure? Impose a declarative BINARY
TREE (hierarchy) on arbitrary tokens (no pretrained order), read the entity-layout geometry (snap regime, matching the
line-vs-cycle 2×2), and test whether the centroid geometry matches **tree (graph) distance** — beating a scrambled-tree
null and a linear template. If yes: topology-follows-specification generalizes from cycles to relational structure
(the renderer draws whatever spec it is given). If null: the renderer is scoped to 1-D/cyclic types (also informative).

7-node binary tree:  n0 -> {n1,n2};  n1 -> {n3,n4};  n2 -> {n5,n6}.
Probe: declare shuffled parent->children edges; query each entity (parent/children/depth) × templates; read last-token
pre-gen residual (thinking-off snap); group by queried entity; centroid RDM vs tree-distance RDM (Spearman).
Controls: scrambled-tree null (permute entity↔node), dRSA(tree − linear-BFS). Saves residuals + stats (Standard #11).
Usage: python -u geom_tree.py <hf_model> [--nscr 6] [--base]
Out: data_dir("geometry")/tree_<tag>.json (+ .npz) + FIG_tree_<tag>.png"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv
SEPEDGES="--sepedges" in sys.argv   # co-occurrence control: one edge per sentence, siblings NOT co-listed (shuffled apart)
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 6
FRACD=0.75
TOKENS=["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle",
        "Garden","Pencil","Marble","Feather","Comet","Harbor","Kettle","Ribbon","Maple","Socket","Pebble","Lantern",
        "Willow","Copper","Meadow","Beacon","Thistle","Quartz","Cedar","Acorn","Domino","Velvet"]  # 35: ≥31 for depth-4
DLEVEL=int(sys.argv[sys.argv.index("--depth")+1]) if "--depth" in sys.argv else 2   # full binary tree of this depth
N=2**(DLEVEL+1)-1
PARENT={i:(i-1)//2 for i in range(1,N)}                       # binary-heap indexing
CHILDREN={i:[c for c in (2*i+1,2*i+2) if c<N] for i in range(N)}
TOK7=TOKENS[:N]   # token list sized to the tree (name kept for body below)
assert len(TOK7)==N, f"need {N} tokens for depth-{DLEVEL}, have {len(TOKENS)}"
QSET=sys.argv[sys.argv.index("--qset")+1] if "--qset" in sys.argv else "rel"
TPLS=["Who is the parent of {e}?","Who are the children of {e}?","How many steps from {e} up to the root?",
      "What is directly above {e} in the hierarchy?","List the children of {e}.","Is {e} the root? If not, name its parent."]
if QSET=="anc":  # CONTROL: ancestor/descendant/membership queries — NO 'parent','child','depth/steps' wording
    TPLS=["List all items above {e}, up to the top item.","List all items below {e}.",
          "Which items share {e}'s immediate group?","Trace the line from {e} up to the top item.",
          "Which items can {e} reach by going downward?","Which items lie in the same branch as {e}?"]
if QSET=="neutral":  # DECISIVE CONTROL: present the entity in the hierarchy context, ask NOTHING structural
    TPLS=["Consider the item {e}.","Take note of the item {e}.","The item under discussion is {e}.",
          "Focus on this item: {e}.","Here is an item from the set: {e}.","Item of interest: {e}."]
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")+("" if QSET=="rel" else f"_{QSET}")+("" if DLEVEL==2 else f"_d{DLEVEL}")+("_sep" if SEPEDGES else ""); os.makedirs(data_dir("geometry"),exist_ok=True)
OUT=os.path.join(data_dir("geometry"), f"tree_{tag}.json")

def tree_dist():  # 7x7 graph path length via depths + LCA
    depth={0:0};
    for n in range(1,N):
        d=0; x=n
        while x!=0: x=PARENT[x]; d+=1
        depth[n]=d
    def anc(n):
        a=[n];
        while n!=0: n=PARENT[n]; a.append(n)
        return a
    D=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            ai,aj=anc(i),anc(j); lca=next(x for x in ai if x in aj)
            D[i,j]=(depth[i]-depth[lca])+(depth[j]-depth[lca])
    return D, depth
TREE_D,DEPTH=tree_dist()
BFS=list(range(N)); LIN=np.array([[abs(BFS.index(i)-BFS.index(j)) for j in range(N)] for i in range(N)],float)

def make_def(perm,rng):  # perm[k] = token assigned to node k. Declare shuffled parent->children edges (all internal nodes).
    internal=[p for p in range(N) if CHILDREN[p]]
    if SEPEDGES:   # one parent->child sentence each; shuffle separates siblings (kills co-occurrence confound)
        edges=[f"- {perm[p]} is the parent of {perm[c]}." for p in internal for c in CHILDREN[p]]
    else:
        edges=[f"- {perm[p]} is the parent of "+" and ".join(perm[c] for c in CHILDREN[p])+"." for p in internal]
    order=list(rng.permutation(len(edges))); shuf="\n".join(edges[j] for j in order)
    pre=("You are operating under a REDEFINED hierarchy. The ONLY valid parent/child relations apply below; the normal "
         "meanings of these words do NOT apply.")
    conv=("\nConvention: the root has no parent; 'steps up to the root' = number of parent-links from the item to the root.")
    return f"{pre}\nHierarchy:\n{shuf}{conv}"
SUFFIX="" if QSET=="neutral" else "\n\nAnswer with ONLY the name(s), nothing else."
def render(tok,c):
    if IS_BASE: return c+("\n" if QSET=="neutral" else "\n\nAnswer: ")
    try: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False)
def cos_rdm(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]

def main():
    rng_o=np.random.default_rng(0); perms=[list(rng_o.permutation(TOK7)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"TREE {MODEL} nscr={NSCR} L={L}/{nl} ntpl={len(TPLS)}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    stats=[]; allc={}
    for si in range(NSCR):
        perm=perms[si]; rng_e=np.random.default_rng(1000+si); defn=make_def(perm,rng_e)
        node_of={perm[k]:k for k in range(N)}                       # which tree node each token occupies
        rows=[(e,ti) for e in TOK7 for ti in range(len(TPLS))]; ent_res={e:[] for e in TOK7}
        for bi in range(0,len(rows),14):
            b=rows[bi:bi+14]; prompts=[render(tok,defn+"\n\n"+TPLS[ti].format(e=e)) for (e,ti) in b]
            enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
            for j,(e,ti) in enumerate(b): ent_res[e].append(hs[j])
        cents=np.stack([np.mean(ent_res[e],0) for e in TOK7])       # 7 entity centroids (token order)
        nodes=[node_of[e] for e in TOK7]                            # node index per centroid
        tree_rdm=np.array([[TREE_D[nodes[i],nodes[j]] for j in range(N)] for i in range(N)])[np.triu_indices(N,1)]
        lin_rdm =np.array([[LIN[nodes[i],nodes[j]]    for j in range(N)] for i in range(N)])[np.triu_indices(N,1)]
        obs=cos_rdm(cents)
        rsa_tree=float(spearmanr(obs,tree_rdm).correlation); rsa_lin=float(spearmanr(obs,lin_rdm).correlation)
        # scrambled-tree null: permute entity↔node assignment, recompute tree RSA
        rng_n=np.random.default_rng(7000+si); nullv=[]
        for _ in range(500):
            pn=list(rng_n.permutation(range(N))); trd=np.array([[TREE_D[pn[i],pn[j]] for j in range(N)] for i in range(N)])[np.triu_indices(N,1)]
            nullv.append(spearmanr(obs,trd).correlation)
        nullv=np.array(nullv); z=(rsa_tree-nullv.mean())/(nullv.std()+1e-9)
        stats.append(dict(si=si,rsa_tree=rsa_tree,rsa_linear=rsa_lin,dRSA_tree_minus_lin=rsa_tree-rsa_lin,
                          null_mean=float(nullv.mean()),null_p95=float(np.percentile(nullv,95)),z=float(z)))
        allc[si]=(cents,nodes)
        print(f"  scr{si+1}: RSA_tree={rsa_tree:+.2f} RSA_lin={rsa_lin:+.2f} dRSA(tree-lin)={rsa_tree-rsa_lin:+.2f} | null {nullv.mean():+.2f}/p95 {np.percentile(nullv,95):+.2f} z={z:+.1f}",flush=True)
    del model; torch.cuda.empty_cache()
    agg=lambda k: float(np.mean([s[k] for s in stats]))
    summ=dict(model=MODEL,base=IS_BASE,nscr=NSCR,L=int(L),tree="binary7",per=stats,
              rsa_tree=agg("rsa_tree"),rsa_tree_std=float(np.std([s["rsa_tree"] for s in stats])),
              rsa_linear=agg("rsa_linear"),dRSA=agg("dRSA_tree_minus_lin"),z=agg("z"),
              nsig=int(sum(s["rsa_tree"]>s["null_p95"] for s in stats)))
    json.dump(summ,open(OUT,"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"tree_{tag}.npz"),**{f"c{si}":allc[si][0] for si in range(NSCR)},
             nodes=np.array([allc[si][1] for si in range(NSCR)]),tok=np.array(TOK7),L=L)
    print(f"== {tag}: RSA_tree {summ['rsa_tree']:+.2f}±{summ['rsa_tree_std']:.2f} (lin {summ['rsa_linear']:+.2f}, dRSA {summ['dRSA']:+.2f}) z={summ['z']:+.1f} sig {summ['nsig']}/{NSCR}",flush=True)
    # figure: PCA of scr0 centroids colored by tree depth, edges drawn; + RSA bars
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6))
    cents,nodes=allc[0]; mu=cents.mean(0); _,_,Vt=np.linalg.svd(cents-mu,full_matrices=False); P=(cents-mu)@Vt.T
    depths=[DEPTH[n] for n in nodes]
    sc=ax1.scatter(P[:,0],P[:,1],c=depths,cmap="viridis",s=140,zorder=3)
    for i in range(N): ax1.annotate(f"{TOK7[i]}\n(n{nodes[i]})",(P[i,0],P[i,1]),fontsize=7,ha="center")
    for i in range(N):  # draw tree edges between centroids
        if nodes[i] in PARENT:
            par=PARENT[nodes[i]]; pj=nodes.index(par); ax1.plot([P[i,0],P[pj,0]],[P[i,1],P[pj,1]],"k-",alpha=.3,zorder=1)
    ax1.set_title(f"{tag} imposed-tree centroids (scr1, PC1-2)\ncolor=depth, lines=parent edges"); ax1.set_xticks([]);ax1.set_yticks([])
    fig.colorbar(sc,ax=ax1,label="tree depth")
    rt=[s["rsa_tree"] for s in stats]; rl=[s["rsa_linear"] for s in stats]; nm=[s["null_mean"] for s in stats]; x=np.arange(NSCR)
    ax2.bar(x-0.25,rt,0.25,label="RSA vs TREE",color="C0"); ax2.bar(x,rl,0.25,label="RSA vs linear",color="C1",alpha=.7)
    ax2.bar(x+0.25,nm,0.25,label="scrambled-tree null",color="gray",alpha=.6)
    ax2.set_xticks(x); ax2.set_xticklabels([f"s{i+1}" for i in range(NSCR)]); ax2.set_ylim(-.3,1); ax2.legend(fontsize=8)
    ax2.set_title("Geometry matches imposed TREE distance?"); ax2.axhline(0,color="k",lw=.5)
    plt.tight_layout(); fn=os.path.join(data_dir("geometry"), f"FIG_tree_{tag}.png"); plt.savefig(fn,dpi=130); print("FIG",os.path.abspath(fn),flush=True)
if __name__=="__main__": main()
