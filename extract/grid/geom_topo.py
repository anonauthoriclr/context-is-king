#!/usr/bin/env python3
"""E7 (construct side): does the geometry render a COMPLEX NOVEL topology — one proven outside {line, cycle, tree, grid}
by its invariants — and does it follow the SPECIFIC graph (not a fixed shape)? Impose an undirected graph on arbitrary
tokens via atomic SHUFFLED edges ("X is directly linked to Y."), read entity geometry (snap regime), and report full-dim
RSA of the centroid RDM against the graph's BFS geodesic vs {line, ring, grid} templates (specificity), plus graded(non-
adjacent) RSA (global geodesic vs local edge co-embedding), a cone-preserving label-shuffle null with EMPIRICAL tail p,
and the topological invariants (b1, degree sequence, branch-node count). Saves centroids + edges + geodesic for the 3-D
MDS acceptance view. Graphs: hypercube Q3 (b1=5, genuinely 3-D), petersen (b1=6, non-planar), random (seeded, b1>=2, no
name), and random-diff (B = A with --diff edges swapped — the differential-edge control for true construction).
Usage: python -u geom_topo.py <hf_model> --graph {hypercube|petersen|random} [--n 9 --b1 3 --seed 0 --diff 0]
                                        [--nscr 6] [--qset rel|neutral]
Out: data_dir("geometry")/topo_<graph><tags>_<model>.json (+ .npz)"""
import os, sys, json, numpy as np, torch
from collections import deque
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
def arg(flag,default,cast=str):
    return cast(sys.argv[sys.argv.index(flag)+1]) if flag in sys.argv else default
GRAPH=arg("--graph","random"); NN=arg("--n",9,int); B1=arg("--b1",3,int); SEED=arg("--seed",0,int)
DIFF=arg("--diff",0,int); NSCR=arg("--nscr",6,int); QSET=arg("--qset","rel")
FRACD=0.75
TOKENS=["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle",
        "Garden","Pencil","Marble"]

# ---------- graph generators (edges on node indices 0..N-1) ----------
def hypercube():  # Q3: 3-bit strings, edge iff Hamming distance 1
    N=8; E={(i,j) for i in range(N) for j in range(i+1,N) if bin(i^j).count("1")==1}; return E,N
def petersen():
    E=set()
    for i in range(5):
        E.add((min(i,(i+1)%5),max(i,(i+1)%5)))                 # outer pentagon
        E.add((i,i+5))                                          # spoke
        a,b=i+5,(i+2)%5+5; E.add((min(a,b),max(a,b)))          # inner pentagram (i -> i+2)
    return E,10
def _connected(E,N):
    if N<=1: return True
    adj={i:set() for i in range(N)}
    for a,b in E: adj[a].add(b); adj[b].add(a)
    seen={0}; dq=deque([0])
    while dq:
        x=dq.popleft()
        for y in adj[x]:
            if y not in seen: seen.add(y); dq.append(y)
    return len(seen)==N
def random_graph(N,b1,seed):  # random spanning tree + b1 extra edges => connected, b1 independent loops
    rng=np.random.default_rng(seed); perm=list(rng.permutation(N)); E=set()
    for i in range(1,N):
        j=perm[int(rng.integers(0,i))]; a,b=perm[i],j; E.add((min(a,b),max(a,b)))
    allp=[(i,j) for i in range(N) for j in range(i+1,N)]; cand=[e for e in allp if e not in E]
    rng.shuffle(cand)
    for e in cand[:b1]: E.add(e)
    return E,N
def apply_diff(E,N,diff,seed):  # B = A with `diff` edges swapped, keeping connectivity; returns (B, actual_symdiff)
    rng=np.random.default_rng(seed+991); E=set(E)
    for _ in range(diff):
        rem=list(E); rng.shuffle(rem)
        for e in rem:
            if _connected(E-{e},N): E=E-{e}; break
        allp=[(i,j) for i in range(N) for j in range(i+1,N)]; absent=[p for p in allp if p not in E]; rng.shuffle(absent)
        E.add(absent[0])
    return E

def build():
    if GRAPH=="hypercube": E,N=hypercube(); name="hypercube"
    elif GRAPH=="petersen": E,N=petersen(); name="petersen"
    else:
        E,N=random_graph(NN,B1,SEED); name=f"random-n{NN}-b{B1}-s{SEED}"
        if DIFF>0: E=apply_diff(E,N,DIFF,SEED); name+=f"-d{DIFF}"
    return E,N,name
def geodesic(E,N):
    adj={i:set() for i in range(N)}
    for a,b in E: adj[a].add(b); adj[b].add(a)
    D=np.full((N,N),0.0)
    for s in range(N):
        seen={s:0}; dq=deque([s])
        while dq:
            x=dq.popleft()
            for y in adj[x]:
                if y not in seen: seen[y]=seen[x]+1; dq.append(y)
        for j in range(N): D[s,j]=seen.get(j,N)  # disconnected -> N (shouldn't happen, all connected)
    return D
def invariants(E,N):
    deg=[0]*N
    for a,b in E: deg[a]+=1; deg[b]+=1
    b1=len(E)-N+1  # connected => C=1
    branch=sum(1 for d in deg if d>=3)
    novel = (b1>=2) or (b1==1 and branch>0)   # outside line(b1=0,deg<=2), cycle(b1=1,all deg2), tree(b1=0)
    return dict(b1=int(b1),E=len(E),degseq=sorted(deg,reverse=True),n_branch=branch,is_novel=bool(novel))

E,N,NAME=build(); GEO=geodesic(E,N); INV=invariants(E,N)
TOKK=TOKENS[:N]; assert len(TOKK)==N, f"need {N} tokens"
LINE=np.array([[abs(i-j) for j in range(N)] for i in range(N)],float)
RING=np.array([[min(abs(i-j),N-abs(i-j)) for j in range(N)] for i in range(N)],float)
GRID=None
if N==9:
    co={i:(i//3,i%3) for i in range(9)}; GRID=np.array([[abs(co[i][0]-co[j][0])+abs(co[i][1]-co[j][1]) for j in range(9)] for i in range(9)],float)
TEMPLATES={"self":GEO,"line":LINE,"ring":RING}|({"grid":GRID} if GRID is not None else {})
QW={"rel":["List all items directly linked to {e}.","Which items are directly connected to {e}?",
           "Name every item linked to {e}.","What does {e} directly connect to?","List {e}'s direct links.",
           "Which items share a direct link with {e}?"],
    "neutral":["Consider the item {e}.","Take note of the item {e}.","The item under discussion is {e}.",
               "Focus on this item: {e}.","Here is an item from the set: {e}.","Item of interest: {e}."]}
TPLS=QW[QSET]
PRE=("You are operating under a REDEFINED link structure. The ONLY valid 'directly linked' relations apply below.")
SUFFIX="" if QSET=="neutral" else "\n\nAnswer with ONLY the name(s), nothing else."
tag=NAME+("" if QSET=="rel" else f"_{QSET}"); mtag=MODEL.split("/")[-1]
os.makedirs(data_dir("geometry"),exist_ok=True); OUT=os.path.join(data_dir("geometry"), f"topo_{tag}_{mtag}.json")

def make_def(perm,rng):  # perm[k]=token at node k; atomic undirected edges, shuffled
    es=[f"- {perm[a]} is directly linked to {perm[b]}." for (a,b) in E]
    order=list(rng.permutation(len(es))); return f"{PRE}\nRelations:\n"+"\n".join(es[j] for j in order)
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False)
def cos_rdm_full(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1-X@X.T
TU=np.triu_indices(N,1)
def up(M): return M[TU]

def main():
    print(f"TOPO graph={NAME} N={N} b1={INV['b1']} novel={INV['is_novel']} degseq={INV['degseq']} | {MODEL}",flush=True)
    rng_o=np.random.default_rng(0); perms=[list(rng_o.permutation(TOKK)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"  L={L}/{nl} nscr={NSCR} qset={QSET}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    stats=[]; allc={}; spec0=None
    for si in range(NSCR):
        perm=perms[si]; rng_e=np.random.default_rng(1000+si); defn=make_def(perm,rng_e)
        if si==0: spec0=defn
        node_of={perm[k]:k for k in range(N)}
        rows=[(e,ti) for e in TOKK for ti in range(len(TPLS))]; ent={e:[] for e in TOKK}
        for bi in range(0,len(rows),18):
            b=rows[bi:bi+18]; prompts=[render(tok,defn+"\n\n"+TPLS[ti].format(e=e)) for (e,ti) in b]
            enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
            for j,(e,ti) in enumerate(b): ent[e].append(hs[j])
        cents=np.stack([np.mean(ent[e],0) for e in TOKK]); nodes=[node_of[e] for e in TOKK]
        full=cos_rdm_full(cents); obs=up(full)
        rsa={k: float(spearmanr(obs, up(np.array([[T[nodes[i],nodes[j]] for j in range(N)] for i in range(N)]))).correlation) for k,T in TEMPLATES.items()}
        selfM=np.array([[GEO[nodes[i],nodes[j]] for j in range(N)] for i in range(N)]); nonadj=(selfM[TU]>=2)
        graded=float(spearmanr(obs[nonadj],selfM[TU][nonadj]).correlation) if nonadj.sum()>2 else float("nan")
        rng_n=np.random.default_rng(7000+si); nullv=[]
        for _ in range(2000):
            pn=list(rng_n.permutation(range(N))); sd=up(np.array([[GEO[pn[i],pn[j]] for j in range(N)] for i in range(N)]))
            nullv.append(spearmanr(obs,sd).correlation)
        nullv=np.array(nullv); p_emp=float((np.sum(nullv>=rsa["self"])+1)/(len(nullv)+1))
        best=max(TEMPLATES,key=lambda k:rsa[k])
        stats.append(dict(si=si,rsa_self=rsa["self"],rsa_line=rsa["line"],rsa_ring=rsa["ring"],
                          rsa_grid=rsa.get("grid",float("nan")),best_template=best,graded_nonadj=graded,
                          null_mean=float(nullv.mean()),p_emp=p_emp))
        allc[si]=(cents,nodes)
        print(f"  scr{si+1}: self={rsa['self']:+.2f} line={rsa['line']:+.2f} ring={rsa['ring']:+.2f} best={best} "
              f"graded={graded:+.2f} | null {nullv.mean():+.2f} p={p_emp:.3f}",flush=True)
    del model; torch.cuda.empty_cache()
    agg=lambda k: float(np.mean([s[k] for s in stats]))
    summ=dict(model=MODEL,graph=NAME,N=N,invariants=INV,qset=QSET,nscr=NSCR,L=int(L),spec_example=spec0,
              edges=sorted([list(e) for e in E]),per=stats,
              rsa_self=agg("rsa_self"),rsa_self_std=float(np.std([s["rsa_self"] for s in stats])),
              rsa_line=agg("rsa_line"),rsa_ring=agg("rsa_ring"),graded_nonadj=agg("graded_nonadj"),
              n_self_best=int(sum(s["best_template"]=="self" for s in stats)),
              n_self_beats_line=int(sum(s["rsa_self"]>s["rsa_line"] for s in stats)),
              nsig_emp=int(sum(s["p_emp"]<0.05 for s in stats)))
    json.dump(summ,open(OUT,"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"topo_{tag}_{mtag}.npz"),**{f"c{si}":allc[si][0] for si in range(NSCR)},
             nodes=np.array([allc[si][1] for si in range(NSCR)]),tok=np.array(TOKK),
             edges=np.array(sorted([list(e) for e in E])),geo=GEO,L=L)
    print(f"== {tag}/{mtag}: self {summ['rsa_self']:+.2f}±{summ['rsa_self_std']:.2f} (line {summ['rsa_line']:+.2f} "
          f"ring {summ['rsa_ring']:+.2f}) graded {summ['graded_nonadj']:+.2f} | self-best {summ['n_self_best']}/{NSCR} "
          f"beats-line {summ['n_self_beats_line']}/{NSCR} sig {summ['nsig_emp']}/{NSCR} | b1={INV['b1']} novel={INV['is_novel']}",flush=True)
if __name__=="__main__": main()
