#!/usr/bin/env python3
"""CONTENT-vs-FORMAT decider (2x2): does the imposed geometry's TOPOLOGY (line vs cycle) track the WRAP CONTENT or the
surface FORMAT? Each cell differs from its base by exactly the wrap edge. Geometry-only (one-shot pre-gen residual,
entity-layout). Saves raw residuals (Standard #11). PRE-REG: content-hypothesis => list & adj_nowrap = LINE (dRSA<0,
wrap-pair far), list_wrap & adj = CYCLE (dRSA>0, wrap-pair close). format-hypothesis => both numbered = line, both edge = cycle.
Usage: python -u wrap_2x2.py [--nscr 2]"""
import os,sys,json,numpy as np,torch,glob
from scipy.stats import spearmanr
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 2
FRACD=0.75; KS=list(range(1,7)); NTPL=3
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "arb7":["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp"],
     "arb12":["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror"]}
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
base=ENT[CONCEPT];N=len(base)
TPLS=["What is {k} steps after {e}?","From {e}, advance {k} steps. Which item?","{e} plus {k} steps =?"]
CONDS=["list","list_wrap","adj_nowrap","adj"]
MODELS=[sys.argv[sys.argv.index("--model")+1]] if "--model" in sys.argv else ["google/gemma-4-31B-it","google/gemma-4-12B-it","Qwen/Qwen3.5-27B","Qwen/Qwen3.5-9B"]
def make_def(mode,order,rng):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    wrapfact=f"\nThe order wraps around: {order[N-1]} is immediately followed by {order[0]}."
    if mode=="list":      return f"{pre}\nOrder:\n{numbered}{conv}"
    if mode=="list_wrap": return f"{pre}\nOrder:\n{numbered}{wrapfact}{conv}"
    edges=[f"- {order[i]} is immediately followed by {order[(i+1)%N]}." for i in range(N)]   # edge N-1 = wrap (order[N-1]->order[0])
    if mode=="adj_nowrap": edges=edges[:N-1]
    perm=list(rng.permutation(len(edges))); shuf="\n".join(edges[j] for j in perm)
    cadj="\nConvention: 'k steps after X' = apply 'immediately followed by' that many times, starting at X."
    return f"{pre}\nThe ONLY valid 'immediately followed by' relations are (the normal order does NOT apply):\n{shuf}{cadj}"
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)
def cosM(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)
def cyc(p): P=np.array(p);D=np.abs(P[:,None]-P[None,:]);return np.minimum(D,N-D).astype(float)
def lin(p): P=np.array(p);return np.abs(P[:,None]-P[None,:]).astype(float)
ut=lambda M:M[np.triu_indices(N,1)]
def _ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0])
def _in(p1,p2,p3,p4): return _ccw(p1,p3,p4)!=_ccw(p2,p3,p4) and _ccw(p1,p2,p3)!=_ccw(p1,p2,p4)
def crs(pts,seq):
    s=list(seq)+[seq[0]];E=[(s[i],s[i+1]) for i in range(len(s)-1)];c=0
    for a in range(len(E)):
        for b in range(a+1,len(E)):
            if set(E[a])&set(E[b]):continue
            if _in(pts[E[a][0]],pts[E[a][1]],pts[E[b][0]],pts[E[b][1]]):c+=1
    return c
def main():
    rng_o=np.random.default_rng(0); orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]
    SUMM={}; CACHE={}
    for M in MODELS:
        tag=M.split("/")[-1]; cfg=AutoConfig.from_pretrained(M); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None); L=int(round(FRACD*nl))
        print(f"=== {tag} L={L}/{nl} ===",flush=True)
        tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        try: model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception:
            from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16,device_map={"":0}).eval()
        Rall=[];Call=[];Sall=[];Eall=[]
        for cond in CONDS:
            for si in range(NSCR):
                order=orders[si]; rng_e=np.random.default_rng(1000+si); defn=make_def(cond,order,rng_e)
                rows=[(e,k,ti) for e in base for k in KS for ti in range(NTPL)]; ent_res={e:[] for e in base}
                for bi in range(0,len(rows),12):
                    b=rows[bi:bi+12]; prompts=[render(tok,defn+"\n\n"+TPLS[ti].format(k=k,e=e)) for (e,k,ti) in b]
                    enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
                    with torch.no_grad(): o=model(**enc,output_hidden_states=True)
                    hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
                    for j,(e,k,ti) in enumerate(b): ent_res[e].append(hs[j]); Rall.append(hs[j].astype(np.float16)); Call.append(CONDS.index(cond)); Sall.append(si); Eall.append(base.index(e))
                cents=np.stack([np.mean(ent_res[e],0) for e in base]); ip=[{x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base]; iseq=list(np.argsort(ip))
                D=cosM(cents); dr=spearmanr(ut(D),ut(cyc(ip))).correlation-spearmanr(ut(D),ut(lin(ip))).correlation
                p2e={ip[bi2]:bi2 for bi2 in range(N)}; wrap=D[p2e[0],p2e[N-1]]; interior=np.mean([D[p2e[i],p2e[i+1]] for i in range(N-1)])
                rsa=spearmanr(ut(D),ut(cyc(ip))).correlation; mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T; X=crs(P[:,[0,1]],iseq)
                SUMM.setdefault(tag,{}).setdefault(cond,[]).append(dict(si=si,dRSA=float(dr),closure=float(wrap/(interior+1e-9)),rsa=float(rsa),X=int(X)))
                if si==0: CACHE[(tag,cond)]=(cents,ip,iseq)
        del model; torch.cuda.empty_cache()
        np.savez(os.path.join(data_dir("behavior"), f"wrap2x2_{tag}_{CONCEPT}.npz"),R=np.stack(Rall),COND=np.array(Call),SI=np.array(Sall),ENT=np.array(Eall),
                 imp=np.array([[ {x.lower():i for i,x in enumerate(orders[si])}[e.lower()] for e in base] for si in range(NSCR)]),conds=np.array(CONDS),L=L)
        s=SUMM[tag]; pr=lambda c:f"dRSA={np.mean([r['dRSA'] for r in s[c]]):+.2f} clos={np.mean([r['closure'] for r in s[c]]):.2f} X={np.mean([r['X'] for r in s[c]]):.1f}"
        print(f"  {tag}: list[{pr('list')}] list_wrap[{pr('list_wrap')}] adj_nowrap[{pr('adj_nowrap')}] adj[{pr('adj')}]",flush=True)
    json.dump({m:{c:SUMM[m][c] for c in CONDS} for m in SUMM},open(os.path.join(data_dir("behavior"), f"wrap2x2_summary_{CONCEPT}.json"),"w"),indent=2)
    # grid plot: rows=models, cols=conditions
    fig,ax=plt.subplots(len(MODELS),len(CONDS),figsize=(3.2*len(CONDS),3.2*len(MODELS))); ax=np.atleast_2d(ax)
    for r,M in enumerate(MODELS):
        tag=M.split("/")[-1]
        for c,cond in enumerate(CONDS):
            cents,ip,iseq=CACHE[(tag,cond)]; mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T
            dr=np.mean([x['dRSA'] for x in SUMM[tag][cond]]); cl=np.mean([x['closure'] for x in SUMM[tag][cond]]); X=crs(P[:,[0,1]],iseq)
            a=ax[r,c]; path=iseq+[iseq[0]]; a.plot(P[path,0],P[path,1],"-",color="#888",lw=1.1); a.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=80,edgecolors="k",linewidths=.4)
            for i,e in enumerate(base): a.annotate(e[:3],(P[i,0],P[i,1]),fontsize=6,ha="center",va="center")
            a.set_xticks([]);a.set_yticks([])
            if r==0: a.set_title(f"{cond}",fontsize=10)
            if c==0: a.set_ylabel(tag.replace('gemma-4-','G').replace('Qwen3.5-','Q'),fontsize=9)
            a.text(.5,-.07,f"dRSA={dr:+.2f} clos={cl:.2f} X={X}",transform=a.transAxes,ha="center",fontsize=7)
    fig.suptitle("CONTENT vs FORMAT 2x2 — wrap flips topology WITHIN format? (rows=models, cols: numbered{list,list_wrap} | edge{adj_nowrap,adj})",fontsize=11)
    fig.tight_layout(); o=os.path.join(data_dir("behavior"), f"FIG_wrap2x2_{CONCEPT}.png"); fig.savefig(o,dpi=125); print("SAVED",os.path.abspath(o),flush=True)
if __name__=="__main__": main()
