#!/usr/bin/env python3
"""PRE-THINKING geometry across MANY scrambles (forward-only, no generation). Natural prompt: open-ended question,
model DEFAULT mode (native thinking scaffold), read residual at the LAST prompt token (= thinking-opener for Gemma/Qwen,
i.e. pre-thinking). Group by input entity → per-scramble imposed-order geometry. Shows whether the structure forms
robustly across orders. Saves residuals (Standard #11).
Usage: python -u natural_geom.py <hf_model> [--nscr 12] [--ntpl 3]"""
import os,sys,numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1]; NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 12
NTPL=int(sys.argv[sys.argv.index("--ntpl")+1]) if "--ntpl" in sys.argv else 3
FRACD=0.75; KS=list(range(1,7)); base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];N=7
TPLS=["What is {k} steps after {e}?","From {e}, advance {k} steps. Which item?","{e} plus {k} steps =?","Starting at {e}, move {k} steps forward. Result?"]
tag=MODEL.split("/")[-1]; os.makedirs(data_dir("behavior"),exist_ok=True)
def deftext(order):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N))
    return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
            f"\nOrder:\n{numbered}\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps.")
def render(tok,c): return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)  # NATURAL default
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
    rng=np.random.default_rng(0); orders=[list(rng.permutation(base)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None); L=int(round(FRACD*nl))
    print(f"NATURAL-GEOM {MODEL} nscr={NSCR} ntpl={NTPL} L={L}/{nl} (forward-only, pre-thinking)",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    allc={}; stats=[]
    for si in range(NSCR):
        order=orders[si]; idx={x.lower():i for i,x in enumerate(order)}; ent_res={e:[] for e in base}
        rows=[(e,k,ti) for e in base for k in KS for ti in range(NTPL)]
        for bi in range(0,len(rows),12):
            b=rows[bi:bi+12]; prompts=[render(tok,deftext(order)+"\n\n"+TPLS[ti].format(k=k,e=e)) for (e,k,ti) in b]
            enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
            for j,(e,k,ti) in enumerate(b): ent_res[e].append(hs[j])
        cents=np.stack([np.mean(ent_res[e],0) for e in base]); ip=[idx[e.lower()] for e in base]; iseq=list(np.argsort(ip))
        D=cosM(cents); rsa=spearmanr(ut(D),ut(cyc(ip))).correlation; dr=rsa-spearmanr(ut(D),ut(lin(ip))).correlation
        p2e={ip[i]:i for i in range(N)}; clos=D[p2e[0],p2e[N-1]]/(np.mean([D[p2e[i],p2e[i+1]] for i in range(N-1)])+1e-9)
        mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T; X=crs(P[:,[0,1]],iseq)
        allc[si]=(cents,ip,iseq); stats.append(dict(si=si,rsa=float(rsa),dRSA=float(dr),closure=float(clos),X=int(X)))
        print(f"  scr{si+1}/{NSCR}: RSA={rsa:+.2f} dRSA={dr:+.2f} clos={clos:.2f} X={X}",flush=True)
    del model; torch.cuda.empty_cache()
    np.savez(os.path.join(data_dir("behavior"), f"naturalgeom_{tag}.npz"),**{f"c{si}":allc[si][0] for si in range(NSCR)},
             imp=np.array([allc[si][1] for si in range(NSCR)]))
    import json; json.dump(stats,open(os.path.join(data_dir("behavior"), f"naturalgeom_{tag}.json"),"w"),indent=2)
    rsas=[s["rsa"] for s in stats]; print(f"== {tag} pre-thinking, {NSCR} scrambles: RSA {np.mean(rsas):+.2f}±{np.std(rsas):.2f} | dRSA {np.mean([s['dRSA'] for s in stats]):+.2f} | clos {np.mean([s['closure'] for s in stats]):.2f}",flush=True)
    # grid plot
    import math; cols=4; rws=math.ceil(NSCR/cols); fig,ax=plt.subplots(rws,cols,figsize=(3.1*cols,3.1*rws)); ax=np.array(ax).reshape(-1)
    for si in range(NSCR):
        cents,ip,iseq=allc[si]; mu=cents.mean(0);_,_,Vt=np.linalg.svd(cents-mu,full_matrices=False);P=(cents-mu)@Vt.T; a=ax[si]
        for i in range(N-1): u,v=iseq[i],iseq[i+1]; a.plot([P[u,0],P[v,0]],[P[u,1],P[v,1]],"-",color="#bbb",lw=1.1)
        u,v=iseq[-1],iseq[0]; a.plot([P[u,0],P[v,0]],[P[u,1],P[v,1]],"-",color="#e00",lw=2.0)
        a.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=70,edgecolors="k",linewidths=.4)
        for i,e in enumerate(base): a.annotate(e[:2],(P[i,0],P[i,1]),fontsize=5,ha="center",va="center")
        a.set_xticks([]);a.set_yticks([]); a.set_title(f"scr{si+1} RSA={stats[si]['rsa']:+.2f}",fontsize=8)
    for k in range(NSCR,len(ax)): ax[k].set_axis_off()
    fig.suptitle(f"{tag} PRE-THINKING geometry (natural prompt, last-token pre-gen), {NSCR} imposed orders — imposed order in PC1-2 (red=wrap edge)",fontsize=11)
    fig.tight_layout();o=os.path.join(data_dir("behavior"), f"FIG_naturalgeom_{tag}.png");fig.savefig(o,dpi=125);print("SAVED",os.path.abspath(o),flush=True)
if __name__=="__main__": main()
