#!/usr/bin/env python3
"""CAUSAL k-patch — is the STEP-COUNT k causally used, and is it read LATER than the entity (operand early / computation
late)? Companion to causal_use.py (entity patch). Same machinery, target = the k-digit token.

Prompt = <imposed order O> + "Advance {k} steps from {E}. Which item?" (snap/answer-only). Overwrite the k-digit token
residual k_src->k_dst (donor = same prompt with k_dst) at decoder layer ell during prefill; generate; check answer ==
succ_O^{k_dst}(E) (the k_dst-hop successor). Restrict to (E,k) where the no-patch baseline is correct at BOTH k_src and
k_dst (clean flip). ALL k-pairs k in {1..KMAX} (operator: "show all pairs work"), full k_src x k_dst matrix at the causal
layer + layer sweep. Controls: random-donor at k-slot; k-donor at a non-k slot.

Usage: python -u causal_k.py <hf_model> [--nscr 2] [--kmax 6] [--concept days]
Out: data_dir("causal")/causalk_<tag>.json + data_dir("causal")/FIG_causalk_<tag>.png
"""
import os, sys, json, re, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 2
KMAX=int(sys.argv[sys.argv.index("--kmax")+1]) if "--kmax" in sys.argv else 6
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
FRACD=0.75
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]}
base=ENT[CONCEPT]; N=len(base); KS=list(range(1,KMAX+1))
TPL="Advance {k} steps from {e}. Which item?"
tag=MODEL.split("/")[-1]; os.makedirs(data_dir("causal"),exist_ok=True)
OUT=os.path.join(data_dir("causal"), f"causalk_{tag}.json")

def make_def(order):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N))
    conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    return f"{pre}\nOrder:\n{numbered}{conv}"
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)
def prompt_of(tok,order,e,k): return render(tok,make_def(order)+"\n\n"+TPL.format(k=k,e=e))
def parse_answer(text):
    best=None; bp=10**9
    for e in base:
        m=re.search(r"\b"+re.escape(e)+r"\b",text,re.I)
        if m and m.start()<bp: best=e; bp=m.start()
    return best
def find_layers(model,nl):
    cands=[mod for _,mod in model.named_modules() if isinstance(mod,nn.ModuleList) and len(mod)==nl]
    for mod in cands:
        if any(hasattr(mod[0],a) for a in ("self_attn","attn","self_attention","attention","mlp")): return mod
    if cands: return cands[0]
    raise RuntimeError("decoder layer list not found")
PATCH={"map":None}
def make_hook():
    def hook(module,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        if h.shape[1]<=1 or PATCH["map"] is None: return out
        for b,item in enumerate(PATCH["map"]):
            if item is None: continue
            pos,vec=item; m=vec.shape[0]; h[b,pos[-m:],:]=vec.to(h.dtype).to(h.device)
        return out
    return hook
def tok_pos(tok,prompts,targets,device):
    """left-pad + locate the TARGET substring (last occurrence) token positions (padded coords)."""
    encs=[tok(p,return_offsets_mapping=True,add_special_tokens=False) for p in prompts]
    ids=[e["input_ids"] for e in encs]; lens=[len(x) for x in ids]; Lmax=max(lens); tp=[]
    for p,t,enc in zip(prompts,targets,encs):
        ci=p.rfind(t); cj=ci+len(t); offs=enc["offset_mapping"]
        tp.append([i for i,(s,e2) in enumerate(offs) if e2>s and s<cj and e2>ci])
    pad=tok.pad_token_id; bids=[]; attn=[]; sh=[]
    for x,ln,pos in zip(ids,lens,tp):
        d=Lmax-ln; bids.append([pad]*d+x); attn.append([0]*d+[1]*ln); sh.append([pp+d for pp in pos])
    return torch.tensor(bids,device=device),torch.tensor(attn,device=device),sh

def main():
    rng_o=np.random.default_rng(0); orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL)
    nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    map_layer=int(round(FRACD*nl))
    sweep=sorted({max(1,min(nl-1,int(round(f*nl)))) for f in [0.1,0.2,0.3,0.45,0.6,0.7,0.75,0.85,0.95]} | {map_layer})
    ctrl_layers=sorted({max(1,min(nl-1,int(round(f*nl)))) for f in [0.2,0.45,0.6]} | {map_layer})
    print(f"CAUSAL-K {MODEL} concept={CONCEPT} nscr={NSCR} kmax={KMAX} nl={nl} sweep={sweep}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    layers=find_layers(model,nl); dev=model.device
    def gen(ids,attn,pm):
        PATCH["map"]=pm
        with torch.no_grad(): g=model.generate(input_ids=ids,attention_mask=attn,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
        PATCH["map"]=None
        return [parse_answer(t) for t in tok.batch_decode(g[:,ids.shape[1]:],skip_special_tokens=True)]

    results={"model":MODEL,"concept":CONCEPT,"nscr":NSCR,"kmax":KMAX,"nl":int(nl),"map_layer":int(map_layer),
             "sweep":sweep,"tpl":TPL,"per_scr":[]}
    for si in range(NSCR):
        order=orders[si]; idx={x.lower():i for i,x in enumerate(order)}
        succ={(e,k):order[(idx[e.lower()]+k)%N] for e in base for k in KS}
        # all (E,k) prompts
        cells=[(e,k) for e in base for k in KS]
        prompts={(e,k):prompt_of(tok,order,e,k) for (e,k) in cells}
        # donor cache: k-digit residual per (e,k) at sweep layers
        donor={};
        for bi in range(0,len(cells),14):
            chunk=cells[bi:bi+14]
            ids,attn,kpos=tok_pos(tok,[prompts[c] for c in chunk],[str(k) for (e,k) in chunk],dev)
            with torch.no_grad(): o=model(input_ids=ids,attention_mask=attn,output_hidden_states=True)
            for j,(e,k) in enumerate(chunk):
                donor[(e,k)]={"pos":kpos[j],"vec":{L:o.hidden_states[L+1][j,kpos[j],:].detach().to(torch.float16).cpu() for L in sweep}}
            del o
        torch.cuda.empty_cache()
        # baseline (no patch) correctness per (e,k)
        base_ok={}
        for bi in range(0,len(cells),14):
            chunk=cells[bi:bi+14]
            ids,attn,_=tok_pos(tok,[prompts[c] for c in chunk],[str(k) for (e,k) in chunk],dev)
            ans=gen(ids,attn,[None]*len(chunk))
            for (e,k),a in zip(chunk,ans): base_ok[(e,k)]=bool(a and a.lower()==succ[(e,k)].lower())
        base_acc={k:float(np.mean([base_ok[(e,k)] for e in base])) for k in KS}
        print(f"  scr{si+1}: baseline acc by k = "+" ".join(f"k{k}={base_acc[k]:.2f}" for k in KS),flush=True)
        # patch cases: all (e,k_src->k_dst), k_src!=k_dst
        kpairs=[(e,ks,kd) for e in base for ks in KS for kd in KS if ks!=kd]
        scr={"order":order,"base_acc_by_k":base_acc,"layers":{}}
        for L in sweep:
            recs=[]; hk=layers[L].register_forward_hook(make_hook())
            for bi in range(0,len(kpairs),14):
                chunk=kpairs[bi:bi+14]
                ids,attn,kpos=tok_pos(tok,[prompts[(e,ks)] for (e,ks,kd) in chunk],[str(ks) for (e,ks,kd) in chunk],dev)
                pm=[]
                for j,(e,ks,kd) in enumerate(chunk):
                    dv=donor[(e,kd)]["vec"][L].to(dev); m=min(dv.shape[0],len(kpos[j])); pm.append((kpos[j],dv[-m:]))
                ans=gen(ids,attn,pm)
                for (e,ks,kd),a in zip(chunk,ans):
                    recs.append({"e":e,"ks":ks,"kd":kd,"ans":a,
                                 "hit_dst":bool(a and a.lower()==succ[(e,kd)].lower()),
                                 "stuck_src":bool(a and a.lower()==succ[(e,ks)].lower()),
                                 "clean":base_ok[(e,ks)] and base_ok[(e,kd)]})
            hk.remove()
            clean=[r for r in recs if r["clean"]]
            rate=lambda rs,k: float(np.mean([r[k] for r in rs])) if rs else 0.0
            scr["layers"][str(L)]={"n_clean":len(clean),"patch_success_clean":rate(clean,"hit_dst"),
                                   "stuck_src_clean":rate(clean,"stuck_src"),"patch_success_all":rate(recs,"hit_dst"),"recs":recs}
            print(f"  scr{si+1} L{L}: k-patch_success(clean n={len(clean)})={scr['layers'][str(L)]['patch_success_clean']:.2f}  stuck_src={scr['layers'][str(L)]['stuck_src_clean']:.2f}",flush=True)
        # controls at ctrl_layers
        scr["controls"]={}
        for L in ctrl_layers:
            ctrl={}
            for mode in ("random","position"):
                hk=layers[L].register_forward_hook(make_hook()); rr=[]
                for bi in range(0,len(kpairs),14):
                    chunk=kpairs[bi:bi+14]
                    ids,attn,kpos=tok_pos(tok,[prompts[(e,ks)] for (e,ks,kd) in chunk],[str(ks) for (e,ks,kd) in chunk],dev)
                    pm=[]
                    for j,(e,ks,kd) in enumerate(chunk):
                        dv=donor[(e,kd)]["vec"][L]
                        if mode=="random":
                            rng=np.random.default_rng(8000+si*1000+L*100+bi+j); m=min(dv.shape[0],len(kpos[j]))
                            rv=torch.tensor(rng.standard_normal(dv[-m:].shape),dtype=torch.float32)
                            rv=rv/(rv.norm(dim=1,keepdim=True)+1e-9)*dv[-m:].float().norm(dim=1,keepdim=True)
                            pm.append((kpos[j],rv.to(dev)))
                        else:
                            pm.append(([max(0,kpos[j][0]-1)],dv.to(dev)[-1:]))
                    ans=gen(ids,attn,pm)
                    for (e,ks,kd),a in zip(chunk,ans):
                        if base_ok[(e,ks)] and base_ok[(e,kd)]: rr.append(bool(a and a.lower()==succ[(e,kd)].lower()))
                hk.remove(); ctrl[mode]=float(np.mean(rr)) if rr else 0.0
            scr["controls"][str(L)]=ctrl
            print(f"  scr{si+1} CONTROLS @L{L}: random={ctrl['random']:.2f} position={ctrl['position']:.2f}",flush=True)
        results["per_scr"].append(scr)
    del model; torch.cuda.empty_cache()
    json.dump(results,open(OUT,"w"),indent=2); print("SAVED",os.path.abspath(OUT),flush=True)
    plot(results)

def plot(R):
    P=R["per_scr"]; sweep=R["sweep"]; KS=list(range(1,R["kmax"]+1))
    avg=lambda L,k: float(np.mean([s["layers"][str(L)][k] for s in P]))
    ys=[avg(L,"patch_success_clean") for L in sweep]
    causal_L=sweep[int(np.argmax(ys))]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.6))
    ax1.plot(sweep,ys,"o-",color="C0",label="k-patch success (clean)")
    ax1.plot(sweep,[avg(L,"stuck_src_clean") for L in sweep],"s--",color="C3",alpha=.6,label="stuck@succ^{ks}")
    if "controls" in P[0]:
        cl=sorted(int(k) for k in P[0]["controls"])
        ax1.plot(cl,[np.mean([s["controls"][str(L)]["random"] for s in P]) for L in cl],"x:",color="gray",label="random-donor")
        ax1.plot(cl,[np.mean([s["controls"][str(L)]["position"] for s in P]) for L in cl],"+:",color="brown",label="position (wrong slot)")
    ax1.axhline(1/7,ls="--",color="k",alpha=.3,label="chance 1/7"); ax1.axvline(causal_L,color="green",alpha=.25,label=f"k causal L{causal_L}")
    ax1.set_xlabel("patch layer (k-digit token)"); ax1.set_ylabel("rate (clean k-pairs)"); ax1.set_ylim(-.02,1.02)
    ax1.set_title(f"k-patch USE test — {R['model'].split('/')[-1]}"); ax1.legend(fontsize=6.5)
    # k_src x k_dst patch-success matrix at causal layer (clean)
    M=np.full((len(KS),len(KS)),np.nan); cnt=np.zeros((len(KS),len(KS)))
    for s in P:
        for r in s["layers"][str(causal_L)]["recs"]:
            if r["clean"]:
                i=r["ks"]-1; j=r["kd"]-1
                M[i,j]=(0 if np.isnan(M[i,j]) else M[i,j]*cnt[i,j])+r["hit_dst"]; cnt[i,j]+=1; M[i,j]/=cnt[i,j]
    im=ax2.imshow(M,vmin=0,vmax=1,cmap="viridis"); ax2.set_xticks(range(len(KS))); ax2.set_xticklabels(KS); ax2.set_yticks(range(len(KS))); ax2.set_yticklabels(KS)
    ax2.set_xlabel("k_dst (patched-to)"); ax2.set_ylabel("k_src (prompt)"); ax2.set_title(f"k-patch success matrix @L{causal_L} (clean)")
    for i in range(len(KS)):
        for j in range(len(KS)):
            if not np.isnan(M[i,j]): ax2.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w",fontsize=7)
    fig.colorbar(im,ax=ax2,fraction=.046)
    R["causal_layer"]=int(causal_L)
    plt.tight_layout(); fn=os.path.join(data_dir("causal"), f"FIG_causalk_{R['model'].split('/')[-1]}.png"); plt.savefig(fn,dpi=130); print("FIG",os.path.abspath(fn),"causal_L",causal_L,flush=True)

if __name__=="__main__": main()
