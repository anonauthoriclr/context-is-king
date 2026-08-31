#!/usr/bin/env python3
"""CAUSAL USE test — OPEN-ENDED / CoT regime (regime-matching, Standard #10). Extends the snap entity-patch
(causal_use.py, k=1 answer-only) to the regime the model NATURALLY reasons in.

Question: does the entity-substitution patch survive into multi-token open-ended reasoning, or does the surface token
("Monday") reassert itself during generation? We patch the QUERY entity residual E_src->E_dst at a live layer (from snap:
L18/L27) during PREFILL — the patched rep enters the KV cache and every generated CoT token attends to it (no re-patch
needed) — then let the model generate OPEN-ENDED (thinking on, no "answer only"), and parse the FINAL answer.

  Metric: final answer == succ_O(E_dst) (context's successor of the DONOR), over pairs where the no-patch baseline is
  correct (succ_O(E_src)). Double dissociation: imposed ctx -> succ_imposed(E_dst); natural ctx -> succ_natural(E_dst).
  Bonus: trace-level mention counts (does the reasoning talk about E_src or E_dst?).

Saves FULL generation traces + per-pair flags (Standard #11).
Usage: python -u causal_cot.py <hf_model> [--layers 18,27] [--maxnew 400] [--nscr 1] [--concept days]
Out: data_dir("causal")/causalcot_<tag>.json + data_dir("causal")/FIG_causalcot_<tag>.png
"""
import os, sys, json, re, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 1
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
MAXNEW=int(sys.argv[sys.argv.index("--maxnew")+1]) if "--maxnew" in sys.argv else 400
LAYERS_ARG=[int(x) for x in sys.argv[sys.argv.index("--layers")+1].split(",")] if "--layers" in sys.argv else None  # None -> full fractional sweep (like snap)
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]}
base=ENT[CONCEPT]; N=len(base)
TPL="What is 1 step after {e}?"
tag=MODEL.split("/")[-1]; os.makedirs(data_dir("causal"),exist_ok=True)
OUT=os.path.join(data_dir("causal"), f"causalcot_{tag}.json")

def make_def(order):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N))
    conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    return f"{pre}\nOrder:\n{numbered}{conv}"
def render(tok,c):  # OPEN-ENDED: thinking ON (Gemma/Qwen default), no "answer only" constraint -> let it reason
    return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def imposed_prompt(tok,order,e): return render(tok,make_def(order)+"\n\n"+TPL.format(e=e))
def natural_prompt(tok,e):       return render(tok,TPL.format(e=e))

def parse_last(text):  # CoT conclusion is usually the LAST entity mention
    best=None; bestpos=-1
    for e in base:
        for m in re.finditer(r"\b"+re.escape(e)+r"\b",text,re.I):
            if m.start()>bestpos: bestpos=m.start(); best=e
    return best
def mentions(text,e): return len(re.findall(r"\b"+re.escape(e)+r"\b",text,re.I))

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
        if h.shape[1]<=1 or PATCH["map"] is None: return out   # prefill only; gen steps attend to patched KV
        for b,item in enumerate(PATCH["map"]):
            if item is None: continue
            pos,vec=item; m=vec.shape[0]; h[b,pos[-m:],:]=vec.to(h.dtype).to(h.device)
        return out
    return hook

def tok_entpos(tok,prompts,entities,device):
    encs=[tok(p,return_offsets_mapping=True,add_special_tokens=False) for p in prompts]
    ids=[e["input_ids"] for e in encs]; lens=[len(x) for x in ids]; Lmax=max(lens); entpos=[]
    for p,ent,enc in zip(prompts,entities,encs):
        ci=p.rfind(ent); cj=ci+len(ent); offs=enc["offset_mapping"]
        entpos.append([i for i,(s,e2) in enumerate(offs) if e2>s and s<cj and e2>ci])
    pad=tok.pad_token_id; bids=[]; attn=[]; shifted=[]
    for x,ln,pos in zip(ids,lens,entpos):
        sh=Lmax-ln; bids.append([pad]*sh+x); attn.append([0]*sh+[1]*ln); shifted.append([pp+sh for pp in pos])
    return (torch.tensor(bids,device=device),torch.tensor(attn,device=device),shifted,lens)

def main():
    rng_o=np.random.default_rng(0); orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL)
    nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    LAYERS=LAYERS_ARG if LAYERS_ARG else sorted({max(1,min(nl-1,int(round(f*nl)))) for f in [0.1,0.2,0.3,0.45,0.6,0.7,0.75,0.85,0.95]})
    print(f"CAUSAL-COT {MODEL} concept={CONCEPT} nscr={NSCR} nl={nl} layers={LAYERS} maxnew={MAXNEW} (OPEN-ENDED, thinking ON)",flush=True)
    # --- block-level checkpoint/resume: each (context,scramble) block saved to a partial as it completes ---
    PARTIAL=os.path.join(data_dir("causal"), f"causalcot_{tag}_partial.json")
    done=json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    todo=[(c,si) for c in ["imposed","natural"] for si in range(NSCR) if f"{c}|{si}" not in done]
    print(f"CHECKPOINT: {len(done)}/{2*NSCR} blocks done; todo={todo} (partial={PARTIAL})",flush=True)
    NEED=len(todo)>0
    tok=model=layers=dev=pairs=None
    if NEED:
        tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception:
            from transformers import AutoModelForImageTextToText
            model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        layers=find_layers(model,nl); dev=model.device
        pairs=[(s,d) for s in range(N) for d in range(N) if s!=d]
    def gen_text(ids,attn,patchmap):
        PATCH["map"]=patchmap
        with torch.no_grad():
            g=model.generate(input_ids=ids,attention_mask=attn,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
        PATCH["map"]=None
        return tok.batch_decode(g[:,ids.shape[1]:],skip_special_tokens=True)
    for ctx in ["imposed","natural"]:
        for si in range(NSCR):
            key=f"{ctx}|{si}"
            if key in done: continue
            order=orders[si]; idx={x.lower():i for i,x in enumerate(order)}
            succ_imp={e:order[(idx[e.lower()]+1)%N] for e in base}; succ_nat={e:base[(base.index(e)+1)%N] for e in base}
            succ=succ_imp if ctx=="imposed" else succ_nat; other=succ_nat if ctx=="imposed" else succ_imp
            prompts=[(imposed_prompt(tok,order,e) if ctx=="imposed" else natural_prompt(tok,e)) for e in base]
            ids,attn,entpos,_=tok_entpos(tok,prompts,base,dev)
            # donor cache (entity residual at chosen layers)
            with torch.no_grad(): o=model(input_ids=ids,attention_mask=attn,output_hidden_states=True)
            donor={ei:{L:o.hidden_states[L+1][ei,entpos[ei],:].detach().to(torch.float16).cpu() for L in LAYERS} for ei in range(N)}
            del o; torch.cuda.empty_cache()
            # baseline open-ended (no patch) -> correctness per entity
            btxt=gen_text(ids,attn,[None]*N)
            base_ok={base[i]:(parse_last(btxt[i]) or "").lower()==succ[base[i]].lower() for i in range(N)}
            base_acc=float(np.mean(list(base_ok.values())))
            scr={"order":order,"base_acc_succ_src":base_acc,"base_traces":{base[i]:btxt[i] for i in range(N)},"layers":{}}
            print(f"  [{ctx}] scr{si+1}: open-ended baseline acc(succ_src)={base_acc:.2f}",flush=True)
            for L in LAYERS:
                recs=[]; hk=layers[L].register_forward_hook(make_hook())
                for bi in range(0,len(pairs),14):
                    chunk=pairs[bi:bi+14]; sp=[base[s] for (s,d) in chunk]
                    bids,battn,bpos,_=tok_entpos(tok,[prompts[s] for (s,d) in chunk],sp,dev); pmap=[]
                    for j,(s,d) in enumerate(chunk):
                        dv=donor[d][L].to(dev); m=min(dv.shape[0],len(bpos[j])); pmap.append((bpos[j],dv[-m:]))
                    txt=gen_text(bids,battn,pmap)
                    for (s,d),t in zip(chunk,txt):
                        a=parse_last(t)
                        recs.append({"src":base[s],"dst":base[d],"ans":a,
                                     "hit_dst":bool(a and a.lower()==succ[base[d]].lower()),
                                     "stuck_src":bool(a and a.lower()==succ[base[s]].lower()),
                                     "cross_dst":bool(a and a.lower()==other[base[d]].lower()),
                                     "na":a is None,"base_ok":base_ok[base[s]] and base_ok[base[d]],
                                     "ment_src":mentions(t,base[s]),"ment_dst":mentions(t,base[d]),"trace":t})
                hk.remove()
                clean=[r for r in recs if r["base_ok"]]
                def rate(rs,k): return float(np.mean([r[k] for r in rs])) if rs else 0.0
                scr["layers"][str(L)]={
                    "n_clean":len(clean),
                    "patch_success_clean":rate(clean,"hit_dst"),"stuck_src_clean":rate(clean,"stuck_src"),
                    "cross_map_clean":rate(clean,"cross_dst"),"na_clean":rate(clean,"na"),
                    "patch_success_all":rate(recs,"hit_dst"),
                    "ment_dst_gt_src":float(np.mean([r["ment_dst"]>r["ment_src"] for r in clean])) if clean else 0.0,
                    "recs":recs}
                s=scr["layers"][str(L)]
                print(f"  [{ctx}] scr{si+1} L{L}: patch_success(clean n={len(clean)})={s['patch_success_clean']:.2f}  stuck_src={s['stuck_src_clean']:.2f}  cross_map={s['cross_map_clean']:.2f}  na={s['na_clean']:.2f}  trace_dst>src={s['ment_dst_gt_src']:.2f}",flush=True)
            done[key]=scr; json.dump(done,open(PARTIAL,"w"),indent=2)
            print(f"  CKPT saved block {key} ({len(done)}/{2*NSCR})",flush=True)
    if NEED and model is not None: del model; torch.cuda.empty_cache()
    results={"model":MODEL,"concept":CONCEPT,"nscr":NSCR,"nl":int(nl),"layers":LAYERS,"maxnew":MAXNEW,
             "regime":"open-ended/CoT (thinking ON)","tpl":TPL,"contexts":{}}
    for ctx in ["imposed","natural"]:
        results["contexts"][ctx]={"per_scr":[done[f"{ctx}|{si}"] for si in range(NSCR)]}
    json.dump(results,open(OUT,"w"),indent=2); print("SAVED",os.path.abspath(OUT),flush=True)
    plot(results)

def plot(R):
    imp=R["contexts"]["imposed"]["per_scr"]; nat=R["contexts"]["natural"]["per_scr"]; sweep=R["layers"]
    avg=lambda scrs,L,k: float(np.mean([s["layers"][str(L)][k] for s in scrs]))
    imp_succ=[avg(imp,L,"patch_success_clean") for L in sweep]
    imp_cross=[avg(imp,L,"cross_map_clean") for L in sweep]
    nat_succ=[avg(nat,L,"patch_success_clean") for L in sweep]
    imp_stuck=[avg(imp,L,"stuck_src_clean") for L in sweep]
    causal_L=sweep[int(np.argmax(imp_succ))]; ml=str(causal_L)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.6))
    ax1.plot(sweep,imp_succ,"o-",color="C0",label="imposed → succ_IMPOSED(dst)")
    ax1.plot(sweep,imp_cross,"s--",color="C3",alpha=.7,label="imposed → succ_natural(dst)")
    ax1.plot(sweep,nat_succ,"^-",color="C2",label="natural → succ_NATURAL(dst)")
    ax1.plot(sweep,imp_stuck,"v:",color="gray",alpha=.6,label="stuck@succ(src) [imposed]")
    ax1.axhline(1/7,ls="--",color="k",alpha=.3,label="chance 1/7"); ax1.axvline(causal_L,color="green",alpha=.25)
    ax1.set_xlabel("patch layer (entity token)"); ax1.set_ylabel("rate (clean subset)"); ax1.set_ylim(-.02,1.02)
    ax1.set_title(f"Entity patch in OPEN-ENDED/CoT — {R['model'].split('/')[-1]}"); ax1.legend(fontsize=6.5)
    x=np.arange(2); w=.35
    imp_self=avg(imp,causal_L,"patch_success_clean"); imp_oth=avg(imp,causal_L,"cross_map_clean")
    nat_self=avg(nat,causal_L,"patch_success_clean"); nat_oth=avg(nat,causal_L,"cross_map_clean")
    ax2.bar(x-w/2,[imp_self,nat_self],w,label="follows ITS OWN context-map",color="C0")
    ax2.bar(x+w/2,[imp_oth,nat_oth],w,label="follows the OTHER map",color="C3",alpha=.7)
    ax2.set_xticks(x); ax2.set_xticklabels(["imposed context","natural context"]); ax2.set_ylim(0,1.05)
    ax2.set_title(f"Dissociation @L{causal_L} (CoT, clean subset)\npatch survives open-ended reasoning"); ax2.legend(fontsize=8)
    for i,v in enumerate([imp_self,nat_self]): ax2.text(i-w/2,v+.02,f"{v:.2f}",ha="center",fontsize=9)
    R["causal_layer"]=int(causal_L)
    plt.tight_layout(); fn=os.path.join(data_dir("causal"), f"FIG_causalcot_{R['model'].split('/')[-1]}.png"); plt.savefig(fn,dpi=130); print("FIG",os.path.abspath(fn),"causal_L",causal_L,flush=True)

if __name__=="__main__": main()
