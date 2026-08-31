#!/usr/bin/env python3
"""CAUSAL USE test (Gate-1 SIGNED OFF 2026-06-23): is the context-built order-map USED to compute the one-shot answer?

Entity-substitution activation patch. Prompt = <context O> + "What is 1 step after E_src?" (k=1, answer-only, the regime
where one-shot reads the map). We OVERWRITE the QUERY entity-token residual E_src->E_dst (donor-cached, same context) at
decoder layer L during prefill, hold the order context fixed, then generate.

  PRIMARY (patch-success): answer becomes succ_O(E_dst) -- the IMPOSED successor of the DONOR entity. If so, the model
  computed the answer FROM the (patched) representation THROUGH the context map, not from the surface token (which still
  literally says E_src).

  DECISIVE CONTROL -- imposed-vs-natural DOUBLE DISSOCIATION: the SAME entity patch under
    - imposed context  -> answer flips to succ_imposed(E_dst)
    - natural context  -> answer flips to succ_natural(E_dst)
  Same patch, different map (set by context) => the CONTEXT-BUILT map is the causal entity->answer transform.

  CONTROLS (at the map layer): random-donor (norm-matched -> rules out "any perturbation flips it"); position-token
  (write the entity vector at a NON-entity slot -> locus is the entity, not anywhere); no-patch baseline (ans==succ_O(src)).

Decisions: site=entity-token (query occ); op=OVERWRITE; sites=ALL entity sub-tokens, right-aligned m=min(len_src,len_dst).
Saves raw per (context, layer, scramble, pair): generated answer + flags + span lengths (Standard #11).

Usage: python -u causal_use.py <hf_model> [--nscr 2] [--concept days] [--maxnew 8]
Out: data_dir("causal")/causaluse_<tag>.json + FIG_causaluse_<tag>.png
"""
import os, sys, json, re, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 2
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
MAXNEW=int(sys.argv[sys.argv.index("--maxnew")+1]) if "--maxnew" in sys.argv else 8
FRACD=0.75
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "months":["January","February","March","April","May","June","July","August","September","October","November","December"],
     "zodiac":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]}
base=ENT[CONCEPT]; N=len(base)
TPL="What is 1 step after {e}?"   # single canonical k=1 template (keep the patch clean)
tag=MODEL.split("/")[-1]; os.makedirs(data_dir("causal"),exist_ok=True)
OUT=os.path.join(data_dir("causal"), f"causaluse_{tag}{'' if CONCEPT=='days' else '_'+CONCEPT}.json")  # days keeps legacy name; others suffixed

# --- prompt builders: imposed = F43 list-mode; natural = no redefinition (pretrained order) ---
def make_def(order):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N))
    conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    return f"{pre}\nOrder:\n{numbered}{conv}"
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)
def imposed_prompt(tok,order,e): return render(tok,make_def(order)+"\n\n"+TPL.format(e=e))
def natural_prompt(tok,e):       return render(tok,TPL.format(e=e))

def parse_answer(text):
    best=None; bestpos=10**9
    for e in base:
        m=re.search(r"\b"+re.escape(e)+r"\b",text,re.I)
        if m and m.start()<bestpos: best=e; bestpos=m.start()
    return best

def find_layers(model,nl):
    cands=[mod for _,mod in model.named_modules() if isinstance(mod,nn.ModuleList) and len(mod)==nl]
    for mod in cands:
        if any(hasattr(mod[0],a) for a in ("self_attn","attn","self_attention","attention","mlp")): return mod
    if cands: return cands[0]
    raise RuntimeError("decoder layer list not found")

# global patch state read by the hook
PATCH={"map":None}  # list over batch idx -> (pos_list, donor_tensor[m,d]) or None
def make_hook():
    def hook(module,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        if h.shape[1]<=1 or PATCH["map"] is None: return out   # only prefill
        for b,item in enumerate(PATCH["map"]):
            if item is None: continue
            pos,vec=item; m=vec.shape[0]
            h[b,pos[-m:],:]=vec.to(h.dtype).to(h.device)
        return out
    return hook

def tok_entpos(tok,prompts,entities,device):
    """tokenize -> left-padded (ids,attn) + per-example QUERY-entity token positions (padded coords) + unpadded len."""
    encs=[tok(p,return_offsets_mapping=True,add_special_tokens=False) for p in prompts]
    ids=[e["input_ids"] for e in encs]; lens=[len(x) for x in ids]; Lmax=max(lens)
    entpos=[]
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
    map_layer=int(round(FRACD*nl))
    sweep=sorted({max(1,min(nl-1,int(round(f*nl)))) for f in [0.1,0.2,0.3,0.45,0.6,0.7,0.75,0.85,0.95]} | {map_layer})
    # controls run where the real patch is LIVE (specificity is only informative there), not only at the dead deep layer
    ctrl_layers=sorted({max(1,min(nl-1,int(round(f*nl)))) for f in [0.2,0.45,0.6]} | {map_layer})
    print(f"CAUSAL-USE {MODEL} concept={CONCEPT} nscr={NSCR} nl={nl} map_layer={map_layer} sweep={sweep}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    layers=find_layers(model,nl); dev=model.device
    pairs=[(s,d) for s in range(N) for d in range(N) if s!=d]   # 42 ordered entity pairs

    def gen(ids,attn,patchmap):
        PATCH["map"]=patchmap
        with torch.no_grad():
            g=model.generate(input_ids=ids,attention_mask=attn,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
        PATCH["map"]=None
        return [parse_answer(t) for t in tok.batch_decode(g[:,ids.shape[1]:],skip_special_tokens=True)]

    results={"model":MODEL,"concept":CONCEPT,"nscr":NSCR,"nl":int(nl),"map_layer":int(map_layer),"sweep":sweep,
             "tpl":TPL,"decisions":{"site":"entity-token-query","op":"overwrite","subtokens":"all-right-aligned"},"contexts":{}}

    for ctx in ["imposed","natural"]:
        ctx_layers = sweep   # full sweep on BOTH contexts -> dissociation measured where the patch is LIVE (early layers)
        per_scr=[]
        for si in range(NSCR):
            order=orders[si]; idx={x.lower():i for i,x in enumerate(order)}
            succ_imp={e:order[(idx[e.lower()]+1)%N] for e in base}
            succ_nat={e:base[(base.index(e)+1)%N] for e in base}
            succ = succ_imp if ctx=="imposed" else succ_nat
            other= succ_nat if ctx=="imposed" else succ_imp     # the "wrong map" successor (cross-check)
            prompts=[(imposed_prompt(tok,order,e) if ctx=="imposed" else natural_prompt(tok,e)) for e in base]
            ids,attn,entpos,lens=tok_entpos(tok,prompts,base,dev)
            # donor cache: residual at QUERY-entity positions, all layers, per entity (single forward)
            with torch.no_grad(): o=model(input_ids=ids,attention_mask=attn,output_hidden_states=True)
            donor={}  # donor[e_idx][layer] = tensor[n_tok,d] (cpu float16)
            for ei in range(N):
                pos=entpos[ei]; donor[ei]={L:o.hidden_states[L+1][ei,pos,:].detach().to(torch.float16).cpu() for L in sweep}
            del o; torch.cuda.empty_cache()
            # no-patch baseline (verify ans==succ(src))
            base_ans=gen(ids,attn,[None]*N)
            base_hit=np.mean([1.0 if (base_ans[i] and base_ans[i].lower()==succ[base[i]].lower()) else 0.0 for i in range(N)])
            scr={"order":order,"base_acc_succ_src":float(base_hit),"layers":{}}
            # --- layer sweep: entity patch E_src->E_dst ---
            h=layers  # alias
            for L in ctx_layers:
                hk=h[L].register_forward_hook(make_hook())
                recs=[]
                for bi in range(0,len(pairs),14):
                    chunk=pairs[bi:bi+14]
                    sp=[base[s] for (s,d) in chunk]
                    bids,battn,bpos,blens=tok_entpos(tok,[prompts[s] for (s,d) in chunk],sp,dev)
                    pmap=[]
                    for j,(s,d) in enumerate(chunk):
                        dv=donor[d][L].to(dev); m=min(dv.shape[0],len(bpos[j]))  # right-align
                        pmap.append((bpos[j],dv[-m:]))
                    ans=gen(bids,battn,pmap)
                    for (s,d),a in zip(chunk,ans):
                        recs.append({"src":base[s],"dst":base[d],"ans":a,
                                     "hit_dst":bool(a and a.lower()==succ[base[d]].lower()),
                                     "stuck_src":bool(a and a.lower()==succ[base[s]].lower()),
                                     "cross_dst":bool(a and a.lower()==other[base[d]].lower()),
                                     "na":a is None})
                hk.remove()
                psucc=float(np.mean([r["hit_dst"] for r in recs]))
                pstuck=float(np.mean([r["stuck_src"] for r in recs]))
                pcross=float(np.mean([r["cross_dst"] for r in recs]))
                pna=float(np.mean([r["na"] for r in recs]))
                scr["layers"][str(L)]={"patch_success":psucc,"stuck_src":pstuck,"cross_map":pcross,"na":pna,"recs":recs}
                print(f"  [{ctx}] scr{si+1} L{L}: patch_success(->succ_{ctx[:3]}(dst))={psucc:.2f}  stuck_src={pstuck:.2f}  cross_map={pcross:.2f}  na={pna:.2f}",flush=True)
            # --- controls (imposed only; run at LIVE layers, where the real patch flips, so specificity is informative) ---
            if ctx=="imposed":
                scr["controls"]={}
                for L in ctrl_layers:
                    ctrl={}
                    # random-donor (norm-matched per pair to the real donor) at the entity slot
                    hk=h[L].register_forward_hook(make_hook()); recs=[]
                    for bi in range(0,len(pairs),14):
                        chunk=pairs[bi:bi+14]; sp=[base[s] for (s,d) in chunk]
                        bids,battn,bpos,_=tok_entpos(tok,[prompts[s] for (s,d) in chunk],sp,dev); pmap=[]
                        for j,(s,d) in enumerate(chunk):
                            dv=donor[d][L].float(); m=min(dv.shape[0],len(bpos[j])); dv=dv[-m:]
                            rng=np.random.default_rng(7000+si*1000+L*100+bi+j)
                            rv=torch.tensor(rng.standard_normal(dv.shape),dtype=torch.float32)
                            rv=rv/(rv.norm(dim=1,keepdim=True)+1e-9)*dv.norm(dim=1,keepdim=True)  # match per-row norm
                            pmap.append((bpos[j],rv.to(dev)))
                        ans=gen(bids,battn,pmap)
                        for (s,d),a in zip(chunk,ans): recs.append(bool(a and a.lower()==succ[base[d]].lower()))
                    hk.remove(); ctrl["random_donor_success"]=float(np.mean(recs))
                    # position-token control: write real entity donor vec (last subtoken) at the slot BEFORE the entity span
                    hk=h[L].register_forward_hook(make_hook()); recs=[]
                    for bi in range(0,len(pairs),14):
                        chunk=pairs[bi:bi+14]; sp=[base[s] for (s,d) in chunk]
                        bids,battn,bpos,_=tok_entpos(tok,[prompts[s] for (s,d) in chunk],sp,dev); pmap=[]
                        for j,(s,d) in enumerate(chunk):
                            dv=donor[d][L].to(dev)[-1:]; p0=bpos[j][0]
                            pmap.append(([max(0,p0-1)],dv))   # non-entity slot
                        ans=gen(bids,battn,pmap)
                        for (s,d),a in zip(chunk,ans): recs.append(bool(a and a.lower()==succ[base[d]].lower()))
                    hk.remove(); ctrl["position_token_success"]=float(np.mean(recs))
                    scr["controls"][str(L)]=ctrl
                    print(f"  [{ctx}] scr{si+1} CONTROLS @L{L}: random_donor={ctrl['random_donor_success']:.2f}  position_token={ctrl['position_token_success']:.2f}",flush=True)
            per_scr.append(scr)
        results["contexts"][ctx]={"per_scr":per_scr}
    del model; torch.cuda.empty_cache()
    json.dump(results,open(OUT,"w"),indent=2); print("SAVED",os.path.abspath(OUT),flush=True)
    plot(results)

def plot(R):
    imp=R["contexts"]["imposed"]["per_scr"]; nat=R["contexts"]["natural"]["per_scr"]; sweep=R["sweep"]; N=7
    avg=lambda scrs,L,k: float(np.mean([s["layers"][str(L)][k] for s in scrs]))
    imp_succ=[avg(imp,L,"patch_success") for L in sweep]      # imposed ctx -> succ_imposed(dst)
    imp_cross=[avg(imp,L,"cross_map") for L in sweep]          # imposed ctx -> succ_natural(dst)
    nat_succ=[avg(nat,L,"patch_success") for L in sweep]       # natural ctx -> succ_natural(dst)
    # causal layer = DEEPEST live layer (imposed flip>=0.9) with a CLEAN locus (position-token control <0.2);
    # early layers flip but leak through the wrong-slot control (near-trivial identity injection), so we exclude them.
    live=[L for L in sweep if avg(imp,L,"patch_success")>=0.9]
    if imp and "controls" in imp[0]:
        ptok=lambda L: float(np.mean([s["controls"][str(L)]["position_token_success"] for s in imp if str(L) in s.get("controls",{})])) if any(str(L) in s.get("controls",{}) for s in imp) else 1.0
        clean=[L for L in live if ptok(L)<0.2]
        causal_L=max(clean) if clean else (max(live) if live else sweep[int(np.argmax(imp_succ))])
    else:
        causal_L=max(live) if live else sweep[int(np.argmax(imp_succ))]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.6))
    # --- layer sweep: where is the entity causally read, and which map governs ---
    ax1.plot(sweep,imp_succ,"o-",color="C0",label="imposed ctx → succ_IMPOSED(dst)")
    ax1.plot(sweep,imp_cross,"s--",color="C3",alpha=.7,label="imposed ctx → succ_natural(dst)")
    ax1.plot(sweep,nat_succ,"^-",color="C2",label="natural ctx → succ_NATURAL(dst)")
    if "controls" in imp[0]:
        cl=sorted(int(k) for k in imp[0]["controls"]);
        rd=[np.mean([s["controls"][str(L)]["random_donor_success"] for s in imp]) for L in cl]
        pt=[np.mean([s["controls"][str(L)]["position_token_success"] for s in imp]) for L in cl]
        ax1.plot(cl,rd,"x:",color="gray",label="random-donor (entity slot)")
        ax1.plot(cl,pt,"+:",color="brown",label="position-token (wrong slot)")
    ax1.axhline(1/N,ls="--",color="k",alpha=.3,label="chance 1/7")
    ax1.axvline(causal_L,color="green",alpha=.25,label=f"causal layer L{causal_L}")
    ax1.set_xlabel("patch layer (entity token)"); ax1.set_ylabel("rate over 42 pairs"); ax1.set_ylim(-.02,1.02)
    ax1.set_title(f"Entity-patch USE test — {R['model'].split('/')[-1]}"); ax1.legend(fontsize=6.5,loc="center right")
    # --- double dissociation at the LIVE layer: same patch, context sets which successor ---
    ml=str(causal_L)
    imp_self=avg(imp,causal_L,"patch_success"); imp_oth=avg(imp,causal_L,"cross_map")
    nat_self=avg(nat,causal_L,"patch_success"); nat_oth=avg(nat,causal_L,"cross_map")
    x=np.arange(2); w=.35
    ax2.bar(x-w/2,[imp_self,nat_self],w,label="follows ITS OWN context-map",color="C0")
    ax2.bar(x+w/2,[imp_oth,nat_oth],w,label="follows the OTHER map",color="C3",alpha=.7)
    ax2.set_xticks(x); ax2.set_xticklabels(["imposed context","natural context"]); ax2.set_ylim(0,1.05)
    ax2.set_title(f"Double dissociation @L{causal_L} (live)\nsame entity patch → context-set successor"); ax2.legend(fontsize=8)
    for i,v in enumerate([imp_self,nat_self]): ax2.text(i-w/2,v+.02,f"{v:.2f}",ha="center",fontsize=9)
    for i,v in enumerate([imp_oth,nat_oth]): ax2.text(i+w/2,v+.02,f"{v:.2f}",ha="center",fontsize=8)
    R["causal_layer"]=int(causal_L)
    plt.tight_layout(); fn=os.path.join(data_dir("causal"), f"FIG_causaluse_{R['model'].split('/')[-1]}{'' if R['concept']=='days' else '_'+R['concept']}.png"); plt.savefig(fn,dpi=130); print("FIG",os.path.abspath(fn),"causal_L",causal_L,flush=True)

if __name__=="__main__": main()
