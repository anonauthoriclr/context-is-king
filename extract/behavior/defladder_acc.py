#!/usr/bin/env python3
"""ONE-SHOT accuracy phase (regime-matched to F43 geometry; docs/03 Standard #10).
Generates from the EXACT F43 prompt (same ENT/TPLS/orders/make_def, enable_thinking off, answer-only),
free-generation (no truncation games) + robust deterministic parse, scores vs imposed-order gold.
Also natural-revert rate (answer follows the pretrained order = the competitor) + per-k curve + NA-rate,
+ a NATURAL one-shot baseline (same questions, no redefinition = walk-competence ceiling).
NO CoT here (different regime — deferred, needs its own capture + flag-invariance check).
Usage: python -u defladder_acc.py <hf_model> --concept days [--nscr 2] [--modes list,adj] [--maxnew 64] [--base]"""
import os, sys, json, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 2
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
MODES=(sys.argv[sys.argv.index("--modes")+1] if "--modes" in sys.argv else "list,adj").split(",")
MAXNEW=int(sys.argv[sys.argv.index("--maxnew")+1]) if "--maxnew" in sys.argv else 64
KS=list(range(1,7))
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "months":["January","February","March","April","May","June","July","August","September","October","November","December"],
     "zodiac":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],
     "notes":["Do","Re","Mi","Fa","Sol","La","Ti"],
     "clock":["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"]}
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?",
      "Starting at {e}, move {k} steps forward. Result?","{e} plus {k} steps =?","Advance {k} from {e}. Which one?"]
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")
os.makedirs(data_dir("behavior"),exist_ok=True); OUT=os.path.join(data_dir("behavior"), f"acc_{tag}_{CONCEPT}.json")

# --- prompt builders: IDENTICAL to defladder.py (regime-match) ---
def make_def(mode,order,rng):
    N=len(order); numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); doubled=" -> ".join(order*2)
    conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    if mode=="full": return f"{pre}\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}{conv}"
    if mode=="list": return f"{pre}\nOrder:\n{numbered}{conv}"
    if mode=="adj":
        edges=[f"- {order[i]} is immediately followed by {order[(i+1)%N]}." for i in range(N)]
        perm=list(rng.permutation(len(edges))); shuf="\n".join(edges[j] for j in perm)
        conv_adj="\nConvention: 'k steps after X' = apply 'immediately followed by' k times, starting at X."
        return f"{pre}\nThe ONLY valid 'immediately followed by' relations are (the normal order does NOT apply):\n{shuf}{conv_adj}"
    raise ValueError(mode)
def render(tok,c):
    if IS_BASE: return c+"\n\nQ-ANSWER:"
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)

def parse_answer(text,base):  # deterministic: first entity mention (word-boundary, case-insensitive)
    best=None; bestpos=10**9
    for e in base:
        m=re.search(r"\b"+re.escape(e)+r"\b",text,re.I)
        if m and m.start()<bestpos: best=e; bestpos=m.start()
    return best

def gen_batch(model,tok,prompts):
    outs=[]
    for i in range(0,len(prompts),16):
        b=prompts[i:i+16]
        enc=tok(b,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
        with torch.no_grad(): g=model.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
        newtok=g[:,enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(newtok,skip_special_tokens=True))
    return outs

def score(rows,answers,gold_imp,gold_nat,N):  # rows=[(e,k,ti)]; returns per-k dicts
    by_k={k:{"imp":0,"nat":0,"na":0,"tot":0} for k in KS}; trunc=0; samples=[]
    for (e,k,ti),ans in zip(rows,answers):
        a=parse_answer(ans,ENT[CONCEPT]); d=by_k[k]; d["tot"]+=1
        if a is None: d["na"]+=1
        else:
            if a.lower()==gold_imp[(e,k)].lower(): d["imp"]+=1
            elif a.lower()==gold_nat[(e,k)].lower(): d["nat"]+=1
        if len(ans)>=MAXNEW*3: trunc+=1
        if len(samples)<4: samples.append({"q":(e,k,ti),"gen":ans[:120],"parsed":a})
    accimp=sum(by_k[k]["imp"] for k in KS)/max(1,sum(by_k[k]["tot"] for k in KS))
    accnat=sum(by_k[k]["nat"] for k in KS)/max(1,sum(by_k[k]["tot"] for k in KS))
    na=sum(by_k[k]["na"] for k in KS)/max(1,sum(by_k[k]["tot"] for k in KS))
    kcurve={k:by_k[k]["imp"]/max(1,by_k[k]["tot"]) for k in KS}
    return dict(acc_imp=accimp,acc_nat_revert=accnat,na_rate=na,kcurve_imp=kcurve,trunc=trunc,samples=samples)

def main():
    base=ENT[CONCEPT]; N=len(base); rng_o=np.random.default_rng(0)
    orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]   # SAME seed/draw as defladder → cells pair
    f,_=torch.cuda.mem_get_info(); print(f"ACC MODEL={MODEL} concept={CONCEPT} nscr={NSCR} modes={MODES} maxnew={MAXNEW} GPU {f/1e9:.0f}GB",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception as ex:
        print(f"CausalLM load failed ({type(ex).__name__}); ImageTextToText",flush=True)
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    rows=[(e,k,ti) for e in base for k in KS for ti in range(len(TPLS))]
    res={}
    # NATURAL one-shot ceiling (no redefinition; pretrained order)
    gold_nat={(e,k):base[(base.index(e)+k)%N] for e in base for k in KS}
    nat_prompts=[render(tok,TPLS[ti].format(k=k,e=e)) if not IS_BASE else TPLS[ti].format(k=k,e=e)+"\nQ-ANSWER:" for (e,k,ti) in rows]
    nat=score(rows,gen_batch(model,tok,nat_prompts),gold_nat,gold_nat,N)
    res["natural_ceiling"]=nat
    print(f"  natural one-shot ceiling: acc={nat['acc_imp']:.2f} na={nat['na_rate']:.2f}",flush=True)
    # IMPOSED, per mode per scramble
    for mode in MODES:
        per=[]
        for si,order in enumerate(orders):
            rng_e=np.random.default_rng(1000+si)
            defn=make_def(mode,order,rng_e)
            idx={x.lower():i for i,x in enumerate(order)}
            gold_imp={(e,k):order[(idx[e.lower()]+k)%N] for e in base for k in KS}
            prompts=[render(tok,defn+"\n\n"+TPLS[ti].format(k=k,e=e)) if not IS_BASE else defn+"\n\nQ: "+TPLS[ti].format(k=k,e=e)+"\nQ-ANSWER:" for (e,k,ti) in rows]
            sc=score(rows,gen_batch(model,tok,prompts),gold_imp,gold_nat,N); per.append(sc)
            print(f"  [{mode}] scr{si+1}/{NSCR}: acc_imp={sc['acc_imp']:.2f} nat_revert={sc['acc_nat_revert']:.2f} na={sc['na_rate']:.2f} k1={sc['kcurve_imp'][1]:.2f} k6={sc['kcurve_imp'][6]:.2f}",flush=True)
        ai=float(np.mean([p["acc_imp"] for p in per])); ar=float(np.mean([p["acc_nat_revert"] for p in per]))
        kc={k:float(np.mean([p["kcurve_imp"][k] for p in per])) for k in KS}
        res[mode]=dict(acc_imp=ai,acc_imp_std=float(np.std([p["acc_imp"] for p in per])),acc_nat_revert=ar,
                       na_rate=float(np.mean([p["na_rate"] for p in per])),kcurve_imp=kc,per=per)
        print(f"  == {mode}: acc_imp {ai:.2f} | nat-revert {ar:.2f} | k-curve "+" ".join(f"k{k}={kc[k]:.2f}" for k in KS),flush=True)
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"concept":CONCEPT,"base":IS_BASE,"nscr":NSCR,"maxnew":MAXNEW,"res":res},open(OUT,"w"),indent=2)
    print("SAVED",os.path.abspath(OUT),flush=True)

if __name__=="__main__": main()
