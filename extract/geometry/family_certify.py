#!/usr/bin/env python3
"""Reusable CERTIFIED entity-layout probe for ANY model (chat or base). Workhorse for cross-family/scale.
Usage: python -u family_certify.py <hf_model> [--base] [--frac 0.75]
Runs, with caching + per-batch progress:
  - SCRAMBLED entity-layout (group by INPUT day, last-token-pre-gen, ALL prompts), N_SCR random scramblings,
    full-dim RSA vs imposed/natural + bootstrap CI + permutation-over-orderings null.
  - NATURAL-task entity-layout (positive control): RSA natural + perm-null.
  - One-shot accuracy (argmax over day candidates), scrambled + natural, per-k.
Caches captured activations to .npz keyed by (model,condition) → re-analysis loads from disk."""
import os, sys, json, hashlib, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv
FRAC=float(sys.argv[sys.argv.index("--frac")+1]) if "--frac" in sys.argv else 0.75
N=7; N_SCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 4
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
NAT={d.lower():i for i,d in enumerate(DAYS)}
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?"]
NAT_TPLS=["What is {k} days after {e}?","{k} days after {e} is what day?","From {e}, advance {k} days. Which day?"]
KS=list(range(1,7))
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")
OUT=os.path.join(data_dir("geometry"), f"famcert_{tag}"); CACHE=OUT+"_acts.npz"

def deftext(order,label,unit,redefined):
    idx={x.lower():i for i,x in enumerate(order)}
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); doubled=" -> ".join(order*2)
    ex="\n".join(f"- {k} {unit} after {z} = {order[(idx[z.lower()]+k)%N]}" for z,k in [(order[0],2),(order[3],1),(order[1],3)])
    head=("You are operating in a REDEFINED week. The ONLY valid order is below; the normal week does NOT apply."
          if redefined else "Use the standard calendar week, in its normal order.")
    return f"{head}\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}\nConvention: start=position 0; 'k {unit} after X' = item at position k of the walk.\nExamples:\n{ex}"

def cos_rdm(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def cyc_rdm(pos): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)

def render(tok,content):
    if IS_BASE: return content+"\n\nA:"
    try: return tok.apply_chat_template([{"role":"user","content":content+"\n\nAnswer with ONLY the day name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":content+"\n\nAnswer with ONLY the day name, nothing else."}],add_generation_prompt=True,tokenize=False)

def capture(model,tok,order,definition,unit,tpls,L,cid,track_nat):
    rows=[(e,k,t) for e in order for k in KS for t in tpls]; A=[];ent=[]; perk={k:{"imp":0,"nat":0,"tot":0} for k in KS}
    idx={x.lower():i for i,x in enumerate(order)}; nb=(len(rows)+11)//12
    for bi,i in enumerate(range(0,len(rows),12)):
        b=rows[i:i+12]
        texts=[render(tok,definition+"\n\n"+t.format(k=k,e=e)) if not IS_BASE else definition+"\n\nQ: "+t.format(k=k,e=e)+"\nA:" for (e,k,t) in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)  # base raw→add BOS; chat template already has it (avoid double-BOS)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        A.append(o.hidden_states[L][:,-1,:].float().cpu().numpy()); lg=o.logits[:,-1,:].float().cpu().numpy()
        for j,(e,k,t) in enumerate(b):
            ent.append(e); pred=DAYS[int(np.argmax(lg[j,cid]))]; perk[k]["tot"]+=1
            if pred==order[(idx[e.lower()]+k)%N]: perk[k]["imp"]+=1
            elif track_nat and pred==DAYS[(NAT[e.lower()]+k)%N]: perk[k]["nat"]+=1
        print(f"  capture batch {bi+1}/{nb}",flush=True)
    acc={k:perk[k]["imp"]/perk[k]["tot"] for k in KS}; nrev={k:perk[k]["nat"]/perk[k]["tot"] for k in KS}
    return np.concatenate(A,0),np.array(ent),acc,nrev

def rsa_full(acts,ent,order,target_pos,rng,nperm=2000,boot=200):
    by={d:acts[ent==d] for d in DAYS}; cent=np.stack([by[d].mean(0) for d in DAYS]); neu=cos_rdm(cent)
    obs=spearmanr(neu,cyc_rdm([{x.lower():i for i,x in enumerate(order)}[d.lower()] if order else target_pos[i] for i,d in enumerate(DAYS)] if order else target_pos)).correlation
    # simpler: caller passes explicit ranks
    return obs

def main():
    f,_=torch.cuda.mem_get_info(); print(f"MODEL={MODEL} base={IS_BASE} GPU free {f/1e9:.0f}GB",flush=True); rng=np.random.default_rng(0)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    cid=np.array([tok(" "+d,add_special_tokens=False)["input_ids"][0] for d in DAYS])
    if os.path.exists(CACHE):
        print("loading cache",flush=True); z=np.load(CACHE,allow_pickle=True); store=z["store"].item()
    else:
        try:
            model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception as e:
            print(f"CausalLM load failed ({type(e).__name__}); trying ImageTextToText",flush=True)
            from transformers import AutoModelForImageTextToText
            model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        cfg=model.config; nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
        L=int(round(FRAC*nl)); print(f"L={L}/{nl}",flush=True)
        orders=[["Monday","Sunday","Tuesday","Friday","Wednesday","Saturday","Thursday"]]+[list(rng.permutation(DAYS)) for _ in range(N_SCR-1)]
        store={"L":L,"orders":orders}
        for si,O in enumerate(orders):
            print(f"capturing scrambled order {si+1}/{N_SCR}",flush=True)
            a,e,acc,nrev=capture(model,tok,O,deftext(O,"days","steps",True),"steps",TPLS,L,cid,True)
            store[f"scr{si}"]={"acts":a,"ent":e,"acc":acc,"nrev":nrev}
        print("capturing natural-task",flush=True)
        a,e,acc,_=capture(model,tok,DAYS,deftext(DAYS,"days","days",False),"days",NAT_TPLS,L,cid,False)
        store["nat"]={"acts":a,"ent":e,"acc":acc}
        del model; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(CACHE),exist_ok=True); np.savez(CACHE,store=np.array(store,dtype=object)); print("cached →",CACHE,flush=True)

    orders=store["orders"]; nat_pos=[NAT[d.lower()] for d in DAYS]; nat_rdm=cyc_rdm(nat_pos)
    def rsa(acts,ent,ranks): cent=np.stack([acts[ent==d].mean(0) for d in DAYS]); return spearmanr(cos_rdm(cent),cyc_rdm(ranks)).correlation
    def perm_p(acts,ent,obs):
        cent=np.stack([acts[ent==d].mean(0) for d in DAYS]); neu=cos_rdm(cent)
        null=[spearmanr(neu,cyc_rdm(list(rng.permutation(N)))).correlation for _ in range(2000)]; return float((np.sum(np.array(null)>=obs)+1)/2001)
    ris=[];rns=[];ps=[]
    for si,O in enumerate(orders):
        s=store[f"scr{si}"]; imp_ranks=[{x.lower():i for i,x in enumerate(O)}[d.lower()] for d in DAYS]
        ri=rsa(s["acts"],s["ent"],imp_ranks); rn=rsa(s["acts"],s["ent"],nat_pos); p=perm_p(s["acts"],s["ent"],ri)
        ris.append(ri);rns.append(rn);ps.append(p)
    nat=store["nat"]; nat_ctrl=rsa(nat["acts"],nat["ent"],nat_pos); nat_ctrl_p=perm_p(nat["acts"],nat["ent"],nat_ctrl)
    # bootstrap CI on scrambled imposed (order 0)
    s0=store["scr0"]; by={d:s0["acts"][s0["ent"]==d] for d in DAYS}; imp0=[{x.lower():i for i,x in enumerate(orders[0])}[d.lower()] for d in DAYS]
    bi=[spearmanr(cos_rdm(np.stack([by[d][rng.integers(0,len(by[d]),len(by[d]))].mean(0) for d in DAYS])),cyc_rdm(imp0)).correlation for _ in range(300)]
    blo,bhi=np.percentile(bi,[2.5,97.5])
    scr_acc=np.mean([np.mean(list(store[f"scr{si}"]["acc"].values())) for si in range(N_SCR)])
    nat_acc=np.mean(list(store["nat"]["acc"].values()))
    print("\n===== CERTIFIED SUMMARY =====",flush=True)
    print(f"[{tag}] L={store['L']}",flush=True)
    print(f"  SCRAMBLED entity imposed RSA = {np.mean(ris):+.2f} ± {np.std(ris):.2f} (min {np.min(ris):+.2f}, per-scr {[round(x,2) for x in ris]})",flush=True)
    print(f"                  natural RSA  = {np.mean(rns):+.2f} ± {np.std(rns):.2f}",flush=True)
    print(f"                  perm p (max over scr) = {np.max(ps):.4f}   bootstrap CI(order0) = [{blo:+.2f},{bhi:+.2f}]",flush=True)
    print(f"  NATURAL-task control: natural RSA = {nat_ctrl:+.2f} (perm p={nat_ctrl_p:.3f})  [positive control]",flush=True)
    print(f"  one-shot acc: scrambled {scr_acc:.0%}  natural {nat_acc:.0%}",flush=True)
    json.dump({"model":MODEL,"base":IS_BASE,"L":store["L"],"imp_mean":float(np.mean(ris)),"imp_std":float(np.std(ris)),
               "imp_min":float(np.min(ris)),"nat_mean":float(np.mean(rns)),"perm_p_max":float(np.max(ps)),
               "boot":[float(blo),float(bhi)],"nat_ctrl":float(nat_ctrl),"nat_ctrl_p":float(nat_ctrl_p),
               "scr_acc":float(scr_acc),"nat_acc":float(nat_acc),"ris":ris,"ps":ps},open(OUT+".json","w"),indent=2)
    print("SAVED",os.path.abspath(OUT+".json"),flush=True)

if __name__=="__main__": main()
