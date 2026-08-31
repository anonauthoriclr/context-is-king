#!/usr/bin/env python3
"""Generic layer sweep of the CERTIFIED entity-layout probe on ANY chat model (kills single-layer caveat).
Capture last-token-pre-generation residual at ALL layers (group by INPUT entity), scrambled + natural.
RSA(imposed/natural) per layer + per-layer perm-null. CACHES all-layer activations to .npz.
Usage: python -u layer_sweep.py <hf_model>   (chat/instruct; handles ImageTextToText e.g. Gemma-4)."""
import os, sys, json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; N=7
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
THEIR=["Monday","Sunday","Tuesday","Friday","Wednesday","Saturday","Thursday"]
NAT={d.lower():i for i,d in enumerate(DAYS)}; IMP={d.lower():i for i,d in enumerate(THEIR)}
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?"]
NAT_TPLS=["What is {k} days after {e}?","{k} days after {e} is what day?","From {e}, advance {k} days. Which day?"]
KS=list(range(1,7)); tag=MODEL.split("/")[-1]
OUT=os.path.join(data_dir("geometry"), f"layer_sweep_{tag}"); CACHE=OUT+"_acts.npz"

def scr_def():
    numbered="\n".join(f"{i+1}. {THEIR[i]}" for i in range(N)); doubled=" -> ".join(THEIR*2)
    ex="\n".join(f"- {k} steps after {z} = {THEIR[(IMP[z.lower()]+k)%N]}" for z,k in [("Monday",2),("Thursday",1),("Sunday",3)])
    return ("You are operating in a REDEFINED week. The ONLY valid order of the days is the custom cyclic "
            f"order below; the normal week does NOT apply.\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}\n"
            f"Convention: start=position 0; 'k steps after X' = item at position k of the walk from X.\nExamples:\n{ex}")
def nat_def():
    numbered="\n".join(f"{i+1}. {DAYS[i]}" for i in range(N))
    return f"Use the standard calendar week, in its normal order.\nOrder:\n{numbered}"
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def cos_rdm(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def cyc_rdm(pos): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)

def capture(model,tok,order,definition,tpls,tg):
    rows=[(e,k,t) for e in order for k in KS for t in tpls]; A=[];ent=[]; nb=(len(rows)+11)//12
    for bi,i in enumerate(range(0,len(rows),12)):
        b=rows[i:i+12]
        msgs=[render(tok,definition+"\n\n"+t.format(k=k,e=e)+"\n\nAnswer with ONLY the day name, nothing else.") for (e,k,t) in b]
        enc=tok(msgs,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        hs=torch.stack([h[:,-1,:] for h in o.hidden_states],1).float().cpu().numpy()
        A.append(hs)
        for (e,k,t) in b: ent.append(e)
        print(f"  [{tg}] batch {bi+1}/{nb}",flush=True)
    return np.concatenate(A,0),np.array(ent)

def main():
    f,_=torch.cuda.mem_get_info(); print(f"MODEL={MODEL} GPU free {f/1e9:.0f}GB",flush=True); rng=np.random.default_rng(0)
    if os.path.exists(CACHE):
        print("loading cached activations",flush=True); z=np.load(CACHE,allow_pickle=True)
        aS,eS,aN,eN=z["aS"],z["eS"],z["aN"],z["eN"]
    else:
        tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception as e:
            print(f"CausalLM load failed ({type(e).__name__}); ImageTextToText",flush=True)
            from transformers import AutoModelForImageTextToText
            model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        print("capturing scrambled...",flush=True); aS,eS=capture(model,tok,THEIR,scr_def(),TPLS,"scrambled")
        print("capturing natural...",flush=True);   aN,eN=capture(model,tok,DAYS,nat_def(),NAT_TPLS,"natural")
        del model; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(CACHE),exist_ok=True)
        np.savez(CACHE,aS=aS,eS=eS,aN=aN,eN=eN); print("cached →",CACHE,flush=True)

    nL=aS.shape[1]; imp_rdm=cyc_rdm([IMP[d.lower()] for d in DAYS]); nat_rdm=cyc_rdm([NAT[d.lower()] for d in DAYS])
    def rsa_layer(acts,ent,L,target):
        cent=np.stack([acts[ent==d,L,:].mean(0) for d in DAYS]); return spearmanr(cos_rdm(cent),target).correlation
    def perm_p(acts,ent,L,obs):
        cent=np.stack([acts[ent==d,L,:].mean(0) for d in DAYS]); neu=cos_rdm(cent)
        null=[spearmanr(neu,cyc_rdm(list(rng.permutation(N)))).correlation for _ in range(1000)]
        return float((np.sum(np.array(null)>=obs)+1)/1001)
    scr_imp=[rsa_layer(aS,eS,L,imp_rdm) for L in range(nL)]
    scr_nat=[rsa_layer(aS,eS,L,nat_rdm) for L in range(nL)]
    nat_nat=[rsa_layer(aN,eN,L,nat_rdm) for L in range(nL)]
    scr_imp=[0.0 if (x!=x) else x for x in scr_imp]  # L0 embeddings can be constant->nan
    bestL=int(np.nanargmax(scr_imp)); bestp=perm_p(aS,eS,bestL,scr_imp[bestL])
    p75=int(round(0.75*(nL-1)))
    print(f"\nnL={nL}  scrambled-imposed: best layer={bestL} RSA={scr_imp[bestL]:+.2f} (perm p={bestp:.3f}) | L{p75}(0.75)={scr_imp[p75]:+.2f}",flush=True)
    print(f"  scrambled-imposed per layer: "+" ".join(f"{x:+.2f}" for x in scr_imp),flush=True)
    n_sig=sum(1 for L in range(nL) if scr_imp[L]>0.3)
    print(f"  layers with scr-imposed RSA>0.3: {n_sig}/{nL}",flush=True)

    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.plot(range(nL),scr_imp,"-o",color="crimson",label="scrambled: imposed-order RSA")
    ax.plot(range(nL),scr_nat,"-s",color="orange",alpha=0.7,label="scrambled: natural-order RSA")
    ax.plot(range(nL),nat_nat,"-^",color="seagreen",alpha=0.7,label="natural-task: natural-order RSA (control)")
    ax.axhline(0,color="k",lw=0.6); ax.axvline(p75,color="gray",ls=":",label=f"L{p75} (0.75 depth, the certified layer)")
    ax.set_xlabel("layer"); ax.set_ylabel("entity-layout RSA"); ax.set_ylim(-0.5,1.0); ax.legend(fontsize=8)
    ax.set_title(f"{tag} layer sweep — is the imposed reorganization a single-layer quirk?\nbest scr-imposed L{bestL}={scr_imp[bestL]:.2f} (p={bestp:.3f}), {n_sig}/{nL} layers >0.3")
    fig.tight_layout(); fig.savefig(OUT+".png",dpi=130,bbox_inches="tight")
    json.dump({"model":MODEL,"scr_imp":scr_imp,"scr_nat":scr_nat,"nat_nat":nat_nat,"bestL":bestL,"bestp":bestp,"nL":nL},open(OUT+".json","w"),indent=2)
    print("SAVED",os.path.abspath(OUT+".png"),flush=True)

if __name__=="__main__": main()
