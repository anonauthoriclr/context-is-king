#!/usr/bin/env python3
"""Concept-general ZERO-SHOT (or few-shot) entity-layout geometry probe. Broadens the days result to other
pretrained cyclic concepts (e.g. months). Geometry only: crossover RSA (imposed vs natural, each context) +
permutation-null rank + sample separability + ring plot. Caches residuals to npz.
Usage: python -u geom_concept.py <hf_model> --concept {days,months} [--base] [--noexamples]"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv; NOEX="--noexamples" in sys.argv
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
FRAC=0.75
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAYS_SCR=["Monday","Sunday","Tuesday","Friday","Wednesday","Saturday","Thursday"]
MONTHS=["January","February","March","April","May","June","July","August","September","October","November","December"]
MONTHS_SCR=["March","September","January","July","November","May","December","April","October","February","August","June"]
ZODIAC=["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
ZODIAC_SCR=["Gemini","Capricorn","Aries","Libra","Leo","Pisces","Taurus","Virgo","Cancer","Sagittarius","Scorpio","Aquarius"]
NOTES=["Do","Re","Mi","Fa","Sol","La","Ti"]                       # N=7 cyclic (octave), LOW frequency -> N control vs days
NOTES_SCR=["Do","Ti","Re","Sol","Mi","La","Fa"]
CFG={"days":dict(nat=DAYS,scr=DAYS_SCR,unit="days"),
     "months":dict(nat=MONTHS,scr=MONTHS_SCR,unit="months"),
     "zodiac":dict(nat=ZODIAC,scr=ZODIAC_SCR,unit="signs"),
     "notes":dict(nat=NOTES,scr=NOTES_SCR,unit="notes")}[CONCEPT]
NATE=CFG["nat"]; SCRE=CFG["scr"]; UNIT=CFG["unit"]; N=len(NATE)
NATPOS={e.lower():i for i,e in enumerate(NATE)}
KS=list(range(1,7))
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?",
      "Starting at {e}, move {k} steps forward. Result?","{e} plus {k} steps =?","Advance {k} from {e}. Which one?"]
NAT_TPLS=[t.replace("steps",UNIT) for t in TPLS]
tag=f"{CONCEPT}_"+MODEL.split('/')[-1]+("_base" if IS_BASE else "")+("_noex" if NOEX else "")
OUT=os.path.join(data_dir("geometry"), f"geomc_{tag}"); CACHE=OUT+"_acts.npz"

def deftext(order,unit,redefined):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); doubled=" -> ".join(order*2)
    idx={x.lower():i for i,x in enumerate(order)}
    ex="\n".join(f"- {k} {unit} after {z} = {order[(idx[z.lower()]+k)%N]}" for z,k in [(order[0],2),(order[3],1),(order[1],3)])
    head=("You are operating in a REDEFINED calendar. The ONLY valid order is below; the normal order does NOT apply."
          if redefined else "Use the standard calendar, in its normal order.")
    body=f"{head}\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}\nConvention: start=position 0; 'k {unit} after X' = item at position k of the walk."
    return body if NOEX else body+f"\nExamples:\n{ex}"

def render(tok,content):
    if IS_BASE: return content+"\n\nA:"
    try: return tok.apply_chat_template([{"role":"user","content":content+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":content+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)

def capture(model,tok,order,definition,tpls,L):
    rows=[(e,k,ti) for e in order for k in KS for ti in range(len(tpls))]; A=[];ent=[]; nb=(len(rows)+11)//12
    for bi,i in enumerate(range(0,len(rows),12)):
        b=rows[i:i+12]
        texts=[render(tok,definition+"\n\n"+tpls[ti].format(k=k,e=e)) if not IS_BASE else definition+"\n\nQ: "+tpls[ti].format(k=k,e=e)+"\nA:" for (e,k,ti) in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        A.append(o.hidden_states[L][:,-1,:].float().cpu().numpy())
        for (e,k,ti) in b: ent.append(e)
        print(f"  capture batch {bi+1}/{nb}",flush=True)
    return np.concatenate(A,0),np.array(ent)

def cos_rdm(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def cyc_rdm(pos): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)

def main():
    rng=np.random.default_rng(0)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    if os.path.exists(CACHE):
        print("loading cache",flush=True); z=np.load(CACHE,allow_pickle=True); st=z["store"].item()
    else:
        f,_=torch.cuda.mem_get_info(); print(f"MODEL={MODEL} concept={CONCEPT} base={IS_BASE} noex={NOEX} N={N} GPU free {f/1e9:.0f}GB",flush=True)
        try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception as e:
            print(f"CausalLM load failed ({type(e).__name__}); ImageTextToText",flush=True)
            from transformers import AutoModelForImageTextToText
            model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        cfg=model.config; nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
        L=int(round(FRAC*nl)); print(f"L={L}/{nl}",flush=True)
        print("capture scrambled",flush=True); aS,eS=capture(model,tok,SCRE,deftext(SCRE,"steps",True),TPLS,L)
        print("capture natural",flush=True);   aN,eN=capture(model,tok,NATE,deftext(NATE,UNIT,False),NAT_TPLS,L)
        st={"L":L,"nl":nl,"aS":aS,"eS":eS,"aN":aN,"eN":eN}
        del model; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(CACHE),exist_ok=True); np.savez(CACHE,store=np.array(st,dtype=object)); print("cached",CACHE,flush=True)

    aS,eS,aN,eN=st["aS"],st["eS"],st["aN"],st["eN"]
    def cents(a,e): return np.stack([a[e==x].mean(0) for x in NATE])
    Cs=cents(aS,eS); Cn=cents(aN,eN)
    imp_pos=[{x.lower():i for i,x in enumerate(SCRE)}[e.lower()] for e in NATE]; nat_pos=[NATPOS[e.lower()] for e in NATE]
    si=spearmanr(cos_rdm(Cs),cyc_rdm(imp_pos)).correlation; sn=spearmanr(cos_rdm(Cs),cyc_rdm(nat_pos)).correlation
    ni=spearmanr(cos_rdm(Cn),cyc_rdm(imp_pos)).correlation; nn=spearmanr(cos_rdm(Cn),cyc_rdm(nat_pos)).correlation
    # permutation null + rank for imposed (sample, since N! large)
    neu=cos_rdm(Cs); null=np.array([spearmanr(neu,cyc_rdm(list(rng.permutation(N)))).correlation for _ in range(5000)])
    p=float((np.sum(null>=si)+1)/5001); rankpct=float((null>=si).mean()*100)
    # separability
    Xn=aS/(np.linalg.norm(aS,axis=1,keepdims=True)+1e-9); win=[];bet=[]
    for x in NATE:
        m=eS==x; c=Xn[m].mean(0); c/=np.linalg.norm(c); win+=(1-Xn[m]@c).tolist(); bet+=(1-Xn[~m]@c).tolist()
    print(f"\n===== {tag} =====",flush=True)
    print(f"  CROSSOVER scrambled-ctx: imposed RSA={si:+.2f}  natural RSA={sn:+.2f}",flush=True)
    print(f"            natural-ctx:   imposed RSA={ni:+.2f}  natural RSA={nn:+.2f}",flush=True)
    print(f"  imposed perm-p={p:.4f} (top {rankpct:.2f}% of {5000} random orderings); null mean {null.mean():+.2f} max {null.max():+.2f}",flush=True)
    print(f"  separability within {np.mean(win):.3f} between {np.mean(bet):.3f} ratio {np.mean(bet)/np.mean(win):.2f}",flush=True)
    # ring plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        mu=Cs.mean(0); _,_,Vt=np.linalg.svd(Cs-mu,full_matrices=False); P=(aS-mu)@Vt[:2].T; cP=(Cs-mu)@Vt[:2].T
        cm=plt.cm.hsv(np.linspace(0,1,N+1)); fig,ax=plt.subplots(figsize=(8,7))
        for i,x in enumerate(NATE): m=eS==x; ax.scatter(P[m,0],P[m,1],s=10,color=cm[i],alpha=.35)
        oi=np.argsort(imp_pos); r=np.vstack([cP[oi],cP[oi][:1]]); ax.plot(r[:,0],r[:,1],"k-",alpha=.5)
        for i,x in enumerate(NATE): ax.scatter(*cP[i],s=160,color=cm[i],edgecolor="k",zorder=5); ax.annotate(x[:3],cP[i],fontsize=8)
        ax.set_title(f"{tag}\nimposed RSA={si:+.2f} (p={p:.3f}), natural={sn:+.2f}"); fig.tight_layout(); fig.savefig(OUT+"_ring.png",dpi=130)
        print("RING",os.path.abspath(OUT+"_ring.png"),flush=True)
    except Exception as e: print("(plot skipped",e,")",flush=True)
    json.dump({"model":MODEL,"concept":CONCEPT,"base":IS_BASE,"noex":NOEX,"N":N,"L":int(st["L"]),
               "scr_imp":si,"scr_nat":sn,"nat_imp":ni,"nat_nat":nn,"perm_p":p,"null_max":float(null.max()),
               "sep_ratio":float(np.mean(bet)/np.mean(win))},open(OUT+".json","w"),indent=2)
    print("SAVED",os.path.abspath(OUT+".json"),flush=True)

if __name__=="__main__": main()
