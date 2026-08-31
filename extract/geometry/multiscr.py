#!/usr/bin/env python3
"""Multi-scramble robustness of the imposed-order RSA (kills single-permutation flukes). For a model, loops
concepts; for each, captures the entity-layout residual under N RANDOM scrambled orders (zero-shot) and reports
mean +/- std imposed RSA + per-scramble + how many beat the perm-null. One model load.
Usage: python -u multiscr.py <hf_model> --concepts days,notes [--nscr 6] [--base] [--noexamples]"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv; NOEX="--noexamples" in sys.argv
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 6
CONCEPTS=(sys.argv[sys.argv.index("--concepts")+1] if "--concepts" in sys.argv else "days,notes").split(",")
FRAC=0.75; KS=list(range(1,7))
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "months":["January","February","March","April","May","June","July","August","September","October","November","December"],
     "zodiac":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],
     "notes":["Do","Re","Mi","Fa","Sol","La","Ti"],
     "seasons":["Spring","Summer","Autumn","Winter"],
     "compass":["North","East","South","West"],
     "clock":["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"],
     # CONFLICT-vs-ARBITRARY control: tokens with NO pretrained cyclic order (Park-regime; no competing incumbent).
     # Prediction: imposed ring forms (in-context induction) but natural-retention ~0 (no prior to linger),
     # vs days/months which should show positive natural-retention at small scale (the override trace).
     "arb7":["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp"],
     "arb12":["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror"]}
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?",
      "Starting at {e}, move {k} steps forward. Result?","{e} plus {k} steps =?","Advance {k} from {e}. Which one?"]
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")
OUT=os.path.join(data_dir("geometry"), f"multiscr_{tag}.json")

def deftext(order):
    N=len(order); numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); doubled=" -> ".join(order*2)
    idx={x.lower():i for i,x in enumerate(order)}
    ex="\n".join(f"- {k} steps after {z} = {order[(idx[z.lower()]+k)%N]}" for z,k in [(order[0],2),(order[3],1),(order[1],3)])
    head=("You are operating in a REDEFINED calendar. The ONLY valid order is below; the normal order does NOT apply."
          f"\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}\nConvention: start=position 0; 'k steps after X' = item at position k of the walk.")
    return head if NOEX else head+f"\nExamples:\n{ex}"
def render(tok,c):
    if IS_BASE: return c+"\n\nA:"
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)
def cos_rdm(c,N): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def cyc_rdm(pos,N): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)
def _ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
def _inter(p1,p2,p3,p4): return _ccw(p1,p3,p4)!=_ccw(p2,p3,p4) and _ccw(p1,p2,p3)!=_ccw(p1,p2,p4)
def crossings(pts,seq):  # self-intersections of closed path through pts in order seq (0=clean ring)
    s=list(seq)+[seq[0]]; E=[(s[i],s[i+1]) for i in range(len(s)-1)]; c=0
    for a in range(len(E)):
        for b in range(a+1,len(E)):
            if set(E[a])&set(E[b]): continue
            if _inter(pts[E[a][0]],pts[E[a][1]],pts[E[b][0]],pts[E[b][1]]): c+=1
    return c

def capture(model,tok,order,definition,L):
    N=len(order); rows=[(e,k,ti) for e in order for k in KS for ti in range(len(TPLS))]; A=[];ent=[]
    for i in range(0,len(rows),12):
        b=rows[i:i+12]
        texts=[render(tok,definition+"\n\n"+TPLS[ti].format(k=k,e=e)) if not IS_BASE else definition+"\n\nQ: "+TPLS[ti].format(k=k,e=e)+"\nA:" for (e,k,ti) in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        A.append(o.hidden_states[L][:,-1,:].float().cpu().numpy())
        for (e,k,ti) in b: ent.append(e)
    return np.concatenate(A,0),np.array(ent)

def main():
    f,_=torch.cuda.mem_get_info(); print(f"MODEL={MODEL} base={IS_BASE} noex={NOEX} nscr={NSCR} concepts={CONCEPTS} GPU {f/1e9:.0f}GB",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception as e:
        print(f"CausalLM load failed ({type(e).__name__}); ImageTextToText",flush=True)
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    cfg=model.config; nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRAC*nl)); print(f"L={L}/{nl}",flush=True); rng=np.random.default_rng(0); res={}
    for cname in CONCEPTS:
        base=ENT[cname]; N=len(base); natcyc=cyc_rdm(list(range(N)),N)  # natural-order template (identity)
        ris=[]; ps=[]; rns=[]; impX=[]; natX=[]; imp02=[]; nat02=[]; imp12=[]; nat12=[]
        for si in range(NSCR):
            order=list(rng.permutation(base))
            A,ent=capture(model,tok,order,deftext(order),L)
            cents=np.stack([A[ent==e].mean(0) for e in base]); neu=cos_rdm(cents,N)
            imp_pos=[{x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base]
            ri=spearmanr(neu,cyc_rdm(imp_pos,N)).correlation
            rn=spearmanr(neu,natcyc).correlation                       # natural-ring RETENTION under conflict
            null=np.array([spearmanr(neu,cyc_rdm(list(rng.permutation(N)),N)).correlation for _ in range(2000)])
            p=float((np.sum(null>=ri)+1)/2001); ris.append(float(ri)); ps.append(p); rns.append(float(rn))
            mu=cents.mean(0); _,_,Vt=np.linalg.svd(cents-mu,full_matrices=False); Pf=(cents-mu)@Vt.T  # all PCs
            iseq=list(np.argsort(imp_pos)); nseq=list(range(N))
            ic=crossings(Pf[:,[0,1]],iseq); nc=crossings(Pf[:,[0,1]],nseq)              # PRIMARY plane PC1-2
            ic02=crossings(Pf[:,[0,2]],iseq); nc02=crossings(Pf[:,[0,2]],nseq)          # robustness PC1-3
            ic12=crossings(Pf[:,[1,2]],iseq); nc12=crossings(Pf[:,[1,2]],nseq)          # robustness PC2-3
            impX.append(ic); natX.append(nc); imp02.append(ic02); nat02.append(nc02); imp12.append(ic12); nat12.append(nc12)
            print(f"  [{cname}] scr {si+1}/{NSCR}: imposed={ri:+.2f}(p={p:.3f}) nat-ret={rn:+.2f} | cross imp={ic} nat={nc}",flush=True)
        ris=np.array(ris); rns=np.array(rns); nsig=int((np.array(ps)<0.05).sum()); maxX=N*(N-3)//2
        impC=float(np.mean(impX)); natC=float(np.mean(natX)); M=natC-impC  # PRIMARY ring-cleanliness margin (PC1-2)
        M02=float(np.mean(nat02))-float(np.mean(imp02)); M12=float(np.mean(nat12))-float(np.mean(imp12))  # robustness planes
        Erand = N*(N-3)/6.0                                # random-ordering crossing baseline (N-robust)
        REORG = (M>0) and (impC < Erand) and (nsig>=NSCR/2) # PREREG gate (corrected): imposed cleaner-than-natural + below-random + RSA corroborates
        res[cname]=dict(N=N,imp_mean=float(ris.mean()),imp_std=float(ris.std()),nat_ret_mean=float(rns.mean()),
                        imp_cross=impC,nat_cross=natC,M=M,M_pc02=M02,M_pc12=M12,reorganized=bool(REORG),maxX=maxX,
                        per=ris.tolist(),nat_per=rns.tolist(),nsig=nsig,nscr=NSCR)
        print(f"  == {cname} (N={N}): imposed {ris.mean():+.2f}±{ris.std():.2f} [{nsig}/{NSCR}sig] | nat-RET {rns.mean():+.2f} | "
              f"CROSS imp={impC:.1f} nat={natC:.1f} → M={M:+.1f} (PC1-3 M={M02:+.1f}, PC2-3 M={M12:+.1f}) | "
              f"PREREG-REORGANIZED={REORG}",flush=True)
    del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump({"model":MODEL,"base":IS_BASE,"noex":NOEX,"res":res},open(OUT,"w"),indent=2)
    print("SAVED",os.path.abspath(OUT),flush=True)

if __name__=="__main__": main()
