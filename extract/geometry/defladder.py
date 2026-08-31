#!/usr/bin/env python3
"""DEFINITION-LADDER geometry probe (NO generation; prepared, trigger manually).

Question: how much does the geometry's reorganization depend on the imposed order being SPELLED OUT
in the prompt? Varies ONE thing — how much order is given — holding model/concept/questions/capture fixed:
  full : numbered list + "Full cycle (wraps):" doubled arrow-chain   (current canonical; order+traversal+wrap explicit)
  list : numbered list ONLY                                          (positions explicit, traversal not drawn out)
  adj  : successor edges only, SHUFFLED, incl wrap                   (distances must be COMPOSED, not read off)

Geometry only: forward pass, read last-token-pre-generation residual, group by INPUT entity → centroids → RSA
(imposed + natural-retention) + self-crossings. Same scrambled target orders reused across the 3 rungs (fair compare).
Caches activations to .npz so re-plotting needs no GPU. Produces FIG_defladder_<tag>_<concept>.png.

Usage (DO NOT run until operator says go):
  python -u defladder.py <hf_model> --concept days [--nscr 3] [--base] [--modes full,list,adj]
"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 3
CONCEPT=sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
MODES=(sys.argv[sys.argv.index("--modes")+1] if "--modes" in sys.argv else "full,list,adj").split(",")
FRAC=0.75; KS=list(range(1,7))
ENT={"days":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
     "months":["January","February","March","April","May","June","July","August","September","October","November","December"],
     "zodiac":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],
     "notes":["Do","Re","Mi","Fa","Sol","La","Ti"],
     "clock":["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"]}
TPLS=["What is {k} steps after {e}?","{k} steps after {e} is?","From {e}, advance {k} steps. Which item?",
      "Starting at {e}, move {k} steps forward. Result?","{e} plus {k} steps =?","Advance {k} from {e}. Which one?"]
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")
CACHE=data_dir("geometry","cache"); os.makedirs(CACHE,exist_ok=True)
os.makedirs(data_dir("geometry"),exist_ok=True)

# ---- the controlled variable: three definition rungs (order given in decreasing detail) ----
def make_def(mode,order,rng):
    N=len(order); numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N)); doubled=" -> ".join(order*2)
    conv="\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    if mode=="full":
        return f"{pre}\nOrder:\n{numbered}\nFull cycle (wraps):\n{doubled}{conv}"
    if mode=="list":
        return f"{pre}\nOrder:\n{numbered}{conv}"
    if mode=="adj":  # successor edges only, shuffled, incl wrap edge; distances must be composed
        edges=[f"- {order[i]} is immediately followed by {order[(i+1)%N]}." for i in range(N)]
        perm=list(rng.permutation(len(edges))); shuf="\n".join(edges[j] for j in perm)
        conv_adj="\nConvention: 'k steps after X' = apply 'immediately followed by' k times, starting at X."
        return f"{pre}\nThe ONLY valid 'immediately followed by' relations are (the normal order does NOT apply):\n{shuf}{conv_adj}"
    raise ValueError(mode)

def render(tok,c):
    if IS_BASE: return c+"\n\nQ-ANSWER:"
    try: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+"\n\nAnswer with ONLY the name, nothing else."}],add_generation_prompt=True,tokenize=False)
def cos_rdm(c,N): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def cyc_rdm(pos,N): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)
def _ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0])>(b[1]-a[1])*(c[0]-a[0])
def _inter(p1,p2,p3,p4): return _ccw(p1,p3,p4)!=_ccw(p2,p3,p4) and _ccw(p1,p2,p3)!=_ccw(p1,p2,p4)
def crossings(pts,seq):
    s=list(seq)+[seq[0]]; E=[(s[i],s[i+1]) for i in range(len(s)-1)]; c=0
    for a in range(len(E)):
        for b in range(a+1,len(E)):
            if set(E[a])&set(E[b]): continue
            if _inter(pts[E[a][0]],pts[E[a][1]],pts[E[b][0]],pts[E[b][1]]): c+=1
    return c

def capture(model,tok,definition,L):  # forward only, NO generation
    rows=[(e,k,ti) for e in ENT[CONCEPT] for k in KS for ti in range(len(TPLS))]; A=[];ent=[]
    for i in range(0,len(rows),12):
        b=rows[i:i+12]
        texts=[render(tok,definition+"\n\n"+TPLS[ti].format(k=k,e=e)) if not IS_BASE else definition+"\n\nQ: "+TPLS[ti].format(k=k,e=e)+"\nQ-ANSWER:" for (e,k,ti) in b]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        A.append(o.hidden_states[L][:,-1,:].float().cpu().numpy())
        for (e,k,ti) in b: ent.append(e)
    return np.concatenate(A,0),np.array(ent)

def main():
    base=ENT[CONCEPT]; N=len(base); rng_o=np.random.default_rng(0)
    orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]      # SHARED across modes (fair compare)
    natcyc=cyc_rdm(list(range(N)),N)
    f,_=torch.cuda.mem_get_info(); print(f"MODEL={MODEL} concept={CONCEPT} N={N} nscr={NSCR} modes={MODES} GPU {f/1e9:.0f}GB",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRAC*nl)); print(f"L={L}/{nl}",flush=True)
    ckpath=lambda mode,si: f"{CACHE}/defladder_{tag}_{CONCEPT}_{mode}_s{si}_L{L}.npz"
    allcached=all(os.path.exists(ckpath(m,si)) for m in MODES for si in range(NSCR))
    model=None
    if allcached:
        print("all activations cached — skipping model load (re-plot only)",flush=True)
    else:
        try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
        except Exception as e:
            print(f"CausalLM load failed ({type(e).__name__}); ImageTextToText",flush=True)
            from transformers import AutoModelForImageTextToText
            model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    summary={}; cents0={}                                            # cents0[mode]=scramble-0 centroids (for the ring plot)
    for mode in MODES:
        ris=[];rns=[];impX=[];natX=[]
        for si,order in enumerate(orders):
            rng_e=np.random.default_rng(1000+si)                     # deterministic edge-shuffle per scramble
            ck=ckpath(mode,si)
            if os.path.exists(ck):
                d=np.load(ck,allow_pickle=True); A=d["A"]; ent=d["ent"]
            else:
                A,ent=capture(model,tok,make_def(mode,order,rng_e),L); np.savez(ck,A=A,ent=ent)
            cents=np.stack([A[ent==e].mean(0) for e in base]); neu=cos_rdm(cents,N)
            imp_pos=[{x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base]
            ris.append(float(spearmanr(neu,cyc_rdm(imp_pos,N)).correlation))
            rns.append(float(spearmanr(neu,natcyc).correlation))
            mu=cents.mean(0); _,_,Vt=np.linalg.svd(cents-mu,full_matrices=False); Pf=(cents-mu)@Vt.T
            iseq=list(np.argsort(imp_pos))
            impX.append(crossings(Pf[:,[0,1]],iseq)); natX.append(crossings(Pf[:,[0,1]],list(range(N))))
            if si==0: cents0[mode]=(cents,imp_pos)
            print(f"  [{mode}] scr{si+1}/{NSCR}: imposed={ris[-1]:+.2f} nat-ret={rns[-1]:+.2f} cross imp={impX[-1]} nat={natX[-1]}",flush=True)
        summary[mode]=dict(imp_mean=float(np.mean(ris)),imp_std=float(np.std(ris)),nat_ret=float(np.mean(rns)),
                           imp_cross=float(np.mean(impX)),nat_cross=float(np.mean(natX)),per=ris)
        s=summary[mode]; print(f"  == {mode}: imposed {s['imp_mean']:+.2f}±{s['imp_std']:.2f} | nat-ret {s['nat_ret']:+.2f} | cross imp={s['imp_cross']:.1f} nat={s['nat_cross']:.1f}",flush=True)
    if model is not None: del model; torch.cuda.empty_cache()

    # ---- PLOT: cols = def rungs; top row = centroids drawn in IMPOSED order, bottom = NATURAL order (scramble-0) ----
    nm=len(MODES); fig,ax=plt.subplots(2,nm,figsize=(4.2*nm,8.4))
    if nm==1: ax=ax.reshape(2,1)
    for ci,mode in enumerate(MODES):
        cents,imp_pos=cents0[mode]; mu=cents.mean(0); _,_,Vt=np.linalg.svd(cents-mu,full_matrices=False); P=(cents-mu)@Vt.T
        iseq=list(np.argsort(imp_pos)); s=summary[mode]
        for ri,(seq,name) in enumerate([(iseq,"imposed order"),(list(range(N)),"natural order")]):
            a=ax[ri,ci]; path=seq+[seq[0]]
            a.plot(P[path,0],P[path,1],"-",color="#888",lw=1,zorder=1)
            a.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=140,zorder=2,edgecolors="k",linewidths=.5)
            for i,e in enumerate(base): a.annotate(e[:3],(P[i,0],P[i,1]),fontsize=8,ha="center",va="center")
            a.set_xticks([]);a.set_yticks([])
            if ri==0: a.set_title(f"{mode}\nimpRSA={s['imp_mean']:+.2f}±{s['imp_std']:.2f} natret={s['nat_ret']:+.2f}\ncross imp={s['imp_cross']:.1f} nat={s['nat_cross']:.1f}",fontsize=9)
            if ci==0: a.set_ylabel(name,fontsize=10)
    fig.suptitle(f"{tag} — {CONCEPT}: geometry vs how much order is spelled out (PC1-2, scramble-0; titles avg over {NSCR})",fontsize=11)
    fig.tight_layout(); out=os.path.join(data_dir("geometry"), f"FIG_defladder_{tag}_{CONCEPT}.png"); fig.savefig(out,dpi=130); print("SAVED",os.path.abspath(out),flush=True)
    json.dump({"model":MODEL,"concept":CONCEPT,"base":IS_BASE,"nscr":NSCR,"L":int(L),"summary":summary},
              open(os.path.join(data_dir("geometry"), f"defladder_{tag}_{CONCEPT}.json"),"w"),indent=2)

if __name__=="__main__": main()
