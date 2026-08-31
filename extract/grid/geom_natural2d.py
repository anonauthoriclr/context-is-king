#!/usr/bin/env python3
"""E6 (store side): does the model NATIVELY store a 2-D grid? Probe the PRETRAINED geometry of chess squares (no
redefinition), the 2-D analogue of the natural weekday cycle. Read last-token pre-gen residual at 0.75 depth under a
neutral chess prompt, group by square -> centroids -> mean-centered cosine RDM, and RSA against 2-D king (Chebyshev) /
rook (Manhattan) templates vs a 1-D reading-order template (the competitor) + file-only / rank-only diagnostics.
PASS (native 2-D store): 2-D beats 1-D and the label-shuffle null (empirical p). KEY CONFOUND = tokenization: 'a1','a2'
share the file letter -> --opaque re-encodes each square as a distinct arbitrary token; if 2-D survives, it is not lexical.
Usage: python -u geom_natural2d.py <hf_model> [--size 3] [--opaque] [--qset neutral|king]
Out: data_dir("geometry")/nat2d_<tag>.json (+ .npz)"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
def arg(flag,default,cast=str): return cast(sys.argv[sys.argv.index(flag)+1]) if flag in sys.argv else default
SIZE=arg("--size",3,int); OPAQUE="--opaque" in sys.argv; QSET=arg("--qset","neutral"); FRACD=0.75
FILES="abcdefgh"[:SIZE]; RANKS=[str(r+1) for r in range(SIZE)]
SQ=[f+r for f in FILES for r in RANKS]            # file-major: a1,a2,...,b1,...
N=len(SQ); COORD={f+r:(fi,ri) for fi,f in enumerate(FILES) for ri,r in enumerate(RANKS)}
OPAQ=["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle",
      "Garden","Pencil","Marble"][:N]
DISP={sq:(OPAQ[i] if OPAQUE else sq) for i,sq in enumerate(SQ)}     # how each square is shown to the model
# templates on the SQ order
def cheb(): return np.array([[max(abs(COORD[a][0]-COORD[b][0]),abs(COORD[a][1]-COORD[b][1])) for b in SQ] for a in SQ],float)
def manh(): return np.array([[abs(COORD[a][0]-COORD[b][0])+abs(COORD[a][1]-COORD[b][1]) for b in SQ] for a in SQ],float)
def fileonly(): return np.array([[abs(COORD[a][0]-COORD[b][0]) for b in SQ] for a in SQ],float)
def rankonly(): return np.array([[abs(COORD[a][1]-COORD[b][1]) for b in SQ] for a in SQ],float)
def line1d(): return np.array([[abs(i-j) for j in range(N)] for i in range(N)],float)   # file-major reading order
TEMPLATES={"king2D":cheb(),"rook2D":manh(),"line1D":line1d(),"file":fileonly(),"rank":rankonly()}
QW={"neutral":["On a chessboard, consider the square {e}.","Look at the square {e} on the board.",
               "The square in question is {e}.","Focus on this square: {e}.","Here is a board square: {e}.","Square of interest: {e}."],
    "king":["Which squares does a king on {e} attack?","List the squares adjacent to {e}.","From {e}, which squares are one step away?",
            "A king stands on {e}; name its neighbors.","Which squares touch {e}?","Name the squares bordering {e}."]}
TPLS=QW[QSET]; SUFFIX="" if QSET=="neutral" else "\n\nAnswer with ONLY the square name(s), nothing else."
PRE=("" if not OPAQUE else "You are operating on a labeled grid; treat each name as a fixed cell identifier.\n")
tag=MODEL.split("/")[-1]+f"_{SIZE}x{SIZE}"+("_opaque" if OPAQUE else "")+("" if QSET=="neutral" else f"_{QSET}")
os.makedirs(data_dir("geometry"),exist_ok=True); OUT=os.path.join(data_dir("geometry"), f"nat2d_{tag}.json")

def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":PRE+c+SUFFIX}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":PRE+c+SUFFIX}],add_generation_prompt=True,tokenize=False)
def cos_rdm_full(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1-X@X.T
TU=np.triu_indices(N,1)
def up(M): return M[TU]

def main():
    print(f"NAT2D chess {SIZE}x{SIZE} N={N} opaque={OPAQUE} qset={QSET} | {MODEL}",flush=True)
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"  L={L}/{nl}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    rows=[(sq,ti) for sq in SQ for ti in range(len(TPLS))]; ent={sq:[] for sq in SQ}
    for bi in range(0,len(rows),18):
        b=rows[bi:bi+18]; prompts=[render(tok,TPLS[ti].format(e=DISP[sq])) for (sq,ti) in b]
        enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
        with torch.no_grad(): o=model(**enc,output_hidden_states=True)
        hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
        for j,(sq,ti) in enumerate(b): ent[sq].append(hs[j])
    cents=np.stack([np.mean(ent[sq],0) for sq in SQ]); full=cos_rdm_full(cents); obs=up(full)
    rsa={k:float(spearmanr(obs,up(T)).correlation) for k,T in TEMPLATES.items()}
    # cone-preserving label-shuffle null on the 2-D king template, empirical p
    rng=np.random.default_rng(7); base=cheb(); nullv=[]
    for _ in range(2000):
        pn=list(rng.permutation(range(N))); sd=up(np.array([[base[pn[i],pn[j]] for j in range(N)] for i in range(N)]))
        nullv.append(spearmanr(obs,sd).correlation)
    nullv=np.array(nullv); p_emp=float((np.sum(nullv>=rsa["king2D"])+1)/(len(nullv)+1))
    best=max(TEMPLATES,key=lambda k:rsa[k])
    del model; torch.cuda.empty_cache()
    summ=dict(model=MODEL,concept="chess",size=SIZE,N=N,opaque=OPAQUE,qset=QSET,L=int(L),squares=SQ,
              rsa=rsa,best_template=best,dRSA_2D_minus_1D=rsa["king2D"]-rsa["line1D"],
              null_mean=float(nullv.mean()),p_emp_2D=p_emp)
    json.dump(summ,open(OUT,"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"nat2d_{tag}.npz"),cents=cents,squares=np.array(SQ),
             coord=np.array([COORD[s] for s in SQ]),L=L)
    print(f"== {tag}: king2D={rsa['king2D']:+.2f} rook2D={rsa['rook2D']:+.2f} line1D={rsa['line1D']:+.2f} "
          f"file={rsa['file']:+.2f} rank={rsa['rank']:+.2f} | best={best} dRSA(2D-1D)={rsa['king2D']-rsa['line1D']:+.2f} "
          f"p_2D={p_emp:.3f}",flush=True)
if __name__=="__main__": main()
