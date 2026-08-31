"""Imposed QWERTY (2-D override on a natural 2-D prior): give a REDEFINED key layout (random key->cell, declared via
shuffled right-of/below relations) that conflicts with real QWERTY, read the key geometry, and head-to-head RSA the
centroid RDM against the IMPOSED-layout 2-D distance vs the NATURAL-QWERTY 2-D distance. Dominance = imposed - natural;
if imposed wins, context overrides the model's native keyboard grid (the keyboard analog of the imposed-days override).
Usage: python -u geom_qwerty_imposed.py <hf_model> [--nscr 6]"""
import os,sys,json,numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1]; NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 6; FRACD=0.75
KEYS=["Q","W","E","A","S","D","Z","X","C"]; N=9
COORD={"Q":(0,0),"W":(0,1),"E":(0,2),"A":(1,0),"S":(1,1),"D":(1,2),"Z":(2,0),"X":(2,1),"C":(2,2)}
NAT=np.array([[abs(COORD[a][0]-COORD[b][0])+abs(COORD[a][1]-COORD[b][1]) for b in KEYS] for a in KEYS],float)  # natural QWERTY Manhattan
QT=["What is immediately right of {e}?","What is immediately below {e}?","What is immediately above {e}?",
    "What is immediately to the left of {e}?","Which row contains {e}? (top/middle/bottom)","Which column contains {e}? (left/middle/right)"]
PRE=("You are operating on a REDEFINED keyboard. The ONLY valid 'right of' / 'below' relations apply below; "
     "the normal QWERTY layout does NOT apply.")
SUF="\n\nAnswer with ONLY the key(s), nothing else."
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c+SUF}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+SUF}],add_generation_prompt=True,tokenize=False)
def cos_rdm(C): X=C-C.mean(0);X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9);return (1-X@X.T)[np.triu_indices(N,1)]
TU=np.triu_indices(N,1)
def main():
    tag=MODEL.split("/")[-1]; cfg=AutoConfig.from_pretrained(MODEL)
    nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None);L=int(round(FRACD*nl))
    print(f"QWERTY-IMPOSED {tag} L={L}/{nl} nscr={NSCR}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL);tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText;model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    rng_o=np.random.default_rng(0); perms=[list(rng_o.permutation(KEYS)) for _ in range(NSCR)]
    stats=[]; allc={}; spec0=None
    for si in range(NSCR):
        perm=perms[si]; rng_e=np.random.default_rng(1000+si); cell={perm[k]:k for k in range(N)}
        es=[]
        for r in range(3):
            for c in range(2): es.append(f"- {perm[r*3+c+1]} is immediately right of {perm[r*3+c]}.")
        for r in range(2):
            for c in range(3): es.append(f"- {perm[(r+1)*3+c]} is immediately below {perm[r*3+c]}.")
        order=list(rng_e.permutation(len(es))); defn=f"{PRE}\nRelations:\n"+"\n".join(es[j] for j in order)
        if si==0: spec0=defn
        rows=[(k,t) for k in KEYS for t in range(len(QT))]; ent={k:[] for k in KEYS}
        prompts=[render(tok,defn+"\n\n"+QT[t].format(e=k)) for (k,t) in rows]
        for bi in range(0,len(prompts),18):
            enc=tok(prompts[bi:bi+18],return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): H=model(**enc,output_hidden_states=True).hidden_states[L][:,-1,:].float().cpu().numpy()
            for j,(k,t) in enumerate(rows[bi:bi+18]): ent[k].append(H[j])
        C=np.stack([np.mean(ent[k],0) for k in KEYS]); obs=cos_rdm(C)
        ic={k:(cell[k]//3,cell[k]%3) for k in KEYS}
        IMP=np.array([[abs(ic[a][0]-ic[b][0])+abs(ic[a][1]-ic[b][1]) for b in KEYS] for a in KEYS],float)
        ri=float(spearmanr(obs,IMP[TU]).correlation); rn=float(spearmanr(obs,NAT[TU]).correlation)
        rng_n=np.random.default_rng(7000+si);nv=[spearmanr(obs,np.array([[IMP[pn[i],pn[j]] for j in range(N)] for i in range(N)])[TU]).correlation for pn in [list(rng_n.permutation(N)) for _ in range(2000)]]
        p=float((np.sum(np.array(nv)>=ri)+1)/2001)
        stats.append(dict(si=si,imp=ri,nat=rn,dom=ri-rn,p=p)); allc[si]=C
        print(f"  scr{si+1}: imposed={ri:+.2f} natural-QWERTY={rn:+.2f} dominance={ri-rn:+.2f} p_imp={p:.3f}",flush=True)
    del model;torch.cuda.empty_cache()
    a=lambda k: float(np.mean([s[k] for s in stats]))
    out=dict(model=MODEL,L=int(L),nscr=NSCR,keys=KEYS,spec_example=spec0,per=stats,
             rsa_imposed=a("imp"),rsa_imposed_std=float(np.std([s["imp"] for s in stats])),rsa_natural=a("nat"),
             dominance=a("dom"),n_imp_beats_nat=int(sum(s["imp"]>s["nat"] for s in stats)),nsig=int(sum(s["p"]<0.05 for s in stats)))
    json.dump(out,open(os.path.join(data_dir("geometry"), f"qwertyimposed_{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"qwertyimposed_{tag}.npz"),**{f"c{si}":allc[si] for si in range(NSCR)},keys=np.array(KEYS),perms=np.array(perms),natcoord=np.array([COORD[k] for k in KEYS]))
    print(f"== {tag}: imposed {a('imp'):+.2f}±{out['rsa_imposed_std']:.2f} natural {a('nat'):+.2f} DOMINANCE {a('dom'):+.2f} | imp-beats-nat {out['n_imp_beats_nat']}/{NSCR} sig {out['nsig']}/{NSCR}",flush=True)
if __name__=="__main__": main()
