#!/usr/bin/env python3
"""VERIFY which 'strong pretrained structure' concepts actually form a clean NATURAL (no-context) ring. For each of the
five concepts the paper lists (weekdays, months, zodiac, clock hours, musical notes), probe with NO imposed order
(neutral mentions + natural k-step queries) and read BOTH at sentence-end (the paper's readout) and at the entity token
(fair 'does the prior exist' check). Report cyclic RSA (natural order) + permutation-null p + participation-ratio
eff-dim at each readout. Forward-only. Usage: python -u verify_pretrained.py <hf_model>
Out: data_dir("geometry")/pretrained_verify_<tag>.json"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL = sys.argv[1]; FRACD = 0.75
CONCEPTS = {
 "weekdays": (["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], "day"),
 "months":   (["January","February","March","April","May","June","July","August","September","October","November","December"], "month"),
 "zodiac":   (["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"], "sign"),
 "clock":    (["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve"], "hour"),
 "notes":    (["Do","Re","Mi","Fa","Sol","La","Ti"], "note"),
}
QUERY   = ["What is {k} steps after {e}?", "From {e}, advance {k} steps. Which {noun}?", "Starting at {e}, move {k} forward. Result?"]
NEUTRAL = ["Consider the {noun} {e}.", "The {noun} under discussion is {e}.", "Here is a {noun}: {e}."]

def render(tok, c):
    try: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False)
def cosM(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1-X@X.T
def cyc_of(p,N): P=np.array(p); D=np.abs(P[:,None]-P[None,:]); return np.minimum(D,N-D).astype(float)
def effdim(c):
    X=c-c.mean(0); l=np.linalg.svd(X,compute_uv=False)**2; return float((l.sum()**2)/(np.sum(l**2)+1e-12))
def last_end(ids, sub):
    Ls=len(sub); best=None
    for i in range(len(ids)-Ls+1):
        if ids[i:i+Ls]==sub: best=i+Ls-1
    return best

def main():
    tag=MODEL.split("/")[-1]; os.makedirs(data_dir("geometry"),exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"VERIFY-PRETRAINED {MODEL} L={L}/{nl}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    out={}
    for cname,(base,noun) in CONCEPTS.items():
        N=len(base); KS=list(range(1,min(7,N)))
        eids={e:(tok.encode(" "+e,add_special_tokens=False) or tok.encode(e,add_special_tokens=False)) for e in base}
        endE={e:[] for e in base}; tokE={e:[] for e in base}; miss=0
        rows=[(e,t,k) for e in base for t in range(len(QUERY)) for k in KS] + [(e,t,None) for e in base for t in range(len(NEUTRAL))]
        prompts=[render(tok, (QUERY[t].format(k=k,e=e,noun=noun) if k is not None else NEUTRAL[t].format(e=e,noun=noun))) for (e,t,k) in rows]
        for bi in range(0,len(prompts),16):
            ch=prompts[bi:bi+16]; rw=rows[bi:bi+16]
            enc=tok(ch,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            H=o.hidden_states[L].float().cpu().numpy(); ids_b=enc["input_ids"].cpu().tolist()
            for j,(e,t,k) in enumerate(rw):
                endE[e].append(H[j,-1,:])
                pos=last_end(ids_b[j],eids[e])
                if pos is None: miss+=1; tokE[e].append(H[j,-1,:])
                else: tokE[e].append(H[j,pos,:])
        def stats(cents):
            C=np.stack([np.mean(cents[e],0) for e in base]); natrdm=cyc_of(range(N),N)
            ut=lambda M:M[np.triu_indices(N,1)]; D=cosM(C); rsa=spearmanr(ut(D),ut(natrdm)).correlation
            rng=np.random.default_rng(0); nv=[spearmanr(ut(D),ut(cyc_of(rng.permutation(N),N))).correlation for _ in range(2000)]
            return float(rsa), float((np.sum(np.array(nv)>=rsa)+1)/2001), effdim(C)
        rE,pE,dE=stats(endE); rT,pT,dT=stats(tokE)
        out[cname]={"N":N,"sentend":{"rsa":rE,"p":pE,"effdim":dE},"daytok":{"rsa":rT,"p":pT,"effdim":dT},"miss":int(miss)}
        verdict = "STRONG" if (rE>=0.6 and dE>=3) else ("TOKEN-ONLY" if rT>=0.6 else "WEAK")
        print(f"  [{cname:8s} N={N:2d}] SENT-END rsa={rE:+.2f} eff-dim={dE:.1f} | DAY-TOK rsa={rT:+.2f} eff-dim={dT:.1f}  -> {verdict}",flush=True)
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"L":L,"nl":nl,"res":out},open(os.path.join(data_dir("geometry"), f"pretrained_verify_{tag}.json"),"w"),indent=2)
    print("SAVED",os.path.abspath(os.path.join(data_dir("geometry"), f"pretrained_verify_{tag}.json")),flush=True)
if __name__=="__main__": main()
