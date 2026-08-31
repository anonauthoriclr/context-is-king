#!/usr/bin/env python3
"""NATURAL (no-context) weekday geometry at SENTENCE-END (last prompt token, pre-generation) --- the paper's main
readout regime, but with NO imposed order. Baseline for the override: does the pretrained ring appear at the integrated
sentence-end state the next-token computation consumes, not only at the day token? Two prompt sets, both with NO
redefinition: (query) k-step questions about the natural days, matched to the main probe; (neutral) plain mentions.
Reports cyclic RSA + permutation-null p + participation-ratio effective dim, at BOTH sentence-end and day-token, and
saves centroids + a PCA ring plot. Forward-only. Usage: python -u natural_sentend.py <hf_model>"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL = sys.argv[1]; FRACD = 0.75
base = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N = 7
KS = [1,2,3,4,5,6]
QUERY   = ["What is {k} steps after {e}?", "From {e}, advance {k} steps. Which item?",
           "Starting at {e}, move {k} steps forward. Result?", "{e} plus {k} steps =?"]
NEUTRAL = ["Consider the day {e}.", "The day under discussion is {e}.",
           "Tell me about the day {e}.", "Here is a day: {e}."]

def render(tok, c):
    try: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False)
def cosM(c): X = c - c.mean(0); X = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1 - X@X.T
def cyc_of(p): P = np.array(p); D = np.abs(P[:,None]-P[None,:]); return np.minimum(D, N-D).astype(float)
ut = lambda M: M[np.triu_indices(N,1)]
def eff_dim(c):
    X = c - c.mean(0); l = np.linalg.svd(X, compute_uv=False)**2
    return float((l.sum()**2)/(np.sum(l**2)+1e-12))
def last_end(ids, sub):
    Ls = len(sub); best = None
    for i in range(len(ids)-Ls+1):
        if ids[i:i+Ls] == sub: best = i+Ls-1
    return best

def main():
    tag = MODEL.split("/")[-1]; os.makedirs(data_dir("geometry"), exist_ok=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    nl = getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L = int(round(FRACD*nl)); print(f"NAT-SENTEND {MODEL} L={L}/{nl}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dayids = {e:(tok.encode(" "+e,add_special_tokens=False) or tok.encode(e,add_special_tokens=False)) for e in base}
    try: model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"":0}).eval()
    natrdm = cyc_of(range(N))
    def collect(templates, kmode):
        endE = {e:[] for e in base}; dayE = {e:[] for e in base}
        rows = ([(e,t,k) for e in base for t in range(len(templates)) for k in KS] if kmode
                else [(e,t,None) for e in base for t in range(len(templates))])
        prompts = [render(tok, templates[t].format(e=e,k=k) if k is not None else templates[t].format(e=e)) for (e,t,k) in rows]
        for bi in range(0, len(prompts), 16):
            ch = prompts[bi:bi+16]; rw = rows[bi:bi+16]
            enc = tok(ch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad(): o = model(**enc, output_hidden_states=True)
            H = o.hidden_states[L].float().cpu().numpy(); ids_b = enc["input_ids"].cpu().tolist()
            for j,(e,t,k) in enumerate(rw):
                endE[e].append(H[j,-1,:])
                pos = last_end(ids_b[j], dayids[e]); dayE[e].append(H[j, pos if pos is not None else -1, :])
        return (np.stack([np.mean(endE[e],0) for e in base]), np.stack([np.mean(dayE[e],0) for e in base]))
    def stats(c):
        D = cosM(c); rsa = spearmanr(ut(D), ut(natrdm)).correlation
        rng = np.random.default_rng(0)
        nv = [spearmanr(ut(D), ut(cyc_of(rng.permutation(N)))).correlation for _ in range(2000)]
        return float(rsa), float((np.sum(np.array(nv) >= rsa)+1)/2001), eff_dim(c)
    out = {}; cents = {}
    for name, tpls, km in [("query", QUERY, True), ("neutral", NEUTRAL, False)]:
        cE, cD = collect(tpls, km); cents[name] = (cE, cD)
        rE,pE,dE = stats(cE); rD,pD,dD = stats(cD)
        out[name] = {"sentend":{"rsa":rE,"p":pE,"effdim":dE}, "daytok":{"rsa":rD,"p":pD,"effdim":dD}}
        print(f"  [{name:7s}] SENTEND rsa={rE:+.2f} p={pE:.3f} effdim={dE:.1f}  |  DAYTOK rsa={rD:+.2f} p={pD:.3f} effdim={dD:.1f}", flush=True)
        np.savez(os.path.join(data_dir("geometry"), f"natsentend_{tag}_{name}.npz"), cent_sentend=cE, cent_daytok=cD, order=np.arange(N))
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"L":L,"nl":nl,"res":out}, open(os.path.join(data_dir("geometry"), f"natsentend_{tag}.json"),"w"), indent=2)
    # ring plot: query condition, sentence-end vs day-token, traced in the natural order
    fig,axs = plt.subplots(1,2,figsize=(8.4,4.2))
    for ax,(key,ttl) in zip(axs, [("sentend","SENTENCE-END (paper readout)"),("daytok","day token")]):
        c = cents["query"][0 if key=="sentend" else 1]
        mu=c.mean(0); _,_,Vt=np.linalg.svd(c-mu, full_matrices=False); P=(c-mu)@Vt.T
        seq=list(range(N))+[0]
        for i in range(N): ax.plot([P[seq[i],0],P[seq[i+1],0]],[P[seq[i],1],P[seq[i+1],1]],"-",color="#c0392b" if i==N-1 else "#888",lw=2.0 if i==N-1 else 1.2)
        ax.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=90,edgecolors="k",linewidths=.5,zorder=3)
        for i,e in enumerate(base): ax.annotate(e[:3],(P[i,0],P[i,1]),fontsize=7,ha="center",va="center")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{ttl}\ncyclic RSA={out['query'][key]['rsa']:+.2f}, eff-dim={out['query'][key]['effdim']:.1f}",fontsize=9)
    fig.suptitle(f"{tag}: NATURAL (no-context) weekday ring, traced in pretrained order (red = wrap edge)",fontsize=11)
    fig.tight_layout(); fn=os.path.join(data_dir("geometry"), f"FIG_natsentend_{tag}.png"); fig.savefig(fn,dpi=140); print("SAVED",os.path.abspath(fn),flush=True)
if __name__=="__main__": main()
