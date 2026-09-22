#!/usr/bin/env python3
"""NATURAL (no-context) weekday geometry at SENTENCE-END (last prompt token, pre-generation) --- the paper's main
readout regime, but with NO imposed order. Baseline for the override: does the pretrained ring appear at the integrated
sentence-end state the next-token computation consumes, not only at the day token? Two prompt sets, both with NO
redefinition: (query) k-step questions about the natural days, matched to the main probe; (neutral) plain mentions.
Reports cyclic RSA + permutation-null p + participation-ratio effective dim, at BOTH sentence-end and day-token, and
saves centroids + a PCA ring plot. Forward-only. Usage: python -u natural_sentend.py <hf_model>"""
import os, sys, json, numpy as np, torch
from itertools import permutations
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL = sys.argv[1]; FRACD = 0.75
base = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N = 7
KS = [1,2,3,4,5,6]
QUERY   = ["What is {k} steps after {e}?", "{k} steps after {e} is?",
           "From {e}, advance {k} steps. Which item?",
           "Starting at {e}, move {k} steps forward. Result?", "{e} plus {k} steps =?",
           "Advance {k} from {e}. Which one?"]
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
def last_end(ids, candidates):
    best = None
    for sub in candidates:
        Ls = len(sub)
        for i in range(len(ids)-Ls+1):
            if ids[i:i+Ls] == sub: best = i+Ls-1
    return best

def main():
    tag = MODEL.split("/")[-1]; os.makedirs("results/geometry", exist_ok=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    nl = getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L = int(round(FRACD*nl)); print(f"NAT-SENTEND {MODEL} L={L}/{nl}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dayids = {e:[x for x in (tok.encode(" "+e,add_special_tokens=False),
                              tok.encode(e,add_special_tokens=False)) if x] for e in base}
    try: model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"":0}).eval()
    natrdm = cyc_of(range(N))
    def collect(templates, kmode):
        rows = ([(e,t,k) for e in base for t in range(len(templates)) for k in KS] if kmode
                else [(e,t,None) for e in base for t in range(len(templates))])
        prompts = [render(tok, templates[t].format(e=e,k=k) if k is not None else templates[t].format(e=e)) for (e,t,k) in rows]
        end_rows, day_rows = [], []
        for bi in range(0, len(prompts), 16):
            ch = prompts[bi:bi+16]; rw = rows[bi:bi+16]
            enc = tok(ch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad(): o = model(**enc, output_hidden_states=True)
            H = o.hidden_states[L].float().cpu().numpy(); ids_b = enc["input_ids"].cpu().tolist()
            for j,(e,t,k) in enumerate(rw):
                pos = last_end(ids_b[j], dayids[e])
                if pos is None:
                    raise RuntimeError(f"Could not locate entity token for {e!r} in prompt {ch[j]!r}")
                end_rows.append(H[j,-1,:]); day_rows.append(H[j,pos,:])
        return np.stack(end_rows), np.stack(day_rows), rows, prompts
    all_perms = list(permutations(range(N)))
    def rsa_only(c):
        return float(spearmanr(ut(cosM(c)), ut(natrdm)).correlation)
    def stats(c):
        D = cosM(c); rsa = rsa_only(c)
        nv = [spearmanr(ut(D), ut(cyc_of(p))).correlation for p in all_perms]
        return float(rsa), float(np.mean(np.asarray(nv) >= rsa)), eff_dim(c)
    def crossed_ci(raw, kmode, nboot=2000):
        rng = np.random.default_rng(20260917); vals = []
        nt, nk = raw.shape[1], raw.shape[2]
        for _ in range(nboot):
            ti = rng.integers(0, nt, nt)
            ki = rng.integers(0, nk, nk) if kmode else np.array([0])
            c = np.take(np.take(raw, ti, axis=1), ki, axis=2).mean((1,2))
            vals.append(rsa_only(c))
        return [float(x) for x in np.percentile(vals, [2.5,97.5])]
    out = {}; cents = {}
    for name, tpls, km in [("query", QUERY, True), ("neutral", NEUTRAL, False)]:
        end_rows, day_rows, rows, prompts = collect(tpls, km)
        nt, nk = len(tpls), len(KS) if km else 1
        shape = (N, nt, nk, end_rows.shape[-1])
        rawE, rawD = end_rows.reshape(shape), day_rows.reshape(shape)
        cE, cD = rawE.mean((1,2)), rawD.mean((1,2)); cents[name] = (cE, cD)
        rE,pE,dE = stats(cE); rD,pD,dD = stats(cD)
        per_hop = ([{"k":k, "sentend_rsa":rsa_only(rawE[:,:,ki].mean(1)),
                     "daytok_rsa":rsa_only(rawD[:,:,ki].mean(1))} for ki,k in enumerate(KS)] if km else [])
        per_template = [{"template":tpls[ti], "sentend_rsa":rsa_only(rawE[:,ti].mean(1)),
                         "daytok_rsa":rsa_only(rawD[:,ti].mean(1))} for ti in range(nt)]
        out[name] = {"sentend":{"rsa":rE,"p_exact":pE,"effdim":dE,"ci95":crossed_ci(rawE,km)},
                     "daytok":{"rsa":rD,"p_exact":pD,"effdim":dD,"ci95":crossed_ci(rawD,km)},
                     "per_hop":per_hop, "per_template":per_template,
                     "n_prompts":len(rows)}
        print(f"  [{name:7s}] SENTEND rsa={rE:+.2f} p={pE:.3f} effdim={dE:.1f}  |  DAYTOK rsa={rD:+.2f} p={pD:.3f} effdim={dD:.1f}", flush=True)
        np.savez_compressed(f"results/geometry/natsentend_{tag}_{name}.npz",
                            cent_sentend=cE.astype(np.float32), cent_daytok=cD.astype(np.float32),
                            raw_sentend=rawE.astype(np.float16), raw_daytok=rawD.astype(np.float16),
                            entities=np.array(base), templates=np.array(tpls),
                            hops=np.array(KS if km else [-1]), prompts=np.array(prompts), order=np.arange(N))
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"L":L,"nl":nl,"res":out}, open(f"results/geometry/natsentend_{tag}.json","w"), indent=2)
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
    fig.tight_layout(); fn=f"results/geometry/FIG_natsentend_{tag}.png"; fig.savefig(fn,dpi=140); print("SAVED",os.path.abspath(fn),flush=True)
if __name__=="__main__": main()
