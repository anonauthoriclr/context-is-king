#!/usr/bin/env python3
"""H6 control: is the cyclic 2x2 "wrap closes the loop" effect the asserted RELATION, or just local token
CO-OCCURRENCE of the two endpoints (which the +wrap sentence co-mentions, and which the closure metric reads)?

Four conditions on the same numbered order (geometry-only, last-token pre-gen residual, entity-layout):
  list           : numbered order, no wrap                              -> LINE baseline
  list_wrap      : + adjacent wrap sentence co-mentioning order[N-1],order[0] (the original cycle cell)
  list_wrap_pos  : + wrap stated BY POSITION ("one step after the last position returns to position 0"),
                   so the two endpoint NAMES are never co-listed  -> tests RELATION-at-distance
  list_comention : + length-matched placebo that co-mentions order[N-1] and order[0] but asserts NO relation
                   ("for reference, two of the items above are X and Y")  -> tests CO-OCCURRENCE alone

DECISION (genuine topology, co-occurrence ruled out):
  list_wrap_pos closes (closure~1, dRSA>=0) like list_wrap, AND list_comention stays open (closure>>1, dRSA<0)
  like list. If list_comention closes -> co-occurrence contributes. If list_wrap_pos fails to close -> the effect
  needed local adjacency.

Idempotent: skips a model if its npz exists (use --force to redo). Caches raw residuals (Standard #11).
Usage: python -u wrap_cooccur_control.py [--model google/gemma-4-31B-it] [--concept days] [--nscr 3] [--force]"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

NSCR = int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 3
FORCE = "--force" in sys.argv
FRACD = 0.75; KS = list(range(1, 7)); NTPL = 3
ENT = {"days": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
       "arb7": ["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp"]}
CONCEPT = sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
base = ENT[CONCEPT]; N = len(base)
TPLS = ["What is {k} steps after {e}?", "From {e}, advance {k} steps. Which item?", "{e} plus {k} steps =?"]
CONDS = ["list", "list_wrap", "list_wrap_pos", "list_comention"]
MODELS = [sys.argv[sys.argv.index("--model")+1]] if "--model" in sys.argv else \
         ["google/gemma-4-31B-it", "Qwen/Qwen3.5-27B"]
OUTDIR = data_dir("behavior")

def make_def(mode, order, rng):
    numbered = "\n".join(f"{i+1}. {order[i]}" for i in range(N))
    conv = "\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre = "You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    body = f"{pre}\nOrder:\n{numbered}"
    if mode == "list":
        return body + conv
    if mode == "list_wrap":   # original: adjacent co-mention of the two endpoint NAMES
        return body + f"\nThe order wraps around: {order[N-1]} is immediately followed by {order[0]}." + conv
    if mode == "list_wrap_pos":  # wrap stated by POSITION; endpoint names never co-listed
        return body + conv + (f"\nThe order is circular: one step after the last position (position {N-1}) "
                              f"returns to position 0.")
    if mode == "list_comention":  # placebo: co-mention both endpoint names, assert NO relation (length-matched)
        return body + conv + f"\nFor reference, two of the items above are {order[N-1]} and {order[0]}."
    raise ValueError(mode)

def render(tok, c):
    msg = [{"role": "user", "content": c + "\n\nAnswer with ONLY the name, nothing else."}]
    try: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

def cosM(c): X = c - c.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); return 1 - X @ X.T
def cyc(p): P = np.array(p); D = np.abs(P[:, None] - P[None, :]); return np.minimum(D, N - D).astype(float)
def lin(p): P = np.array(p); return np.abs(P[:, None] - P[None, :]).astype(float)
ut = lambda M: M[np.triu_indices(N, 1)]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng_o = np.random.default_rng(0); orders = [list(rng_o.permutation(base)) for _ in range(NSCR)]
    SUMM = {}
    for M in MODELS:
        tag = M.split("/")[-1]
        outnpz = f"{OUTDIR}/wrap_cooccur_{tag}_{CONCEPT}.npz"
        if os.path.exists(outnpz) and not FORCE:
            print(f"SKIP {tag} (exists: {outnpz})", flush=True); continue
        cfg = AutoConfig.from_pretrained(M)
        nl = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
        L = int(round(FRACD * nl)); print(f"=== {tag} L={L}/{nl} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        try: model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
        except Exception:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
        Rall = []; Call = []; Sall = []; Eall = []
        for cond in CONDS:
            for si in range(NSCR):
                order = orders[si]; rng_e = np.random.default_rng(1000 + si); defn = make_def(cond, order, rng_e)
                rows = [(e, k, ti) for e in base for k in KS for ti in range(NTPL)]; ent_res = {e: [] for e in base}
                for bi in range(0, len(rows), 12):
                    b = rows[bi:bi+12]; prompts = [render(tok, defn + "\n\n" + TPLS[ti].format(k=k, e=e)) for (e, k, ti) in b]
                    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
                    with torch.no_grad(): o = model(**enc, output_hidden_states=True)
                    hs = o.hidden_states[L][:, -1, :].float().cpu().numpy()
                    for j, (e, k, ti) in enumerate(b):
                        ent_res[e].append(hs[j]); Rall.append(hs[j].astype(np.float16))
                        Call.append(CONDS.index(cond)); Sall.append(si); Eall.append(base.index(e))
                cents = np.stack([np.mean(ent_res[e], 0) for e in base])
                ip = [{x.lower(): i for i, x in enumerate(order)}[e.lower()] for e in base]
                D = cosM(cents)
                dr = spearmanr(ut(D), ut(cyc(ip))).correlation - spearmanr(ut(D), ut(lin(ip))).correlation
                p2e = {ip[b2]: b2 for b2 in range(N)}
                wrap = D[p2e[0], p2e[N-1]]; interior = np.mean([D[p2e[i], p2e[i+1]] for i in range(N-1)])
                rsa = spearmanr(ut(D), ut(cyc(ip))).correlation
                SUMM.setdefault(tag, {}).setdefault(cond, []).append(
                    dict(si=si, dRSA=float(dr), closure=float(wrap/(interior+1e-9)), rsa=float(rsa)))
        del model; torch.cuda.empty_cache()
        np.savez(outnpz, R=np.stack(Rall), COND=np.array(Call), SI=np.array(Sall), ENT=np.array(Eall),
                 imp=np.array([[{x.lower(): i for i, x in enumerate(orders[si])}[e.lower()] for e in base] for si in range(NSCR)]),
                 conds=np.array(CONDS), L=L)
        s = SUMM[tag]
        def pr(c): return f"dRSA={np.mean([r['dRSA'] for r in s[c]]):+.2f} clos={np.mean([r['closure'] for r in s[c]]):.2f}"
        print(f"  {tag}:  list[{pr('list')}]  list_wrap[{pr('list_wrap')}]  "
              f"list_wrap_pos[{pr('list_wrap_pos')}]  list_comention[{pr('list_comention')}]", flush=True)
        # verdict
        cl = {c: np.mean([r['closure'] for r in s[c]]) for c in CONDS}
        dr_ = {c: np.mean([r['dRSA'] for r in s[c]]) for c in CONDS}
        pos_closes = cl['list_wrap_pos'] < 1.3 and dr_['list_wrap_pos'] >= 0
        com_open = cl['list_comention'] >= 1.3 and dr_['list_comention'] < 0
        print(f"  VERDICT {tag}: wrap_pos closes={pos_closes}  comention stays open={com_open}  "
              f"=> {'co-occurrence RULED OUT' if (pos_closes and com_open) else 'inspect'}", flush=True)
        SUMM[tag]["_verdict"] = dict(pos_closes=bool(pos_closes), com_open=bool(com_open))
    json.dump(SUMM, open(f"{OUTDIR}/wrap_cooccur_summary_{CONCEPT}.json", "w"), indent=2)
    print("SAVED", f"{OUTDIR}/wrap_cooccur_summary_{CONCEPT}.json", flush=True)

if __name__ == "__main__": main()
