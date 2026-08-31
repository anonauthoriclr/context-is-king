#!/usr/bin/env python3
"""Behavioral accuracy of the IMPOSED order under all THREE prompt regimes, per model (for a 3-regime Fig 7).
For days (list mode), score the imposed-order k-curve (k=1..6) under:
  direct   : answer immediately ("Answer with ONLY the name")          enable_thinking=False, maxnew=16
  scripted : dictated step-by-step ("work ONE STEP AT A TIME -> FINAL") enable_thinking=False, maxnew=2048
  natural  : the model's own open-ended thinking (no instruction)       enable_thinking=True,  maxnew=2048
Same redefined order + seed as defladder/wrap (cells pair with the geometry). Parses FINAL else last entity mentioned.
Idempotent: skips a model whose output json exists (--force to redo). NOTE: Gemma-4 E2B/E4B/12B are gemma4_unified and
do NOT load on the gemma4-venv transformers (5.5.3); 31B-it and Qwen-3.5 load. Run the loadable models.

Usage: python -u behavior_3regime.py [--model google/gemma-4-31B-it] [--concept days] [--maxnew 2048] [--force]"""
import os, sys, json, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

CONCEPT = sys.argv[sys.argv.index("--concept")+1] if "--concept" in sys.argv else "days"
MAXNEW = int(sys.argv[sys.argv.index("--maxnew")+1]) if "--maxnew" in sys.argv else 2048
FORCE = "--force" in sys.argv
NTPL = 2; KS = list(range(1, 7))
ENT = {"days": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]}
base = ENT[CONCEPT]; N = len(base)
TPLS = ["What is {k} steps after {e}?", "From {e}, advance {k} steps. Which item?"]
MODELS = [sys.argv[sys.argv.index("--model")+1]] if "--model" in sys.argv else \
         ["google/gemma-4-31B-it", "Qwen/Qwen3.5-27B"]
REGIMES = ["direct", "scripted", "natural"]
OUTDIR = data_dir("behavior")

def make_def(order, rng):  # zero-shot list, identical convention to defladder
    numbered = "\n".join(f"{i+1}. {order[i]}" for i in range(N))
    conv = "\nConvention: start=position 0; 'k steps after X' = item reached by moving forward k steps."
    pre = "You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    return f"{pre}\nOrder:\n{numbered}{conv}"

def render(tok, defn, q, regime):
    if regime == "direct":
        body = defn + "\n\n" + q + "\n\nAnswer with ONLY the name, nothing else."; think = False
    elif regime == "scripted":
        body = defn + "\n\n" + q + "\n\nWork through it ONE STEP AT A TIME along the order above, then end with a line exactly: FINAL: <name>"; think = False
    else:  # natural: no instruction; let the model reason in its own style
        body = defn + "\n\n" + q; think = True
    try: return tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False, enable_thinking=think)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False)

def parse_final(text):
    pat = "|".join(re.escape(e) for e in base)
    m = re.findall(r"FINAL[:\s]*\**\s*(" + pat + r")", text, re.I)
    if m: return next(e for e in base if e.lower() == m[-1].lower())
    allm = list(re.finditer(r"\b(" + pat + r")\b", text, re.I))
    if allm: return next(e for e in base if e.lower() == allm[-1].group(1).lower())
    return None

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    order = list(np.random.default_rng(0).permutation(base))  # si=0, same as defladder
    idx = {x.lower(): i for i, x in enumerate(order)}
    gold_imp = {(e, k): order[(idx[e.lower()] + k) % N] for e in base for k in KS}
    gold_nat = {(e, k): base[(base.index(e) + k) % N] for e in base for k in KS}
    defn = make_def(order, np.random.default_rng(1000))
    rows = [(e, k, ti) for e in base for k in KS for ti in range(NTPL)]
    for M in MODELS:
        tag = M.split("/")[-1]; outp = f"{OUTDIR}/behavior3_{tag}_{CONCEPT}.json"
        if os.path.exists(outp) and not FORCE:
            print(f"SKIP {tag} (exists)", flush=True); continue
        try: cfg = AutoConfig.from_pretrained(M)
        except Exception as e:
            print(f"SKIP {tag}: config load failed ({type(e).__name__}: {str(e)[:80]})", flush=True); continue
        nl = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
        print(f"=== {tag} (nl={nl}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        try: model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
        except Exception:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
        res = {}; trace_fh = open(f"{OUTDIR}/traces_{tag}_{CONCEPT}.jsonl", "w")  # full traces -> offline re-scoring
        for regime in REGIMES:
            mn = 16 if regime == "direct" else MAXNEW
            prompts = [render(tok, defn, TPLS[ti].format(k=k, e=e), regime) for (e, k, ti) in rows]
            by_k = {k: {"imp": 0, "nat": 0, "na": 0, "tot": 0} for k in KS}; trunc = 0; samples = []
            bs = 8 if regime != "direct" else 24
            for i in range(0, len(prompts), bs):
                b = prompts[i:i+bs]; br = rows[i:i+bs]
                enc = tok(b, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad(): g = model.generate(**enc, max_new_tokens=mn, do_sample=False, pad_token_id=tok.pad_token_id)
                new = g[:, enc["input_ids"].shape[1]:]; txt = tok.batch_decode(new, skip_special_tokens=True)
                for (e, k, ti), t, row in zip(br, txt, new):
                    a = parse_final(t); d = by_k[k]; d["tot"] += 1
                    if a is None: d["na"] += 1
                    elif a.lower() == gold_imp[(e, k)].lower(): d["imp"] += 1
                    elif a.lower() == gold_nat[(e, k)].lower(): d["nat"] += 1
                    cut = int(row.shape[0]) >= mn
                    if cut: trunc += 1
                    # full trace for offline re-scoring (never regenerate)
                    trace_fh.write(json.dumps({"regime": regime, "e": e, "k": k, "ti": ti,
                        "gold_imp": gold_imp[(e, k)], "gold_nat": gold_nat[(e, k)],
                        "truncated": cut, "parse": a, "text": t}) + "\n")
                    if len(samples) < 3: samples.append({"q": [e, k, ti], "ans": a, "tail": t[-160:]})
                print(f"  [{regime}] {min(i+bs,len(prompts))}/{len(prompts)}", flush=True)
            tot = sum(by_k[k]["tot"] for k in KS)
            kc = {k: by_k[k]["imp"] / max(1, by_k[k]["tot"]) for k in KS}
            res[regime] = dict(acc_imp=sum(by_k[k]["imp"] for k in KS)/max(1, tot),
                               acc_nat_revert=sum(by_k[k]["nat"] for k in KS)/max(1, tot),
                               na_rate=sum(by_k[k]["na"] for k in KS)/max(1, tot),
                               kcurve_imp=kc, trunc_rate=trunc/max(1, tot), maxnew=mn, samples=samples)
            print(f"  == {tag} {regime}: acc_imp={res[regime]['acc_imp']:.2f} nat-revert={res[regime]['acc_nat_revert']:.2f} "
                  f"na={res[regime]['na_rate']:.2f} trunc={res[regime]['trunc_rate']:.2f} | "
                  + " ".join(f"k{k}:{kc[k]:.2f}" for k in KS), flush=True)
        trace_fh.close()
        del model; torch.cuda.empty_cache()
        json.dump({"model": M, "concept": CONCEPT, "si": 0, "regimes": res}, open(outp, "w"), indent=2)
        print("SAVED", os.path.abspath(outp), "+ traces_%s_%s.jsonl" % (tag, CONCEPT), flush=True)

if __name__ == "__main__": main()
