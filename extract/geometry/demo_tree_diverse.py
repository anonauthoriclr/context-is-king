#!/usr/bin/env python3
"""Same imposed depth-4 tree as demo_tree.py (deterministic seed-0 tree/perm), but read under DIVERSE,
non-structural query TYPES (question / command / narrative / exclamation / statement / request) rather
than near-identical neutral paraphrases. Saves activations + the template texts to a NEW npz (does not
clobber the cached neutral run). No GPU generation -- forward-only hidden states.
Usage: python demo_tree_diverse.py <hf_model> [--depth 4]  -> data_dir("geometry")/demo_tree_diverse_<tag>_d<D>.npz"""
import sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL = sys.argv[1]
D = int(sys.argv[sys.argv.index("--depth")+1]) if "--depth" in sys.argv else 4
FRACD = 0.75; N = 2**(D+1)-1
PARENT = {i: (i-1)//2 for i in range(1, N)}; CHILDREN = {i: [c for c in (2*i+1, 2*i+2) if c < N] for i in range(N)}
TOKENS = ["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle",
          "Garden","Pencil","Marble","Feather","Comet","Harbor","Kettle","Ribbon","Maple","Socket","Pebble","Lantern",
          "Willow","Copper","Meadow","Beacon","Thistle","Quartz","Cedar","Acorn","Domino","Velvet"][:N]
# DIVERSE, non-structural query types (never mention the hierarchy) — one per register:
TPLS = ["Consider the item {e}.",          # neutral reference
        "What do you think of {e}?",        # open question
        "Tell me about {e}.",               # command
        "Yesterday I dreamed about {e}.",   # narrative
        "I absolutely love {e}!",           # emotional exclamation
        "Is {e} nearby?",                   # yes/no question
        "Once, there was {e}.",             # story opening
        "Please remember {e}.",             # instruction
        "{e}, again?!",                     # exclamation
        "Today's topic: {e}.",              # announcement
        "Could you explain {e}?",           # polite request
        "Nobody mentioned {e}."]            # statement
rng = np.random.default_rng(0); perm = list(rng.permutation(TOKENS)); node_of = {perm[k]: k for k in range(N)}
def make_def():
    internal = [p for p in range(N) if CHILDREN[p]]
    edges = [f"- {perm[p]} is the parent of " + " and ".join(perm[c] for c in CHILDREN[p]) + "." for p in internal]
    shuf = "\n".join(edges[j] for j in rng.permutation(len(edges)))
    return ("You are operating under a REDEFINED hierarchy. The ONLY valid parent/child relations apply below; the normal "
            "meanings of these words do NOT apply.\nHierarchy:\n" + shuf)
defn = make_def()
cfg = AutoConfig.from_pretrained(MODEL); nl = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None); L = int(round(FRACD*nl))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
try: model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
except Exception:
    from transformers import AutoModelForImageTextToText; model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
def render(c):
    try: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}], add_generation_prompt=True, tokenize=False)
rows = [(e, ti) for e in TOKENS for ti in range(len(TPLS))]; R = []; NODE = []
print(f"DEMO-TREE-DIVERSE {MODEL} depth={D} N={N} ntpl={len(TPLS)} L={L}/{nl}", flush=True)
for bi in range(0, len(rows), 14):
    b = rows[bi:bi+14]; prompts = [render(defn + "\n\n" + TPLS[ti].format(e=e)) for (e, ti) in b]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad(): o = model(**enc, output_hidden_states=True)
    hs = o.hidden_states[L][:, -1, :].float().cpu().numpy()
    for j, (e, ti) in enumerate(b): R.append(hs[j]); NODE.append(node_of[e])
del model; torch.cuda.empty_cache()
R = np.array(R, np.float32); NODE = np.array(NODE)
mu = R.mean(0); U, S, Vt = np.linalg.svd(R-mu, full_matrices=False); P = (R-mu) @ Vt[:3].T
tag = MODEL.split("/")[-1]
_outp=os.path.join(data_dir("geometry"), f"demo_tree_diverse_{tag}_d{D}.npz")
np.savez(_outp,
         R=R.astype(np.float16), NODE=NODE, perm=np.array(perm), P=P, L=L, tpls=np.array(TPLS))
print(f"WROTE {_outp} | rows {len(R)} ntpl {len(TPLS)}", flush=True)
