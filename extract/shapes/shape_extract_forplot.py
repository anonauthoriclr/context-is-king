"""Extract day-centroids for ONE representative scramble (seed 0) across conditions, for 3D-PCA
eyeballing. Saves centroids + metadata to npz (no plotting here). Usage: python shape_extract_forplot.py <hf_model>"""
import os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1] if len(sys.argv)>1 else "google/gemma-4-31B-it"; FRACD=0.75
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N=7
NEUTRAL=["Consider {e}.","Take note of {e}.","The item is {e}.","Focus on {e}.","Here is an item: {e}.","Regarding {e}."]
DIVERSE=["What do you think of {e}?","Tell me about {e}.","Yesterday I recalled {e}.","I really like {e}!","{e}, once more.","Today's note: {e}."]
QSET=NEUTRAL+DIVERSE; CHILD={0:[1,2],1:[3,4],2:[5,6]}; DEPTH=[0,1,1,2,2,2,2]
def cyc_rule(o): return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal "
    "order does NOT apply. It is a closed cycle (it wraps).\nOrder:\n"+"\n".join(f"- {o[i]} is immediately followed by {o[(i+1)%N]}." for i in range(N))+"\n\n")
def line_rule(o): return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal "
    f"order does NOT apply. It is a one-way list, NOT a cycle: it does not wrap, and {o[-1]} has no day after it.\nOrder:\n"+
    "\n".join(f"- {o[i]} is immediately followed by {o[i+1]}." for i in range(N-1))+"\n\n")
def tree_rule(a,rng):
    edges=[f"- {a[p]} is the parent of {a[c[0]]} and {a[c[1]]}." for p,c in CHILD.items()]
    return ("You are operating under a REDEFINED hierarchy. The ONLY valid parent/child relations apply below; the "
            "normal meanings of these words do NOT apply.\nHierarchy:\n"+"\n".join(edges[j] for j in rng.permutation(len(edges)))+"\n\n")

cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
L=int(round(FRACD*nl)); tag=MODEL.split("/")[-1]; print(f"EXTRACT-FORPLOT {tag} L={L}/{nl}",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
dayids={e:[s for s in [tok.encode(" "+e,add_special_tokens=False),tok.encode(e,add_special_tokens=False)] if s] for e in DAYS}
def last_pos(ids,e):
    best=None
    for sub in dayids[e]:
        Ls=len(sub)
        for i in range(len(ids)-Ls+1):
            if ids[i:i+Ls]==sub: best=i+Ls-1
    return best
try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
except Exception:
    from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
def render(c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def centroids(pbd,pos):  # pos: 'day' or 'end'
    rows=[(e,i) for e in DAYS for i in range(len(pbd[e]))]; allp=[pbd[e][i] for e,i in rows]; acc={e:[] for e in DAYS}
    for bi in range(0,len(allp),8):
        enc=tok([render(x) for x in allp[bi:bi+8]],return_tensors="pt",padding=True).to(model.device)
        with torch.no_grad(): H=model(**enc,output_hidden_states=True).hidden_states[L].float().cpu().numpy()
        ids_b=enc["input_ids"].cpu().tolist()
        for j,(e,i) in enumerate(rows[bi:bi+8]):
            if pos=="day": dp=last_pos(ids_b[j],e); acc[e].append(H[j,dp if dp is not None else -1])
            else: acc[e].append(H[j,-1])
    return np.stack([np.mean(acc[e],0) for e in DAYS])
def neutral(rule): return {e:[rule+q.format(e=e) for q in QSET] for e in DAYS}

rng=np.random.default_rng(0); order=list(rng.permutation(DAYS)); ipos=[order.index(d) for d in DAYS]
perm=list(rng.permutation(DAYS)); assign=dict(enumerate(perm)); node_of={perm[k]:k for k in range(N)}
dvec=[DEPTH[node_of[d]] for d in DAYS]
tree_edges=np.array([(DAYS.index(assign[p]), DAYS.index(assign[c])) for p,ch in CHILD.items() for c in ch])
out={}
out["NAT_day"]=centroids(neutral(""),"day")
out["CYCLE_end"]=centroids(neutral(cyc_rule(order)),"end")
out["LINE_end"]=centroids(neutral(line_rule(order)),"end")
out["TREE_end"]=centroids(neutral(tree_rule(assign,rng)),"end")
SP=data_dir("shapes"); os.makedirs(SP, exist_ok=True)
np.savez(f"{SP}/shape_plot_{tag}.npz", days=np.array(DAYS), ipos=np.array(ipos), depth=np.array(dvec),
         tree_edges=tree_edges, nat_natorder=np.arange(N), **out)
print(f"SAVED shape_plot_{tag}.npz",flush=True)
