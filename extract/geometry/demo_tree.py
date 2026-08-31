#!/usr/bin/env python3
"""Interactive 3D PCA POINT-CLOUD demo of an imposed HIERARCHY (spin it yourself). One tree, many neutral paraphrases →
dense per-sample residual cloud; PCA→3D; plotly scatter3d colored by depth + parent edges + rich hover (token, depth,
path). Self-contained HTML (plotly inlined). Neutral queries (ask nothing structural) = the intrinsic rendering.
Usage: python demo_tree.py <hf_model> [--depth 3] [--ntpl 24]  ->  data_dir("geometry")/demo_tree_<tag>_d<D>.html"""
import sys, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import plotly.graph_objects as go
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1]
D=int(sys.argv[sys.argv.index("--depth")+1]) if "--depth" in sys.argv else 3
NTPL=int(sys.argv[sys.argv.index("--ntpl")+1]) if "--ntpl" in sys.argv else 24
FRACD=0.75; N=2**(D+1)-1
PARENT={i:(i-1)//2 for i in range(1,N)}; CHILDREN={i:[c for c in (2*i+1,2*i+2) if c<N] for i in range(N)}
def anc(n):
    a=[n]
    while n!=0:n=PARENT[n];a.append(n)
    return a
DEP={n:len(anc(n))-1 for n in range(N)}
TOKENS=["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle",
        "Garden","Pencil","Marble","Feather","Comet","Harbor","Kettle","Ribbon","Maple","Socket","Pebble","Lantern",
        "Willow","Copper","Meadow","Beacon","Thistle","Quartz","Cedar","Acorn","Domino","Velvet"][:N]
TPLS=["Consider the item {e}.","Take note of {e}.","The item is {e}.","Focus on {e}.","Here is an item: {e}.",
      "Item of interest: {e}.","Note this item: {e}.","Regarding the item {e}.","We turn to {e}.","Look at {e}.",
      "The item named {e}.","Observe the item {e}.","About the item {e}.","This concerns {e}.","Attend to {e}.",
      "The relevant item is {e}.","Item under review: {e}.","Consider {e} now.","The item {e}, specifically.",
      "Examine {e}.","Recall the item {e}.","The chosen item: {e}.","Point to the item {e}.","Item: {e}."][:NTPL]
rng=np.random.default_rng(0); perm=list(rng.permutation(TOKENS))      # one fixed tree
node_of={perm[k]:k for k in range(N)}
def make_def():
    internal=[p for p in range(N) if CHILDREN[p]]
    edges=[f"- {perm[p]} is the parent of "+" and ".join(perm[c] for c in CHILDREN[p])+"." for p in internal]
    shuf="\n".join(edges[j] for j in rng.permutation(len(edges)))
    return ("You are operating under a REDEFINED hierarchy. The ONLY valid parent/child relations apply below; the normal "
            "meanings of these words do NOT apply.\nHierarchy:\n"+shuf)
defn=make_def()
cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None); L=int(round(FRACD*nl))
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
except Exception:
    from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
def render(c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
rows=[(e,ti) for e in TOKENS for ti in range(len(TPLS))]; R=[]; NODE=[]
print(f"DEMO-TREE {MODEL} depth={D} N={N} ntpl={len(TPLS)} L={L}/{nl}",flush=True)
for bi in range(0,len(rows),14):
    b=rows[bi:bi+14]; prompts=[render(defn+"\n\n"+TPLS[ti].format(e=e)) for (e,ti) in b]
    enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
    with torch.no_grad(): o=model(**enc,output_hidden_states=True)
    hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
    for j,(e,ti) in enumerate(b): R.append(hs[j]); NODE.append(node_of[e])
del model; torch.cuda.empty_cache()
R=np.array(R,np.float32); NODE=np.array(NODE)
mu=R.mean(0); U,S,Vt=np.linalg.svd(R-mu,full_matrices=False); P=(R-mu)@Vt[:3].T; ev=(S**2/(S**2).sum())[:3]
tag=MODEL.split("/")[-1]
np.savez(os.path.join(data_dir("geometry"), f"demo_tree_{tag}_d{D}.npz"),R=R.astype(np.float16),NODE=NODE,perm=np.array(perm),P=P,L=L)
# hover: token + depth + path root->leaf
def path(n):
    a=anc(n)[::-1]; return " → ".join(perm[x] for x in a)
hov=[f"{perm[n]}<br>depth {DEP[n]}<br>{path(n)}" for n in NODE]
pts=go.Scatter3d(x=P[:,0],y=P[:,1],z=P[:,2],mode="markers",
    marker=dict(size=3.5,color=[DEP[n] for n in NODE],colorscale="Viridis",opacity=0.8,colorbar=dict(title="depth")),
    text=hov,hoverinfo="text",name="per-sample residuals")
cent=np.stack([P[NODE==n].mean(0) for n in range(N)])
edges_x=[];edges_y=[];edges_z=[]
for c,p in PARENT.items():
    edges_x+=[cent[c,0],cent[p,0],None];edges_y+=[cent[c,1],cent[p,1],None];edges_z+=[cent[c,2],cent[p,2],None]
lines=go.Scatter3d(x=edges_x,y=edges_y,z=edges_z,mode="lines",line=dict(color="rgba(0,0,0,0.35)",width=3),name="tree edges (centroids)")
cdots=go.Scatter3d(x=cent[:,0],y=cent[:,1],z=cent[:,2],mode="markers+text",
    marker=dict(size=6,color=[DEP[n] for n in range(N)],colorscale="Viridis",line=dict(color="black",width=1)),
    text=[perm[n] for n in range(N)],textposition="top center",textfont=dict(size=8),name="nodes")
fig=go.Figure([pts,lines,cdots])
fig.update_layout(title=f"Spin the imposed hierarchy — depth-{D} tree, {N} nodes ({D+1} levels)<br>"
    f"<sub>{tag}, arbitrary tokens, NEUTRAL queries (ask nothing); per-sample residuals, 3D PCA "
    f"(EV {ev.round(2)}). Color = tree depth. Drag to rotate.</sub>",
    scene=dict(xaxis_title="PC1",yaxis_title="PC2",zaxis_title="PC3"),template="plotly_white",width=950,height=800)
out=os.path.join(data_dir("geometry"), f"demo_tree_{tag}_d{D}.html"); fig.write_html(out,include_plotlyjs=True,full_html=True)
print("WROTE",out,"| points",len(P),"| EV",ev.round(3),flush=True)
