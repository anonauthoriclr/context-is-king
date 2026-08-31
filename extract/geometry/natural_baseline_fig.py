#!/usr/bin/env python3
"""BASELINE FIGURE: with NO imposed order, do the pretrained relations already form the ring at the SENTENCE-END readout
(the paper's readout, not the entity token)? Three cyclic concepts (weekdays/months/zodiac), neutral mentions + natural
k-step queries, NO redefinition. Read residual at last prompt token, pre-generation. Reports cyclic RSA (natural order)
+ permutation-null p + participation-ratio eff-dim; saves centroids + a fig1-style ring panel.
Usage: python -u natural_baseline_fig.py <hf_model>   Out: data_dir("geometry")/natbase_<tag>.{json,npz,png}"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL = sys.argv[1]; FRACD = 0.75
CONCEPTS = {
 "weekdays": (["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], "day"),
 "months":   (["January","February","March","April","May","June","July","August","September","October","November","December"], "month"),
 "zodiac":   (["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"], "sign"),
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

def main():
    tag=MODEL.split("/")[-1]; os.makedirs(data_dir("geometry"),exist_ok=True)
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"NAT-BASELINE {MODEL} L={L}/{nl}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    out={}; cents_all={}
    for cname,(base,noun) in CONCEPTS.items():
        N=len(base); KS=list(range(1,min(7,N)))
        entE={e:[] for e in base}
        rows=[(e,t,k) for e in base for t in range(len(QUERY)) for k in KS] + [(e,t,None) for e in base for t in range(len(NEUTRAL))]
        def build(e,t,k):
            return render(tok, (QUERY[t].format(k=k,e=e,noun=noun) if k is not None else NEUTRAL[t].format(e=e,noun=noun)))
        prompts=[build(e,t,k) for (e,t,k) in rows]
        for bi in range(0,len(prompts),16):
            ch=prompts[bi:bi+16]; rw=rows[bi:bi+16]
            enc=tok(ch,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            H=o.hidden_states[L][:,-1,:].float().cpu().numpy()   # SENTENCE-END (last prompt token)
            for j,(e,t,k) in enumerate(rw): entE[e].append(H[j])
        C=np.stack([np.mean(entE[e],0) for e in base]); cents_all[cname]=C
        natrdm=cyc_of(range(N),N); D=cosM(C)
        ut=lambda M:M[np.triu_indices(N,1)]
        rsa=spearmanr(ut(D),ut(natrdm)).correlation
        rng=np.random.default_rng(0); nv=[spearmanr(ut(D),ut(cyc_of(rng.permutation(N),N))).correlation for _ in range(2000)]
        p=(np.sum(np.array(nv)>=rsa)+1)/2001; ed=effdim(C)
        out[cname]={"N":N,"rsa":float(rsa),"p":float(p),"effdim":ed}
        print(f"  [{cname:8s}] SENTENCE-END natural-order RSA={rsa:+.2f} p={p:.3f} eff-dim={ed:.1f}",flush=True)
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"L":L,"nl":nl,"res":out},open(os.path.join(data_dir("geometry"), f"natbase_{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"natbase_{tag}.npz"),**{cname:cents_all[cname] for cname in CONCEPTS})
    # fig1-style panel: natural-order ring per concept, at sentence-end
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    for ax,(cname,(base,noun)) in zip(axs,CONCEPTS.items()):
        C=cents_all[cname]; N=len(base); mu=C.mean(0); _,_,Vt=np.linalg.svd(C-mu,full_matrices=False); P=(C-mu)@Vt.T
        seq=list(range(N))+[0]
        for i in range(N): ax.plot([P[seq[i],0],P[seq[i+1],0]],[P[seq[i],1],P[seq[i+1],1]],"-",color="#c0392b" if i==N-1 else "#1b7837",lw=2.1 if i==N-1 else 1.5,zorder=1)
        ax.scatter(P[:,0],P[:,1],c=range(N),cmap="hsv",s=70,edgecolors="k",linewidths=.4,zorder=2)
        for i,e in enumerate(base): ax.annotate(e[:3],(P[i,0],P[i,1]),fontsize=6,ha="center",va="center")
        ax.set_xticks([]);ax.set_yticks([]); ax.set_title(f"{cname}\nnat-RSA={out[cname]['rsa']:+.2f}, eff-dim={out[cname]['effdim']:.1f}",fontsize=9)
    fig.suptitle(f"{tag}: NO imposed order — pretrained relations at SENTENCE-END (2D PCA; traced in natural order, red=wrap)",fontsize=11)
    fig.tight_layout(); fn=os.path.join(data_dir("geometry"), f"natbase_{tag}.png"); fig.savefig(fn,dpi=140); print("SAVED",os.path.abspath(fn),flush=True)
if __name__=="__main__": main()
