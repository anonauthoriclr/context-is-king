"""Natural 2-D grid probe: does the model natively represent a QWERTY key block as a 2-D spatial grid? Read the geometry of
a 3x3 key block (Q W E / A S D / Z X C) under a keyboard context, grouped by key, and RSA the centroid RDM against
keyboard-2-D distance (Chebyshev/Manhattan on row,col) vs 1-D nulls (reading-order row-major, and alphabetical). The 2-D
signature: column-neighbors (Q-A-Z) are close despite being far in any 1-D order. If kbd-2-D beats the 1-D nulls + the
cone-preserving null, the model stores a spatial keyboard grid (natural-grid analog of the weekday cycle). Single-letter
keys avoid the chess a1/a2 lexical confound. Usage: python -u geom_qwerty.py <hf_model>"""
import os,sys,json,numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1]; FRACD=0.75
KEYS=["Q","W","E","A","S","D","Z","X","C"]; N=9   # reading (row-major) order
COORD={"Q":(0,0),"W":(0,1),"E":(0,2),"A":(1,0),"S":(1,1),"D":(1,2),"Z":(2,0),"X":(2,1),"C":(2,2)}
PROBES=["On a standard QWERTY keyboard, consider the key {K}.","Press the {K} key on the keyboard.",
        "The {K} key in the standard keyboard layout.","When typing, you reach for the key {K}.",
        "Locate the letter {K} on the keyboard.","Here is a key from the keyboard layout: {K}."]
def cheb(): return np.array([[max(abs(COORD[a][0]-COORD[b][0]),abs(COORD[a][1]-COORD[b][1])) for b in KEYS] for a in KEYS],float)
def manh(): return np.array([[abs(COORD[a][0]-COORD[b][0])+abs(COORD[a][1]-COORD[b][1]) for b in KEYS] for a in KEYS],float)
def read1d(): return np.array([[abs(i-j) for j in range(N)] for i in range(N)],float)
ALPHA=sorted(KEYS); aidx={k:i for i,k in enumerate(ALPHA)}
def alpha1d(): return np.array([[abs(aidx[a]-aidx[b]) for b in KEYS] for a in KEYS],float)
TPL={"kbd_cheb":cheb(),"kbd_manh":manh(),"read_1d":read1d(),"alpha_1d":alpha1d()}
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def cos_rdm(C): X=C-C.mean(0);X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9);return (1-X@X.T)[np.triu_indices(N,1)]
def main():
    tag=MODEL.split("/")[-1]; cfg=AutoConfig.from_pretrained(MODEL)
    nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None);L=int(round(FRACD*nl))
    print(f"QWERTY {tag} L={L}/{nl}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL);tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText;model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    rows=[(k,t) for k in KEYS for t in range(len(PROBES))];ent={k:[] for k in KEYS}
    prompts=[render(tok,PROBES[t].format(K=k)) for (k,t) in rows]
    for bi in range(0,len(prompts),18):
        enc=tok(prompts[bi:bi+18],return_tensors="pt",padding=True).to(model.device)
        with torch.no_grad(): H=model(**enc,output_hidden_states=True).hidden_states[L][:,-1,:].float().cpu().numpy()
        for j,(k,t) in enumerate(rows[bi:bi+18]): ent[k].append(H[j])
    C=np.stack([np.mean(ent[k],0) for k in KEYS]); obs=cos_rdm(C)
    tu=np.triu_indices(N,1)
    rsa={n:float(spearmanr(obs,T[tu]).correlation) for n,T in TPL.items()}
    rng=np.random.default_rng(7);base=cheb();nv=[]
    for _ in range(2000):
        pn=list(rng.permutation(range(N)));nv.append(spearmanr(obs,np.array([[base[pn[i],pn[j]] for j in range(N)] for i in range(N)])[tu]).correlation)
    nv=np.array(nv);p=float((np.sum(nv>=rsa["kbd_cheb"])+1)/2001);best=max(TPL,key=lambda n:rsa[n])
    del model;torch.cuda.empty_cache()
    out=dict(model=MODEL,L=int(L),keys=KEYS,rsa=rsa,best=best,
             dRSA_kbd_minus_read=rsa["kbd_cheb"]-rsa["read_1d"],dRSA_kbd_minus_alpha=rsa["kbd_cheb"]-rsa["alpha_1d"],p_emp_kbd=p)
    json.dump(out,open(os.path.join(data_dir("geometry"), f"qwerty_{tag}.json"),"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"qwerty_{tag}.npz"),cents=C,keys=np.array(KEYS),coord=np.array([COORD[k] for k in KEYS]),L=L)
    print(f"== {tag}: kbd_cheb={rsa['kbd_cheb']:+.2f} kbd_manh={rsa['kbd_manh']:+.2f} read_1d={rsa['read_1d']:+.2f} "
          f"alpha_1d={rsa['alpha_1d']:+.2f} | best={best} dRSA(kbd-read)={rsa['kbd_cheb']-rsa['read_1d']:+.2f} "
          f"dRSA(kbd-alpha)={rsa['kbd_cheb']-rsa['alpha_1d']:+.2f} p_kbd={p:.3f}",flush=True)
if __name__=="__main__": main()
