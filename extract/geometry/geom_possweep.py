"""Position sweep: under imposed context + complex probe, read day geometry at fractions from the day-token position (f=0)
to the sentence-final token (f=1), grouped by day. RSA vs imposed-cyclic and natural-cyclic at each fraction → shows the
natural->imposed transition as integration accrues over sequence position. Saves centroids at f=0 and f=1 (scr0) + the
imposed order for ring plots. Usage: python -u geom_possweep.py <hf_model> [--nscr 3]"""
import os,sys,json,numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1]; NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 3; FRACD=0.75
base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N=7
COMPLEX=[
 "Last week brought a string of small surprises, and the most memorable of them unfolded on {e} when an unexpected visitor finally arrived at the old house by the harbor.",
 "In the quiet town beside the slow river the long-awaited harvest festival had been set for {e}, and preparations had quietly been underway for several weeks before anyone noticed.",
 "She had always preferred {e} above every other day of the week, ever since the summer her grandmother first taught her to bake dark bread in the old stone oven.",
 "After hours of increasingly heated debate the planning committee finally settled on {e}, a compromise that satisfied almost no one in the room but kept the fragile project moving forward.",
 "By the time {e} came around at last the snow had largely melted, the mountain roads were clear again, and the long-delayed northern journey could finally begin in earnest.",
 "Every surviving account of the strange incident agreed on a single stubborn detail, namely that it had happened on {e}, although the witnesses disagreed about nearly everything else."]
FRACS=[0.0,0.2,0.4,0.6,0.8,1.0]
def imposed_def(o):
    return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
            "\nOrder:\n"+"\n".join(f"{i+1}. {o[i]}" for i in range(N))+"\n\n")
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def cyc(p): P=np.array(p); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)
def cos_rdm(C): X=C-C.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def last_end(ids,sub):
    L=len(sub); b=None
    for i in range(len(ids)-L+1):
        if ids[i:i+L]==sub: b=i+L-1
    return b
def main():
    tag=MODEL.split("/")[-1]; cfg=AutoConfig.from_pretrained(MODEL)
    nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None); L=int(round(FRACD*nl))
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dayids={e:(tok.encode(" "+e,add_special_tokens=False) or tok.encode(e,add_special_tokens=False)) for e in base}
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText; model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    nat_rdm=cyc(list(range(N))); print(f"POSSWEEP {tag} L={L}/{nl} nscr={NSCR}",flush=True)
    rng=np.random.default_rng(0); orders=[list(rng.permutation(base)) for _ in range(NSCR)]
    C0_all=[]; C1_all=[]; ord_all=[]; ipos_all=[]; rimp_all=[]; rnat_all=[]
    for si in range(NSCR):
        o=orders[si]; ipos=[{x.lower():i for i,x in enumerate(o)}[e.lower()] for e in base]; imp_rdm=cyc(ipos)
        rows=[(e,t) for e in base for t in range(len(COMPLEX))]; prompts=[render(tok,imposed_def(o)+COMPLEX[t].format(e=e)) for (e,t) in rows]
        entF={f:{e:[] for e in base} for f in FRACS}
        for bi in range(0,len(prompts),12):
            ch=prompts[bi:bi+12]; rw=rows[bi:bi+12]; enc=tok(ch,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): H=model(**enc,output_hidden_states=True).hidden_states[L].float().cpu().numpy()
            ids_b=enc["input_ids"].cpu().tolist(); seq=H.shape[1]
            for j,(e,t) in enumerate(rw):
                ids=ids_b[j]; dp=last_end(ids,dayids[e])
                if dp is None: dp=seq-1
                lp=seq-1
                for f in FRACS:
                    pos=int(round(dp+f*(lp-dp))); entF[f][e].append(H[j,pos,:])
        C0=C1=None; ri=[]; rn=[]
        for f in FRACS:
            C=np.stack([np.mean(entF[f][e],0) for e in base]); obs=cos_rdm(C)
            ri.append(float(spearmanr(obs,imp_rdm).correlation)); rn.append(float(spearmanr(obs,nat_rdm).correlation))
            if f==0.0: C0=C
            if f==1.0: C1=C
        C0_all.append(C0); C1_all.append(C1); ord_all.append(o); ipos_all.append(ipos); rimp_all.append(ri); rnat_all.append(rn)
        print(f"  scr{si}: imp@0={ri[0]:+.2f} imp@1={ri[-1]:+.2f} nat@0={rn[0]:+.2f} nat@1={rn[-1]:+.2f} order={o}",flush=True)
    np.savez(os.path.join(data_dir("geometry"), f"possweep_{tag}_all.npz"),fracs=np.array(FRACS),tok=np.array(base),
             cents0=np.array(C0_all),cents1=np.array(C1_all),orders=np.array(ord_all),ipos_all=np.array(ipos_all),
             rimp_all=np.array(rimp_all),rnat_all=np.array(rnat_all))
    print("SAVED",os.path.abspath(os.path.join(data_dir("geometry"), f"possweep_{tag}_all.npz")),flush=True)
if __name__=="__main__": main()
