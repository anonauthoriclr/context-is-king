#!/usr/bin/env python3
"""Query-induction control (v2, complex probes): does the IMPOSED day-ring appear under LONG, complex sentences that
embed the day MID-sentence (content after it, so the day is NOT the final token)? Read geometry at TWO positions:
(a) the day-token position (the day's contextualized representation), and (b) the genuine sentence-final token (not the
day). If imposed-cyclic RSA dominates natural at either readout under these rich probes, the reorganization is intrinsic,
not cued by a k-hop question and not an artifact of the day being the last token. {imposed,natural} x {neutral,complex}.
Cone-preserving null. Usage: python -u geom_storyprobe.py <hf_model> [--nscr 4]   Out: data_dir("geometry")/storyprobe_<tag>.json"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 4
FRACD=0.75
base=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N=len(base)
PROBES={
  "neutral":["Consider the day {e} for a moment.","Take note of the day {e}, please.",
             "The day under discussion here is {e} today.","Let us focus on the day {e} now.",
             "Here is a day to think about: {e} indeed.","The day of interest right now is {e}."],
  "complex":[ # day embedded mid-sentence, substantial content AFTER it (day is not the final token)
    "Last week brought a string of small surprises, and the most memorable of them unfolded on {e} when an unexpected visitor finally arrived at the old house by the harbor.",
    "In the quiet town beside the slow river the long-awaited harvest festival had been set for {e}, and preparations had quietly been underway for several weeks before anyone noticed.",
    "She had always preferred {e} above every other day of the week, ever since the summer her grandmother first taught her to bake dark bread in the old stone oven.",
    "After hours of increasingly heated debate the planning committee finally settled on {e}, a compromise that satisfied almost no one in the room but kept the fragile project moving forward.",
    "By the time {e} came around at last the snow had largely melted, the mountain roads were clear again, and the long-delayed northern journey could finally begin in earnest.",
    "Every surviving account of the strange incident agreed on a single stubborn detail, namely that it had happened on {e}, although the witnesses disagreed about nearly everything else."]}
def imposed_def(order):
    numbered="\n".join(f"{i+1}. {order[i]}" for i in range(N))
    pre="You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal order does NOT apply."
    return f"{pre}\nOrder:\n{numbered}\n\n"
def render(tok,c):
    try: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c}],add_generation_prompt=True,tokenize=False)
def cyc(pos): P=np.array(pos); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[np.triu_indices(N,1)].astype(float)
def cos_rdm(C): X=C-C.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)[np.triu_indices(N,1)]
def last_subseq_end(ids, sub):  # end index (inclusive) of the LAST occurrence of token-subseq `sub` in `ids`
    L=len(sub); best=None
    for i in range(len(ids)-L+1):
        if ids[i:i+L]==sub: best=i+L-1
    return best

def main():
    tag=MODEL.split("/")[-1]; os.makedirs(data_dir("geometry"),exist_ok=True); OUT=os.path.join(data_dir("geometry"), f"storyprobe_{tag}.json")
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"STORYPROBE-v2 {MODEL} L={L}/{nl} nscr={NSCR}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dayids={e:(tok.encode(" "+e,add_special_tokens=False) or tok.encode(e,add_special_tokens=False)) for e in base}
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    natpos=list(range(N)); nat_rdm=cyc(natpos)
    def centroids(defn,probelist):  # returns (cents_daypos, cents_lasttok), each [N,d]; n_missed day positions
        rows=[(e,t) for e in base for t in range(len(probelist))]; entD={e:[] for e in base}; entL={e:[] for e in base}; miss=0
        prompts=[render(tok,defn+probelist[t].format(e=e)) for (e,t) in rows]
        for bi in range(0,len(prompts),12):
            chunk=prompts[bi:bi+12]; rws=rows[bi:bi+12]
            enc=tok(chunk,return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            H=o.hidden_states[L].float().cpu().numpy()  # [b, seq, d] (left-padded)
            ids_b=enc["input_ids"].cpu().tolist(); am=enc["attention_mask"].cpu().tolist()
            for j,(e,t) in enumerate(rws):
                ids=ids_b[j]; entL[e].append(H[j,-1,:])  # last token (left-padded => index -1 is real last)
                pos=last_subseq_end(ids,dayids[e])
                if pos is None: miss+=1; entD[e].append(H[j,-1,:])  # fallback to last
                else: entD[e].append(H[j,pos,:])
        cD=np.stack([np.mean(entD[e],0) for e in base]); cL=np.stack([np.mean(entL[e],0) for e in base])
        return cD,cL,miss
    def nullp(obs,imp_rdm,seed):
        rng=np.random.default_rng(seed); ri=spearmanr(obs,imp_rdm).correlation
        nv=[spearmanr(obs,cyc(list(rng.permutation(N)))).correlation for _ in range(2000)]
        return float((np.sum(np.array(nv)>=ri)+1)/2001)
    res={}; rng_o=np.random.default_rng(0); orders=[list(rng_o.permutation(base)) for _ in range(NSCR)]
    for probe,plist in PROBES.items():
        cDn,cLn,_=centroids("",plist)  # NATURAL context (Control 1)
        res[f"natural_{probe}"]={"rsa_nat_daypos":float(spearmanr(cos_rdm(cDn),nat_rdm).correlation),
                                 "rsa_nat_lasttok":float(spearmanr(cos_rdm(cLn),nat_rdm).correlation)}
        agg={k:[] for k in["impD","natD","impL","natL","pD","pL"]}
        for si in range(NSCR):
            order=orders[si]; ipos=[ {x.lower():i for i,x in enumerate(order)}[e.lower()] for e in base]; imp_rdm=cyc(ipos)
            cD,cL,miss=centroids(imposed_def(order),plist); oD=cos_rdm(cD); oL=cos_rdm(cL)
            agg["impD"].append(spearmanr(oD,imp_rdm).correlation); agg["natD"].append(spearmanr(oD,nat_rdm).correlation)
            agg["impL"].append(spearmanr(oL,imp_rdm).correlation); agg["natL"].append(spearmanr(oL,nat_rdm).correlation)
            agg["pD"].append(nullp(oD,imp_rdm,7000+si)); agg["pL"].append(nullp(oL,imp_rdm,8000+si))
        m=lambda k: float(np.mean(agg[k]))
        res[f"imposed_{probe}"]=dict(
            daypos=dict(imp=m("impD"),nat_ret=m("natD"),dominance=m("impD")-m("natD"),nsig=int(sum(p<0.05 for p in agg["pD"]))),
            lasttok=dict(imp=m("impL"),nat_ret=m("natL"),dominance=m("impL")-m("natL"),nsig=int(sum(p<0.05 for p in agg["pL"]))),
            nscr=NSCR, day_pos_missed=int(miss))
        print(f"  [{probe}] NAT(daypos/last)={res['natural_'+probe]['rsa_nat_daypos']:+.2f}/{res['natural_'+probe]['rsa_nat_lasttok']:+.2f} "
              f"| IMP daypos: imp={m('impD'):+.2f} nat={m('natD'):+.2f} dom={m('impD')-m('natD'):+.2f} sig{sum(p<0.05 for p in agg['pD'])}/{NSCR}"
              f" | IMP lasttok: imp={m('impL'):+.2f} nat={m('natL'):+.2f} dom={m('impL')-m('natL'):+.2f} sig{sum(p<0.05 for p in agg['pL'])}/{NSCR}",flush=True)
    del model; torch.cuda.empty_cache()
    json.dump({"model":MODEL,"L":int(L),"nscr":NSCR,"res":res},open(OUT,"w"),indent=2); print("SAVED",os.path.abspath(OUT),flush=True)
if __name__=="__main__": main()
