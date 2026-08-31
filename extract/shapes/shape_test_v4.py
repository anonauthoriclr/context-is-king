"""Shapes-not-orders FIRMED v4 = v3 + intrinsic dimensionality (participation ratio, lambda2/lambda1)
per condition per scramble (line-vs-ring discriminator), + saves scramble-0 centroids for figures.
Same 7 weekdays, 5 conditions, 10 scrambles, 12 queries, day-token + last-token. Usage: python shape_test_v4.py <hf_model>"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir
MODEL=sys.argv[1] if len(sys.argv)>1 else "google/gemma-4-31B-it"; FRACD=0.75; NSCR=10
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; N=7
NEUTRAL=["Consider {e}.","Take note of {e}.","The item is {e}.","Focus on {e}.","Here is an item: {e}.","Regarding {e}."]
DIVERSE=["What do you think of {e}?","Tell me about {e}.","Yesterday I recalled {e}.","I really like {e}!","{e}, once more.","Today's note: {e}."]
QSET=NEUTRAL+DIVERSE; KHOP=["What is {k} days after {e}?","{k} days after {e} is?","Starting at {e}, go {k} days forward. Which day?"]
DEPTH=np.array([0,1,1,2,2,2,2]); CHILD={0:[1,2],1:[3,4],2:[5,6]}; tri=np.triu_indices(N,1)
def cyc(p): P=np.array(p); D=np.abs(P[:,None]-P[None,:]); D=np.minimum(D,N-D); return D[tri].astype(float)
def lin(p): P=np.array(p); return np.abs(P[:,None]-P[None,:])[tri].astype(float)
def cos_rdm(C): X=C-C.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return 1-X@X.T
def closure(C,order): D=cos_rdm(C); s=list(order); adj=[D[s[i],s[i+1]] for i in range(N-1)]; return float(D[s[-1],s[0]]/(np.mean(adj)+1e-9))
def dimstats(C): X=C-C.mean(0); lam=np.linalg.svd(X,compute_uv=False)**2; lam=lam/lam.sum(); return float(1.0/np.sum(lam**2)), float(lam[1]/lam[0])
def cyc_rule(o): return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal "
    "order does NOT apply. It is a closed cycle (it wraps).\nOrder:\n"+"\n".join(f"- {o[i]} is immediately followed by {o[(i+1)%N]}." for i in range(N))+"\n\n")
def line_rule(o): return ("You are operating in a REDEFINED calendar. The ONLY valid order applies below; the normal "
    f"order does NOT apply. It is a one-way list, NOT a cycle: it does not wrap, and {o[-1]} has no day after it.\nOrder:\n"+
    "\n".join(f"- {o[i]} is immediately followed by {o[i+1]}." for i in range(N-1))+"\n\n")
def tree_rule(a,rng):
    edges=[f"- {a[p]} is the parent of {a[c[0]]} and {a[c[1]]}." for p,c in CHILD.items()]
    return ("You are operating under a REDEFINED hierarchy. The ONLY valid parent/child relations apply below; the normal "
            "meanings of these words do NOT apply.\nHierarchy:\n"+"\n".join(edges[j] for j in rng.permutation(len(edges)))+"\n\n")

def main():
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); tag=MODEL.split("/")[-1]; print(f"SHAPE-V4 {tag} L={L}/{nl} nscr={NSCR} |Q|={len(QSET)}",flush=True)
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
    def centroids(pbd):
        rows=[(e,i) for e in DAYS for i in range(len(pbd[e]))]; allp=[pbd[e][i] for e,i in rows]; aD={e:[] for e in DAYS}; aE={e:[] for e in DAYS}
        for bi in range(0,len(allp),8):
            enc=tok([render(x) for x in allp[bi:bi+8]],return_tensors="pt",padding=True).to(model.device)
            with torch.no_grad(): H=model(**enc,output_hidden_states=True).hidden_states[L].float().cpu().numpy()
            ids_b=enc["input_ids"].cpu().tolist()
            for j,(e,i) in enumerate(rows[bi:bi+8]):
                dp=last_pos(ids_b[j],e); aD[e].append(H[j,dp if dp is not None else -1]); aE[e].append(H[j,-1])
        return np.stack([np.mean(aD[e],0) for e in DAYS]), np.stack([np.mean(aE[e],0) for e in DAYS]), aD, aE
    def _cloud(a):  # dict entity->list[vec] -> (points[P,d], entity_label[P]); the cloud behind each centroid
        return np.concatenate([np.stack(a[e]) for e in DAYS]), np.array([e for e in DAYS for _ in a[e]])
    def neutral(rule): return {e:[rule+q.format(e=e) for q in QSET] for e in DAYS}
    def khop(): return {e:[q.format(k=k,e=e) for q in KHOP for k in (1,2,3)] for e in DAYS}

    rng=np.random.default_rng(0); rows=[]; cent0={}
    def rec(nm,pos,si,C,ipos,dtmpl=None):
        o=cos_rdm(C)[tri]; order=(np.argsort(ipos).tolist() if ipos is not None else list(range(N)))
        pr,l2=dimstats(C); ip=ipos if ipos is not None else list(range(N))
        rows.append((nm,pos,si,spearmanr(o,cyc(ip)).correlation,spearmanr(o,lin(ip)).correlation,
                     (spearmanr(o,dtmpl).correlation if dtmpl is not None else np.nan),closure(C,order),pr,l2))
    CD,CE,aD,aE=centroids(neutral(""));    rec("NAT-neutral","day",-1,CD,None); rec("NAT-neutral","end",-1,CE,None); cent0["NAT_day"]=CD
    cent0["NAT_day_pts"],cent0["NAT_day_pts_ent"]=_cloud(aD)   # point cloud behind the natural ring
    CD,CE,aD,aE=centroids(khop());         rec("NAT-structural","day",-1,CD,None); rec("NAT-structural","end",-1,CE,None)
    for si in range(NSCR):
        order=list(rng.permutation(DAYS)); ipos=[order.index(d) for d in DAYS]
        for nm,rule in [("CYCLE",cyc_rule(order)),("LINE",line_rule(order))]:
            CD,CE,aD,aE=centroids(neutral(rule)); rec(nm,"day",si,CD,ipos); rec(nm,"end",si,CE,ipos)
            if si==0:
                cent0[nm+"_end"]=CE; cent0[nm+"_ipos"]=np.array(ipos)
                cent0[nm+"_end_pts"],cent0[nm+"_end_pts_ent"]=_cloud(aE)
        perm=list(rng.permutation(DAYS)); assign=dict(enumerate(perm)); node_of={perm[k]:k for k in range(N)}
        dvec=np.array([DEPTH[node_of[d]] for d in DAYS]); dtmpl=np.abs(dvec[:,None]-dvec[None,:])[tri].astype(float)
        CD,CE,aD,aE=centroids(neutral(tree_rule(assign,rng))); rec("TREE","day",si,CD,ipos,dtmpl); rec("TREE","end",si,CE,ipos,dtmpl)
        if si==0:
            cent0["TREE_end"]=CE; cent0["tree_depth"]=dvec; cent0["tree_edges"]=np.array([(DAYS.index(assign[p]),DAYS.index(assign[c])) for p,ch in CHILD.items() for c in ch])
            cent0["TREE_end_pts"],cent0["TREE_end_pts_ent"]=_cloud(aE)
        print(f"  scr{si} done",flush=True)
    import collections; agg=collections.defaultdict(list)
    for r in rows: agg[(r[0],r[1])].append(r[3:])
    def ms(x): x=np.array(x,float); x=x[~np.isnan(x)]; return (np.mean(x),np.std(x)) if len(x) else (np.nan,0.0)
    print("\n cond            pos   cyc-RSA      depth-RSA    closure      partR(effdim)  l2/l1")
    for nm in ["NAT-neutral","NAT-structural","CYCLE","LINE","TREE"]:
        for pos in ["day","end"]:
            a=np.array(agg[(nm,pos)],float); cy=ms(a[:,0]); dp=ms(a[:,2]); cl=ms(a[:,3]); pr=ms(a[:,4]); l2=ms(a[:,5])
            print(f" {nm:14s} {pos:3s}  {cy[0]:+.2f}±{cy[1]:.2f}  {dp[0]:+.2f}±{dp[1]:.2f}  {cl[0]:.2f}±{cl[1]:.2f}  {pr[0]:.2f}±{pr[1]:.2f}   {l2[0]:.2f}±{l2[1]:.2f}")
    SP=data_dir("shapes"); os.makedirs(SP,exist_ok=True)
    np.savez(f"{SP}/shape_v4cent_{tag}.npz", days=np.array(DAYS), **cent0)
    json.dump([{"cond":r[0],"pos":r[1],"scr":r[2],"cyc":r[3],"lin":r[4],"depth":(None if r[5]!=r[5] else r[5]),"closure":r[6],"partR":r[7],"l2l1":r[8]} for r in rows],
              open(f"{SP}/shape_v4_{tag}.json","w"),indent=2)
    print(f"\nSAVED shape_v4_{tag}.json + shape_v4cent_{tag}.npz",flush=True)
if __name__=="__main__": main()
