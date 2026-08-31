#!/usr/bin/env python3
"""E5 (construction-vs-store discriminator): does the renderer draw a genuinely 2-D topology that CANNOT be a stored
primitive? Impose a declarative 3x3 GRID on arbitrary tokens ('right of' / 'below' relations), read the entity-layout
geometry (snap regime, same as geom_tree.py), and test whether the 9 centroids match **2-D grid (Manhattan) distance**,
BEATING a reading-order LINE template and a RING template. The line template is the key null: a 3x3 read row-major is
also consistent with the model laying tokens in a 1-D line; the 2-D signature is that COLUMN-neighbors (A-D-G) stay close
despite being reading-distance 3 apart. PASS => account (3) on-demand construction (a 3x3 grid of arbitrary tokens is no
plausible pretrained primitive); grid==line => 1-D collapse (scope back); null => 2-D is the rendering boundary.

Grid cells 0..8 row-major: coord(i)=(i//3, i%3).  right-of: (r,c)->(r,c+1);  below: (r,c)->(r+1,c).
Probe: declare the grid (visual block) or shuffled atomic edges (--sepedges, co-occurrence control); query each entity
(right/below/above/left/row/col) x templates; read last-token pre-gen residual (thinking-off snap); group by entity;
centroid RDM vs grid/line/ring RDM (Spearman). Null: cone-preserving label-shuffle (permute token<->cell), EMPIRICAL
tail p-value (not sigma). Controls: nscr token->cell scrambles, --sepedges, --qset neutral, column signature.
Usage: python -u geom_grid.py <hf_model> [--nscr 6] [--sepedges] [--qset rel|neutral] [--base]
Out: data_dir("geometry")/grid_<tag>.json (+ .npz) + FIG_grid_<tag>.png"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

MODEL=sys.argv[1]; IS_BASE="--base" in sys.argv
SEPEDGES="--sepedges" in sys.argv   # co-occurrence control: atomic shuffled edges, row/col mates NOT co-listed
NSCR=int(sys.argv[sys.argv.index("--nscr")+1]) if "--nscr" in sys.argv else 6
FRACD=0.75
TOKENS=["Apple","River","Chair","Cloud","Tiger","Bridge","Lamp","Anchor","Violin","Pepper","Saddle","Mirror","Candle"]
R=3; C=3; N=R*C                                   # 3x3 grid
TOK9=TOKENS[:N]; assert len(TOK9)==N
COORD={i:(i//C,i%C) for i in range(N)}            # cell index -> (row,col), row-major
QSET=sys.argv[sys.argv.index("--qset")+1] if "--qset" in sys.argv else "rel"
ROWNM=["top","middle","bottom"]; COLNM=["left","middle","right"]
TPLS=["What is immediately right of {e}?","What is immediately below {e}?","What is immediately above {e}?",
      "What is immediately to the left of {e}?","Which row contains {e}? (top/middle/bottom)",
      "Which column contains {e}? (left/middle/right)"]
if QSET=="neutral":  # DECISIVE CONTROL: present entity in the grid context, ask NOTHING structural (intrinsic geometry)
    TPLS=["Consider the item {e}.","Take note of the item {e}.","The item under discussion is {e}.",
          "Focus on this item: {e}.","Here is an item from the set: {e}.","Item of interest: {e}."]
tag=MODEL.split("/")[-1]+("_base" if IS_BASE else "")+("" if QSET=="rel" else f"_{QSET}")+("_sep" if SEPEDGES else "")
os.makedirs(data_dir("geometry"),exist_ok=True); OUT=os.path.join(data_dir("geometry"), f"grid_{tag}.json")

def grid_dist():  # 9x9 Manhattan distance on (row,col); + line (reading-order) and ring templates
    D=np.array([[abs(COORD[i][0]-COORD[j][0])+abs(COORD[i][1]-COORD[j][1]) for j in range(N)] for i in range(N)],float)
    LIN=np.array([[abs(i-j) for j in range(N)] for i in range(N)],float)              # reading-order line
    RNG=np.array([[min(abs(i-j),N-abs(i-j)) for j in range(N)] for i in range(N)],float)  # ring on reading order
    return D,LIN,RNG
GRID_D,LIN_D,RING_D=grid_dist()
VNB=[(i,j) for i in range(N) for j in range(N) if i<j and COORD[i][1]==COORD[j][1] and abs(COORD[i][0]-COORD[j][0])==1]  # vertical neighbors (grid-dist 1, reading-dist 3)
HNB=[(i,j) for i in range(N) for j in range(N) if i<j and COORD[i][0]==COORD[j][0] and abs(COORD[i][1]-COORD[j][1])==1]  # horizontal neighbors (grid-dist 1, reading-dist 1)
FAR=[(i,j) for i in range(N) for j in range(N) if i<j and GRID_D[i,j]>=3]            # grid-distant pairs

def make_def(perm,rng):  # perm[k] = token assigned to cell k. Either visual grid block, or shuffled atomic edges.
    pre=("You are operating under a REDEFINED spatial layout. The ONLY valid 'right of' / 'below' relations apply below; "
         "the normal meanings of these words do NOT apply.")
    conv=("\nConvention: 'right of X' = the next item in X's row; 'below X' = the next item in X's column.")
    if SEPEDGES:   # atomic relations, one per line, shuffled => row/col mates are NOT co-listed (co-occurrence control)
        edges=[]
        for r in range(R):
            for c in range(C-1): edges.append(f"- {perm[r*C+c+1]} is immediately right of {perm[r*C+c]}.")
        for r in range(R-1):
            for c in range(C): edges.append(f"- {perm[(r+1)*C+c]} is immediately below {perm[r*C+c]}.")
        order=list(rng.permutation(len(edges))); body="\n".join(edges[j] for j in order)
        return f"{pre}\nRelations:\n{body}{conv}"
    rows="\n".join(f"Row {r} ({ROWNM[r]}): " + "  ".join(perm[r*C+c] for c in range(C)) for r in range(R))
    return f"{pre}\nLayout:\n{rows}{conv}"
SUFFIX="" if QSET=="neutral" else "\n\nAnswer with ONLY the name(s), nothing else."
def render(tok,c):
    if IS_BASE: return c+("\n" if QSET=="neutral" else "\n\nAnswer: ")
    try: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role":"user","content":c+SUFFIX}],add_generation_prompt=True,tokenize=False)
def cos_rdm_full(c): X=c-c.mean(0); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); return (1-X@X.T)
def upper(M): return M[np.triu_indices(N,1)]

def main():
    rng_o=np.random.default_rng(0); perms=[list(rng_o.permutation(TOK9)) for _ in range(NSCR)]
    cfg=AutoConfig.from_pretrained(MODEL); nl=getattr(cfg,"num_hidden_layers",None) or getattr(getattr(cfg,"text_config",None),"num_hidden_layers",None)
    L=int(round(FRACD*nl)); print(f"GRID {MODEL} nscr={NSCR} L={L}/{nl} sep={SEPEDGES} qset={QSET}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    try: model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0}).eval()
    stats=[]; allc={}
    for si in range(NSCR):
        perm=perms[si]; rng_e=np.random.default_rng(1000+si); defn=make_def(perm,rng_e)
        cell_of={perm[k]:k for k in range(N)}                       # which grid cell each token occupies
        rows=[(e,ti) for e in TOK9 for ti in range(len(TPLS))]; ent_res={e:[] for e in TOK9}
        for bi in range(0,len(rows),18):
            b=rows[bi:bi+18]; prompts=[render(tok,defn+"\n\n"+TPLS[ti].format(e=e)) for (e,ti) in b]
            enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=IS_BASE).to(model.device)
            with torch.no_grad(): o=model(**enc,output_hidden_states=True)
            hs=o.hidden_states[L][:,-1,:].float().cpu().numpy()
            for j,(e,ti) in enumerate(b): ent_res[e].append(hs[j])
        cents=np.stack([np.mean(ent_res[e],0) for e in TOK9])       # 9 entity centroids (token order)
        cells=[cell_of[e] for e in TOK9]                            # cell index per centroid
        full=cos_rdm_full(cents); obs=upper(full)
        grid_rdm=upper(np.array([[GRID_D[cells[i],cells[j]] for j in range(N)] for i in range(N)]))
        lin_rdm =upper(np.array([[LIN_D [cells[i],cells[j]] for j in range(N)] for i in range(N)]))
        ring_rdm=upper(np.array([[RING_D[cells[i],cells[j]] for j in range(N)] for i in range(N)]))
        rsa_grid=float(spearmanr(obs,grid_rdm).correlation)
        rsa_lin =float(spearmanr(obs,lin_rdm ).correlation)
        rsa_ring=float(spearmanr(obs,ring_rdm).correlation)
        # cone-preserving label-shuffle null: permute token<->cell, recompute grid RSA; EMPIRICAL tail p
        rng_n=np.random.default_rng(7000+si); nullv=[]
        for _ in range(2000):
            pn=list(rng_n.permutation(range(N)))
            grd=upper(np.array([[GRID_D[pn[i],pn[j]] for j in range(N)] for i in range(N)]))
            nullv.append(spearmanr(obs,grd).correlation)
        nullv=np.array(nullv); p_emp=float((np.sum(nullv>=rsa_grid)+1)/(len(nullv)+1))
        # column signature: vertical-neighbor closeness (2-D) vs horizontal-neighbor; both grid-dist 1 => similar in a GRID,
        # but in a LINE vertical neighbors (reading-dist 3) are far. cos = 1 - cosine-distance.
        cosM=1.0-full
        cos_v=float(np.mean([cosM[i,j] for i,j in VNB])); cos_h=float(np.mean([cosM[i,j] for i,j in HNB]))
        cos_far=float(np.mean([cosM[i,j] for i,j in FAR]))
        stats.append(dict(si=si,rsa_grid=rsa_grid,rsa_linear=rsa_lin,rsa_ring=rsa_ring,
                          dRSA_grid_minus_lin=rsa_grid-rsa_lin,null_mean=float(nullv.mean()),
                          null_p95=float(np.percentile(nullv,95)),p_emp=p_emp,
                          cos_vert=cos_v,cos_horiz=cos_h,cos_far=cos_far,
                          col_sig=cos_v-cos_far,vh_ratio=cos_v/ (cos_h+1e-9)))
        allc[si]=(cents,cells)
        print(f"  scr{si+1}: RSA_grid={rsa_grid:+.2f} RSA_lin={rsa_lin:+.2f} RSA_ring={rsa_ring:+.2f} "
              f"dRSA(grid-lin)={rsa_grid-rsa_lin:+.2f} | null {nullv.mean():+.2f} p={p_emp:.3f} | "
              f"cos v/h/far {cos_v:.2f}/{cos_h:.2f}/{cos_far:.2f}",flush=True)
    del model; torch.cuda.empty_cache()
    agg=lambda k: float(np.mean([s[k] for s in stats]))
    summ=dict(model=MODEL,base=IS_BASE,nscr=NSCR,L=int(L),grid="3x3",sepedges=SEPEDGES,qset=QSET,per=stats,
              rsa_grid=agg("rsa_grid"),rsa_grid_std=float(np.std([s["rsa_grid"] for s in stats])),
              rsa_linear=agg("rsa_linear"),rsa_ring=agg("rsa_ring"),dRSA_grid_lin=agg("dRSA_grid_minus_lin"),
              col_sig=agg("col_sig"),vh_ratio=agg("vh_ratio"),
              nsig_emp=int(sum(s["p_emp"]<0.05 for s in stats)),
              nbeats_line=int(sum(s["dRSA_grid_minus_lin"]>0 for s in stats)))
    json.dump(summ,open(OUT,"w"),indent=2)
    np.savez(os.path.join(data_dir("geometry"), f"grid_{tag}.npz"),**{f"c{si}":allc[si][0] for si in range(NSCR)},
             cells=np.array([allc[si][1] for si in range(NSCR)]),tok=np.array(TOK9),L=L)
    print(f"== {tag}: RSA_grid {summ['rsa_grid']:+.2f}±{summ['rsa_grid_std']:.2f} (lin {summ['rsa_linear']:+.2f}, "
          f"ring {summ['rsa_ring']:+.2f}, dRSA grid-lin {summ['dRSA_grid_lin']:+.2f}) col_sig {summ['col_sig']:+.2f} "
          f"sig {summ['nsig_emp']}/{NSCR} beats_line {summ['nbeats_line']}/{NSCR}",flush=True)
    # figure: PCA of scr0 centroids, color=row, marker=col, grid edges drawn; + RSA bars (grid/line/ring/null)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6))
    cents,cells=allc[0]; mu=cents.mean(0); _,_,Vt=np.linalg.svd(cents-mu,full_matrices=False); P=(cents-mu)@Vt.T
    rowc=[COORD[c][0] for c in cells]; colc=[COORD[c][1] for c in cells]; mks=["o","s","^"]
    for i in range(N):
        ax1.scatter(P[i,0],P[i,1],c=[rowc[i]],cmap="viridis",vmin=0,vmax=2,s=150,marker=mks[colc[i]],zorder=3,edgecolor="k",linewidth=.4)
        ax1.annotate(f"{TOK9[i]}\n(r{rowc[i]}c{colc[i]})",(P[i,0],P[i,1]),fontsize=7,ha="center")
    pos={cells[i]:(P[i,0],P[i,1]) for i in range(N)}                # draw grid edges between adjacent cells
    for a in range(N):
        for b in range(N):
            if a<b and GRID_D[a,b]==1 and a in pos and b in pos:
                ax1.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],"k-",alpha=.25,zorder=1)
    ax1.set_title(f"{tag} imposed-grid centroids (scr1, PC1-2)\ncolor=row, marker=col, lines=grid edges"); ax1.set_xticks([]);ax1.set_yticks([])
    rg=[s["rsa_grid"] for s in stats]; rl=[s["rsa_linear"] for s in stats]; rr=[s["rsa_ring"] for s in stats]; nm=[s["null_mean"] for s in stats]; x=np.arange(NSCR)
    ax2.bar(x-0.3,rg,0.2,label="RSA vs GRID",color="C0"); ax2.bar(x-0.1,rl,0.2,label="RSA vs line",color="C1",alpha=.8)
    ax2.bar(x+0.1,rr,0.2,label="RSA vs ring",color="C2",alpha=.7); ax2.bar(x+0.3,nm,0.2,label="null mean",color="gray",alpha=.6)
    ax2.set_xticks(x); ax2.set_xticklabels([f"s{i+1}" for i in range(NSCR)]); ax2.set_ylim(-.3,1); ax2.legend(fontsize=8)
    ax2.set_title("Geometry matches imposed 2-D GRID (beats line)?"); ax2.axhline(0,color="k",lw=.5)
    plt.tight_layout(); fn=os.path.join(data_dir("geometry"), f"FIG_grid_{tag}.png"); plt.savefig(fn,dpi=130); print("FIG",os.path.abspath(fn),flush=True)
if __name__=="__main__": main()
