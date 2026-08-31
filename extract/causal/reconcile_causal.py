#!/usr/bin/env python3
"""RECONCILIATION (CPU, reads existing JSON only): does the paper's 'no flip at 2B' claim survive, and was the SAME
method applied across all model scales? We compare, per model/concept:
  (A) PAPER RULE  [causal_use.py::plot]: live = {L: imposed patch_success>=0.9}; clean = {L in live: position_token_ctrl<0.2};
      causal_L = max(clean) [deepest clean-and-live]. Report imposed own/cross + natural own + controls THERE.
  (B) EARLY-READ  [my re-analysis]: the layer maximizing imposed patch_success (ties -> shallowest). Report own + the
      position_token control THERE, to expose whether the 'flip' at that layer is a clean entity read or trivial injection.
Prints one block per file so we can see where the two rules diverge and whether the divergence is the clean-locus filter.
Run: python reconcile_causal.py"""
import json, glob, numpy as np
from collections import OrderedDict
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

def avg(scrs, L, k):
    vs=[s["layers"][str(L)][k] for s in scrs if str(L) in s["layers"]]
    return float(np.mean(vs)) if vs else None

def ptok_at(imp, L):
    """position-token control averaged over scrambles at layer L (1.0 if not measured there)."""
    vs=[s["controls"][str(L)]["position_token_success"] for s in imp
        if "controls" in s and str(L) in s["controls"]]
    return float(np.mean(vs)) if vs else None

def rdon_at(imp, L):
    vs=[s["controls"][str(L)]["random_donor_success"] for s in imp
        if "controls" in s and str(L) in s["controls"]]
    return float(np.mean(vs)) if vs else None

def analyze(path):
    d=json.load(open(path)); nl=d["nl"]; sweep=d["sweep"]
    imp=d["contexts"]["imposed"]["per_scr"]; nat=d["contexts"]["natural"]["per_scr"]
    depth=lambda L: round(L/nl,2)
    rows=[]
    for L in sweep:
        rows.append(dict(L=L, dep=depth(L),
            imp_own=avg(imp,L,"patch_success"), imp_cross=avg(imp,L,"cross_map"),
            nat_own=avg(nat,L,"patch_success"),
            ptok=ptok_at(imp,L), rdon=rdon_at(imp,L)))
    # --- Rule A: paper (deepest clean-and-live) ---
    live=[r for r in rows if r["imp_own"] is not None and r["imp_own"]>=0.9]
    clean=[r for r in live if r["ptok"] is not None and r["ptok"]<0.2]
    A = clean[-1] if clean else (live[-1] if live else None)   # deepest
    # --- Rule B: early-read (max imp_own, shallowest on tie) ---
    cand=[r for r in rows if r["imp_own"] is not None]
    B = max(cand, key=lambda r:(r["imp_own"], -r["L"])) if cand else None
    return dict(model=d["model"].split("/")[-1], concept=d.get("concept","days"), nl=nl, A=A, B=B,
                nat_at_A=(A and A["nat_own"]))

def fmt(r):
    if r is None: return "n/a"
    def g(x): return "  . " if x is None else f"{x:.2f}"
    return (f"L{r['L']:>2d}(d{r['dep']:.2f}) own={g(r['imp_own'])} cross={g(r['imp_cross'])} "
            f"nat_own={g(r['nat_own'])} ptok={g(r['ptok'])} rdon={g(r['rdon'])}")

def main():
    files=sorted(glob.glob(os.path.join(data_dir("causal"), "causaluse_*.json")))
    # order: days first grouped by family/scale, then months
    def key(f):
        return (("months" in f), f)
    for f in sorted(files, key=key):
        r=analyze(f)
        print(f"\n=== {r['model']:18s} [{r['concept']}]  nl={r['nl']} ===")
        print(f"  A paper (deepest clean-live): {fmt(r['A'])}")
        print(f"  B early  (max imp_own)      : {fmt(r['B'])}")
        # verdict flags
        if r["A"] and r["B"]:
            gap = r["B"]["imp_own"] - (r["A"]["imp_own"] or 0)
            leak = r["B"]["ptok"]
            note=[]
            if r["A"]["imp_own"] is None or r["A"]["imp_own"]<0.5: note.append("A:imposed-arm does NOT flip cleanly")
            if r["nat_at_A"] is not None and r["nat_at_A"]>=0.9: note.append("A:natural-arm clean (double-dissoc OK)")
            elif r["nat_at_A"] is not None: note.append(f"A:natural-arm={r['nat_at_A']:.2f} (dissoc weak)")
            if leak is not None and leak>=0.2 and r["B"]["imp_own"]>=0.9:
                note.append(f"B's flip LEAKS (ptok={leak:.2f}>=0.2) => not a clean entity read")
            if note: print("   -> " + " | ".join(note))
if __name__=="__main__": main()
