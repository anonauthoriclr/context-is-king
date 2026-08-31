"""
Recompute CoT causal-patching metrics from subagent-judged answers.

Reads:  causalcot_<tag>.json  + judge_out/*.json  ({hash: {"answer": <entity|NONE>, "concluded": bool}})
Writes: causalcot_<tag>_judged.json  and prints a summary (imposed/natural patch_success per layer,
        baseline acc from the judge, and judge-vs-regex agreement).

Usage: python apply_judge.py data_dir("causal")/causalcot_Qwen3.5-27B.json
"""
import os, sys, json, glob, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

NATURAL = {
    "days":   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "months": ["January","February","March","April","May","June","July","August",
               "September","October","November","December"],
}
def h_(t): return hashlib.md5(t.encode()).hexdigest() if t else None
def norm(a): return (a or "").strip().lower()
def mean(xs): xs=[float(x) for x in xs]; return sum(xs)/len(xs) if xs else 0.0

def main():
    path = sys.argv[1]
    d = json.load(open(path))
    concept = d["concept"]; base = NATURAL[concept]; N = len(base)
    bidx = {e.lower(): i for i, e in enumerate(base)}

    judged = {}
    for jf in glob.glob(os.path.join(os.path.dirname(path), "judge_out", "*.json")):
        for h, v in json.load(open(jf)).items():
            a = v.get("answer")
            judged[h] = None if (a is None or norm(a) in ("none","")) else a
    print(f"loaded {len(judged)} judged answers")

    def ans(t):
        return judged.get(h_(t))

    out = {"model": d["model"], "concept": concept, "maxnew": d.get("maxnew"),
           "layers": d["layers"], "contexts": {}}
    agree = tot = 0
    for ctx in ("imposed", "natural"):
        out["contexts"][ctx] = {"per_scr": []}
        for scr in d["contexts"][ctx]["per_scr"]:
            order = scr["order"]; oidx = {e.lower(): i for i, e in enumerate(order)}
            succ_imp = {e: order[(oidx[e.lower()]+1) % N] for e in base}
            succ_nat = {e: base[(bidx[e.lower()]+1) % N] for e in base}
            succ  = succ_imp if ctx == "imposed" else succ_nat
            other = succ_nat if ctx == "imposed" else succ_imp
            base_ok = {e: (norm(ans(scr["base_traces"][e])) == norm(succ[e])) for e in base}
            sscr = {"order": order, "base_acc_judge": mean([base_ok[e] for e in base]), "layers": {}}
            for L, lv in scr["layers"].items():
                recs = []
                for r in lv["recs"]:
                    a = ans(r.get("trace"))
                    recs.append({"src": r["src"], "dst": r["dst"], "ans_judge": a, "ans_regex": r.get("ans"),
                                 "hit_dst":   bool(a and norm(a) == norm(succ[r["dst"]])),
                                 "stuck_src": bool(a and norm(a) == norm(succ[r["src"]])),
                                 "cross_dst": bool(a and norm(a) == norm(other[r["dst"]])),
                                 "na": a is None,
                                 "base_ok": base_ok[r["src"]] and base_ok[r["dst"]]})
                    if a is not None:
                        agree += (norm(a) == norm(r.get("ans"))); tot += 1
                clean = [x for x in recs if x["base_ok"]]
                rate = lambda rs, k: mean([x[k] for x in rs]) if rs else 0.0
                sscr["layers"][L] = {"n_clean": len(clean),
                    "patch_success_clean": rate(clean, "hit_dst"), "stuck_src_clean": rate(clean, "stuck_src"),
                    "cross_map_clean": rate(clean, "cross_dst"), "na_clean": rate(clean, "na"),
                    "patch_success_all": rate(recs, "hit_dst")}
            out["contexts"][ctx]["per_scr"].append(sscr)
    out["judge_vs_regex_agreement"] = (agree / tot) if tot else None

    outp = path.replace(".json", "_judged.json"); json.dump(out, open(outp, "w"), indent=2)
    print(f"\n=== JUDGED SUMMARY -> {outp} ===")
    print(f"judge vs regex agreement (committed answers): "
          f"{out['judge_vs_regex_agreement']:.3f} (n={tot})" if tot else "no answers")
    for ctx in ("imposed", "natural"):
        for si, s in enumerate(out["contexts"][ctx]["per_scr"]):
            print(f"[{ctx} scr{si+1}] baseline acc (judge)={s['base_acc_judge']:.2f}  order={s['order']}")
            for L, v in s["layers"].items():
                print(f"    L{L:>3}: patch={v['patch_success_clean']:.2f} (n_clean={v['n_clean']}) "
                      f"stuck={v['stuck_src_clean']:.2f} cross={v['cross_map_clean']:.2f} na={v['na_clean']:.2f}")

if __name__ == "__main__":
    main()
