"""
Prep CoT causal-patching traces for subagent judging (no API — Claude subagents judge the batches).

Reads a causalcot_<tag>.json (with full 'trace' + 'base_traces'), dedups every unique trace by hash,
and writes them in batches to judge_batches/batch_XXX.jsonl. A subagent reads each batch file and
writes {hash: {"answer": <entity|NONE>, "concluded": bool}} to judge_out/batch_XXX.json. Then
apply_judge.py recomputes the metrics from those answers.

Usage: python prep_judge_batches.py data_dir("causal")/causalcot_Qwen3.5-27B.json [--batch 40] [--tail 4000]
"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from paths import data_dir

BATCH = int(sys.argv[sys.argv.index("--batch")+1]) if "--batch" in sys.argv else 40
TAIL  = int(sys.argv[sys.argv.index("--tail")+1])  if "--tail"  in sys.argv else 4000
NATURAL = {
    "days":   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "months": ["January","February","March","April","May","June","July","August",
               "September","October","November","December"],
}

def main():
    path = sys.argv[1]
    d = json.load(open(path))
    concept = d["concept"]; entities = NATURAL[concept]
    outdir = os.path.join(os.path.dirname(path), "judge_batches")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(path), "judge_out"), exist_ok=True)

    seen = {}  # hash -> trace (tail)
    def add(t):
        if not t: return
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen: seen[h] = t[-TAIL:]
    for ctx in ("imposed","natural"):
        for scr in d["contexts"][ctx]["per_scr"]:
            for t in scr.get("base_traces", {}).values(): add(t)
            for lv in scr["layers"].values():
                for r in lv["recs"]: add(r.get("trace"))

    items = [{"h": h, "trace": t} for h, t in seen.items()]
    nb = (len(items) + BATCH - 1) // BATCH
    for i in range(nb):
        chunk = items[i*BATCH:(i+1)*BATCH]
        with open(os.path.join(outdir, f"batch_{i:03d}.jsonl"), "w") as f:
            for it in chunk:
                f.write(json.dumps(it) + "\n")
    json.dump({"concept": concept, "entities": entities, "n_traces": len(items), "n_batches": nb,
               "tail": TAIL},
              open(os.path.join(os.path.dirname(path), "judge_meta.json"), "w"), indent=2)
    print(f"{len(items)} unique traces -> {nb} batches of <= {BATCH} in {outdir}")
    print(f"entities ({concept}): {entities}")

if __name__ == "__main__":
    main()
