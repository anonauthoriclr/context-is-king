#!/bin/bash
# Fires after E2 (Qwen-CoT) frees the GPU: E4 tree (lead, 3 models) → E1 arb-token 2x2. File-skip/resumable. gemma4-venv.
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
D=results; log(){ echo "[$(date +%H:%M:%S)] $*"; }
gpu_free(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit ($1>5000)?0:1}'; do sleep 15; done; }
run(){ local f=$1 lbl=$2; shift 2; if [ ! -f $D/.$f ]; then log "$lbl START"; "$V/bin/python" -u "$@" > $D/log_$f.txt 2>&1 && touch $D/.$f && log "$lbl DONE" || { log "$lbl FAILED (log_$f.txt)"; }; gpu_free; fi; }
log "waiting for E2 (causal_cot.py Qwen) to finish + GPU free"
while pgrep -f "causal_cot.py Qwen" >/dev/null; do sleep 30; done
gpu_free; log "GPU free -> E4 (tree) then E1 (arb 2x2)"
# E4 — the renderer killer: does an imposed TREE render as tree-distance geometry?
run e4_tree_31b  "E4 tree Gemma-31B"   geom_tree.py google/gemma-4-31B-it --nscr 6
run e4_tree_e2b  "E4 tree Gemma-E2B (floor)" geom_tree.py google/gemma-4-E2B-it --nscr 6
run e4_tree_qwen "E4 tree Qwen-27B (cross-family)" geom_tree.py Qwen/Qwen3.5-27B --nscr 6
# E1 — arbitrary-token 2x2 (airtights pillar 1b: wrap closes a cycle on structure-free tokens)
run e1_arb7   "E1 arb7 2x2 Gemma-31B"  wrap_2x2.py --concept arb7  --model google/gemma-4-31B-it --nscr 3
run e1_arb12  "E1 arb12 2x2 Gemma-31B" wrap_2x2.py --concept arb12 --model google/gemma-4-31B-it --nscr 3
log "E4 + E1 DONE"
