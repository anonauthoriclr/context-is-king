#!/bin/bash
# Solidify the causal patch finding (task 3): Gemma CoT 2nd scramble + Qwen-27B cross-model (snap + CoT).
# File-based step skipping; sequential (one model at a time, each fits one A100). Resumable.
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
mkdir -p results/causal
log(){ echo "[$(date +%H:%M:%S)] $*"; }
D=results/causal

# A: Gemma-31B CoT, nscr=2 (solidify the open-ended result with a 2nd scramble)
if [ ! -f $D/.done_A ]; then
  log "A START Gemma-31B CoT nscr=2"
  "$V/bin/python" -u causal_cot.py google/gemma-4-31B-it --layers 6,18,27 --maxnew 400 --nscr 2 > $D/run_A_gemma_cot2.log 2>&1 \
    && touch $D/.done_A && log "A DONE" || { log "A FAILED"; exit 1; }
fi
# B: Qwen-27B snap, nscr=2 (cross-model double dissociation)
if [ ! -f $D/.done_B ]; then
  log "B START Qwen-27B snap nscr=2"
  "$V/bin/python" -u causal_use.py Qwen/Qwen3.5-27B --nscr 2 --maxnew 8 > $D/run_B_qwen_snap.log 2>&1 \
    && touch $D/.done_B && log "B DONE" || { log "B FAILED"; exit 1; }
fi
# C: Qwen-27B CoT, nscr=2 (cross-model regime-match)
if [ ! -f $D/.done_C ]; then
  log "C START Qwen-27B CoT nscr=2"
  "$V/bin/python" -u causal_cot.py Qwen/Qwen3.5-27B --layers 6,19,29 --maxnew 400 --nscr 2 > $D/run_C_qwen_cot.log 2>&1 \
    && touch $D/.done_C && log "C DONE" || { log "C FAILED"; exit 1; }
fi
log "ALL SOLIDIFY STEPS DONE"
