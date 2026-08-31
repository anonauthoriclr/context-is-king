#!/bin/bash
# After step A (Gemma-31B CoT) frees the GPU: Gemma-12B entity patch (snap+CoT, gemma4u-venv, gemma4_unified arch)
# + k-patch all-pairs on Gemma-31B (gemma4-venv). File-skip/resumable. Qwen deferred (per operator).
cd "${ROOT:-/path/to/context-is-king/extract}"
VU="${VENV:-/path/to/gemma4u-venv}"   # tf5.12.1, loads gemma4_unified (12B)
V="${VENV:-/path/to/gemma4-venv}"      # tf5.5.3, loads gemma4 (31B) + qwen3_5
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
D=results/causal; log(){ echo "[$(date +%H:%M:%S)] $*"; }
gpu_free(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit ($1>5000)?0:1}'; do sleep 15; done; }

log "waiting for step A (.done_A) to free the GPU"
while [ ! -f $D/.done_A ]; do sleep 20; done
gpu_free; log "GPU free"

# D: Gemma-12B snap entity patch (capability gradient: E2B < 12B < 31B)
if [ ! -f $D/.done_D ]; then
  log "D START Gemma-12B snap"
  "$VU/bin/python" -u causal_use.py google/gemma-4-12B-it --nscr 2 --maxnew 8 > $D/run_D_gemma12_snap.log 2>&1 \
    && touch $D/.done_D && log "D DONE" || { log "D FAILED"; exit 1; }
  gpu_free
fi
# E: Gemma-12B CoT entity patch
if [ ! -f $D/.done_E ]; then
  log "E START Gemma-12B CoT"
  "$VU/bin/python" -u causal_cot.py google/gemma-4-12B-it --layers 5,14,22 --maxnew 400 --nscr 2 > $D/run_E_gemma12_cot.log 2>&1 \
    && touch $D/.done_E && log "E DONE" || { log "E FAILED"; exit 1; }
  gpu_free
fi
# F: k-patch all-pairs on Gemma-31B (operand-early / computation-late test)
if [ ! -f $D/.done_F ]; then
  log "F START k-patch Gemma-31B"
  "$V/bin/python" -u causal_k.py google/gemma-4-31B-it --nscr 2 --kmax 6 > $D/run_F_kpatch_gemma31.log 2>&1 \
    && touch $D/.done_F && log "F DONE" || { log "F FAILED"; exit 1; }
fi
log "GEMMA-12B + K-PATCH DONE"
