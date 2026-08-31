#!/bin/bash
# Causal suite (Qwen deferred): full-sweep CoT on both layers, capability gradient, k-patch. Sequential, file-skip/resumable.
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"     # gemma4 (31B) + qwen3_5
VU="${VENV:-/path/to/gemma4u-venv}"   # gemma4_unified (12B)
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
D=results/causal; log(){ echo "[$(date +%H:%M:%S)] $*"; }
gpu_free(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit ($1>5000)?0:1}'; do sleep 15; done; }
run(){ # $1=flag $2=label $3=venv-python ; rest=cmd
  local f=$1 lbl=$2 py=$3; shift 3
  if [ ! -f $D/$f ]; then log "$lbl START"; "$py" -u "$@" > $D/log_$f.txt 2>&1 && touch $D/$f && log "$lbl DONE" || { log "$lbl FAILED (see log_$f.txt)"; exit 1; }; gpu_free; fi
}
run .s1 "1 Gemma-31B CoT full-sweep nscr=2" "$V/bin/python"  causal_cot.py google/gemma-4-31B-it --maxnew 400 --nscr 2
run .s2 "2 Gemma-12B snap nscr=2"           "$VU/bin/python" causal_use.py google/gemma-4-12B-it --nscr 2 --maxnew 8
run .s3 "3 Gemma-12B CoT full-sweep nscr=2" "$VU/bin/python" causal_cot.py google/gemma-4-12B-it --maxnew 400 --nscr 2
run .s4 "4 k-patch Gemma-31B all-pairs nscr=2" "$V/bin/python" causal_k.py google/gemma-4-31B-it --nscr 2 --kmax 6
log "CAUSAL SUITE DONE"
