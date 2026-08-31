#!/bin/bash
# Qwen-27B cross-model causal (auto-triggers after the Gemma suite's last step .s4 frees the GPU). gemma4-venv runs qwen3_5.
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
D=results/causal; log(){ echo "[$(date +%H:%M:%S)] $*"; }
gpu_free(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit ($1>5000)?0:1}'; do sleep 15; done; }
run(){ local f=$1 lbl=$2; shift 2; if [ ! -f $D/$f ]; then log "$lbl START"; "$V/bin/python" -u "$@" > $D/log_$f.txt 2>&1 && touch $D/$f && log "$lbl DONE" || { log "$lbl FAILED"; exit 1; }; gpu_free; fi; }
log "waiting for Gemma suite (.s4) to finish"
while [ ! -f $D/.s4 ]; do sleep 30; done
gpu_free; log "suite done, GPU free -> Qwen-27B"
run .q1 "Qwen-27B snap nscr=2"            causal_use.py Qwen/Qwen3.5-27B --nscr 2 --maxnew 8
run .q2 "Qwen-27B CoT full-sweep nscr=2"  causal_cot.py Qwen/Qwen3.5-27B --nscr 2 --maxnew 600
log "QWEN DONE"
