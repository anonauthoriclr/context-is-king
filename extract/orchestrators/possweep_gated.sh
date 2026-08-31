#!/bin/bash
set -u
cd "${ROOT:-/path/to/context-is-king/extract}"
VPY="${VENV:-/path/to/gemma4u-venv/bin/python}"
LOG=logs/possweep_rerun.log; mkdir -p logs
echo "[gated] $(date) waiting for >=62GB free GPU" >> "$LOG"
while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$free" -ge 62000 ] && break; sleep 30; done
echo "[gated] $(date) GPU ${free}MiB free -- running possweep nscr=8" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VPY" -u geom_possweep.py google/gemma-4-31B-it --nscr 8 >> "$LOG" 2>&1
echo "[gated] $(date) done rc=$?" >> "$LOG"; touch results/geometry/.possweep_done
