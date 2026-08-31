#!/bin/bash
set -u
cd "${ROOT:-/path/to/context-is-king/extract}"
VPY="${VENV:-/path/to/gemma4u-venv/bin/python}"
LOG=logs/diverse_gated.log; mkdir -p logs
echo "[gated] $(date) waiting for >=62GB free GPU" >> "$LOG"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 62000 ] && break
  sleep 60
done
echo "[gated] $(date) GPU ${free}MiB free — running diverse extraction" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VPY" -u demo_tree_diverse.py google/gemma-4-31B-it --depth 4 >> "$LOG" 2>&1
echo "[gated] $(date) done rc=$?" >> "$LOG"
touch results/geometry/.diverse_done
