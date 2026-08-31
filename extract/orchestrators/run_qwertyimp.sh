#!/bin/bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/qwertyimp.log; echo "=== qwerty-imposed start $(date) ===" | tee -a "$LOG"
for M in Qwen/Qwen3.5-27B google/gemma-4-31B-it; do
  [ -f "results/geometry/qwertyimposed_$(basename $M).json" ] && { echo "SKIP $(basename $M)"|tee -a "$LOG"; continue; }
  $PY -u geom_qwerty_imposed.py "$M" --nscr 6 2>&1 | tee -a "$LOG"
done
echo "=== qwerty-imposed done $(date) ===" | tee -a "$LOG"
