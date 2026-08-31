#!/bin/bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/qwerty.log; echo "=== qwerty start $(date) ===" | tee -a "$LOG"
for M in google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
  [ -f "results/geometry/qwerty_$(basename $M).json" ] && { echo "SKIP $(basename $M)"|tee -a "$LOG"; continue; }
  $PY -u geom_qwerty.py "$M" 2>&1 | tee -a "$LOG"
done
echo "=== qwerty done $(date) ===" | tee -a "$LOG"
