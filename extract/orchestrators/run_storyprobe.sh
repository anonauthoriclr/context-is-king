#!/bin/bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/storyprobe.log
echo "=== storyprobe start $(date) ===" | tee -a "$LOG"
for M in google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
  [ -f "results/geometry/storyprobe_$(basename $M).json" ] && { echo "SKIP $(basename $M)"|tee -a "$LOG"; continue; }
  $PY -u geom_storyprobe.py "$M" --nscr 4 2>&1 | tee -a "$LOG"
done
echo "=== storyprobe done $(date) ===" | tee -a "$LOG"
