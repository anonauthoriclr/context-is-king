#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4u-venv/bin/python}"
for M in google/gemma-4-31B-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3.5-27B; do
  TAG=$(basename "$M")
  [ -f "results/behavior/natural_summary_${TAG}.json" ] && { echo "[skip] $TAG"; continue; }
  echo "[run ] $TAG $(date)"; $PY -u natural_regime.py "$M" --nscr 2 --ntpl 2 --maxnew 2048 > "logs/natural_${TAG}.log" 2>&1
  [ -f "results/behavior/natural_summary_${TAG}.json" ] && echo "[ ok ] $TAG" || echo "[FAIL] $TAG"
done
echo "NATURAL ALL DONE $(date)"
