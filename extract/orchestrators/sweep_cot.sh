#!/usr/bin/env bash
# CoT-regime sweep (defladder_cot): high-ceiling decoupled cells, days, list+adj, si=0.
# Part B flag-invariance (vs F43 cache) + Part A CoT behavior (2048 tok). Large->small, file-skip.
set -u
cd "$(dirname "$0")"
VENV="${VENV:-/path/to/gemma4u-venv}"
PY=$VENV/bin/python
CONCEPT="${1:-days}"
MODELS=( google/gemma-4-31B-it google/gemma-4-12B-it Qwen/Qwen3.5-27B )
mkdir -p logs results/behavior
echo "COT SWEEP START $(date) concept=$CONCEPT models=${#MODELS[@]}"
for M in "${MODELS[@]}"; do
  TAG=$(basename "$M")
  DONE="results/behavior/cot_${TAG}_${CONCEPT}_s0.json"
  if [ -f "$DONE" ]; then echo "[skip] $TAG"; continue; fi
  echo "[run ] $TAG  $(date)"
  $PY -u defladder_cot.py "$M" --concept "$CONCEPT" --modes list,adj --si 0 --ntpl 2 --maxnew 2048 \
      > "logs/cot_${TAG}_${CONCEPT}.log" 2>&1
  if [ -f "$DONE" ]; then echo "[ ok ] $TAG"; else echo "[FAIL] $TAG (see log)"; fi
done
echo "COT SWEEP ALL DONE $(date)"
