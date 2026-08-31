#!/usr/bin/env bash
# k=1 clean-cell sweep: instruct ladder, large->small. Per-model file-skip. Saves raw+traces+summary per model.
set -u
cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4u-venv/bin/python}"
MODE="${1:-list}"
MODELS=( google/gemma-4-31B-it Qwen/Qwen3.5-27B google/gemma-4-12B-it Qwen/Qwen3.5-9B
         google/gemma-4-E4B-it Qwen/Qwen3.5-4B google/gemma-4-E2B-it meta-llama/Llama-3.1-8B-Instruct )
mkdir -p logs results/behavior
echo "K1 SWEEP START $(date) mode=$MODE models=${#MODELS[@]}"
for M in "${MODELS[@]}"; do
  TAG=$(basename "$M")
  DONE="results/behavior/k1summary_${TAG}_${MODE}.json"
  if [ -f "$DONE" ]; then echo "[skip] $TAG"; continue; fi
  echo "[run ] $TAG $(date)"
  $PY -u k1_extract.py "$M" --mode "$MODE" --nscr 3 > "logs/k1_${TAG}_${MODE}.log" 2>&1
  if [ -f "$DONE" ]; then echo "[ ok ] $TAG"; else echo "[FAIL] $TAG"; fi
done
echo "K1 SWEEP COMPUTE DONE $(date)"
$PY -u k1_plot.py --mode "$MODE" > "logs/k1_plot_${MODE}.log" 2>&1; cat "logs/k1_plot_${MODE}.log"
echo "K1 SWEEP ALL DONE $(date)"
