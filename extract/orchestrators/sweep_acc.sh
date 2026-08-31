#!/usr/bin/env bash
# ONE-SHOT accuracy sweep (defladder_acc, regime-matched to F43). Concept-generic: $1 (default days).
# Rolls LARGE -> SMALL so high-signal big models land first. Per-model file-based skip.
set -u
cd "$(dirname "$0")"
VENV="${VENV:-/path/to/gemma4u-venv}"
PY=$VENV/bin/python
CONCEPT="${1:-days}"; MODES=list,adj; NSCR=2
MODELS=(                         # large -> small
  google/gemma-4-31B-it
  Qwen/Qwen3.5-27B
  google/gemma-4-12B-it
  Qwen/Qwen3.5-9B
  google/gemma-4-E4B-it
  Qwen/Qwen3.5-4B
  google/gemma-4-E2B-it
  meta-llama/Llama-3.1-8B-Instruct
)
mkdir -p logs results/behavior
echo "ACC SWEEP START $(date) concept=$CONCEPT modes=$MODES nscr=$NSCR models=${#MODELS[@]}"
for M in "${MODELS[@]}"; do
  TAG=$(basename "$M")
  DONE="results/behavior/acc_${TAG}_${CONCEPT}.json"
  if [ -f "$DONE" ]; then echo "[skip] $TAG (json exists)"; continue; fi
  echo "[run ] $TAG  $(date)"
  $PY -u defladder_acc.py "$M" --concept "$CONCEPT" --nscr "$NSCR" --modes "$MODES" \
      > "logs/acc_${TAG}_${CONCEPT}.log" 2>&1
  if [ -f "$DONE" ]; then echo "[ ok ] $TAG"; else echo "[FAIL] $TAG (see log)"; fi
done
echo "ACC SWEEP COMPUTE DONE $(date) — building geometry-vs-behavior summary"
$PY -u defladder_acc_plot.py --concept "$CONCEPT" > "logs/acc_sweep_${CONCEPT}.log" 2>&1
cat "logs/acc_sweep_${CONCEPT}.log"
echo "ACC SWEEP ALL DONE $(date)"
