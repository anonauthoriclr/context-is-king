#!/usr/bin/env bash
# Cross-model GEOMETRY sweep (defladder, no generation): instruct ladder, modes=list,adj.
# Concept-generic: pass concept as $1 (default days). SSH-resilient: per-model file-based skip;
# defladder also skips model-load when acts cached.
set -u
cd "$(dirname "$0")"
VENV="${VENV:-/path/to/gemma4u-venv}"
PY=$VENV/bin/python
CONCEPT="${1:-days}"; MODES=list,adj; NSCR=3
MODELS=(
  google/gemma-4-E2B-it
  google/gemma-4-E4B-it
  google/gemma-4-12B-it
  google/gemma-4-31B-it
  Qwen/Qwen3.5-4B
  Qwen/Qwen3.5-9B
  Qwen/Qwen3.5-27B
  meta-llama/Llama-3.1-8B-Instruct
)
mkdir -p logs results/geometry
echo "SWEEP START $(date) concept=$CONCEPT modes=$MODES nscr=$NSCR models=${#MODELS[@]}"
for M in "${MODELS[@]}"; do
  TAG=$(basename "$M")
  DONE="results/geometry/defladder_${TAG}_${CONCEPT}.json"
  if [ -f "$DONE" ]; then echo "[skip] $TAG (json exists)"; continue; fi
  echo "[run ] $TAG  $(date)"
  $PY -u defladder.py "$M" --concept "$CONCEPT" --nscr "$NSCR" --modes "$MODES" \
      > "logs/defladder_${TAG}_${CONCEPT}.log" 2>&1
  if [ -f "$DONE" ]; then echo "[ ok ] $TAG"; else echo "[FAIL] $TAG (see log)"; fi
done
echo "SWEEP COMPUTE DONE $(date) — building summary"
$PY -u defladder_sweep_plot.py --concept "$CONCEPT" > "logs/defladder_sweep_${CONCEPT}.log" 2>&1
cat "logs/defladder_sweep_${CONCEPT}.log"
echo "SWEEP ALL DONE $(date)"
