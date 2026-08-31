#!/usr/bin/env bash
# After the CoT sweep frees the GPU: render CoT-prompt geometry rings (one-shot vs CoT, list vs adj) per model,
# then refresh the unified CoT k-curve plot. Forward-only captures (cheap).
set -u
cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4u-venv/bin/python}"
echo "waiting for CoT sweep (defladder_cot.py) to finish... $(date)"
while pgrep -f defladder_cot.py >/dev/null; do sleep 15; done
echo "GPU free $(date) — rendering CoT geometry rings"
for M in google/gemma-4-31B-it google/gemma-4-12B-it Qwen/Qwen3.5-27B; do
  TAG=$(basename "$M")
  echo "[geom] $TAG $(date)"
  $PY -u cot_geom_rings.py "$M" --concept days --si 0 > "logs/cotgeom_${TAG}.log" 2>&1
  grep -E "SAVED|Error|Traceback" "logs/cotgeom_${TAG}.log" | tail -2
done
echo "refreshing unified k-curve plot"
$PY -u defladder_cot_plot.py --concept days > logs/cot_plot.log 2>&1; tail -8 logs/cot_plot.log
echo "COTGEOM ALL DONE $(date)"
