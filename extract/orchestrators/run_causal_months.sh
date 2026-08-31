#!/bin/bash
# T2.1b: replicate the causal entity-patch dissociation on a SECOND concept (months, N=12) across the Gemma ladder +Qwen.
# Output suffixed _months (days files untouched). single-load per model, file-skip. gemma4-venv.
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/causal_months.log
echo "=== causal months start $(date) ===" | tee -a "$LOG"
for M in google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
  OUT="results/causal/causaluse_$(basename $M)_months.json"
  if [ -f "$OUT" ]; then echo "SKIP $(basename $M)" | tee -a "$LOG"; continue; fi
  echo ">>> $(basename $M) months @ $(date)" | tee -a "$LOG"
  $PY -u causal_use.py "$M" --concept months --nscr 2 2>&1 | tee -a "$LOG"
done
echo "=== causal months done $(date) ===" | tee -a "$LOG"
