#!/usr/bin/env bash
# cross-concept topology check: defladder geometry (forward-only) list+adj on zodiac/clock/notes, capable models.
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4u-venv/bin/python}"
for C in zodiac clock notes; do
  for M in google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
    TAG=$(basename "$M")
    [ -f "results/geometry/defladder_${TAG}_${C}.json" ] && { echo "[skip] $TAG/$C"; continue; }
    echo "[run ] $TAG/$C $(date)"; $PY -u defladder.py "$M" --concept "$C" --nscr 3 --modes list,adj > "logs/defladder_${TAG}_${C}.log" 2>&1
    [ -f "results/geometry/defladder_${TAG}_${C}.json" ] && echo "[ ok ] $TAG/$C" || echo "[FAIL] $TAG/$C"
  done
done
echo "XCONCEPT GEOM DONE $(date)"
