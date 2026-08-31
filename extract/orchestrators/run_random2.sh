#!/bin/bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/random2.log
echo "=== random rerun start $(date) ===" | tee -a "$LOG"
$PY -u topo_driver.py Qwen/Qwen3.5-27B --configs random:0:0,random:0:2 --nscr 6 2>&1 | tee -a "$LOG"
$PY -u topo_driver.py google/gemma-4-31B-it --configs random:0:0,random:0:2 --nscr 6 2>&1 | tee -a "$LOG"
echo "=== differential ===" | tee -a "$LOG"
for M in Qwen3.5-27B gemma-4-31B-it; do
  A="results/geometry/topo_random-n9-b3-s0_${M}.npz"; B="results/geometry/topo_random-n9-b3-s0-d2_${M}.npz"
  [ -f "$A" ] && [ -f "$B" ] && { echo "-- $M --" | tee -a "$LOG"; $PY analyze_topo.py "$A" "$B" 2>&1 | tee -a "$LOG"; }
done
echo "=== random rerun done $(date) ===" | tee -a "$LOG"
