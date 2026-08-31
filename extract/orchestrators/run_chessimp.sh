#!/bin/bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/chessimp_run.log
echo "=== chess-imposed start $(date) ===" | tee -a "$LOG"
for M in google/gemma-4-E2B-it Qwen/Qwen3.5-27B google/gemma-4-31B-it; do
  $PY -u topo_driver.py "$M" --configs chess-imposed --nscr 6 2>&1 | tee -a "$LOG"
done
echo "=== SUMMARY ===" | tee -a "$LOG"
for f in results/geometry/chessimposed_*.json; do
  $PY -c "import json;d=json.load(open('$f'));print('%-22s imposed %+.2f±%.2f natural %+.2f dominance %+.2f | imp-beats-nat %d/%d sig %d/%d'%(d['model'].split('/')[-1],d['rsa_imposed'],d['rsa_imposed_std'],d['rsa_natural'],d['dominance'],d['n_imp_beats_nat'],d['nscr'],d['nsig_imp'],d['nscr']))" 2>/dev/null | tee -a "$LOG"
done
echo "=== chess-imposed done $(date) ===" | tee -a "$LOG"
