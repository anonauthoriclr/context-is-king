#!/bin/bash
# E5 CONTROLS (under review): discriminate genuine spec-following graph construction from local edge-co-embedding, and test
# gating per graph. {grid, ring} x {Gemma-E2B (small), Gemma-31B (flag)}, atomic shuffled edges, nscr 6. Forward-only,
# file-skip resumable. Saves exact spec text + centroids for offline audit. gemma4-venv. Run from exp01 dir.
set -u
cd "$(dirname "$0")"
V="${VENV:-/path/to/gemma4-venv}"
PY=$V/bin/python
LOG=results/geometry/grid_ctrl.log
mkdir -p results/geometry
echo "=== E5 CTRL start $(date) ===" | tee -a "$LOG"
run () {  # graph  model
  local G="$1"; local M="$2"; local TAG="$(basename "$M")"
  local OUT="results/geometry/graph_${G}_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $G/$TAG (exists)" | tee -a "$LOG"; return; fi
  echo ">>> RUN $G/$TAG @ $(date)" | tee -a "$LOG"
  $PY -u geom_graph.py "$M" --graph "$G" --nscr 6 2>&1 | tee -a "$LOG"
}
for M in google/gemma-4-E2B-it google/gemma-4-31B-it; do
  run grid "$M"
  run ring "$M"
done
echo "=== E5 CTRL done $(date) ===" | tee -a "$LOG"
echo "--- CONTROL SUMMARY ---" | tee -a "$LOG"
for f in results/geometry/graph_*.json; do
  $PY -c "import json;d=json.load(open('$f'));print('%-22s imposed=%-4s | self %+.2f  grid %+.2f  ring %+.2f  line %+.2f | graded(nonadj) %+.2f | self-best %d/%d beats-line %d/%d sig %d/%d'%(d['model'].split('/')[-1],d['graph'],d['rsa_self'],d['rsa_grid'],d['rsa_ring'],d['rsa_line'],d['rsa_graded_nonadj'],d['n_self_is_best'],d['nscr'],d['n_self_beats_line'],d['nscr'],d['nsig_emp'],d['nscr']))" 2>/dev/null | tee -a "$LOG"
done
