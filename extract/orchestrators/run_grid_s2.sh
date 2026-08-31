#!/bin/bash
# E5 Stage 2 (capability gate): small ladders x {sepedges, neutral} -- the two CLEAN conditions (the visual block confounds
# line vs grid, so we skip it). Qwen 4B/9B and Gemma E2B/E4B. Together with Stage-1 flagships: Gemma E2B->E4B->31B and
# Qwen 4B->9B->27B gating curves on the 2-D grid. Forward-only, file-skip resumable. gemma4-venv. Run from exp01 dir.
set -u
cd "$(dirname "$0")"
V="${VENV:-/path/to/gemma4-venv}"
PY=$V/bin/python
LOG=results/geometry/grid_s2.log
mkdir -p results/geometry
echo "=== E5 Stage2 start $(date) ===" | tee -a "$LOG"
run () {  # model  tagsuffix  extra-flags...
  local M="$1"; local TAGSUF="$2"; shift 2
  local TAG="$(basename "$M")${TAGSUF}"
  local OUT="results/geometry/grid_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $TAG (exists)" | tee -a "$LOG"; return; fi
  echo ">>> RUN $TAG : geom_grid.py $M --nscr 6 $* @ $(date)" | tee -a "$LOG"
  $PY -u geom_grid.py "$M" --nscr 6 "$@" 2>&1 | tee -a "$LOG"
}
for M in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B google/gemma-4-E2B-it google/gemma-4-E4B-it; do
  run "$M" "_sep"      --sepedges ;      # decisive condition (line confound removed)
  run "$M" "_neutral"  --qset neutral ;  # intrinsic
done
echo "=== E5 Stage2 done $(date) ===" | tee -a "$LOG"
echo "--- FULL LADDER SUMMARY (all grid_*.json) ---" | tee -a "$LOG"
for f in results/geometry/grid_*.json; do
  $PY -c "import json;d=json.load(open('$f'));print('%-32s RSA_grid %+.2f±%.2f  lin %+.2f  dRSA(g-l) %+.2f  sig %d/%d  beats_line %d/%d'%(d['model'].split('/')[-1]+('/'+d['qset'] if d['qset']!='rel' else '')+('+sep' if d['sepedges'] else ''),d['rsa_grid'],d['rsa_grid_std'],d['rsa_linear'],d['dRSA_grid_lin'],d['nsig_emp'],d['nscr'],d['nbeats_line'],d['nscr']))" 2>/dev/null | tee -a "$LOG"
done
