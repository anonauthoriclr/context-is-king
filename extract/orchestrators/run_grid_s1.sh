#!/bin/bash
# E5 Stage 1: 2-D grid geometry on both FLAGSHIPS x {rel-visual, sepedges co-occ control, neutral intrinsic}.
# Forward-only, ~minutes each. File-skip resumable (grid_<tag>.json). gemma4-venv. Run from exp01 dir.
set -u
cd "$(dirname "$0")"
V="${VENV:-/path/to/gemma4-venv}"
PY=$V/bin/python
LOG=results/geometry/grid_s1.log
mkdir -p results/geometry
echo "=== E5 Stage1 start $(date) ===" | tee -a "$LOG"
run () {  # args: model  tagsuffix  extra-flags...
  local M="$1"; local TAGSUF="$2"; shift 2
  local TAG="$(basename "$M")${TAGSUF}"
  local OUT="results/geometry/grid_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $TAG (exists)" | tee -a "$LOG"; return; fi
  echo ">>> RUN $TAG : geom_grid.py $M --nscr 6 $* @ $(date)" | tee -a "$LOG"
  $PY -u geom_grid.py "$M" --nscr 6 "$@" 2>&1 | tee -a "$LOG"
}
for M in google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
  run "$M" ""          ;                 # rel + visual grid block (default)
  run "$M" "_sep"      --sepedges ;      # co-occurrence control
  run "$M" "_neutral"  --qset neutral ;  # intrinsic (no structural query)
done
echo "=== E5 Stage1 done $(date) ===" | tee -a "$LOG"
echo "--- SUMMARY ---" | tee -a "$LOG"
for f in results/geometry/grid_*.json; do
  $PY -c "import json,sys;d=json.load(open('$f'));print('%-34s RSA_grid %+.2f±%.2f  lin %+.2f  ring %+.2f  dRSA(g-l) %+.2f  col_sig %+.2f  sig %d/%d  beats_line %d/%d'%(d['model'].split('/')[-1]+('/'+d['qset'] if d['qset']!='rel' else '')+('+sep' if d['sepedges'] else ''),d['rsa_grid'],d['rsa_grid_std'],d['rsa_linear'],d['rsa_ring'],d['dRSA_grid_lin'],d['col_sig'],d['nsig_emp'],d['nscr'],d['nbeats_line'],d['nscr']))" 2>/dev/null | tee -a "$LOG"
done
