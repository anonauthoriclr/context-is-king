#!/bin/bash
# E6 (chess native 2-D / store) + E7 (complex novel topology / construct). Forward-only, file-skip resumable. gemma4-venv.
set -u
cd "$(dirname "$0")"
V="${VENV:-/path/to/gemma4-venv}"
PY=$V/bin/python
LOG=results/geometry/topo_run.log
mkdir -p results/geometry
echo "=== TOPO run start $(date) ===" | tee -a "$LOG"
FLAG="google/gemma-4-31B-it Qwen/Qwen3.5-27B"
SMALL="google/gemma-4-E2B-it"

topo () {  # model  graph-args...   (skip if json exists)
  local M="$1"; shift
  local pre="$($PY -c "import sys;a=sys.argv[1:];g=a[a.index('--graph')+1] if '--graph' in a else 'random';n=''.join(['-d'+a[a.index('--diff')+1]] if '--diff' in a and a[a.index('--diff')+1]!='0' else []);import re;print(g+ ('-n'+a[a.index('--n')+1]+'-b'+a[a.index('--b1')+1]+'-s'+a[a.index('--seed')+1] if g=='random' else '')+n)" "$@" 2>/dev/null)"
  local OUT="results/geometry/topo_${pre}_$(basename "$M").json"
  if [ -f "$OUT" ]; then echo "SKIP topo $pre/$(basename "$M")" | tee -a "$LOG"; return; fi
  echo ">>> topo $(basename "$M") $* @ $(date)" | tee -a "$LOG"
  $PY -u geom_topo.py "$M" "$@" --nscr 6 2>&1 | tee -a "$LOG"
}
nat2d () {  # model  extra...
  local M="$1"; shift; local sz=3; local op=""
  [[ "$*" == *"--opaque"* ]] && op="_opaque"
  local OUT="results/geometry/nat2d_$(basename "$M")_3x3${op}.json"
  if [ -f "$OUT" ]; then echo "SKIP nat2d $(basename "$M")$op" | tee -a "$LOG"; return; fi
  echo ">>> nat2d $(basename "$M") $* @ $(date)" | tee -a "$LOG"
  $PY -u geom_natural2d.py "$M" --size 3 "$@" 2>&1 | tee -a "$LOG"
}

# ---- E7 showcase: hypercube Q3 ----
for M in $FLAG $SMALL; do topo "$M" --graph hypercube; done
# ---- E7 discriminator: random A (s0) + diff-2 partner B ----
for M in $FLAG; do
  topo "$M" --graph random --n 9 --b1 3 --seed 0 --diff 0
  topo "$M" --graph random --n 9 --b1 3 --seed 0 --diff 2
done
# ---- E7 exotic (optional, flagship only): petersen ----
topo google/gemma-4-31B-it --graph petersen
# ---- E6 store side: chess native 2-D + opaque control ----
for M in $FLAG $SMALL; do nat2d "$M"; done
for M in $FLAG; do nat2d "$M" --opaque; done

echo "=== differential cross-RSA (E7) ===" | tee -a "$LOG"
for M in $FLAG; do
  b=$(basename "$M")
  A="results/geometry/topo_random-n9-b3-s0_${b}.npz"; B="results/geometry/topo_random-n9-b3-s0-d2_${b}.npz"
  [ -f "$A" ] && [ -f "$B" ] && { echo "-- $b --" | tee -a "$LOG"; $PY analyze_topo.py "$A" "$B" 2>&1 | tee -a "$LOG"; }
done

echo "=== TOPO SUMMARY ($(date)) ===" | tee -a "$LOG"
for f in results/geometry/topo_*.json; do
  $PY -c "import json;d=json.load(open('$f'));i=d['invariants'];print('%-40s self %+.2f±%.2f (line %+.2f ring %+.2f) graded %+.2f | self-best %d/%d sig %d/%d | b1=%d novel=%s'%(d['graph']+'/'+d['model'].split('/')[-1],d['rsa_self'],d['rsa_self_std'],d['rsa_line'],d['rsa_ring'],d['graded_nonadj'],d['n_self_best'],d['nscr'],d['nsig_emp'],d['nscr'],i['b1'],i['is_novel']))" 2>/dev/null | tee -a "$LOG"
done
for f in results/geometry/nat2d_*.json; do
  $PY -c "import json;d=json.load(open('$f'));r=d['rsa'];print('%-34s king2D %+.2f rook2D %+.2f line1D %+.2f | best=%s dRSA(2D-1D) %+.2f p_2D=%.3f'%(d['model'].split('/')[-1]+('+opaque' if d['opaque'] else ''),r['king2D'],r['rook2D'],r['line1D'],d['best_template'],d['dRSA_2D_minus_1D'],d['p_emp_2D']))" 2>/dev/null | tee -a "$LOG"
done
echo "=== TOPO run done $(date) ===" | tee -a "$LOG"
