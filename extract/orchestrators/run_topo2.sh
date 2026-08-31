#!/bin/bash
# E6+E7 single-load driver: each model loaded ONCE (cold loads were ~15-17 min). file-skip resumable. gemma4-venv.
set -u
cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/topo2_run.log
mkdir -p results/geometry
echo "=== TOPO2 start $(date) ===" | tee -a "$LOG"
# E2B first (fast) — only the new configs (hypercube + chess already done)
$PY -u topo_driver.py google/gemma-4-E2B-it --configs hypercube-scaffold,chess-opaque --nscr 6 2>&1 | tee -a "$LOG"
# Qwen-27B (one load)
$PY -u topo_driver.py Qwen/Qwen3.5-27B --configs hypercube,hypercube-scaffold,random:0:0,random:0:2,chess,chess-opaque --nscr 6 2>&1 | tee -a "$LOG"
# Gemma-31B (one load; bare hypercube already exists -> skipped)
$PY -u topo_driver.py google/gemma-4-31B-it --configs hypercube,hypercube-scaffold,random:0:0,random:0:2,petersen,chess,chess-opaque --nscr 6 2>&1 | tee -a "$LOG"

echo "=== differential cross-RSA (E7) ===" | tee -a "$LOG"
for M in Qwen3.5-27B gemma-4-31B-it; do
  A="results/geometry/topo_random-n9-b3-s0_${M}.npz"; B="results/geometry/topo_random-n9-b3-s0-d2_${M}.npz"
  [ -f "$A" ] && [ -f "$B" ] && { echo "-- $M --" | tee -a "$LOG"; $PY analyze_topo.py "$A" "$B" 2>&1 | tee -a "$LOG"; }
done

echo "=== TOPO SUMMARY ($(date)) ===" | tee -a "$LOG"
for f in results/geometry/topo_*.json; do
  $PY -c "import json;d=json.load(open('$f'));i=d['invariants'];print('%-42s self %+.2f±%.2f (line %+.2f ring %+.2f) graded %+.2f | self-best %d/%d sig %d/%d | b1=%d'%(d['graph']+'/'+d['model'].split('/')[-1],d['rsa_self'],d['rsa_self_std'],d['rsa_line'],d['rsa_ring'],d['graded_nonadj'],d['n_self_best'],d['nscr'],d['nsig_emp'],d['nscr'],i['b1']))" 2>/dev/null | tee -a "$LOG"
done
for f in results/geometry/nat2d_*.json; do
  $PY -c "import json;d=json.load(open('$f'));r=d['rsa'];print('%-34s king2D %+.2f rook2D %+.2f line1D %+.2f | best=%s dRSA(2D-1D) %+.2f p_2D=%.3f'%(d['model'].split('/')[-1]+('+opaque' if d['opaque'] else ''),r['king2D'],r['rook2D'],r['line1D'],d['best_template'],d['dRSA_2D_minus_1D'],d['p_emp_2D']))" 2>/dev/null | tee -a "$LOG"
done
echo "=== TOPO2 done $(date) ===" | tee -a "$LOG"
