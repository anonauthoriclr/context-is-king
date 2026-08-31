#!/bin/bash
# Clean BASE-model scale ladder (no instruction-tuning) on the multiscr protocol = directly comparable to the instruct
# table. Breaks the tuning/family confound on "scale is the gate." Small->large; file-skip. gemma4-venv.
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4-venv/bin/python}"
LOG=results/geometry/baseladder.log
echo "=== base ladder start $(date) ===" | tee -a "$LOG"
run () {  # model
  local M="$1"; local TAG="$(basename "$M")_base"
  local OUT="results/geometry/multiscr_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $TAG (exists)" | tee -a "$LOG"; return; fi
  echo ">>> $TAG @ $(date)" | tee -a "$LOG"
  $PY -u multiscr.py "$M" --base --concepts months,zodiac,days --nscr 10 2>&1 | tee -a "$LOG"
}
for M in google/gemma-4-E2B google/gemma-4-E4B Qwen/Qwen3.5-2B-Base Qwen/Qwen3.5-9B-Base google/gemma-4-31B; do run "$M"; done
echo "=== BASE LADDER SUMMARY ===" | tee -a "$LOG"
for f in results/geometry/multiscr_*_base.json; do
  $PY -c "import json;d=json.load(open('$f'));r=d['res'];print('%-26s '%(d['model'].split('/')[-1])+'  '.join('%s:imp%+.2f/nat%+.2f/sig%s'%(c,r[c].get('imp_mean',0),r[c].get('nat_ret_mean',0),r[c].get('nsig','?')) for c in ['months','zodiac','days'] if c in r))" 2>/dev/null | tee -a "$LOG"
done
echo "=== base ladder done $(date) ===" | tee -a "$LOG"
