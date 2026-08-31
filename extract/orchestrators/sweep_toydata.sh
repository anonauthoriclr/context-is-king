#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
PY="${VENV:-/path/to/gemma4u-venv/bin/python}"
for M in google/gemma-4-31B-it Qwen/Qwen3.5-27B; do
  TAG=$(basename "$M")
  [ -f "results/behavior/toydata_${TAG}.npz" ] && { echo "[skip] $TAG"; continue; }
  echo "[run ] $TAG $(date)"; $PY -u toy_data.py "$M" --nord 24 > "logs/toydata_${TAG}.log" 2>&1
  [ -f "results/behavior/toydata_${TAG}.npz" ] && echo "[ ok ] $TAG" || echo "[FAIL] $TAG"
done
echo "TOYDATA ALL DONE $(date)"
