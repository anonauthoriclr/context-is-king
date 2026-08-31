#!/usr/bin/env bash
# Edit my_prompt.py, then run:  bash run_my_prompt.sh
# Coexistence-safe: single GPU, bf16, expandable segments, aborts if <18GB free.
cd "$(dirname "$0")" || exit 1
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CIK_MAXTOK="${CIK_MAXTOK:-220}" \
  "${VENV:-/path/to/projects/.venv/bin/python}" run_my_prompt.py 2>&1 \
  | grep -v -E "checkpoint|it/s\]|deprecated|generation flags|TRANSFORMERS|Setting pad"
