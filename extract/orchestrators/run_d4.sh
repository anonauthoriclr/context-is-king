#!/bin/bash
# Depth-4 (31-node, 5-level) imposed-tree geometry, 6 scrambles, neutral queries — Gemma + Qwen.
# Settles whether the renderer's branch fidelity holds at the deepest level (the single-scramble demo folded; n=1).
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
gf(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{exit ($1>5000)?0:1}';do sleep 15;done; }
# skip a model if its depth-4 output already exists (resume-safe)
[ -f results/geometry/tree_gemma-4-31B-it_neutral_d4.npz ] || { "$V/bin/python" -u geom_tree.py google/gemma-4-31B-it --nscr 6 --qset neutral --depth 4 > results/geometry/log_d4_31b.txt 2>&1; gf; }
[ -f results/geometry/tree_Qwen3.5-27B_neutral_d4.npz ]    || "$V/bin/python" -u geom_tree.py Qwen/Qwen3.5-27B --nscr 6 --qset neutral --depth 4 > results/geometry/log_d4_qwen.txt 2>&1
echo DONE_D4
