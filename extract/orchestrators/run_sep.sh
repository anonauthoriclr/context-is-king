#!/bin/bash
# CO-OCCURRENCE CONTROL: re-declare the tree with one edge per sentence so siblings are NOT co-listed (shuffled apart).
# If the sibling bin stays tight => rendered hierarchy; if it jumps to cousin level => it was text co-occurrence.
# depth-3 neutral, 6 scrambles, both families. Waits for the current d4 job to free the GPU first (sequential).
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
gf(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{exit ($1>5000)?0:1}';do sleep 15;done; }
gf   # wait until the depth-4 run releases the GPU
[ -f results/geometry/tree_gemma-4-31B-it_neutral_d3_sep.npz ] || { "$V/bin/python" -u geom_tree.py google/gemma-4-31B-it --nscr 6 --qset neutral --depth 3 --sepedges > results/geometry/log_sep_31b.txt 2>&1; gf; }
[ -f results/geometry/tree_Qwen3.5-27B_neutral_d3_sep.npz ]    || "$V/bin/python" -u geom_tree.py Qwen/Qwen3.5-27B --nscr 6 --qset neutral --depth 3 --sepedges > results/geometry/log_sep_qwen.txt 2>&1
echo DONE_SEP
