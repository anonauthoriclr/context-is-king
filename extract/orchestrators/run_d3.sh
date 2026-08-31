#!/bin/bash
cd "${ROOT:-/path/to/context-is-king/extract}"
V="${VENV:-/path/to/gemma4-venv}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
gf(){ while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{exit ($1>5000)?0:1}';do sleep 15;done; }
"$V/bin/python" -u geom_tree.py google/gemma-4-31B-it --nscr 6 --qset neutral --depth 3 > results/geometry/log_d3_31b.txt 2>&1; gf
"$V/bin/python" -u geom_tree.py Qwen/Qwen3.5-27B --nscr 6 --qset neutral --depth 3 > results/geometry/log_d3_qwen.txt 2>&1
echo DONE_D3
