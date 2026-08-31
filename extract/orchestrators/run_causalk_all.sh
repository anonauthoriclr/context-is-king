#!/bin/bash
# k-sweep (causal_k.py, step-count patching, days) across the full ladder. Resumable: skips any model whose
# causalk_<tag>.json already exists. One process per model -> GPU freed between models. 12B needs gemma4u-venv.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G="${VENV:-/path/to/gemma4-venv/bin/python}"
GU="${VENV:-/path/to/gemma4u-venv/bin/python}"

run() {  # $1=python  $2=hf_model  $3=tag
  out="results/causal/causalk_$3.json"
  if [ -f "$out" ]; then echo "### SKIP $3 (exists) ###"; return; fi
  echo "### RUN $3 ###"
  if "$1" -u causal_k.py "$2" --nscr 2 --kmax 6 --concept days; then echo "### DONE $3 ###"; else echo "### FAIL $3 ###"; fi
}

run "$G"  google/gemma-4-E2B-it  gemma-4-E2B-it
run "$G"  google/gemma-4-E4B-it  gemma-4-E4B-it
run "$GU" google/gemma-4-12B-it  gemma-4-12B-it
run "$G"  Qwen/Qwen3.5-4B        Qwen3.5-4B
run "$G"  Qwen/Qwen3.5-9B        Qwen3.5-9B
run "$G"  Qwen/Qwen3.5-27B       Qwen3.5-27B
echo "### ALL DONE ###"
