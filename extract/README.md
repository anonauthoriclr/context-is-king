# Extraction layer (GPU)

These scripts regenerate the cached artifacts in `../data/` from model weights.
You only need this if you want to rebuild the activations from scratch; to just
reproduce the figures, use the cached data (see the top-level README).

## Method in one paragraph

Each script loads an instruction-tuned model with HuggingFace `transformers`
(bf16, single GPU), runs a forward pass with a hook, and reads the **last-token,
pre-generation residual** at a late layer (`round(0.75 · n_layers)`). Activations
are grouped by the input entity into centroids, from which we compute a
mean-centered, full-dimensional cosine RDM and score it against a candidate
structure's template (RSA). Residuals are cached to `.npz` so re-plotting needs
no GPU. There is **no external extraction service and no vLLM** — everything is
self-contained here.

## Environments

Gemma-4 loading is version-sensitive, so the paper used two venvs (see
`../requirements-extract.txt`):

| models | transformers | venv (paper) |
|--------|--------------|--------------|
| Qwen-3.5 4B/9B/27B, Gemma-4-31B-it, Llama-3.1-8B-Instruct | `5.5.3` | `gemma4-venv` |
| Gemma-4 unified E2B / E4B / 12B (`model_type=gemma4_unified`) | `5.12.1` | `gemma4u-venv` |

Export the interpreter once so the orchestrators pick it up:

```bash
export VENV=/path/to/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Models (HF ids)

`google/gemma-4-{E2B,E4B,12B,31B}-it`, `Qwen/Qwen3.5-{4B,9B,27B}`,
`meta-llama/Llama-3.1-8B-Instruct` (plus base variants for the base-model claim).
Gemma-4-31B needs a large-memory GPU (bf16).

## Layout

- `geometry/` — RSA geometry: `defladder.py` (centroid caches), `multiscr.py`
  (per-model imposed vs natural RSA), `layer_sweep.py` / `qwen_layer_sweep.py`,
  `geom_tree.py` / `demo_tree*.py` (imposed trees), `geom_possweep.py`
  (readout-position), `natural_geom.py` / `natural_sentend.py` (no-context
  baseline), `geom_concept.py`, `geom_storyprobe.py`, `family_certify.py`,
  `verify_pretrained.py`, `xconcept_rings.py`, `adj_rings.py`.
- `behavior/` — accuracy & structure behavior: `defladder_acc.py`,
  `behavior_3regime.py`, `wrap_2x2.py`, `wrap_cooccur_control.py`.
- `causal/` — activation patching: `causal_use.py`, `causal_cot.py`,
  `causal_k.py`, plus `apply_judge.py` / `prep_judge_batches.py` and
  `reconcile_causal.py` (consolidation).
- `shapes/` — exp03 topology-type: `shape_test_v4.py`, `shape_extract_forplot.py`.
- `grid/` — appendix 2-D / topology controls: `geom_qwerty*.py`,
  `geom_natural2d.py`, `geom_grid.py`, `geom_topo.py`, `ph_topology.py`
  (persistent homology; needs `ripser`).
- `orchestrators/` — resumable shell sweeps over the model ladder. They honor
  `$VENV` and `$ROOT`. Examples: `sweep_geom.sh` (8-model geometry),
  `sweep_acc.sh`, `run_causal_all.sh`, `run_baseladder.sh`, `run_d4.sh`.

## Typical commands

```bash
# geometry sweep across the ladder for one concept
bash orchestrators/sweep_geom.sh days

# a single model / concept
$VENV geometry/multiscr.py Qwen/Qwen3.5-27B --concepts days,months

# causal patching on Gemma-4-31B
$VENV causal/causal_use.py google/gemma-4-31B-it --nscr 2

# imposed depth-4 tree, neutral queries
$VENV geometry/geom_tree.py google/gemma-4-31B-it --depth 4 --qset neutral
```

Outputs land under `../data/` (matching the layout the figure scripts expect).
See `../ARTIFACT_MAP.md` for exactly which script feeds which paper object.
