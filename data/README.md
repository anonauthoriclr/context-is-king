# Cached artifacts

These are the pre-computed activation caches and result summaries that back the
paper's figures and tables. They let you reproduce every figure and number
**without a GPU or model weights**.

Everything here is **committed to the repo** (JSON summaries + `.npz` activation
caches, ~300 MB total), so the package is self-contained: `git clone` and every
figure and table reproduces with no external download. To rebuild these
artifacts from model weights instead, see `../extract/`.

## Layout (mirrors the original `results/` tree)

```
data/
├── geometry/                 # RSA / centroid caches, layer sweeps, trees, possweep
│   ├── multiscr_<model>.json         # per-model imposed vs natural RSA  -> tab:rsa, fig_dominance
│   ├── layer_sweep_*.json            # full-layer RSA sweep            -> fig_layersweep
│   ├── possweep_<tag>_all.npz        # readout-position sweep          -> fig_possweep
│   ├── tree_*_neutral*.npz           # imposed depth trees            -> fig_hierarchy
│   ├── demo_tree_*_d4.npz            # depth-4 tree demo              -> fig_hierarchy A / queryreloc
│   └── cache/defladder_*_list_s0_L*.npz   # centroid caches          -> fig1_flip, fig_dominance, fig_kmarg
├── behavior/                 # accuracy, regimes, wrap 2x2, co-occurrence
│   ├── acc_<model>_days.json         -> fig_behavior
│   ├── behavior3_<model>_days.json   -> fig_regimes, tab:regimes
│   ├── k1summary_*_list.json         -> rsa-vs-behavior correlation
│   ├── wrap2x2_*.npz / _summary*.json-> fig_topology, tab:arb
│   └── wrap_cooccur_*.npz            -> tab:cooccur
├── causal/                   # activation-patching results
│   ├── causaluse_<tag>.json          -> fig_boundary, tab:causalladder
│   ├── causalcot_<tag>.json          -> tab:causalladder (CoT rows)
│   └── causalk_<tag>.json            -> step-count patching
├── shapes/                   # exp03: same 7 weekdays as ring/line/tree
│   ├── shape_v4cent_<tag>.npz        -> fig_shapes
│   └── shape_v4_<tag>.json           -> tab:shapes
└── soundness_ci.md           # per-scramble confidence intervals (appendix)
```

`ARTIFACT_MAP.md` (repo root) has the full paper-object -> figure-script ->
data-artifact -> extraction-script mapping.

## Overriding the data location

Every figure script resolves data through the `CIK_DATA` environment variable,
defaulting to this directory:

```bash
CIK_DATA=/path/to/data python figures/fig_dominance.py
```
