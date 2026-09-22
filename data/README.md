# Cached artifacts

These are the figure-ready activation caches and machine-readable result
summaries that back the paper. They let you reproduce every figure **without a
GPU or model weights** and inspect the per-scramble or aggregate outputs behind
each table.

Everything here is **committed to the repo** (JSON/CSV summaries plus selected
`.npz` figure caches, about 175 MB for the complete package). The release is
deliberately curated: large raw activation arrays used only for table analyses,
including the matched persistent-homology tensors, are omitted when compact
per-scramble or aggregate outputs preserve the reported result. To rebuild the
omitted raw artifacts from model weights, see `../extract/`.

## Layout (mirrors the original `results/` tree)

```
data/
├── geometry/                 # RSA / centroid caches, layer sweeps, trees, possweep
│   ├── multiscr_<model>.json         # per-model imposed vs natural RSA  -> tab:rsa, fig_dominance
│   ├── layer_sweep_*.json            # full-layer RSA sweep            -> fig_layersweep
│   ├── possweep_<tag>_all.npz        # readout-position sweep          -> fig_possweep
│   ├── tree_*_neutral*.npz           # imposed depth trees            -> fig_hierarchy
│   ├── demo_tree_*_d4.npz            # depth-4 tree demo              -> fig_hierarchy A / queryreloc
│   ├── natsentend_<tag>_query.npz     # no-rule final-token baseline    -> fig_shapes
│   ├── natsentend_<tag>.json           # compact no-rule statistics      -> tab:shapes
│   ├── ph_matched/                    # compact matched-PH outputs       -> tab:phmatched
│   └── cache/defladder_*_list_s0_L*.npz   # centroid caches          -> fig1_flip, fig_dominance, fig_kmarg
├── behavior/                 # accuracy, regimes, wrap 2x2, co-occurrence
│   ├── acc_<model>_days.json         -> fig_behavior
│   ├── behavior3_<model>_days.json   -> fig_regimes, tab:regimes
│   ├── k1summary_*_list.json         -> rsa-vs-behavior correlation
│   ├── wrap2x2_*.npz / _summary*.json-> fig_topology, tab:arb
│   └── wrap_cooccur_summary_days.json -> tab:cooccur
├── causal/                   # activation-patching results
│   ├── causaluse_<tag>.json          -> fig_crossover, tab:causalladder
│   ├── crossorder_<tag>_n20*.json    -> fig_crossorder
│   ├── causalcot_<tag>.json          -> tab:causalladder (CoT rows)
│   └── causalk_<tag>.json            -> step-count patching
├── shapes/                   # exp03: same 7 weekdays as ring/line/tree
│   ├── shape_v4cent_<tag>.npz        -> fig_shapes
│   └── shape_v4_<tag>.json           -> tab:shapes (imposed rows)
├── table_summaries.json      # compact outputs for tab:robust and tab:proj
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
