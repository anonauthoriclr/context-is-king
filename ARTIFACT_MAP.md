# Artifact map

Every figure and table in the paper, traced to the script that draws it, the
cached data it reads (`data/…`), and the extraction script that produced that
data (`extract/…`). `PROVENANCE.md` is the authority on which regime each number
was measured under.

Figure scripts resolve data through `CIK_DATA` (default `./data`). Run them from
`figures/`.

## Figures

| Paper object | figure script (`figures/`) | data (`data/`) | extraction (`extract/`) |
|---|---|---|---|
| Fig 1 `fig:flip` — cyclic concepts under a conflicting order | `fig1_flip.py` | `geometry/cache/defladder_<tag>_<concept>_list_s0_L*.npz` | `geometry/defladder.py` |
| `fig:defs` — what a specification is | `fig_defs.py` (`_aaai`) | *none — schematic drawn in code* | — |
| `fig:dominance` — imposed vs residual-natural RSA | `fig_dominance.py` (`_aaai`) | `geometry/multiscr_<model>.json` + `geometry/cache/defladder_<model>_days_list_s0_L*.npz` | `geometry/multiscr.py` + `geometry/defladder.py` |
| `fig:possweep` — readout-position sweep | `geom_possweep_plot.py` | `geometry/possweep_<tag>_all.npz` | `geometry/geom_possweep.py` |
| `fig:hierarchy` — imposed depth-4 tree (A/B/C) | `fig_hierarchy.py` | `geometry/demo_tree_gemma-4-31B-it_d4.npz`; `geometry/tree_{gemma,Qwen}_neutral*.npz` | `geometry/demo_tree.py`; `geometry/geom_tree.py` |
| `fig:shapes` — 7 weekdays: ring / cycle / tree | `build_paper_fig.py` (abstract: `build_abstract_fig.py`) | `shapes/shape_v4cent_<tag>.npz` + `shapes/shape_v4_<tag>.json` | `shapes/shape_test_v4.py` |
| `fig:layersweep` — full-layer RSA sweep | `fig_layersweep.py` | `geometry/layer_sweep_gemma-4-31B-it.json` + `geometry/qwen_layer_sweep.json` | `geometry/layer_sweep.py` + `geometry/qwen_layer_sweep.py` |
| `fig:kmarg` — imposed ring per hop-value k | `k_marg_plot.py` | `geometry/cache/defladder_gemma-4-31B-it_days_list_s0_L*.npz` | `geometry/defladder.py` |
| `fig:behavior` — accuracy by hop count k | `fig_behavior.py` | `behavior/acc_<model>_days.json` | `behavior/defladder_acc.py` |
| `fig:regimes` — accuracy by prompt regime | `fig_regimes.py` (`_aaai`) | `behavior/behavior3_<model>_days.json` | `behavior/behavior_3regime.py` |
| `fig:boundary` — identity→binding depth (App.) | `make_paper_figs.py` (`boundary_fig`) | `causal/causaluse_gemma-4-31B-it.json` | `causal/causal_use.py` |
| `fig:queryreloc` — depth-4 tree across 36 queries (App.) | `demo_tree_queryfig.py` | pooled tree caches (24 neutral + 12 diverse) | `geometry/demo_tree.py` + `geometry/demo_tree_diverse.py` |
| `fig:topology` — form×wrap 2×2 + closure (App.) | `fig_topology.py` | `behavior/wrap2x2_<tag>.npz` + `behavior/wrap2x2_summary.json` | `behavior/wrap_2x2.py` |
| AAAI-only: scale-is-the-gate | `make_paper_figs.py` (`scale_fig`) | `geometry/multiscr_*.json` | `geometry/multiscr.py` |

## Tables

| Table | shows | data (`data/`) | extraction (`extract/`) |
|---|---|---|---|
| `tab:queries` | query paraphrases | *static* | — |
| `tab:regimes` | accuracy by regime + over-budget | `behavior/behavior3_<model>_days.json` | `behavior/behavior_3regime.py` |
| `tab:patchlayers` | swept layer indices | *static* | — |
| `tab:causalladder` | patch Imp/Nat/clean-locus across ladder | `causal/causaluse_*.json`, `causal/causalcot_*.json`, `causal/causalk_*.json` | `causal/causal_use.py`, `causal_cot.py`, `causal_k.py` (+ `reconcile_causal.py`, `apply_judge.py`) |
| `tab:robust` | imposed−natural under non-structural probes; RSA by k | `geometry/multiscr_*.json`, defladder caches | `geometry/multiscr.py`, `figures/k_marginalize.py`, `geometry/geom_storyprobe.py` |
| `tab:rsa` | per-model imposed vs natural RSA | `geometry/multiscr_<model>.json` | `geometry/multiscr.py` |
| `tab:proj` | RSA at 2D/3D/full + variance | defladder/multiscr caches | `figures/soundness_pass.py` over `geometry/defladder.py` |
| `tab:arb` | 2×2 on arbitrary tokens | `behavior/wrap2x2_summary_arb7.json`, `_arb12` | `behavior/wrap_2x2.py --concept arb7/arb12` |
| `tab:cooccur` | wrap vs endpoint co-occurrence | `behavior/wrap_cooccur_<tag>_days.npz` | `behavior/wrap_cooccur_control.py` |
| `tab:shapes` | shapes on weekdays (RSA, eff-dim) | `shapes/shape_v4_<tag>.json` | `shapes/shape_test_v4.py` |

## Key quantitative claims

| claim | source data | extraction |
|---|---|---|
| RSA dominance (days imposed +0.90 direct / +0.58 free-form; months +0.81) | `geometry/multiscr_*.json` | `geometry/multiscr.py` (+ `natural_geom.py`) |
| Relabeling crossover (imposed +0.84 / natural −0.04; rank 15/5040) | defladder / multiscr centroid caches | `geometry/geom_concept.py` |
| Anisotropy-matched null (10/10 scrambles p<0.05) | permutation null | inside `geometry/multiscr.py`; CIs via `figures/soundness_pass.py` |
| Patching crossover (imp→imp 1.00; nat→nat 1.00; chance 0.14; depth 0.46→0.70) | `causal/causaluse_*.json` | `causal/causal_use.py` (+ `causal_cot.py`, `causal_k.py`) |
| Scale-gating (Gemma E2B 0.60→31B 0.87; Qwen reverses @9B; Llama-8B 0.08) | `geometry/multiscr_*.json` | `geometry/multiscr.py` |
| RSA↔behavior ρ=0.83 / planarity ρ=0.93 | `behavior/k1summary_*_list.json` | `figures/make_paper_figs.py`, `soundness_pass.py` |
| Persistent-homology H1 loop | — | `extract/grid/ph_topology.py` (needs `ripser`) |
