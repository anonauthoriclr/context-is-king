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
| Fig. 1 `fig:possweep` — context takes over downstream | `fig_possweep.py` | `geometry/possweep_gemma-4-31B-it_all.npz` | `geometry/geom_possweep.py` |
| `fig:defs` — what a specification is | `fig_defs.py` (`_aaai`) | *none — schematic drawn in code* | — |
| `fig:dominance` — imposed vs residual-natural RSA | `fig_dominance.py` (`_aaai`) | `geometry/multiscr_<model>.json` + `geometry/cache/defladder_<model>_days_list_s0_L*.npz` | `geometry/multiscr.py` + `geometry/defladder.py` |
| `fig:flip` — cyclic concepts under a conflicting order | `fig1_flip.py` | `geometry/cache/defladder_gemma-4-31B-it_<concept>_list_s0_L45.npz` | `geometry/defladder.py` |
| `fig:hierarchy` — imposed depth-4 tree (A/B/C) | `fig_hierarchy.py` | `geometry/demo_tree_gemma-4-31B-it_d4.npz`; `geometry/tree_{gemma,Qwen}_neutral*.npz` | `geometry/demo_tree.py`; `geometry/geom_tree.py` |
| `fig:shapes` — 7 weekdays: natural ring / imposed cycle / imposed tree | `build_abstract_fig.py` | `geometry/natsentend_gemma-4-31B-it_query.npz`; `shapes/shape_v4cent_gemma-4-31B-it.npz`; `shapes/shape_v4_gemma-4-31B-it.json` | `geometry/natural_sentend.py`; `shapes/shape_test_v4.py` |
| `fig:crossover` — entity-patching depth sweep | `fig_crossover.py` | `causal/causaluse_gemma-4-31B-it.json` | `causal/causal_use.py` |
| `fig:layersweep` — full-layer RSA sweep | `fig_layersweep.py` | `geometry/layer_sweep_gemma-4-31B-it.json` + `geometry/qwen_layer_sweep.json` | `geometry/layer_sweep.py` + `geometry/qwen_layer_sweep.py` |
| `fig:kmarg` — imposed ring per hop-value k | `k_marg_plot.py` | `geometry/cache/defladder_gemma-4-31B-it_days_list_s0_L*.npz` | `geometry/defladder.py` |
| `fig:behavior` — accuracy by hop count k | `fig_behavior.py` | `behavior/acc_<model>_days.json` | `behavior/defladder_acc.py` |
| `fig:regimes` — accuracy by prompt regime | `fig_regimes_aaai.py` | `behavior/behavior3_<model>_days.json` | `behavior/behavior_3regime.py` |
| `fig:crossorder` — cross-order identity-versus-relation control | `fig_cross_order_binding.py` | `causal/crossorder_gemma-4-31B-it_n20.json`; `causal/crossorder_gemma-4-31B-it_n20_extension.json` | `causal/causal_cross_order.py` |
| `fig:queryreloc` — depth-4 tree across 36 queries (App.) | `demo_tree_queryfig.py` | pooled tree caches (24 neutral + 12 diverse) | `geometry/demo_tree.py` + `geometry/demo_tree_diverse.py` |
| `fig:topology` — form×wrap 2×2 + closure (App.) | `fig_topology.py` | `behavior/wrap2x2_<tag>.npz` + `behavior/wrap2x2_summary.json` | `behavior/wrap_2x2.py` |

## Tables

| Table | shows | data (`data/`) | extraction (`extract/`) |
|---|---|---|---|
| `tab:queries` | query paraphrases | *static* | — |
| `tab:regimes` | accuracy by regime + over-budget | `behavior/behavior3_<model>_days.json` | `behavior/behavior_3regime.py` |
| `tab:patchlayers` | swept layer indices | *static* | — |
| `tab:causalladder` | patch Imp/Nat/clean-locus across ladder | `causal/causaluse_*.json`, `causal/causalcot_*.json`, `causal/causalk_*.json` | `causal/causal_use.py`, `causal_cot.py`, `causal_k.py` (+ `reconcile_causal.py`, `apply_judge.py`, `rescore_cot_visible_final.py`) |
| `tab:robust` | imposed−natural under non-structural probes; RSA by k | `table_summaries.json`; Gemma cache for the displayed k-panels | `geometry/geom_storyprobe.py`; `figures/k_marginalize.py` |
| `tab:rsa` | per-model imposed vs natural RSA | `geometry/multiscr_<model>.json` | `geometry/multiscr.py` |
| `tab:proj` | RSA at 2D/3D/full + variance | `table_summaries.json` | `figures/soundness_pass.py` over `geometry/defladder.py` |
| `tab:arb` | 2×2 on arbitrary tokens | `behavior/wrap2x2_summary_arb7.json`, `_arb12` | `behavior/wrap_2x2.py --concept arb7/arb12` |
| `tab:cooccur` | wrap vs endpoint co-occurrence | `behavior/wrap_cooccur_summary_days.json` | `behavior/wrap_cooccur_control.py` |
| `tab:phmatched` | matched cycle-vs-tree persistent homology | `geometry/ph_matched/ph_matched_results.json`; `ph_matched_paired_results.json`; summary CSV | `geometry/ph_matched_extract.py`; `ph_matched_analyze.py`; `ph_matched_paired.py` |
| `tab:shapes` | shapes on weekdays (RSA, eff-dim) | `shapes/shape_v4_<tag>.json`; `geometry/natsentend_<tag>.json` for the matched no-rule row | `shapes/shape_test_v4.py`; `geometry/natural_sentend.py` |

## Key quantitative claims

| claim | source data | extraction |
|---|---|---|
| RSA dominance (days imposed +0.90 direct / +0.58 free-form; months +0.81) | `geometry/multiscr_*.json` | `geometry/multiscr.py` (+ `natural_geom.py`) |
| Relabeling crossover (imposed +0.84 / natural −0.04; rank 15/5040) | defladder / multiscr centroid caches | `geometry/geom_concept.py` |
| Anisotropy-matched null (10/10 scrambles p<0.05) | permutation null | inside `geometry/multiscr.py`; CIs via `figures/soundness_pass.py` |
| Patching crossover (imp→imp 1.00; nat→nat 1.00; chance 0.14; depth 0.46→0.70) | `causal/causaluse_*.json` (20 imposed orders/model; fixed natural control repeated 6× Gemma, 2× Qwen); Qwen CoT three-order summary in `causal/causalcot_Qwen3.5-27B_imposed3_visible_final.json` | `causal/causal_use.py` (+ `causal_cot.py`, `causal_k.py`, `rescore_cot_visible_final.py`) |
| Scale-gating (Gemma E2B 0.60→31B 0.87; Qwen reverses @9B; Llama-8B 0.08) | `geometry/multiscr_*.json` | `geometry/multiscr.py` |
| RSA↔behavior ρ=0.83 / planarity ρ=0.93 | `behavior/k1summary_*_list.json` | `figures/make_paper_figs.py`, `soundness_pass.py` |
| Matched persistent-homology contrast | `geometry/ph_matched/*.json` + summary CSV | `extract/geometry/ph_matched_{extract,analyze,paired}.py` (analysis needs `ripser`) |
| 2-D grid extension (App. J): native and imposed QWERTY; arbitrary-token grid | `grid/qwerty_*.json`; `grid/qwertyimposed_*.json`; `grid/grid_gemma-4-31B-it_sep.json` | `grid/geom_qwerty.py`; `grid/geom_qwerty_imposed.py`; `grid/geom_grid.py` |
