# Measurement provenance and claim scope

**Current as of 2026-09-22.** This file records the experimental regime behind
the paper's reported results. It is not a research diary: superseded framings
and abandoned hypotheses are intentionally omitted.
`ARTIFACT_MAP.md` maps each paper figure and table to its plotting code, cached
data, and extraction code; this file explains what those measurements mean.

## 1. Shared measurement conventions

- **Primary readout.** Relational geometry is measured from the residual stream
  at the final prompt token, immediately before generation. This state follows
  the rule, query, and any regime instruction. It is therefore a downstream,
  task-conditioned representation, not the entity token in isolation.
- **Probe layer.** The standard geometry readout is at
  `round(0.75 * n_layers)`. This fraction was selected from full-layer sweeps on
  Gemma-31B and Qwen-27B, then fixed across models and concepts. The full sweeps
  are cached in `data/geometry/layer_sweep_*.json`.
- **Entity centroids.** In the main cyclic experiment, each entity centroid
  averages six query paraphrases at each hop count `k=1,...,6`, for 36 prompt
  states per entity and imposed order.
- **RSA.** Every reported representational similarity analysis uses the
  full-dimensional, mean-centered cosine representational dissimilarity matrix
  (RDM). The statistic is Spearman correlation between its off-diagonal entries
  and those of the candidate relational-distance template. PCA is used for
  visualization and for the separately reported planarity diagnostics, not to
  choose the dimensions used by RSA.
- **Independent unit.** Scrambles or token-to-position assignments are the
  independent units for uncertainty. Headline confidence intervals are computed
  across those units rather than across pooled prompts or entity pairs.
- **Models.** The main scale comparison uses eight instruction-tuned checkpoints:
  Gemma-4 E2B, E4B, 12B, and 31B; Qwen-3.5 4B, 9B, and 27B; and
  Llama-3.1-8B-Instruct. Base-model caches are supplementary and are not part of
  the paper's eight-model scaling claim.

## 2. Prompt and generation regimes

| regime | instruction and model mode | geometry readout or behavioral endpoint |
|---|---|---|
| **direct** | `Answer with ONLY the name, nothing else.`; native thinking disabled where supported | final prompt token for geometry; generated short answer for behavior |
| **scripted reasoning** | explicit instruction to traverse step by step and finish with `FINAL: name`; native thinking disabled | parsed final answer after the prescribed traversal |
| **free-form** | no trailing answer-format instruction; native thinking enabled | generated final answer; when geometry is measured, the readout is still the final prompt token before reasoning begins |
| **non-structural probe** | the same contextual rule followed by a bare mention or short story that asks nothing about the relation | final prompt token |

The direct and scripted regimes are deliberately different tasks. Results from
the scripted regime are not described as native chain-of-thought. The free-form
geometry result is a **pre-reasoning** readout under a native-thinking scaffold;
the paper does not measure entity-layout geometry during the generated reasoning
trace.

## 3. Evidence ledger

### 3.1 Conflicting cyclic orders and scale

- **Extraction:** `extract/geometry/multiscr.py`
- **Cached outputs:** `data/geometry/multiscr_<model>.json`
- **Regime:** direct, zero-shot (`--noexamples`), final pre-generation state.
- **Sampling:** 10 imposed orders per model and concept; six paraphrases and six
  hop counts per entity.
- **Reported quantity:** imposed-order RSA and residual natural-order RSA in the
  same state. The per-scramble values support the confidence intervals in
  `data/soundness_ci.md`.
- **Scope:** this establishes context-dependent relational organization and its
  variation across the tested checkpoints. It is not, by itself, a causal claim
  about the RSA geometry.

The free-form Gemma-31B robustness check uses 12 imposed orders and reads the
last prompt token before native reasoning begins (`extract/geometry/natural_geom.py`;
compact statistics in `data/soundness_ci.md`). Non-structural mention and story
controls are produced by `extract/geometry/geom_storyprobe.py`; compact outputs
are in `data/table_summaries.json`.

### 3.2 Where context takes over

- **Extraction:** `extract/geometry/geom_possweep.py`
- **Cache:** `data/geometry/possweep_gemma-4-31B-it_all.npz`
- **Comparison:** the queried entity token versus successive downstream prompt
  positions, ending at the pre-generation state.
- **Interpretation:** pretrained organization can remain visible at the entity
  token while the integrated downstream state follows the imposed order. This is
  position-resolved representational evidence, not a claim that the entity-token
  representation itself is overwritten.

### 3.3 Hierarchies, closure, and matched topology checks

- **Hierarchy extraction:** `extract/geometry/geom_tree.py` and
  `extract/geometry/demo_tree.py`; six randomized assignments for the main
  hierarchy analyses. Depth RSA and sibling-versus-cousin distances are measured
  in the full RDM. PCA panels are illustrations.
- **Co-occurrence control:** the hierarchy is restated as individually shuffled
  edges so siblings are not co-listed. The control supports depth organization
  and a reduced but remaining branch-distance effect; it does not show a faithful
  low-dimensional drawing of every tree edge.
- **Wrap factorial:** `extract/behavior/wrap_2x2.py`; two scrambles per cell. It
  crosses surface form (numbered list versus adjacency statements) with whether
  the wrap relation is stated. Closure is graded, not a binary topology verdict.
- **Endpoint co-occurrence control:** `extract/behavior/wrap_cooccur_control.py`;
  three scrambles. Co-mention without a wrap relation leaves both models open;
  relation-only closure is stronger in Gemma than Qwen.
- **Matched persistent homology:** the cyclic and hierarchical conditions use the
  same arbitrary tokens, assignments, six query templates, layer, and readout at
  `N=7` and `N=15`, with 10 paired scrambles for Gemma-31B and Qwen-27B. Cyclic
  minus hierarchical normalized top-`H1` persistence is positive in all four
  cells (`+0.083` to `+0.140`); crossed-bootstrap intervals exclude zero and exact
  paired tests give `p <= 0.027`. Only Qwen-27B at `N=15` independently clears
  the stricter query-shuffle null, and longest-versus-second-longest persistence
  is not robust. Persistent homology is therefore comparative confirmation, not
  standalone proof of one dominant loop. Results are in
  `data/geometry/ph_matched/`.

### 3.4 Same entities under different specifications

- **Extraction:** `extract/shapes/shape_test_v4.py` and
  `extract/geometry/natural_sentend.py`.
- **Conditions:** the same weekdays under no rule, an imposed cycle, an imposed
  line, or an imposed hierarchy at the fixed `0.75 * n_layers` readout.
- **Sampling:** each imposed condition averages 12 neutral/diverse query types
  over 10 randomized token-to-position assignments. The no-rule baseline averages
  six query phrasings over six hop counts.
- **Scope:** the comparison shows that one token set can support distinct
  relational organizations. The 3-D PCA camera in the figure is illustrative;
  the reported RSA is computed in the full-dimensional RDM.

### 3.5 Activation patching

- **Direct intervention:** `extract/causal/causal_use.py`. At one layer, the full
  residual at the queried entity position is overwritten with the donor entity's
  residual from the same context. All matched entity subtokens are patched,
  right-aligned. Success means that the generated answer becomes the donor
  entity's successor under the current context.
- **Flagship sampling:** 20 imposed orders and all 42 ordered unequal weekday
  pairs per order for each flagship model. The fixed natural-order control is
  repeated six times for Gemma-31B and twice for Qwen-27B. The caches are
  `data/causal/causaluse_gemma-4-31B-it.json` and
  `data/causal/causaluse_Qwen3.5-27B.json`.
- **Controls:** a norm-matched random donor tests nonspecific perturbation; an
  adjacent-position injection tests whether the entity slot is necessary. The
  scale-ladder table uses the common two-order subset per model-concept cell and
  fixed normalized layer fractions.
- **Cross-order control:** `extract/causal/causal_cross_order.py` patches an entity
  between prompts specifying different orders. Twenty independent order pairs
  distinguish transfer of entity identity from transfer of the donor prompt's
  relation; results are in `data/causal/crossorder_gemma-4-31B-it_n20*.json`.
- **Reasoning checks:** Gemma scale comparisons use two orders per checkpoint.
  The Qwen-27B free-form patch uses three imposed orders; only visible final
  answers are scored, and truncation is reported rather than treated as an
  incorrect successor. Step-count patching uses two scrambles.
- **Scope:** these interventions causally localize context-dependent binding of
  entity identity, and separately of the requested step count, into the successor
  computation. They do **not** intervene on the RSA-defined manifold and do not
  establish that manifold as the causal carrier.

### 3.6 Behavioral validation

- **Direct hop sweep:** `extract/behavior/defladder_acc.py`; two imposed orders,
  hop counts `k=1,...,6`, compared with the natural order at the same hop counts.
- **Three regimes:** `extract/behavior/behavior_3regime.py`; direct, scripted, and
  free-form generation are scored separately. Qwen generations that exhaust the
  2048-token budget without a conclusion are reported as over-budget; the parser
  reads only the visible final response.
- **Scope:** these measurements distinguish a represented relational organization
  from the ability to traverse it under a particular prompting and generation
  regime. Cross-model RSA-behavior correlation is convergent validity, not a
  within-model causal effect.

### 3.7 Two-dimensional extension

- **Extraction:** `extract/grid/geom_qwerty.py`,
  `extract/grid/geom_qwerty_imposed.py`, and `extract/grid/geom_grid.py`.
- **Caches:** `data/grid/`.
- **Sampling and tests:** imposed QWERTY and arbitrary-token grids use six
  randomized assignments. The native-QWERTY permutation test compares the
  reported Manhattan-distance RSA against the matched Manhattan template.
- **Scope:** this appendix extends formation and override beyond one-dimensional
  cycles. Native grid structure is weaker and model-dependent, so it is not used
  as the paper's primary evidence.

## 4. Claim boundaries

The released evidence supports the paper's claims only at the following levels:

1. Context changes the measured downstream relational organization.
2. The organization tracks cycles, hierarchy depth and branch relationships,
   graded closure, and a two-dimensional grid under the tested specifications.
3. Entity and step-count patching causally localize inputs to the
   context-conditioned successor computation.
4. Clean prior suppression and the full patching crossover appear more reliably
   in the larger tested checkpoints, but the trend is family-dependent and
   nonmonotonic.

The evidence does not determine whether the downstream organization is built
anew or produced by context-dependent reconfiguration of stored structure. It
does not establish the RSA manifold as a causal mediator, a universal scaling
law, faithful embedding of every tree edge, or geometry during native generated
reasoning.

## 5. Reproduction and retained artifacts

- `figures/make_all.py` regenerates the paper figures from the checked-in caches
  without model weights.
- `ARTIFACT_MAP.md` is the exact figure/table-to-code-and-data index.
- `data/README.md` explains which figure-ready caches and compact table summaries
  are retained. Large raw activation tensors used only for table analyses are
  omitted when compact per-scramble or aggregate results preserve the reported
  measurement; the corresponding GPU extraction scripts remain in `extract/`.
- A displayed panel may use a representative scramble when its caption says so.
  Reported means, confidence intervals, and tests use the aggregate artifact and
  the statistical unit specified above, not the displayed example alone.
