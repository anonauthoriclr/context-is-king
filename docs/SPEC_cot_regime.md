# SPEC — CoT regime: does reasoning traverse the imposed ring? (+ flag-invariance of the geometry)

> **Status: DRAFT @ Gate 1 (methods agreement). Nothing runs until operator sign-off.**
> Author: orchestrator, 2026-06-18. Motivated by F44 (one-shot can't traverse the imposed metric ring its
> representation composed). Built on docs/03 **Standard #10**: a different regime (reasoning on) may route through
> different circuits — re-capture geometry under the CoT prompt and verify before pairing CoT-geometry with CoT-behavior.

## 1. Question

F44: the representation composes a clean imposed *metric* ring (F43), but **one-shot behavior only reads the
adjacency** (k=1), collapsing at k≥2 — even at 31B with a perfect natural ceiling. Natural multi-step is ~1.0 one-shot,
so the live question is purely imposed k≥2:

**(A behavior) Does step-by-step reasoning let the model traverse the imposed ring it already built** (rescue imposed
k≥2)?
**(B geometry, Standard #10) Does enabling reasoning *change the imposed geometry*** at our probe layer/position, or is
the `enable_thinking` flag geometry-neutral? Only after answering B can CoT-behavior be validly paired with a geometry.

## 2. Hypotheses (pre-registered)

- **H-rescue.** Imposed CoT k≥2 ≫ imposed one-shot k≥2 (reasoning traverses the ring). F34 hinted CoT→~100% on capable
  models; here we test it k-resolved and paired with geometry. (If CoT does NOT rescue, that's a strong negative — the
  imposed structure is not usable even with reasoning.)
- **H-flaginvariance.** At the **pre-generation** position (end of prompt, before any reasoning), enabling reasoning does
  NOT materially change the imposed concept-geometry (same redefinition context) — predict high centroid-cosine vs the
  F43 one-shot centroids + matched RSA. If it DOES diverge, one-shot and CoT have different geometries and must not share.
- **H-ring-use (exploratory, the deep one).** If CoT rescues behavior, does ring quality *predict* rescue across cells
  (ring is the substrate reasoning reads), or is rescue independent of the ring (reasoning re-derives from the text)?

## 3. Design — two parts, same prompts/orders as F43/F44 (so everything pairs)

**Part B first (geometry flag-invariance) — it gates the pairing:**
- Re-capture the entity-layout residual under the **CoT prompt** (`enable_thinking=True` where supported, else the
  "think step by step, end with `FINAL: <name>`" instruction), at the **same layer L=0.75·depth and the same
  pre-generation position** as F43 (end of prompt, before reasoning tokens). Group by INPUT entity, marginalize k/template.
- Compare to the **cached F43 one-shot centroids** (`results/geometry/cache/`): (a) per-entity centroid cosine
  (cot-prompt vs one-shot-prompt, same entity), (b) imposed-RSA under each, ΔRSA. Also recompute the ring/crossings.
- **Verdict:** centroid-cosine > ~0.9 AND |ΔRSA| < ~0.1 ⇒ flag geometry-neutral at pre-gen → CoT behavior may be paired
  with the (shared) pre-gen geometry. Otherwise ⇒ divergent; pair CoT behavior only with CoT-captured geometry, and
  report the divergence as a finding in itself (the flag reroutes circuits).
- **CAVEAT / deeper follow-up (ii):** the state that actually *drives* the CoT answer is **post-reasoning** (the token
  before `FINAL:`), not pre-generation. Pre-gen flag-invariance tests whether the flag perturbs the *input* geometry;
  a full account also needs the post-reasoning entity-layout capture (per-(e,k) reasoning differs, position varies →
  group by input entity at the last pre-FINAL token, marginalize k). Mark as Part B-ii, optional second pass.

**Part A (CoT behavior):**
- Generate with reasoning enabled, greedy, **`max_new_tokens` ~2048** (F34 lesson: short budgets truncated CoT → false
  24%; report truncation rate per cell). Parse the concluded answer: **`FINAL:` cue first**, then last-entity-after-cue
  heuristic, **LLM-judge only for the ambiguous residual** (reasoning names many entities → first-entity is wrong).
- Score imposed **k-curve** + natural-revert + NA, exactly as F44, so the imposed CoT k-curve overlays the F44 one-shot
  k-curve cell-for-cell.

## 4. Metrics & decision rules (pre-committed)

- **H-rescue supported** iff for the high-ceiling cells (Gemma-31B/12B, Qwen-27B) imposed CoT k6 > 0.7 where one-shot
  k6 < 0.2 (a clear rescue), with the natural-CoT control near ceiling (sanity).
- **H-flaginvariance:** report centroid-cosine + ΔRSA per cell; classify geometry-neutral vs divergent by §3 thresholds.
- **H-ring-use (exploratory):** Spearman(imposed-RSA[F43], CoT-rescue = CoT_k≥2 − one-shot_k≥2) across cells; positive ⇒
  ring quality predicts how much reasoning can traverse. Reported as suggestive, not confirmatory.

## 5. Controls (binding)

1. **Natural CoT** (same questions, no redefinition) — should stay ~ceiling; confirms CoT parsing/budget aren't lossy.
2. **Token-budget adequacy** — report truncation rate; if a cell truncates non-trivially, raise budget and re-run it
   (never score a truncated CoT as wrong).
3. **Parse validation** — smoke a handful of raw CoT generations + parsed FINALs before the full run; report NA-rate;
   escalate to LLM-judge only if the rule-based parser is unreliable on the smoke.
4. **Same orders/prompts as F43/F44** (shared seed) — enforced; verify the order set matches before pairing.

## 6. Conditions (reduced grid — CoT is the cost driver at ~2048 tok)

- Focus the **clean-decoupling, high-ceiling cells**: Gemma-31B-it, Gemma-12B-it, Qwen-27B × {list, adj} × days,
  1 scramble first (acc is low-variance). Roll large→small / add scrambles only if rescue is borderline.
- Part B geometry capture on the same 3 models (cheap, forward-only). Part B-ii deferred unless B shows divergence or
  H-ring-use needs it.
- No silent caps: log which cells got CoT, truncation + NA rates, budget used.

## 7. Confounds & limits (named)

- **Regime mismatch is the whole point** — do NOT pair CoT behavior with the F43 one-shot geometry until Part B clears it.
- **Pre-gen vs post-reasoning geometry** — pre-gen flag-invariance ≠ the state driving the answer; B-ii is the honest
  completion. State which geometry any pairing uses.
- **CoT rescue ≠ ring use** — reasoning could re-derive the walk from the textual definition independent of the ring;
  H-ring-use is suggestive only. A causal test (ablate/steer the ring, see if CoT degrades) is a separate interventions
  experiment.
- **Judge dependence** — keep parsing rule-based where possible; if judged, log judge + spot-check.
- **`enable_thinking` availability differs by family** (Qwen native flag; Gemma/Llama via instruction) — that itself is
  a regime difference; report per-family which mechanism was used and don't pool the *mechanism* silently.

## 8. Reproducibility / outputs

- `defladder_cot.py` (sibling of `defladder_acc.py`): CoT prompt builder; Part B capture (pre-gen residual, compare to
  cached F43 centroids → centroid-cosine + ΔRSA + crossings); Part A generate (2048 tok) + FINAL/judge parse + k-curve.
- `results/behavior/cot_<tag>_<concept>.json` (k-curve, natural-revert, NA, truncation) + `results/geometry/
  flaginvariance_<tag>_<concept>.json` (cosine, ΔRSA per entity).
- Figures: imposed k-curve **one-shot vs CoT** overlay per model; flag-invariance (cosine/ΔRSA) bar; RSA-vs-CoT-rescue
  scatter. Findings entry → Gate 2 before doctrine.
