# SPEC — Accuracy phase: does the imposed GEOMETRY track BEHAVIOR? (geometry↔function link)

> **Status: DRAFT @ Gate 1 (methods agreement). Nothing runs until operator sign-off.**
> Author: orchestrator, 2026-06-18. Phase 2 of the def-ladder × scale program (Phase 1 geometry = F43).
> Pairs cell-for-cell with F43 so every accuracy number sits next to its imposed-RSA / crossings.

## 1. Question

F43 measured the *geometry* (the imposed cyclic ring) across model × scale × concept × presentation-format.
The open question: **does that reorganized geometry come with correct behavior, or is it geometry-without-function?**
The sharp case F43 surfaced: **small Qwen forms a clean ring under *adjacency* (4B/9B days adj RSA 0.53/0.56)** — does it
also *answer* the imposed-walk questions correctly, or is the ring present but behaviorally inert (the F32 base-model
decoupling, now tested *within* instruct models)?

## 2. Hypotheses (pre-registered)

- **H-link.** Across cells, one-shot imposed accuracy tracks imposed geometry (Spearman(RSA, acc) > 0). Scale that
  cleans the ring also makes the walk computable.
- **H-decouple (the interesting one).** There exist cells with a clean ring (cross≈0 / RSA>0.5) but LOW one-shot
  accuracy — geometry present/ahead of function. Predicted at **small Qwen under adjacency** and as the residual at
  mid-scale. This would replicate F32's geometry-without-function *inside* the instruct family.
- **H-format.** Adjacency (better geometry at small scale, F43-R1) → also better accuracy? Or is accuracy
  format-insensitive (the model computes the walk from either presentation)? Either result is informative.
- **H-kcurve (behavioral metric-vs-lookup).** k=1 (immediate successor) = adjacency *lookup*; k≥2 = *composing* the
  walk. A model that succeeds at k=1 but collapses at k≥2 is a behavioral lookup; graded success across k is
  behavioral metric. This is the behavioral twin of the geometric metric-vs-lookup question.

## 3. Generation / data

- **Reuse F43's exact prompts and scrambled orders** (same `rng` seed → same `orders`), so each accuracy cell pairs
  with the F43 geometry cell (same model, concept, mode, scramble). The geometry residual is already cached; this phase
  adds the generated answer for the same prompts.
- **ONE-SHOT ONLY (regime-matched to F43 geometry — Standard #10).** F43 geometry was captured in the one-shot regime
  (`enable_thinking=False`, "answer-only"); its valid behavioral partner is one-shot behavior, generated from that same
  state. **CoT is explicitly OUT of scope here** — it is a different regime whose geometry must be re-captured and
  flag-invariance-checked before pairing; that is a separate spec (`SPEC_cot_regime.md`, deferred), not this run.
- **Two one-shot measures (both regime-matched, free-generation):**
  1. **imposed one-shot (PRIMARY, geometry-paired).** Generate from the *exact* F43 prompt — no prompt change. The F43
     residual is the precursor of this answer (same forward pass). Paired cell-for-cell with RSA.
  2. **natural one-shot (baseline ceiling).** Same prompt, natural order, no redefinition — can the model one-shot a
     walk at all (separates non-override from non-competence). Same regime ⇒ valid comparison.
- **Free-generation, not truncation** (operator point): do NOT cap at ~12 tokens — let it generate and robustly parse
  the answer (avoids truncation artifacts). Under the one-shot prompt the model answers immediately anyway, so this is
  purely a parsing-robustness upgrade, still one-shot.
- **Robust parse, deterministic-first:** scan for the entity name from the concept set (case-insensitive,
  word-boundary); for the rare multi-entity one-shot output, take the post-conclusion-cue / final mention. **LLM-judge
  only as fallback** for the ambiguous residual (rule-based final-answer extractor first; escalate to a small local
  judge only if a smoke sample shows it's unreliable). Report NA-rate per cell — never silently score NA as wrong.

## 4. Metrics & decision rule (pre-committed)

- **Primary: one-shot imposed accuracy per cell** = fraction of answers == imposed-order gold (the k-steps-after under
  the imposed order). Reported per (model, concept, mode), and as the **k-curve** (accuracy by k=1..6).
- **Natural-revert rate** = fraction == natural-order gold (the competitor) — behavioral analog of F43's
  natural-retention; ties function to the suppression story.
- **The link:** Spearman(imposed-RSA[F43 cell], imposed-acc[this cell]) across all cells; plus a 2-D scatter
  (RSA vs acc) with cells labeled, so **decoupled cells (high RSA / low acc) are visible.**
- **Decision rules (pre-commit):**
  - **H-link supported** iff Spearman(imposed-RSA, imposed-one-shot-acc) > 0 with bootstrap CI_low > 0 over cells.
  - **H-decouple supported** iff ≥1 cell has (cross≈0 OR RSA>0.5) AND imposed-one-shot-acc < 0.5·(natural one-shot
    ceiling) — clean ring, model CAN one-shot a walk (natural ceiling decent) but does NOT one-shot the imposed one →
    geometry present, imposed *automaticity* absent (F32-style, inside instruct). Whether reasoning would rescue it
    (incompetence vs automaticity-gap) is the deferred CoT-regime experiment's question, NOT resolvable here.
  - **H-kcurve:** classify each cell behaviorally lookup (k=1 ≫ k≥2, e.g. k1>0.8 & mean(k≥2)<0.3) vs metric (graded) —
    the behavioral twin of metric-vs-lookup, one-shot regime.

## 5. Controls (binding)

1. **Natural-task positive control (per model, the ceiling).** Same `k-steps-after` questions with NO redefinition
   (natural order). Gives baseline walk-competence — imposed accuracy must be read *relative* to it (a model that
   can't do the natural walk can't be expected to do the imposed one; that's incompetence, not non-override).
2. **No silent NA.** Report NA-rate per cell; high NA invalidates that cell's accuracy.
3. **≥2 scrambles** (accuracy is lower-variance than RSA; nscr=2 for a first read, bump to match geometry if a cell is
   borderline). Same orders as F43.
4. **k-stratified** reporting always (never a single pooled accuracy — k=1 lookup would mask k≥2 collapse).

## 6. Conditions (roll LARGE → SMALL)

- Instruct ladder × {days, months} × {list, adj}, **descending size** (Gemma-31B-it, Qwen-27B → … → E2B / Qwen-4B /
  Llama-8B-it) so the expensive, high-signal large models land first and we can stop early if the link is already clear.
- Per-model also run the **natural one-shot control** (1 pass, no scramble needed).
- Compute note (no silent caps): per model, one-shot is ~N·6·6 × 2 modes × 2 scrambles (days ≈1k, months ≈1.7k).
  Free-generation but the one-shot prompt answers fast, so a modest `max_new_tokens` cap (e.g. 64) covers it with margin
  — log the truncation rate; if any cell truncates non-trivially, raise the cap for that cell. Roll large→small.

## 7. Confounds & limits (named)

- **Parse noise** → robust first-entity parser + NA-rate reported; spot-check raw generations for one cell.
- **One-shot ≠ capability** (F34) → we explicitly claim *automaticity*; the CoT/capability ceiling is the natural-task
  control's job, not this metric's.
- **Format on accuracy vs on geometry** may differ (a model could compute the walk from a list it doesn't lay out as a
  ring) — that dissociation is exactly H-format, reported, not assumed.
- **Pairing assumes same prompts/orders as F43** — enforced by reusing the seed; verify the order set matches before scoring.
- Static-vs-causal: a positive RSA↔acc link is **correlational** ("geometry tracks function"), not causal — the
  patch/steer interventions-rung experiment remains separate.

## 8. Reproducibility / outputs

- New script `defladder_acc.py` (sibling of `defladder.py`): same ENT/TPLS/orders/`make_def`; replaces the forward-only
  `capture` with greedy generate + parse; writes `results/behavior/acc_<tag>_<concept>.json` (per mode, per scramble,
  per-k accuracy + natural-revert + NA-rate) and the natural-task ceiling.
- Combine script → `FIG_geom_vs_acc_<concept>.png`: (a) RSA-vs-accuracy scatter (cells labeled, decoupled cells
  highlighted); (b) k-curves per model×mode; (c) imposed-acc vs scale overlaid on the F43 RSA-vs-scale panel.
- Orchestrator `sweep_acc.sh <concept>` mirroring `sweep_geom.sh` (large→small order, file-based skip).
- Findings entry on completion → Gate 2 before doctrine.
