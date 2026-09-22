# SPEC — Causal USE test: is the context-built map *used* to compute the answer? (Gate-1 SIGNED OFF)

> Status: SIGNED OFF @ Gate-1 (methods agreement, 2026-06-23 — operator: "sounds right plan. im with you"). Purpose:
> defeat the tautology objection ("geometry is a passive probe-readout") by showing the **context-built map causally
> mediates the one-shot answer** — the interventions rung. This is CORE science (not application).
>
> **LOCKED DESIGN DECISIONS (the 3 open choices, resolved):**
> 1. **Patch site = entity-token** (inject identity; map stays intact). NOT pre-gen-slot.
> 2. **Patch op = OVERWRITE** with donor residual. NOT difference-vector steer (deferred to optional secondary).
> 3. **Multi-token entities = patch ALL query entity sub-tokens**, right-aligned (m=min(len_src,len_dst) last tokens),
>    since the last sub-token aggregates entity identity. Per-pair span lengths logged.
> **Patch occurrence = the QUERY entity (last occurrence in the prompt), not the context-list occurrence.**
> Implemented in `causal_use.py`.

## 1. Question / hypothesis
**H-use:** the one-shot answer to "k steps after E" is computed by applying the **context-built order-map** to the
entity representation E. If we *patch* the entity representation E→E′ (holding the prompt's order context fixed), the
answer should become **succ_context(E′)** — the imposed successor of the *patched* entity. And critically, the **same
patch yields different answers under different context-built maps** (imposed vs natural), proving the *context-built* map
is the causal transform from entity → answer.

## 2. Intervention (activation patching, entity-substitution)
- Prompt: `<order context O>` + "1 steps after **E_src**?" (k=1, the regime where one-shot works; answer-only).
- Donor: a clean run of the *same* prompt with **E_dst** in the entity slot → cache its residual at the **entity-token
  position(s)**, every layer.
- PATCH: in the E_src run, at layer ℓ and the **entity-token position**, **overwrite** the residual with the donor
  (E_dst) residual; leave the order context + query structure untouched; then generate the one-shot answer.
- Read: parse the generated answer.

## 3. Primary metric — patch-success (answer follows the PATCHED entity through the map)
For (E_src→E_dst) pairs: **answer == succ_O(E_dst)** (the imposed successor of the *donor*). Report rate vs:
- baseline (no patch) answer == succ_O(E_src),
- chance (1/N).
**Layer sweep ℓ** → the causal locus (expect the flip to take hold at/after the map layer ~0.75 depth).

## 4. The decisive control — imposed-vs-natural double dissociation
Run the identical entity patch E_src→E_dst under two contexts:
- **Imposed** order O → predict answer flips to **succ_O(E_dst)**.
- **Natural** order (no "Order:" redefinition) → predict answer flips to **succ_natural(E_dst)**.
**Same entity-patch, different map (set by context) → different answer.** This is the clean proof that the
*context-built* map (not the token, not a fixed circuit) is the causal entity→answer transform. (If the answer ignored
context and followed natural succ in both, the map isn't what's used.)

## 5. Additional controls
- **Specificity / random-donor:** patch with a random vector (matched norm) or an unrelated-token residual → answer
  should NOT become any specific succ (rules out "any perturbation flips it").
- **Position-token control:** patch at a non-entity token (e.g. the "steps" token) → no systematic flip (locus is the
  entity slot, not anywhere).
- **No-context-patch baseline:** confirm the *unpatched* one-shot answer is succ_O(E_src) (the thing we're flipping).
- (Optional, stronger) **directional steer:** add the map's (pos p → p+1) difference vector at the pre-gen slot → answer
  shifts by one position (tests the metric axis is causally used). Harder; secondary.

## 6. Conditions
- Models: **Gemma-31B-it** (clean map) primary; **Qwen-27B** replication. days (N=7).
- Order O: ≥2 scrambles. Pairs: all E_src≠E_dst (7×6=42) × scrambles. k=1.
- Layer sweep: a grid around 0.5–0.9 depth (e.g. every ~4 layers) + the map layer.
- Save raw (Standard #11): per (pair, layer, context) the generated answer + success flags + the donor/patched ids.

## 7. Confounds named
- **Token vs map:** the prompt still literally says E_src; if the answer follows succ_O(E_dst) it used the *patched
  representation*, not the surface token → that IS the point. The position-token control + random-donor rule out
  trivial perturbation effects.
- **Trivial state replacement:** we patch ONLY the entity-token position (not the whole pre-gen state), so we're
  injecting entity *identity*, not the answer — the model must still apply the (untouched) context map.
- **Ceiling:** k=1 baseline is ~1.0, which is what we *want* here (clean baseline to flip); this is a causal-flip
  metric, not an accuracy-vs-quality correlation (so no saturation problem, unlike the correlational USE variant).
- This is correlation-free: patching IS the intervention; a successful flip is causal by construction.

## 8. Decision rule (pre-committed)
- **H-use supported** iff at the causal layer, patch-success (answer→succ_O(E_dst)) ≫ chance AND ≫ random-donor control,
  AND the **imposed-vs-natural double dissociation** holds (same patch → succ_O vs succ_natural). 
- Layer profile reported (where the map becomes causally read).

## 9. Outputs / repro
- `causal_use.py` (hooked generation: cache donor residuals; forward-with-patch via a layer hook that overwrites the
  entity-token position; generate; parse). Save raw answers + flags + a layer-sweep curve + the imposed/natural bars.
- Figures: patch-success vs layer (with random-donor baseline); imposed-vs-natural double-dissociation bar.
- Finding → Gate-2 before doctrine.

## Open design choices to confirm before running
1. **Entity-token patch vs pre-gen-slot patch** — I recommend entity-token (cleaner: injects identity, map stays intact;
   pre-gen-slot risks trivial answer-replacement). OK?
2. **Patch = overwrite vs add-donor-minus-src (activation-difference steer)** — overwrite is simplest/cleanest; the
   difference-vector version is gentler but noisier. Start with overwrite?
3. **Multi-token entities** (e.g. "Wednesday" may be >1 token) — patch all entity sub-tokens, or just the last? I'd patch
   all sub-tokens of the entity span. OK?
