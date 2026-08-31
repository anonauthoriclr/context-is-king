# PROVENANCE & ACCOUNTABILITY — what actually ran, and how it affects claims (2026-06-22)

Purpose: after a sprawling session (+ one wrong claim by the orchestrator about the thinking flag), record the
**exact regime each experiment used** and **which claims are solid vs mislabeled vs never-measured.** Verified facts
only; assumptions flagged.

## A. Verified per-model thinking behavior (from `per_model_thinking.py`, 2026-06-22)
Instrument: chat-template tail + open-ended generation (no format instruction), default mode.

| model | `enable_thinking` flag | default template | OPEN-ENDED natural behavior |
|---|---|---|---|
| Gemma-4-31B-it | accepted & **functional** | opens `<|channel>thought` → **thinks by default** | reasons ~92 tok, step-walks the order, **correct (Tuesday)** |
| Qwen3.5-27B | accepted & **functional** | opens `<think>` → **thinks by default** | reasons **>512 tok (didn't finish)** — long native reasoning |
| Llama-3.1-8B-it | accepted but **no-op** | standard `assistant\n\n`, **no thinking channel** | still reasons in plain output ~83 tok, **correct (Tuesday)** |

**Key takeaways:** (1) "flag accepted (no TypeError)" is UNINFORMATIVE — all 3 accept it; template-tail + behavior are
the real tells. (2) Gemma & Qwen **think by default**; Llama has no thinking channel but **reasons in plain output**.
(3) **All three, open-ended, naturally reason and answer correctly** — the snap "automata" behavior is IMPOSED by us,
not their default.

## B. The shared code regime (`render()` in defladder*/cot*/k1*/toy_data/wrap_2x2)
- Automata/one-shot reads: prompt + **"Answer with ONLY the name"**, `enable_thinking=False`, residual at **last prompt
  token (pre-generation)**. `enable_thinking=False` was accepted AND functional → for Gemma/Qwen **native thinking was
  suppressed** (verified: one-shot outputs were bare 2-token answers, F44 check).
- "CoT" reads: prompt + **"Work through it ONE STEP AT A TIME … FINAL:"**, `enable_thinking=False`, residuals along the
  generated tokens. → **thinking-suppressed + dictated enumeration.** NOT native thinking, NOT natural reasoning.

## C. Per-experiment provenance & claim status

| finding | exact regime | what it actually measured | status |
|---|---|---|---|
| **F43** def-ladder rings | answer-only, thinking-OFF, pre-gen | the **forced-snap / automatic** pre-gen geometry | SOLID (correctly the automatic regime) |
| **F44** one-shot accuracy (k≥2 collapse) | answer-only, thinking-OFF, generate | accuracy **when forced to snap-answer** | SOLID empirically (bare answers verified) — but re-word: "forced-snap fails k≥2," NOT "the model can't" (natural reasoning succeeds, §A) |
| **F45** "CoT regime" geometry / regime-dependence | **step-by-step DICTATED, thinking-OFF**, pre-gen + along-gen | geometry under a *dictated-enumeration* prompt, native thinking suppressed | **MISLABELED**: this is "snap-prompt vs dictated-enumeration-prompt, both thinking-off" — NOT "automata vs native-CoT". Regime-difference is real *between these two prompts*; do not call it native thinking. |
| **ring-formation along CoT** | dictated enumeration, thinking-OFF | RSA along a *prescribed* walk | confounded by our instruction (we dictated the steps); inconclusive anyway |
| **toy regime-lock** (1.00 in-dist, chance cross) | answer-only-pregen vs reason-instruction-pregen, thinking-OFF | probe transfer between **two prompt conditions** (thinking-off) | SOLID as "probe doesn't transfer across these two prompt conditions"; **mislabel risk**: it's not automata-vs-native-thinking |
| **2×2 wrap (line vs cycle)** | answer-only, thinking-OFF, pre-gen | definition-content effect on the **automatic** pre-gen geometry | SOLID (within the automatic regime) |
| **causal USE — snap** (`causal_use.py`, 2026-06-23) | entity-substitution patch E_src→E_dst, answer-only, k=1, Gemma-31B, 2 scr, 42 pairs | does the context-defined relation **causally mediate** the answer | SOLID. Double dissociation @L27 (clean locus): imposed→succ_imp(dst)=1.00, natural→succ_nat(dst)=1.00; other-map=chance(0.14); controls clean (random 0.14, position 0.05–0.07). Flip live L6–L27, **answer locked by L42+**. SCOPE: mediation+localization, NOT manifold-axis causality (linear-steer-on-ring intractable). |
| **causal USE — CoT/open-ended** (`causal_cot.py`, 2026-06-23) | same patch, **thinking ON, open-ended 400-tok**, Gemma-31B, 1 scr, 42 pairs | does the patch **survive into natural reasoning** (Standard #10 regime-match) | SOLID. Final answer→succ_O(dst)=1.00 (imposed, layer-robust L6–27), dissociation holds; **patch drives the trace** (mentions dst>src 86%; model recites imposed order + wraps about the PATCHED entity though prompt says src). Capability contrast: E2B patch DIES in CoT (surface reasserts, 0.00). |
| **E4 renderer-generalization: imposed TREE** (`geom_tree.py`, 2026-06-24) | declarative binary tree on **arbitrary tokens**, entity-layout pre-gen, 6 scr, Gemma-31B/E2B/Qwen-27B; 3 query sets (rel/anc/neutral) | does the renderer draw a **non-cyclic** (hierarchy) structure | SOLID. Renders **depth** (RSA +0.53 Gemma / +0.57 Qwen; parent↔child farthest) + **branch** (leaf sib<cous, true-pairing 6/6, p≈0.0014). **INTRINSIC**: depth-RSA rises rel→anc→neutral 0.53→0.61→**0.92** / 0.57→0.69→**0.89** (strongest with NO structural query). **Triple-controlled**: line-free leaf test, neutral queries, within-token-pair flip 12/12 (Δ≈−0.9, not token semantics). **Capability-gated** (E2B collapses). METRIC LESSON: hierarchy embeds as DEPTH-bands, NOT tree-path (path-RSA −0.26 was a false negative). SCOPE: salient low-dim projection (depth+branch); full 2-D tree compressed. **DEPTH-3 replication** (15 nodes, 4 levels, neutral): depth-RSA +0.82/+0.71, nested branch among 8 leaves monotone sib<cousin<2nd-cousin both families → renders full nesting. |
| **E1 arbitrary-token 2×2** (`wrap_2x2.py --concept arb7/arb12`, 2026-06-24) | structure-free tokens, no pretrained ring, Gemma-31B | is topology-follows-spec re-selecting a pretrained ring? | SOLID. NO: wrap drives closure on arb tokens (list dRSA −0.12/clos 2.7=line; adj +0.31/1.1=cycle; arb12 same). Tautology objection CLOSED. |
| **E4 depth-4 scaling** (`geom_tree.py --depth 4 --qset neutral`, 2026-06-24) | 31-node 5-level binary tree, arbitrary tokens, neutral queries, 6 scr, Gemma-31B + Qwen-27B | does depth/branch hold at the deepest level | SOLID. depth-RSA **+0.729 Gemma / +0.630 Qwen** (all 6 scr 0.53–0.76); leaf genealogy gradient **monotone to the top-level split, both families** (sib<cousin<2nd<3rd). The single-scramble "fold-back" was noise (**RETRACTED**). Qwen note: gradient is steep at the sibling step then flat beyond (cousin≈2nd≈3rd) — Qwen branch is sibling-dominated. analyzer = `analyze_genealogy.py` (depth-general; old `analyze_tree.py` is N=7-hardcoded). |
| **E4 co-occurrence control** (`geom_tree.py --sepedges --depth 3 --qset neutral`, 2026-06-24) | one edge per sentence, siblings shuffled apart (NEVER co-listed), 6 scr, Gemma-31B + Qwen-27B | is sibling closeness rendered or just text co-occurrence | SOLID. **Depth-RSA unchanged** (0.825→0.823 Gemma, 0.705→0.718 Qwen) → depth has ZERO co-occurrence dependence. **Siblings loosen but stay tightest & monotone** (Gemma 0.141→0.216, Qwen 0.146→0.364); sibling–cousin gap −57%/−34% → branch is part surface-driven, part rendered; **report controlled numbers**. branch-RSA stays + every scr (Gemma +0.25, Qwen +0.36). **NO low-dim branch geometry**: in 3 PCs siblings only moderately nearer (−27%/−38%), top-level split near-flat (operator confirmed on demos) → branch = METRIC, depth = the figure. |

## D. What's solid / mislabeled / never-measured
- **SOLID:** the *automatic / forced-snap* regime results (F43 rings, F44 snap-accuracy, 2×2 topology) — all at pre-gen
  under a verified thinking-off snap prompt. The probe regime-lock is solid *as transfer between our two prompt conditions.*
- **MISLABELED (the F45 era, still true):** the *old* "CoT geometry" (defladder_cot / cot_extract, F45) was **thinking-OFF
  + dictated enumeration** — re-label "dictated-reasoning prompt (thinking-off)." Distinct from the NEW runs below.
- **NEVER MEASURED → PARTIALLY CLOSED [2026-06-23 UPDATE].** Two newer runs measure the natural/native-thinking *prompt*:
  (i) **`natural_geom.py`** reads geometry under the **default native-thinking scaffold** (verified: render() passes NO
  `enable_thinking=False`), at the **pre-thinking token** (the thinking-opener, before any reasoning is generated) — 12
  scrambles, imposed RSA **0.639±0.048**, dRSA −0.23, all lines. So natural-PROMPT *pre-thinking* geometry IS now measured.
  (ii) **`causal_cot.py`** runs the causal patch under **thinking-ON open-ended generation** (verified reasoning traces) —
  genuinely native reasoning, NOT dictated. **STILL never measured:** entity-layout GEOMETRY read *during/after* native
  thinking (mid-reasoning residual). The draft (§3/§4/§6) refers to these NEW runs and is consistent with this update; the
  pre-2026-06-23 "never measured / thinking-off" wording above was stale for them.

## F. CONSOLIDATED THREAD CLAIM — the verified spine

**[2026-06-25 FRAMING RECALIBRATION (Gate-3, operator-approved). Spine content unchanged; emphasis moved.]** Two adversarial
critics (whole-paper + novelty gatekeeper) converged: "renderer, NOT store" as a *title/verdict* over-reaches — §6 itself
concedes capability-gating can't separate a computed map from a context-gated readout of a store, and the "store" it beats
is a strawman. **DECISION:** lead with **topology-follows-specification** (the one un-scooped result); keep "render/rendering"
as the *mechanistic* frame; demote the anti-realist "renderer not store" to an **interpretation argued in §9** (parsimony:
construction-on-demand over arbitrary tokens), not a proven dichotomy. [SUPERSEDED by the 2026-06-25b block below: final
title is "Context Is King, Scale Is the Gate: How In-Context Specification Sets the Geometry of Concepts" and "render" was
fully retired.] Draft retitled + abstract/§1/§9 rewritten; realist foil (Othello/Li–Nanda/Engels) added
to §2. NEW EVIDENCE folded in (both CPU, saved data, agent-computed + verified):
- **Dominance head-to-head** (`multiscr_*.json` + `naturalgeom_*.npz`): in the SAME probed state, Gemma-31B days imposed-RSA
  vs residual natural-RSA = 0.639 vs 0.059 (gap **+0.58**, natural prompt) / 0.870 vs −0.033 (gap **+0.90**, snap); months
  snap 0.811 vs 0.049. Capability-gated: weak Qwen-9B days gap **−0.33** (natural ring RETAINED) → "dominates on conflict"
  is a capable-model claim. Quantifies the previously-adjectival "dominates."
- **Predictive co-occurrence** (`k1summary_*_list.json`, 8 models/3 families): imposed-RSA rank-predicts one-shot k=1
  adjacency accuracy, Spearman **ρ=0.83, p=0.011** — renderers read the map correctly, non-renderers (Llama-8B RSA≈0, acc
  0.26) fail. CAVEAT: ceiling-compressed (5 capable models ~0.9–1.0), family-confounded, within-family weak/inverted →
  report as capability-gated co-occurrence, NOT a tight law.

**[2026-06-25b — TITLE + "RENDER" RETIRED (operator decision after same-reviewer head-to-head vs the "Latent Undertow"
paper).]** Same reviewer graded both: Latent Undertow 7 (focus + one quantified deployment number + finished/labeled gaps);
this paper 6 as-framed but novelty 8, "deficit is the fixable kind → 7–8 after refocus+figures." Operator elevated
**capability-gating to a co-equal PILLAR** (not demoted): paper is now **two pillars — (1) construction+topology,
(2) capability-gated computation / "scale is the gate"** with the methodological consequence ("small-model interp does not
extrapolate up") as the headline so-what; the 8-model within-family scale ladders (Gemma E2B 0.60→E4B 0.72→31B 0.87 days;
Qwen 4B 0.46→9B 0.61→27B 0.71 months) disentangle scale from family (answers the critic). **"Renderer" framing fully
RETIRED** (operator unsure of it) — title now **"Context Is King, Scale Is the Gate: How In-Context Specification Sets the
Geometry of Concepts"**; every render/renderer/rendering purged from draft → construct/build/compute on demand; §9 keeps the
computed-not-retrieved interpretation without the renderer word. Draft fully reworked (title/abstract/§1/§4–9), FIGURES +
draft-header synced.

### Pre-recalibration spine (2026-06-22, content still verified):
**In sufficiently capable models, the pre-reasoning (pre-generation) representation is a CONTEXT-CONSTRUCTED MAP of the
imposed relational structure** — robust across **regimes** (snap / instructed / natural), **orders** (12 scrambles),
and **concepts** (days/months) — and its **TOPOLOGY is dictated by how the context specifies the relations** (numbered
list → LINE; adjacency-with-wrap → CYCLE; 2×2 + direct closure test, cross-model). Forming this map for a *conflicting*
order is **CAPABILITY-GATED**: small/weak models fail (Llama-8B natural RSA 0.08; small-Qwen days-list ~0). **Reading the
map one-shot recovers only local/adjacency structure (1-hop); multi-step traversal requires reasoning** (k≥2: snap fails,
reasoning succeeds). **The context-defined relation CAUSALLY MEDIATES the answer** (entity-substitution patching; double
dissociation imposed-vs-natural; **regime-matched across snap + open-ended CoT**; capability-gated — dies in weak model) —
*scoped to mediation/localization, NOT manifold-axis causality.* Anti-realist reading: the cyclic/linear geometry is a
*readout of context-specified relations*, not an intrinsic world-model.

EVIDENCE: F43 (multi-scramble, cross-family, cross-concept); **2×2 + direct wrap-pair closure** (cross-model, list→line/
adj→cycle, content not format); **12-order pre-thinking robustness** (Gemma-31B, RSA 0.64±0.05, all lines, dRSA −0.23);
**3-regime check** (automata/instruct/natural all LINE under list, same order); **F44 + natural behavior** (snap=1-hop
verified bare answers, reasoning=multi-hop, Gemma-31B natural acc 1.00 / Llama-8B 0.57); F42 anisotropy null; F29 crossover.

RETIRED CLAIMS (with reason — do not ship):
- "days = rigid outlier / scale+instruct-gating as a general law" → **Qwen-specific** (F39, cross-family sweep).
- "automata representation is cleaner/superior to reasoning" → **both capable regimes hold the structure** (§A + 3-regime check).
- "natural → ring vs instruct → no-ring" → **all LINE under list; topology = format, not regime** (compare3_gemma).
- "clean ring" from RSA alone (multiple over-reads) → **require dRSA(cyc−lin)<0 + closure ratio + plot** for any ring/cycle claim. RSA + X=0 are NOT sufficient.
- "regime-dependent geometry" as a deep claim → reinterpret: **same structure, different subspace**; the cross-regime probe non-transfer may be **partly a shallow position/token effect** (different scaffold tokens). Only the **regime-LOCK probe non-transfer** (practical: match the regime) + the **cheap generation-free read** survive.
- "ring crystallizes along the CoT" → **inconclusive + over-controlled** (dictated step-by-step; not native/natural).

PAPER SHAPE (converging): **Core = context-constructs-the-map (anti-realist) + topology-follows-specification + capability-gated
+ representation-vs-computation.** **Downstream = suffix-probing**: the pre-reasoning representation is a context-built map of
input attributes, so a small generation-free "snap" suffix yields a cheap read of it (KV-fork) — a **cost/method** result
(NOT representation-superiority), with the **regime-matching** caution and the **input-attribute vs reasoning-driven** boundary.

## E. Required correction going forward
Re-run the "reasoning" side **honestly**: open-ended prompt (no format dictation), **native thinking ON** for Gemma/Qwen
(default), measure BOTH (a) answer accuracy and (b) imposed-structure geometry in the natural regime. Keep the automatic
(forced-snap) regime as the verified concentrated baseline. Only then can we make any "automata vs reasoning" claim.
