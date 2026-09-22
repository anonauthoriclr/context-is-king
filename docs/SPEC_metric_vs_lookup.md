# SPEC — Metric vs Lookup: is the context-imposed ring a genuine metric manifold?

> **Status: DRAFT @ Gate 1 (methods agreement). Nothing runs until operator sign-off.**
> Author: orchestrator, 2026-06-17. Reuses `multiscr.py` capture + `cos_rdm`/`cyc_rdm`.
> Climbs the claims ladder one rung on H0.3 (anti-circularity / metric) — *characterize*
> the representational mechanism, not "invert Park." See `docs/03` H0.3 (currently PARTIAL:
> the held-out-pairs interpolation test "was not run cleanly").

## 1. Question

When context declaratively imposes a conflicting cyclic order, does the reorganized
concept-geometry encode the **full modular-distance metric** of the imposed cycle — a
genuine *closed metric ring* that supports interpolation — or only **local adjacency**
(a successor table / lookup) that we are over-reading as a ring via RSA?

This is the realist-vs-artifact question at the level of the *imposed* structure:
RSA-vs-cyclic-template can be inflated by adjacency alone (Diaconis–Goel–Holmes 2008:
local-only distances produce a **horseshoe/arch**, not a closed cycle). We have never
separated *graded metric* from *adjacency* for the imposed order.

## 2. Hypotheses (pre-registered)

- **H1 (metric).** In large-instruct models the reorganized geometry is a **closed metric
  ring**: representational distance grows monotonically with imposed modular distance
  *beyond adjacency*, and a **cyclic** template fits better than a **linear/open** one.
- **H2 (scale upgrades lookup→metric) — the mechanism claim.** Metric-ness *increases with
  scale*: small models reorganize to **adjacency-only / horseshoe** (graded structure
  absent or open) even where crude mean-RSA is positive; large models close the metric ring.
  This would explain why small-scale mean-RSA flip-flopped (a horseshoe gives unstable RSA).
- **H0 (lookup null).** If the geometry is a successor table, graded-beyond-adjacency ≈ 0,
  cyclic≈linear fit, and held-out interpolation ≈ chance.

Three diagnosable regimes (the test is designed to separate all three):

| regime | M1 graded (d≥2) | M2 closure (cyclic−linear) | M3 interpolation |
|---|---|---|---|
| **Lookup** (adjacency only) | ≈ 0 | ≈ 0 | ≈ chance |
| **Horseshoe** (open metric) | > 0 | **< 0** | partial |
| **Metric ring** (closed) | > 0 | **> 0** | ≫ chance |

## 3. Data / capture (reuse `multiscr.py`)

- One capture per (model, concept, imposed order): entity-layout residual, last-token
  pre-generation, **grouped by INPUT entity, marginalized over k and templates** — i.e. the
  existing `capture()` → N centroids in residual space at L = round(0.75·n_layers).
- ≥ **6 random imposed scramblings** (reuse the multiscr loop); report the distribution,
  not one permutation. Cache acts to `.npz` keyed by (model, concept, order, L) per the
  caching standard — these centroids feed all three metrics + the positive control.
- Distances are **mean-centered cosine** (`cos_rdm`, anti-anisotropy), as in all prior RSA.

Let `p_i` = imposed position of entity i (0..N−1). Templates over pairs (i,j):
- cyclic: `d_cyc(i,j) = min(|p_i−p_j|, N−|p_i−p_j|)`  (current `cyc_rdm`)
- linear: `d_lin(i,j) = |p_i−p_j|`  (open chain — the horseshoe template) **[new]**

## 4. Metrics & statistics

**M1 — graded-beyond-adjacency (the metric core).**
`rho_far = Spearman( cosRDM[pairs with d_cyc ≥ 2], d_cyc[same pairs] )`.
Drops the d_cyc=1 adjacency pairs (the "lookup" signal) — tests whether *non-adjacent*
distances are graded. Companion plot: mean cosine-distance per d_cyc bin (1..⌊N/2⌋) — a
metric ring gives a monotone staircase; lookup gives flat for d_cyc ≥ 2.
- Bootstrap CI by resampling **entities** (recompute centroids' RDM on the resample).
- **Pass (graded):** CI_low > 0.

**M2 — closure (ring vs horseshoe).**
`dRSA = Spearman(cosRDM, d_cyc) − Spearman(cosRDM, d_lin)` over **all** pairs.
A closed ring prefers the cyclic template (wrap-around adjacency is close); a horseshoe is
an open arch that the linear template fits better (its ends — positions 0 and N−1 — sit far
apart). Bootstrap CI over entities.
- **Closed ring:** CI_low > 0.  **Horseshoe:** CI_high < 0.  **Indistinct/lookup:** CI spans 0.

**M3 — held-out interpolation (H0.3's unrun test; the strongest single discriminator).**
Leave-one-entity-out *triangulation*: embed the N−1 retained centroids on a circle
(classical MDS → 2D → angle), then place the held-out entity using **only its distances to
the retained ones** (out-of-sample MDS projection) and read its recovered angle.
- Statistic: **circular correlation** between recovered angles and true imposed positions
  for held-out entities, scored **up to global rotation + reflection** (the imposed ring is
  defined only up to its dihedral symmetry — align by best of the 2N rotations×reflections).
- Null: permute imposed positions (1000×) → chance circular-correlation distribution.
- **Pass (interpolable metric):** observed > 95th percentile of null.

## 5. Controls (binding)

1. **Positive control — natural order.** Run the full M1/M2/M3 battery on the *unscrambled*
   (pretrained) ring. It MUST register as metric-ring (graded, closed, interpolable). If the
   known-good ring fails the battery, the battery is broken → abort interpretation.
2. **Anisotropy null (F42 style).** Label-shuffle entities on the *real* (anisotropic)
   activations, recompute M1/M2 → must collapse to ≈ 0. Confirms graded/closure structure
   is not a cone artifact.
3. **Synthetic lookup null (discriminator power).** Construct successor-only synthetic
   centroids (one-hot "next") and verify the battery classifies them **Lookup** (M1≈0,
   M2≈0, M3≈chance). Confirms the test *can* see the negative — otherwise a pass is
   uninterpretable. (Cheap; no GPU.)
4. **Multi-scramble.** ≥6 imposed orders; report mean ± spread of M1/M2/M3, not a single run.

## 6. Conditions

- **Primary (strong, clean reorg → there IS a ring to interrogate):** Gemma-4-31B-it,
  Qwen3.5-27B-it × {days, months}. days = the hard case; months = the malleable contrast.
- **Scale axis (tests H2 — the mechanism):** Gemma E2B/E4B + Qwen-4B/9B × {days, months}.
  Prediction: M1/M2/M3 rise with scale; small models pass crude RSA but read as
  horseshoe/lookup (M2 ≤ 0). This is the headline result if it holds.
- **Definition-format robustness (stricter metric test):** primary uses the existing
  full-cycle definition (`deftext`, shows "Full cycle (wraps): …" — comparable to all prior
  runs). Add an **adjacency-only** definition variant (state only successor relations, do
  NOT lay out the full traversal) so far-pair distances must be *composed*, not echoed from
  the prompt text. Run on the primary models only. If metric survives adjacency-only def →
  the representation composes the metric (strong); if it collapses → the metric is
  text-scaffolded (still a finding, honestly reported).
- **Layer:** L = 0.75·depth (consistent with prior); spot-check the layer plateau (F31) on
  one primary cell.
- **N ≥ 7** (exclude N=4). Use N=12 (months) for higher power on the borderline reads.

## 7. Decision rules (pre-committed)

Per (model, concept), classify by (M1, M2) signs with bootstrap CIs, M3 corroborating, into
**Metric-ring / Horseshoe / Lookup** per the §2 table. Then report the **scale profile** of
the classification per family (the H2 test). Pre-commit:
- **H1 supported** iff primary large-instruct cells are Metric-ring (M1 CI_low>0 AND
  M2 CI_low>0 AND M3 > null-95th).
- **H2 supported** iff the regime monotonically upgrades Lookup/Horseshoe → Metric-ring with
  scale in ≥1 family (and crude mean-RSA does *not* track that transition — i.e. RSA is
  positive in cells we classify non-metric).
- **Lookup/horseshoe at large scale** (H1 fails) is a real, publishable negative — it would
  bound the realist reading of the imposed structure. Report either way.

## 8. Confounds & limits (named)

- **Definition spoon-feeds the order** (full-cycle text lays out the metric) → the
  adjacency-only condition (§6) is the control; primary result is reported as
  "metric present in representation given full-cycle context," adjacency-only as the
  stricter "metric composed."
- **Anisotropic cone** → mean-centered cosine + anisotropy null (§5.2).
- **Cyclic symmetry** → M3 scored up to rotation+reflection only (absolute phase is not
  identifiable and is not claimed).
- **Low N power** (days N=7) → bootstrap + months N=12 as the higher-power read; never
  classify on a single scramble.
- **k/entity confound** (the F22 trap) → already neutralized by input-entity grouping with
  k,template marginalized in `capture()`.
- This is **static + representational**. "Supports interpolation" is a geometric property of
  the representation, **not** a behavioral/causal claim. Causal (does the metric drive the
  walk) is the separate interventions-rung experiment.

## 9. Reproducibility / outputs

- Extend `multiscr.py` (or a sibling `metric_lookup.py`) with: `d_lin` template; M1 (far-Spearman
  + per-bin profile); M2 (dRSA + CIs); M3 (LOO-MDS circular-corr + perm null); the 4 controls.
- Cache centroids to `.npz` per (model, concept, order, L); analysis reads from cache.
- Unbuffered per-scramble progress (operator standard).
- Figures: (a) cosine-distance vs d_cyc bin staircase per cell; (b) regime-by-scale grid
  (Lookup/Horseshoe/Metric) per family; (c) LOO interpolation recovery scatter (recovered vs
  true position) for one primary cell. Findings entry on completion → Gate 2 before doctrine.
