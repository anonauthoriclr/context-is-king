# Soundness pass — per-scramble CIs + k>=2 predictive (2026-06-25, CPU)

## A. Headline numbers with 95% CIs

### Dominance: imposed-RSA vs residual-natural-RSA (snap, paired by scramble)

| model | concept | imposed-RSA | natural-RSA | gap (imp−nat) |
|---|---|---|---|---|
| Qwen3.5-27B | days | +0.671 ± 0.030 (n=10) | -0.010 ± 0.182 (n=10) | **+0.681 ± 0.184 (n=10)** |
| Qwen3.5-27B | months | +0.706 ± 0.039 (n=10) | +0.120 ± 0.093 (n=10) | **+0.586 ± 0.087 (n=10)** |
| Qwen3.5-4B | days | +0.315 ± 0.109 (n=10) | +0.241 ± 0.122 (n=10) | **+0.074 ± 0.184 (n=10)** |
| Qwen3.5-4B | months | +0.460 ± 0.070 (n=10) | +0.377 ± 0.042 (n=10) | **+0.083 ± 0.082 (n=10)** |
| Qwen3.5-9B | days | +0.105 ± 0.132 (n=10) | +0.439 ± 0.118 (n=10) | **-0.334 ± 0.146 (n=10)** |
| Qwen3.5-9B | months | +0.605 ± 0.050 (n=10) | +0.188 ± 0.062 (n=10) | **+0.416 ± 0.090 (n=10)** |
| gemma-4-31B | days | +0.870 ± 0.019 (n=10) | -0.033 ± 0.215 (n=10) | **+0.902 ± 0.213 (n=10)** |
| gemma-4-31B | months | +0.811 ± 0.018 (n=10) | +0.049 ± 0.088 (n=10) | **+0.763 ± 0.084 (n=10)** |
| gemma-4-E2B | days | +0.599 ± 0.083 (n=10) | -0.061 ± 0.140 (n=10) | **+0.660 ± 0.183 (n=10)** |
| gemma-4-E2B | months | +0.587 ± 0.095 (n=10) | +0.232 ± 0.085 (n=10) | **+0.355 ± 0.151 (n=10)** |
| gemma-4-E4B | days | +0.720 ± 0.055 (n=10) | +0.056 ± 0.265 (n=10) | **+0.665 ± 0.262 (n=10)** |
| gemma-4-E4B | months | +0.583 ± 0.098 (n=10) | +0.308 ± 0.105 (n=10) | **+0.275 ± 0.164 (n=10)** |
| gemma-4-31B | days **(natural prompt)** | +0.639 ± 0.032 (n=12) | +0.059 ± 0.148 (n=12) | **+0.580 ± 0.157 (n=12)** |

### Depth-RSA (imposed tree, neutral, per-scramble)
| tree | model | depth-RSA |
|---|---|---|
| d3 | Gemma | +0.825 ± 0.037 (n=6) |
| d3 | Qwen | +0.705 ± 0.056 (n=6) |
| d4 | Gemma | +0.729 ± 0.028 (n=6) |
| d4 | Qwen | +0.630 ± 0.075 (n=6) |

### Causal flip (entity-substitution, per-scramble) — thin base, flagged
| model | layer | patch_success | cross-map |
|---|---|---|---|
| gemma-4-31B | L6 (flip locus, d=0.10) | +1.000 ± 0.000 (n=2) | +0.143 ± 1.815 (n=2) |

*Causal n is only 2 scrambles (snap) — CI degenerate; per-scramble values are [1.0, 1.0] (own-map flip) vs [0.29, 0.0] (other-map=chance). Flagged as a limitation; +scrambles queued (GPU).*

## B. Does construction quality predict MULTI-HOP (k>=2) reading, not just k=1?

| k (hops) | n models | Spearman ρ | p | sig@.05 | acc range (min–max) |
|---|---|---|---|---|---|
| k=1 | 8 | +0.83 | 0.011 | yes | 0.25–1.00 |
| k=2 | 8 | +0.69 | 0.058 | no | 0.14–0.90 |
| k=3 | 8 | +0.61 | 0.108 | no | 0.02–0.46 |
| k=4 | 8 | +0.66 | 0.076 | no | 0.10–0.50 |

*Note: regime-matched snap (list). Only k=1 is significant at .05 (n=8); k>=2 point estimates stay positive but underpowered. CONFOUND: the 8 models span only 3 families (Gemma high-RSA/high-acc, Qwen mid, Llama low) — the rank correlation is closer to a 3-cluster separation than 8 independent points. Report as directional, not a law.*