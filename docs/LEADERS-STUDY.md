# Domain Leader Study — szl-cram (CRAM estate reimplementation)

OUROBOROS phase-3, cohort-02 tranche-1, Kimi ruling W26-C2T1-KIMI-20261123-ASK (GO). PROPOSE-ONLY study card: documentation only, no implementation in this PR.

## Current state (repo inspection 2026-11-23)
Python repo; description: CRAM (arXiv:2611.17309) estate reimplementation, deep audit, frontier extensions; pure numpy, receipt-bound. Topics: cram, szl-estate, szl-holdings. A prior sibling lane (t11) measured its CI/permissions surface this same night (rank-1 unit, workflow-scope push wall) — no conflict with this docs-only card.

## Domain leaders studied
| Leader | What they have | Key sources |
| --- | --- | --- |
| CRAM original: Prefrontal-AI | The paper Classical Energies Learn to Reason 1,000x Faster than Looped Models (Compact Reasoning Associative Memory) + reference implementation: a learned energy solving logic puzzles (Sudoku-Extreme, maze-unique) in seconds, orders faster than looped models | arxiv.org/abs/2611.17309; github.com/Prefrontal-AI/CRAM (accessed 2026-11-23) |
| NumPy NEP 0038 / NumPy 2.x SIMD | The canonical performance doctrine for pure-numpy numeric code: SIMD-optimized ufuncs, memory layout discipline, broadcasting rules — how vectorized kernels get fast honestly | numpy NEP 0038 (numpy/numpy doc/neps/nep-0038-SIMD-optimizations.rst) (accessed 2026-11-23) |

## What the leaders do better
- The reference repo publishes its puzzle suites and benchmarks alongside the paper, so any reimplementation can be scored against the original, not against itself.
- NumPy's perf doctrine is written down (NEP process, memory layout, SIMD coverage) so vectorization claims are testable per-ufunc.

## Gaps
- Estate reimplementation is receipt-bound, but no documented parity harness against the Prefrontal-AI reference suite (Sudoku-Extreme/maze-unique class) was found; a reimplementation without external parity is self-referential.

## Ranked #1 absorb proposal (NOT implemented in this PR)
A parity + perf receipt harness: reproduce reference-suite cases (citing arXiv:2611.17309) and record per-case solve time + numpy 2 SIMD/memory-layout profile in each run receipt. Expected benefit: deep-audit claims become externally grounded; frontier extensions measurable against the paper's own numbers. Cost/risk: low-medium; additive harness, licensing of reference cases to be checked before inclusion. Validation: harness green on the published cases with receipts emitted. Rollback: harness removal only.

## Sources
- CRAM paper: https://arxiv.org/abs/2611.17309 (accessed 2026-11-23)
- CRAM reference: https://github.com/Prefrontal-AI/CRAM (accessed 2026-11-23)
- NumPy NEP 0038: https://github.com/numpy/numpy/blob/main/doc/neps/nep-0038-SIMD-optimizations.rst (accessed 2026-11-23)
