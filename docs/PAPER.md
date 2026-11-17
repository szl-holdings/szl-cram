# Spurious Deep Minima in Positives-Only Energy Fits: an audit-driven replication of CRAM

**Status: draft.** Working title. Author list and venue TBD (owner-gated).
Numbers below are receipt-bound to `results/` (cram_results.json, cram_results2.json,
cram_results16.json, diag*.json) in this repository.

## Abstract (draft)

Hoover, Blayney & Whittington (arXiv:2611.17309) propose CRAM: classical
Hopfield–Tank energies trained by backpropagation-free pseudolikelihood (PL), reported
to reach 99.996% exact-solve on extreme Sudoku at ~1000× lower training cost than looped
Transformers. We report a faithful pure-numpy replication whose learned model reaches
0.67% (1/150) on boards of matched difficulty, and we localize the discrepancy. The
failure is informative: (1) perturbed solutions relax back at 97.5% (78/80), so solutions
are strong attractors of the learned energy; (2) direct energy probes reveal spurious
minima *at and below solution depth* (diag9: minima below solE = −242.69 on wrong boards);
(3) interpolant analysis shows the learned cost matrix contains *negative off-diagonal
entries* (min −0.5506), i.e., the training objective rewards decoy configurations in
peer pairs. Positives-only PL fitting on permutation-structured constraint problems
therefore buys attractor depth with decoy minima, capping extreme-difficulty solve rates
below a one-weight hand-set energy (8.67% vs 0.67% here). We release all energies,
datasets, search variants, and an energy-certificate format intended as a governance
primitive for relaxation-based solvers.

## 1. Audit of the source

Line-by-line verification of the paper's math (energy Eq. 1, PL gradient, mean-field
anneal): sound. FLOPs accounting and baseline comparisons: honest, with one exception:
the headline "42 ms training" is a warm-run in-process median against cold-start
baselines; honest cold-start comparison is ~2–5 s. The 1000× claim survives at the
order-of-magnitude level.

## 2. Replication setup

- 150 extreme boards, 24–27 clues, unique solutions (dig + check; `data.py`).
- PL fit: 131,949 parameters, fitted threshold −208.25 (7.6 s, cold process, laptop CPU).
- Searches: paper-style anneal; MFA v2 (β-anneal + quorum + polish); focused Gibbs;
  hand-set one-weight energy for all of the above.

## 3. Results (receipted)

| Configuration | Exact /150 | Time |
| --- | --- | --- |
| Learned, paper-style search | 0 | 234 s |
| Learned, MFA v2 | 1 (0.67%) | 695 s |
| Learned, 60-board anneal | 0 | 162 s |
| Hand-set, MFA v2 | 13 (8.67%) | 1032 s |
| Hand-set, focused Gibbs | 4 (2.67%) | 98 s |

16×16 scale probe: PL fit 2.56M params, threshold −475.24; learned 0/30, hand-set 1/30.

## 4. Why the learned energy fails where it should win

Diagnostic chain (diag5a–diag9):

1. **Attractor strength (diag8):** 78/80 perturbed solutions relax back to the solution.
   The energy is not "weak" at solutions.
2. **Spurious depth (diag9):** random boards reach energies *below* solution energy
   solE = −242.69; wrong boards sit below the fitted threshold −208.25. The landscape
   has deep non-solution minima.
3. **Decoy substrate (interp9):** the fitted cost interpolant is argmax-diagonal on
   100% of 1620 (state, peer-state) pairs (mean diagonal 2.1119) but its minimum
   off-diagonal cost is **−0.5506 < 0**: for some peer pairs the *wrong* joint
   configuration is energetically rewarded. Positives-only PL has no margin term, so
   nothing penalizes these entries; the anneal then parks in decoy minima on
   extreme-underconstrained boards.

## 5. Contributions

1. The decoy-minima mechanism above, with receipts.
2. An honest replication corpus (extreme-only, unique-solution, all scripts).
3. The energy certificate `{answer, E_final, threshold, margin, sweeps, valid}` as a
   portable proof object for relaxation solvers.
4. R5 generalization: the same certificate format on graph coloring
   (32 containers / 8 GPUs / 94 edges, E=0 in 0.01 s).

## 6. Limitations

CPU numpy, no GPU; absolute times are not comparable to the paper's H200 runs; ratios
and accuracies are the targets. Our PL fit is documented-not-claimed-verbatim (the paper
does not release code at audit time); we verify math against the text.

## 7. Reproducibility

Everything needed is in this repo: `results/`, `STUDY.md`, `receipt.json`, `szl_cram` package (`pip install szl-cram`), and the dig/fit/search scripts under
`results/scripts/`. Datasets and fitted models on Hugging Face (SZLHOLDINGS org).
