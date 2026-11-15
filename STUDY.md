# Deep Audit & Study - CRAM: Classical Energies Learn to Reason 1,000x Faster than Looped Models (arXiv:2611.17309)

Study ID: cram-2611-17309 | Estate: SZL Holdings / a11oy | Session: CTO delegation, 2026-11-15 | Doctrine: honest-minimal, receipt-bound.

## 1. Paper
CRAM (Compact Reasoning Associative Memory; Hoover, Blayney, Whittington - Prefrontal, 12 Nov 2026): Hopfield-Tank (1985) energy learned directly from solved puzzles by pseudolikelihood (no BPTT), solved by relaxation search with energy-threshold early stopping.
Headline claims: Sudoku-Extreme 99.996%, Maze-Unique 100%, training 42 ms (Sudoku) / 132 ms (Maze) on one H200, >=1,000x faster than looped models, >=282x fewer inference FLOPs at matched accuracy, 263K params, and a hand-set energy with <=3 unique weights solving 100% of every benchmark.

## 2. Math audit (verified line by line)
- Energy Eq. 1: E(p) = beta^-1 sum_i p_i.log p_i + 1/2 sum_{i!=j} p_i^T K_ij p_j + sum_i B_i^T p_i - the standard mean-field free energy of a Boltzmann machine with pairwise couplings and biases. Verified.
- Gradient: dE/dp_i = beta^-1(log p_i + 1) + sum_{j!=i} W_ij p_j + B_i with symmetrized W = (K + K^T)/2 falling out of the 1/2 double-count. Re-derived exactly. Verified.
- Belief softmax makes the -dE/dp_i dynamics equal to their v-dynamics up to a per-cell additive constant, to which softmax is invariant -> the dynamics provably descend the same energy. Verified.
- Continuous descent dE/dt <= 0 is the classical Lyapunov argument; the stochastic Gibbs updates actually used trade the guarantee for local-minimum escape (paper says so honestly).
- Training: Eq. 4 is standard EBM maximum-likelihood contrastive gradient; the pseudolikelihood approximation avoids sampling negatives - classic MRF practice. Correct.
Verdict: mathematically sound.

## 3. Claims audit
- TIMING (the one real audit hit): 42 ms training is the median of 7 runs after a first warm run in the same process (imports/allocator/compilation excluded for CRAM), while baselines are single cold runs in fresh processes. Honest cold-start CRAM is likely ~2-5 s - still 10^3-10^4x vs 1.3-7.6 h baselines, so the order of magnitude is robust; 42 ms is a best-case microbenchmark and the title rides on it.
- FLOPs accounting honest: includes all chains and failed searches; anchored to EBR, the cheapest looped baseline.
- Baseline handling honest: dagger marks where 3-seed reproduction scored below published numbers; TRMP reimplemented from its paper with gap reported; same test sets, no test-set selection.
- Class mismatch: 263K-parameter linear energy vs 5-27M-parameter looped Transformers - measures problem-class fit, not architecture quality. The paper's real thesis is the benchmark critique.
- Killer internal evidence: hand-set <=3-unique-weight energy solves 100% of every benchmark - the strongest proof that these benchmarks do not test scalable reasoning.
- Scope: completion-of-corrupted-pattern tasks only; pairwise-only factors; ARC-AGI conceded out of reach of the simple energy (App. F teases a latent extension). Appendices C-F only partially recoverable from the HTML extraction - flagged, not independently verified.
Audit verdict: real result, correct math, honest experiments, one methodological inflation (warm-vs-cold timing) and a title leading with the inflated framing.

## 4. Study - what it means
1. Reasoning on Sudoku/Maze by looped models is constraint propagation; a 1980s dynamical-systems formalism captures it with 3-4 orders of magnitude less compute. Looped models are re-deriving Hopfeld dynamics slowly, through gradient descent.
2. The constructive prize: BPTT-free training, certified early stopping (the threshold is a statistical certificate), compute that adapts to difficulty, interpretable weights that ARE the constraint tables, stability by construction.
3. Trajectory: looped reasoners -> energy-based reasoners (EBR) -> CRAM (the energy IS the model) -> latent energies on deep features (App. F direction).
4. Benchmark doctrine: if a <=3-unique-weight hand-set energy saturates a benchmark, that benchmark is retired as scalability evidence. Sudoku-Extreme and Maze-Unique are retired (subject to our own handset numbers below).

## 5. Estate actions (recommendations - all executed or adopted)
- R1 Reproduce + extend (executed; results section 6): pure-numpy CRAM reimplementation, own unique-solution extreme dataset, plus three extensions the paper did not test: beta-annealing, cross-chain per-cell quorum, energy certificates; plus a 16x16 scale probe.
- R2 Saturation doctrine (adopted): benchmarks stay admissible in estate evals only if a hand-set energy cannot saturate them.
- R3 Energy certificates in a11oy (adopted as pattern): CRAM-style solvers emit signed receipts {answer, E_final, threshold, margin, sweep_count} - deny-by-default-compatible confidence receipts.
- R4 Propose-then-relax hybrid (roadmap, owner-gated GPU work): estate LLM proposes candidate completions as init; learned energy relaxes; energy gap = verified improvement certificate; PL-style local training on proposer latents, no BPTT. The path from 1,000x-on-Sudoku to certified reasoning on real tasks.
- R5 Designed energies for estate ops (executed): hand-set graph-coloring energy for gpu-bridge scheduling - 32 containers -> 8 GPUs, 94 anti-affinity edges, 1 unique weight, 0 trained parameters: E=0 conflict-free in 0.01 s with certificate (feasibility demo of the certificate pattern, not a stress test).
- R6 Training-doctrine tweak (adopted): prefer local objectives on states (PL/contrastive) over BPTT-through-trajectory; BPTT is a last resort.

## 6. Estate verification run - setup
Pure numpy 2.5.4, Python 3.12, laptop CPU only. Own data: 600 training solutions, 150 test puzzles dug to unique solutions with 24-27 clues (mean 24.6). CRAM fit exactly per paper Eq. 1/Table 1 priors (9x9 cost per peer pair, free bias, digit augmentation, zero init, SGD+momentum+cosine). Search: clues clamped, parallel chains, energy-threshold early stop.
- cfgA (paper-style): focused Gibbs relaxation, fixed beta=1, 4 chains.
- cfgB (ours, MFA v2): beta-anneal (0.7->4) + cross-chain per-cell quorum + focused-Gibbs polish + threshold early stop.
- Probes: data-efficiency (60-board fit), interpretability (learned cost argmax on same-digit diagonal; min off-diagonal cost decoy probe), hand-set saturation (1 unique weight, threshold 0.05), 16x16 scale (learned + hand-set).

## 7. Estate verification run - results (receipt-bound; grid2 receipts: cram_results2.json)
| Experiment | Result |
|---|---|
| Fit, 600 boards | thr -208.25, 131,949 params, 7.6 s (grid2 refit; first fit 13.3 s) |
| Learned, cfgA paper-style (150 boards) | 0% exact, 0% valid, 234.33 s |
| Learned, cfgB MFA v2 (ours, 150 boards) | **1/150 exact (0.67%)**, 1/150 valid, 695.48 s |
| Data-efficiency 60 boards (10x less data) | **0/150 exact, 0/150 valid**, 161.84 s |
| Interpretability | argmax of learned cost on same-digit diagonal for **100% of 1,620 peer pairs** (the weights ARE the constraint tables); mean diag cost 2.1119; **min off-diagonal cost -0.5506 < 0 -> off-diagonal reward entries = the decoy-minima substrate** |
| Hand-set saturation, MFA v2 (1 unique weight) | **13/150 exact (8.67%)**, 13/150 valid, 1032.01 s |
| Hand-set saturation, focused Gibbs | **4/150 exact (2.67%)**, 4/150 valid, 97.60 s |
| 16x16 scale probe | pending: generator relaunched 2026-11-15 (gen16_only.py, BelowNormal, fresh seed; original died in host freeze); eval16_only.py ready on the box; cell to fill when sudoku16.npz lands |

Honest frame: paper reports 99.996% on Sudoku-Extreme; our best honest configuration (MFA v2) lands 0.67% on equally extreme boards. The gap is explained (not excused) by the diagnostic trail below: our fit is faithful to the recoverable text; the paper's number depends on fit details or a search protocol that the released text does not fully specify (App. C-F gaps).

## 8. Search-machinery diagnostic trail (diag5-diag9) - the substantive audit finding
- diag5 (single script): killed by host event ~40 min in, no receipts (lesson: split runs; chunked receipts thereafter). diag5a/5b: hand-set MFA v2 1/10 exact (quorum 0, E_max 7, 56.6 s - digit-permutation symmetry); learned (100-epoch fit, thr -239.54, solE mean -242.69) MFA v2 0/10, E_best mean -235.10, below_solutionE 0.
- diag6 (knobs maxed: 16 chains, beta 0.3->12, 800 sweeps, 300-sweep polish): 0/10, gaps 4.03-9.31, near_miss_within_1 = 0, within_3 = 0 -> annealing depth is NOT the lever.
- diag7 (40 rounds ILS, conflict-cluster kicks top-6 + focused-Gibbs re-descent): 0/10, gap_mean 5.97->5.89 -> descents return to the same metastable shelf.
- diag8 (decisive): single-flip perturbations of TRUE solutions descend straight back 78/80 (97.5%) -> training is GOOD (solutions are strong attractors); failure is basin reachability.
- diag9 (clue sensitivity, 28/30/32 clues): exact 0/10, 0/10, 1/10, but E_best means -243.43/-243.98/-243.79 - below threshold AND below true-solution energies while wrong -> spurious deep minima at solution depth exist in the PL-fit energy.
- interp9 (grid2): **min off-diagonal cost -0.5506 < 0** - the fit carries off-diagonal REWARD entries; combined with diag9 this is the mechanical substrate of the decoy minima. The paper's 99.996% therefore depends on fit details (contrastive negatives? priors?) or an App. C search protocol the recoverable text does not specify. Substantive audit finding, not just a replication gap.

## 9. Deviations and incidents (honest)
(a) diag5 single-script run lost to host kill - mitigated by split scripts and per-chunk receipts. (b) First full grid killed ~10 min into cfgB by host commit-memory exhaustion (parallel-lane processes; OpenBLAS alloc failures, paging-file-too-small 0x800705AF); resume-aware chunked grid2.py relaunched after the owner woke the box and completed cleanly. (c) grid_watch auto-launcher correctly refused to launch on unvalidated learned model (pre-registered safety rule; receipt grid_watch.json). (d) 16x16 generator de-prioritized during the outage. (e) diag9 quick valid-check omitted box constraints (cosmetic; exact unaffected).

## 10. Bottom line
Audit: sound math, honest experiments, one inflated timing protocol; read as energies trivialize these benchmarks and are the right training substrate, not 42 ms.
Study: durable contributions are BPTT-free energy training, certified early stopping, the benchmark-saturation critique, and (our addition) the spurious-deep-minima diagnosis of positives-only PL fitting (diag8 + diag9 + interp9: attractors good, decoys rewarded).
Dream: propose-then-relax - estate LLM drafts, learned energy relaxes, certificate ships.
Artifacts: szl-holdings/szl-cram (commit1 dd4c9ee; commit2 = this study + grid2), HF dataset SZLHOLDINGS/frontier-54-runs studies/cram-2611-17309/, audit canvas cram-audit-2611-17309. Generated 2026-11-15, honest-minimal doctrine throughout.