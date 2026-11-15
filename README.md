# szl-cram

CRAM (Compact Reasoning Associative Memory, arXiv:2611.17309, Hoover/Blayney/Whittington) —
estate reimplementation, deep audit, and frontier extensions. Pure numpy, honest-minimal,
receipt-bound. Audit study: `STUDY.md`. Results: `results/`.

## What this repo is

1. **Independent audit** of arXiv:2611.17309 (math re-derived line-by-line; claims checked;
   one timing-protocol inflation documented honestly in STUDY.md).
2. **Reproduction** on estate hardware (laptop CPU, pure numpy, no framework): unique-solution
   extreme Sudoku (24-27 clues), learned CRAM vs hand-set saturation probe.
3. **Extensions the paper did not test**: beta-annealing (deterministic annealing),
   cross-chain per-cell quorum (khipu-consensus pattern), energy certificates
   (E, threshold, margin shipped with every answer), and a 16x16 scale probe.
4. **Doctrine outputs**: benchmark-saturation rule (hand-set energy test retires a benchmark),
   energy-certificate pattern for a11oy, propose-then-relax hybrid roadmap (R4, owner-gated
   GPU phase), designed energies for estate ops (gpu-bridge allocation as graph coloring).

## Files

- `cram.py` — the energy (Eq. 1), PL training (no BPTT), Gibbs relaxation search, hand-set weights.
- `gen_sudoku.py` — unique-solution puzzle generator (propagation backtracking + dig).
- `train_eval.py` — full experiment grid; writes `cram_results.json` receipts.
- `STUDY.md` — the deep audit + study + frontier plan.
- `results/` — receipts as run (never edited).

## Honest notes

- Search machinery evolved across three receipted iterations (results/diag2-diag5.json):
  v1 plain Gibbs (E~45 Nash plateau) -> v2 focused Gibbs + stuck-restarts + mixing ->
  v3 mean-field annealing (uniform-init belief descent, annealed beta+noise) + focused polish.
  The paper's exact search protocol (App. C-F gaps) is not fully recoverable; ours is documented, not claimed-verbatim.
- Cold-process fit times reported, not in-process medians (see STUDY.md audit hit).
- No GPU was used; all numbers are laptop CPU numpy. Slower than the paper's H200 absolute
  times by construction; ratios and accuracies are the comparison targets.

Apache-2.0. Doctrine v11. Receipts hash-committed. Never fabricated: BLOCKED > invented.