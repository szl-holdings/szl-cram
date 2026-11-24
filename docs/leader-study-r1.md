# szl-cram domain leader study r1 (propose-only)

Date: 2026-11-24. Method: web-verified leader landscape + governing-seat ratification (Kimi nonce T8-RULING-20261124-K1). Study pinned to szl-cram HEAD 920b809700834217675916cb72caa2760b4ff226, fetched 2026-11-24. This card is documentation only; it changes no code, CI, config, or existing PR status. All absorb actions are FUTURE PROPOSALS (non-authorizing); implementation and merges remain separately reviewed and gated. L Conjecture 1: OPEN (advisory, never a closed theorem). No speedup, energy, or proven-trust claims; upstream performance claims are labeled as upstream claims only.

## Domain: Receipt-bound pure-numpy reimplementation and deep audit of CRAM (arXiv:2611.17309), using classical Hopfield-Tank energies, one hand-set weight, zero trained parameters, and machine-checkable energy certificates with honest answer, E_final, threshold, and margin fields.

## Leaders studied (web-verified 2026-11-24)
1. NumPy (https://github.com/numpy/numpy) - moat: A mature array-computing core with documented numerical semantics and reproducibility conventions relevant to a pure-numpy implementation.
2. Hypothesis (https://github.com/HypothesisWorks/hypothesis) - moat: Property-based testing that generates adversarial examples and shrinks failures, useful for expressing certificate invariants without asserting unverified results.
3. JAX (https://github.com/jax-ml/jax) - moat: Composable array transformations and hardware-backend documentation that can inform future compatibility review while leaving the pure-numpy core unchanged.

## Absorb actions (FUTURE PROPOSALS, non-authorizing)
1. (NumPy) Propose a documentation-only note on determinism and reproducibility conventions for the pure-numpy core. Cost: doc-only, ~half day; limit: note only.
2. (Hypothesis) Propose a future design note on property-based certificate checking (energy bounds as invariants). Cost: doc-only, ~1 day; limit: note only.
3. (JAX) Propose a documentation-only review of any future hardware-compatibility planning, upstream claims labeled as such. Cost: doc-only, ~half day; limit: note only, pure-numpy core preserved.
4. Baseline preserve honestly: zero trained parameters stays true; certificates stay machine-checkable with honest margins.

## Do not do (per governing seat)

Do not merge, bypass the signature wall, or claim implementation beyond this card.
