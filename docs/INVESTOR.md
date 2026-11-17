# Investor brief — szl-cram

**One line:** we turn a 1980s dynamical-systems insight (constraint problems are energy
problems) into certified, ultra-cheap reasoning at the edge, with every scientific claim
receipt-bound.

## The wedge

Modern AI burns GPU loops to re-learn what constraint propagation already solves.
arXiv:2611.17309 (CRAM) showed classical Hopfield–Tank energies trained without
backprop-through-time can match looped Transformers on Sudoku-class reasoning at
~1000× less training cost. We audited that paper line-by-line, replicated it honestly
(we publish our failures), found where its learned fits break (decoy minima), and
packaged what survives:

1. **A reasoning engine with no inference cost:** one hand-set weight, zero trained
   parameters, runs at the edge (Cloudflare Worker) with zero cold-start, ~0 marginal
   cost per solve.
2. **The certificate primitive:** every answer ships with
   `{answer, E_final, threshold, margin, sweeps, valid}` — a cheap, tamper-evident
   governance object for any relaxation-based solver. This generalizes far beyond
   Sudoku (we demo it on GPU-scheduling graph coloring today).
3. **Honesty as differentiation:** we publish the 0/150 and 1/150 rows next to the
   13/150 one. In a market drowning in benchmark fraud, receipt-bound claims are the
   brand.

## Traction-ready surface (live after owner flips)

| Asset | Where |
| --- | --- |
| Open-source core (Apache-2.0) | github.com/szl-holdings/szl-cram |
| PyPI package `szl-cram` | pypi.org (trusted publishing, queued) |
| Edge API with free/pro tiers | Cloudflare Worker (deploy workflow, fail-closed) |
| Models + datasets | Hugging Face SZLHOLDINGS org |
| Interactive demo + pricing page | static frontend / HF Space |
| Publication draft | docs/PAPER.md (decoy-minima finding) |

## Monetization (see docs/MONETIZATION.md)

- **Free:** in-browser solver, 20 edge solves/day, open-source core. Funnel and credibility.
- **Pro API (private beta):** unmetered edge solves, learned-energy endpoints, batch +
  certificate export. API-key gated; billing owner-side (Stripe / x402).
- **Enterprise:** designed energies for your constraint domain (scheduling, allocation,
  placement), certificates wired into your governance stack. stephen@szlholdings.com.

## Risks we state plainly

- Sudoku is a wedge, not a market. The thesis is the certificate format and designed
  energies for industrial constraints; the wedge proves the engine and the honesty brand.
- The paper's learned-fit claims did not replicate at their headline numbers; our
  product leads with the hand-set energy and says so.
- Edge monetization depends on owner-side flips (Cloudflare secrets, PyPI publisher,
  payment rails). The estate is deliberately fail-closed: nothing ships silently.

## Estate context

SZL Holdings / a11oy: honest-minimal, receipt-bound, fail-closed. Every number in every
doc traces to a hash-committed receipt in `results/`. Doctrine v11.
