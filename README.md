# szl-cram

**Classical energies that reason, with a certificate on every answer.**

`szl-cram` is a pure-numpy package, edge worker, and API that solve constraint-satisfaction
problems (Sudoku first) with a 1985-style Hopfield–Tank energy: **one hand-set weight,
zero trained parameters**, relaxation search, and a machine-checkable
**energy certificate** `{answer, E_final, threshold, margin, sweeps, valid}` on every answer.

It ships with a full, receipt-bound audit and reproduction of
*CRAM: Classical Energies Learn to Reason 1,000× Faster than Looped Models*
(arXiv:2611.17309, Hoover/Blayney/Whittington), plus the estate's frontier extensions.

```bash
pip install szl-cram
python -c "from szl_cram import CramSudoku; m=CramSudoku(9,3); m.set_hand(); print('ready: 1 unique weight, 0 trained params')"
szl-cram demo
```

## What's in the box

| Layer | Path | What it is |
| --- | --- | --- |
| Python package | `src/szl_cram/` | Core energy, PL training, focused-Gibbs + MFA v2 search, certificates, graph coloring (R5 generalization), data gen. Pure numpy, Python ≥3.11 |
| CLI | `szl-cram` | `demo` / `solve` / `fit` / `color` |
| Edge worker | `worker/` | Cloudflare Worker: solves at the edge, free/pro tier metering, zero cold-start |
| API backend | `backend/` | FastAPI `/solve` with certificate + quorum output |
| Frontend | `frontend/index.html` | Dependency-free static demo (also the HF Space content) |
| Tests | `tests/`, `worker/test/` | Deterministic, fast; no network |
| Docs | `docs/` | Paper draft, investor brief, deployment, monetization |
| Results | `results/` | Receipted audit artifacts (JSON, hash-committed) |

## Why it matters (30-second version)

- Benchmarked "reasoning" on Sudoku-class tasks is constraint propagation. A
  Hopfield-style dynamical system captures it with **3–4 orders of magnitude less
  compute** than looped Transformers. Our audit confirms the paper's core claim.
- Hand-set ≤3-weight energies saturate many benchmarks. **That is a benchmark critique,
  not a product claim** — we publish the failures with the wins (see the table below).
- The energy certificate is the durable idea: a cheap, verifiable proof object
  `{answer, E_final, threshold, margin}` for any relaxation-based solver. Designed for
  deny-by-default governance: no certificate, no commit.

## Honest results (receipted, all in `results/`)

150 extreme boards (24–27 clues, unique solutions), pure numpy, laptop CPU:

| Configuration | Exact | Time |
| --- | --- | --- |
| Learned energy, paper-style search | 0/150 | 234 s |
| Learned energy, our MFA v2 (β-anneal + quorum + polish) | 1/150 (0.67%) | 695 s |
| Hand-set energy, MFA v2 | **13/150 (8.67%)** | 1032 s |
| Hand-set energy, focused Gibbs | 4/150 (2.67%) | 98 s |
| 16×16 scale probe, learned / hand-set | 0/30 · 1/30 | ~430 s |

Key audit findings (full trail in `STUDY.md`):

1. **The math is sound.** Energy, pseudolikelihood gradients, and baselines all verify
   line-by-line against the paper.
2. **One real audit hit:** the paper's "42 ms training" is a warm-run in-process median
   compared against cold-start baselines. Honest cold-start is ~2–5 s. The speedup's
   *order of magnitude* survives; the headline number does not.
3. **Solutions are strong attractors** (diag8: 78/80 perturbed solutions relax back), but
   **positives-only PL fits create spurious deep minima at solution depth** (diag9) with
   off-diagonal reward entries (interp9: min off-diagonal cost **−0.5506**). This decoy
   substrate is why learned fits underperform the hand-set energy at the extreme end, and
   is the subject of our paper draft (`docs/PAPER.md`).

## Quick start (developer)

```bash
git clone https://github.com/szl-holdings/szl-cram
cd szl-cram
pip install -e . pytest
python -m pytest tests/ -v
szl-cram demo
```

```python
from szl_cram import CramSudoku, board_valid, make_certificate

m = CramSudoku(9, 3, seed=0)
m.set_hand()                                 # one weight, zero training
answers, quorum, E = m.search([puzzle], chains=4, max_sweeps=200, seed=7)
valid = board_valid(answers[0], 9, 3)
cert = make_certificate(m, answers[0], float(E.min()), 0.5, valid,
                        model_id="handset-1w")
print(cert.to_json(indent=2))   # {answer, E_final, threshold, margin, sweeps, valid, ...}
```

Run the edge solver locally:

```bash
cd worker && npm install && npm test && npx wrangler dev
```

## Deployment & release (owner flips)

- **PyPI**: canonical estate path — bump version on main, tag `v<version>` + release;
  `publish-pypi.yml` builds, gates (`szl_pypi_release_gate.py`), and publishes via
  GitHub OIDC trusted publishing with PEP 740 attestations. See `docs/DEPLOYMENT.md`.
- **Cloudflare**: `deploy-cloudflare.yml` tests then deploys the worker; fails closed
  until the owner adds `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID` repo secrets.
- **Hugging Face**: models (`model100.npz`, `grid2_models.npz`) and datasets
  (`sudoku9.npz`, `sudoku16.npz`) live under the SZLHOLDINGS org; see `docs/DEPLOYMENT.md`.

## Doctrine

Honest-minimal, receipt-bound, fail-closed. BLOCKED is reported, never invented. If a
hand-set ≤3-weight energy saturates a benchmark, the benchmark is retired as scalability
evidence. BPTT is a last resort.

Apache-2.0. SZL Holdings / a11oy.
