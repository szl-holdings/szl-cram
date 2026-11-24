# szl-cram

**Classical energies that reason, with a certificate on every answer.**

`szl-cram` is a pure-numpy package, edge worker, and API that solve constraint-satisfaction
problems (Sudoku first) with a 1985-style Hopfield–Tank energy: **one hand-set weight,
zero trained parameters**, relaxation search, and a machine-checkable
**energy certificate** `{answer, E_final, threshold, margin, sweeps, valid}` on every answer.

It ships a **bounded, receipt-backed independent reproduction attempt** of
*CRAM: Classical Energies Learn to Reason 1,000× Faster than Looped Models*
(arXiv:2611.17309, Hoover/Blayney/Whittington), plus the estate's frontier extensions.

In an active Python 3.11 or newer virtual environment, install from source and run the demo.

```bash
git clone https://github.com/szl-holdings/szl-cram
cd szl-cram
python -m pip install .
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

- The authors report large compute reductions on selected Sudoku/Maze tasks using
  learned classical energies. **Our independent CPU/NumPy attempt did not reproduce
  their accuracy** (learned 1/150 on our different 150-board set). Its timing,
  hardware, puzzle distribution and search are not controlled matches to the
  original experiment; we cannot claim a SZL 1,000× speedup.
- Hand-set ≤3-weight energies saturate many benchmarks. **That is a benchmark critique,
  not a product claim** — we publish the failures with the wins (see the table below).
- The energy certificate is the durable idea: a machine-readable witness
  `{answer, E_final, threshold, margin, sweeps, valid}` for a relaxation-based
  solver. **A low energy or positive margin is not an independently checked
  correct Sudoku solution.** Require an external hard-constraint checker and a
  trusted source/receipt before promoting any solve. No certificate, no claim.

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
2. **Timing-method discrepancy:** the paper's "42 ms training" is reported as a
   warm-run in-process median compared with cold-start baselines. An estimated
   **2–5 s cold start** is from our historical analysis, not a matched rerun on
   the original hardware. We do not certify a cross-system speedup.
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
- **Cloudflare**: `deploy-cloudflare.yml` is a source-controlled deployment path;
  current live serving and owner configuration remain separate admission evidence.
  It must fail closed without authorized, scoped credentials.
- **Hugging Face**: models (`model100.npz`, `grid2_models.npz`) and datasets
  (`sudoku9.npz`, `sudoku16.npz`) live under the SZLHOLDINGS org; see `docs/DEPLOYMENT.md`.

## Doctrine

Honest-minimal, receipt-bound, fail-closed. BLOCKED is reported, never invented. If a
hand-set ≤3-weight energy saturates a benchmark, the benchmark is retired as scalability
evidence. BPTT is a last resort.

Apache-2.0. SZL Holdings / a11oy.
