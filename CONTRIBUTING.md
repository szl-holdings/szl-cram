# Contributing to szl-cram

Estate rules, in order:

1. **Honest-minimal.** Claims are receipt-bound: every number in docs, README, and the
   frontend traces to a JSON artifact in `results/` (hash-committed). Never publish a
   number you cannot trace. `BLOCKED` beats invented.
2. **Fail-closed.** Deploy/publish workflows fail loudly on missing owner flips; never
   "helpfully" bypass a gate (PyPI trusted publishing, Cloudflare secrets, tag rulesets).
3. **Tests are deterministic and fast.** `python -m pytest tests/ -v` (CPU, no network,
   no fixtures to download) and `cd worker && npm test`. Both must pass before merge.
4. **No heavy deps.** Core is numpy-only. Backend extras live in `backend/requirements.txt`;
   worker deps in `worker/package.json` (wrangler locked).
5. **Don't break the study surface.** `STUDY.md`, `receipt.json`, and `results/` are
   audit artifacts; changes to them require a new receipt, not an edit.
6. **Certificate format is a contract.** `{answer, E_final, threshold, margin, sweeps,
   valid, model, n, box}` — additive changes only.
7. **Canonical files are immutable:** `.github/scripts/szl_pypi_release_gate.py` is
   byte-identical across the estate; `.github/workflows/publish-pypi.yml` may only have
   its `SZL_PACKAGES` line edited.

PRs to `main` (protected). For scientific-result changes, include the receipt JSON.
Apache-2.0. SZL Holdings / a11oy.
