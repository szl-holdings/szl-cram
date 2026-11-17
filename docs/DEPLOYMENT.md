# Deployment & release runbook (szl-cram)

Estate rule: every automated path is fail-closed and receipt-bound. Owner flips are
explicit, 30-second actions. Nothing publishes silently.

## 1. PyPI (`szl-cram`) — canonical path szl.pypi-release.v2

The repo already carries the canonical workflow (`.github/workflows/publish-pypi.yml`,
edited only in `SZL_PACKAGES: "szl-cram=.:pypi"`) and the canonical gate
(`.github/scripts/szl_pypi_release_gate.py`, byte-identical across the estate).

Automated path (no tokens anywhere, by design):

1. Bump `[project].version` in `pyproject.toml` on main (PR).
2. `gh release create v<version>` (tag on main; org ruleset restricts `refs/tags/v*`
   to admins).
3. The release event triggers the workflow: build, gate (twine check, venv import,
   PEP 740 attestations), publish via GitHub OIDC trusted publishing.

Owner flips (one-time, ~30 seconds each):

1. **pypi.org → account settings → Publishing → add pending publisher:**
   - PyPI project name: `szl-cram`
   - Owner: `szl-holdings`  Repository: `szl-cram`
   - Workflow: `publish-pypi.yml`  Environment: `pypi`
2. **GitHub org ruleset:** ensure tag ruleset `refs/tags/v*` admin-only exists
   (already estate-standard).
3. **Manifest:** add a `"queued"` entry for `szl-cram` to
   `szl-org-health/governance/pypi/pypi-packages.v1.json` (PR to the private repo).

Until flip 1 is done, the workflow fails closed with "skip-existing: false" errors on
first publish. That is intended.

## 2. Cloudflare Worker (`szl-cram-solver`)

Deploy workflow: `.github/workflows/deploy-cloudflare.yml` — tests first, deploys on
main pushes, **fails closed** until the owner adds two repo-level secrets:

1. Cloudflare dashboard → create an API token scoped to Workers Scripts (account-level
   or repo-scoped per estate pattern).
2. GitHub → szl-holdings/szl-cram → Settings → Secrets and variables → Actions →
   add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
3. Re-run the workflow (or push to main). It deploys
   `worker/wrangler.jsonc` (`szl-cram-solver`) via SHA-pinned
   `cloudflare/wrangler-action@9acf94ace14e7dc412b076f2c5c20b8ce93c79cd` and reads
   back the deployed revision.

Worker routes: `/solve` (free tier: in-memory 20/day/IP), `/solve` with
`x-szl-api-key` from `PRO_API_KEYS` secret (pro tier, unmetered), `/usage`, `/pricing`,
`/health`. Static demo (`frontend/index.html`) is served by the worker at `/` when
wired (assets binding or inline; see worker/src/index.mjs).

## 3. Hugging Face (SZLHOLDINGS org)

- `SZLHOLDINGS/szl-cram` (model repo): `model100.npz`, `grid2_models.npz` + model card
  with the honest numbers table.
- `SZLHOLDINGS/cram-sudoku-extreme` (dataset): `sudoku9.npz` (150 extreme boards),
  `sudoku16.npz` (250/30 16×16 boards) + dataset card.
- `SZLHOLDINGS/szl-cram-demo` (static Space): `frontend/index.html` + README with
  `sdk: static`.
- Binary uploads go through `huggingface_hub` on the owner laptop (authenticated);
  text cards can go through the HF connector. Never touch `frontier-54-runs`.

## 4. FastAPI backend (optional self-host)

```bash
cd backend && pip install -r requirements.txt && uvicorn app:app --port 8000
```

`POST /solve` accepts a board (81 ints, 0 or -1 = empty; 0-8 digits), returns the
certificate and quorum-certificate JSON. Stateless; put it behind any gateway.

## 5. Versioning

SemVer. v0.1.0 = study engine + CLI + worker + backend + frontend + docs. Later minors:
16×16 endpoints, learned-energy endpoints, certificate verification helper library.
