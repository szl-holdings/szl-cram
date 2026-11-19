# SZL × TufaZip × CRAM — source-owned synthetic dataset intake v0

**Authority:** Research proposal only. Parent protected `szl-cram/main@3f8894caad893961167af39c7f3d9f40383fe7cf`. This adds an opt-in offline data-intake check, **not** a trainer, model, additional energy solver, deployment, new proof, or Hugging Face publication.

## Upstream methods

* **Tufa Labs / Tommy He — TufaZip:** next-symbol probabilities and arithmetic encoding, mirrored online adaptation, replay, compact Cycle2 offline priors. The benchmark **105,388,353 bytes** includes model and decoder, not just payload. Exact source `Tufalabs/tufazip@b9d678db3a0f726e117c8cab51927975830395ac`, MIT code. The actual `enwik9` benchmark bytes and weights are **not included or copied**. Its published decoder requires 2 B200 GPUs, CUDA 13 and Python 3.12.3; no attempt to reproduce the full benchmark.
* **Prefrontal-AI — CRAM (Hoover / Blayney / Whittington):** arXiv:2611.17309, explicit Hopfield–Tank energies, pseudolikelihood and energy-aware early stopping for finite puzzles. Exact `Prefrontal-AI/CRAM@62bbe830c7d6f87f2ba18f30a28039258b986454`, Apache-2.0 code. No official datasets or model checkpoints are imported.
* **SZL original study:** `szl-cram/main@3f8894caad893961167af39c7f3d9f40383fe7cf` already contains a separate CPU NumPy replication, including hard-negative / spurious-minima observations and honest 1/150 learned-board result on its own 150-board test. Do not duplicate it or rewrite its published evidence.

## The new original synthetic dataset (generated outside GitHub)

The conversation-provided **SZL_TufaZip_CRAM_Frontier_ProvingGround_v0.zip** contains:
* `data-v1/train.jsonl`: **1,320** finite labeled train rows.
* `data-v1/train_sft.jsonl`: **1,320** direct messages-format SFT examples; **train-only**.
* `data-v1/dev.jsonl`: **310** held-out rows.
* `data-v1/test.jsonl`: **290** held-out rows.
* Independent generator/verifier, 20 offline stdlib unit tests, source/licensing catalogue, manifest/digests and synthetic CPU compression probe.

Three task families: (i) 3-color graph edge-conflict energy, (ii) ideal Shannon bit cost under a declared `(1/2,1/4,1/8,1/8)` distribution, (iii) model-transfer **complete artifact** accounting (`payload+decoder+prior`).

Split by source family, not individual rows. For graphs, conservative 1-WL degree/neighbor fingerprints prevent isomorphic structural graphs being sent to separate splits; they may over-group nonisomorphic graphs. **Synthetic finite software arithmetic, not model accuracy or general theorem.**

### Exact intake identity

| Artifact | SHA-256 |
| --- | --- |
| `data-v1/manifest.json` | `d8cb77a517e69ded7314747df5e68db3481aaba3639a8529ed1e3eb47c448904` |
| `data-v1/train_sft.jsonl` | `36048599efb5abc1b4ca0ef21ac1eae072c7412728f1dacd9cbe437b623df502` |
| Full chat proving-ground ZIP | `98820ec43494f4cec2c02234a9d1c2bb8e874d9f67dd6f105443206f732137a0` |
| Train-only chat ZIP | `4216b3e1130d3cba49f05fe9456d9e2ee936a9eb87ab80639698f0b58caa3c4d` |
| Unsigned review receipt | `a160372984702b209f903d996122139d9b6883c641d147de81b1c2915ede9cbc` |

The ZIPs are **conversation artifacts, not GitHub/HF releases**; do not reference them as externally published URLs. No source/weights from TufaZip or CRAM official corpora were copied into these rows.

## Offline review command

Unpack the train-only ZIP into a private owner-reviewed work directory. The ZIP contains `data-v1/manifest.json` and `data-v1/train_sft.jsonl`. From the owning repo, run:

```sh
python -m szl_cram.reasoning_compression_intake --bundle /absolute/path/to/unpacked/data-v1
```

Only on explicit source-owner approval, optionally **materialize local source-verified SFT input** without starting a trainer:

```sh
python -m szl_cram.reasoning_compression_intake \
  --bundle /absolute/path/to/unpacked/data-v1 \
  --export /private/qualified-inputs/szl-reasoning-compression-v0.jsonl \
  --owner-admitted-export
```

The export is create-only; an existing file is never overwritten. **This is not permission to start a paid job, modify production, or publish a model.** Training/production budgets, base-model revision, dataset licensing, checkpoint durability, installed toolchain, signed provenance and external evaluation each require distinct admission.

## Adaptation gates

1. **TufaZip lesson:** log actual bits-per-token and *full* artifact size, not payload-only wins. A trained prior only helps if its stored weights plus decoder add less than the saved payload. Avoid trying the 2×B200, 7-day enwik9 decoder on the A6000 fleet as a trivial canary.
2. **CRAM lesson:** couple an exact checker to candidate energy, calibrate abstention/halting on held-out cases. Investigate **margin-based hard negatives** against the existing positives-only pseudolikelihood and hand-set energy without claiming broad reasoning improvement from Sudoku/Maze.
3. **SZL invariants:** `szl-formulas` owns formula identities, `lutar-lean` owns formal machine-checked proof, `szl-gpu-bridge` owns Salad/Massed scheduling, `szl-papers` owns scholarly claims, and `szl-cram` owns this finite solver's experimental code. Hashes in this intake are **not** DSSE signatures; `Lambda` is an advisory conjecture, not authority.
4. **Validation:** compare frozen baseline/candidate on 310 dev and 290 untouched test rows; report per-domain exact-match and coverage, error/abstention, false-green rates and hardware/energy costs independently. Do not train on dev/test or externally copyrighted benchmark text.

**Status:** 20/20 standalone independent Python tests passed in the conversation's isolated runtime; this PR adds an additional 16 source-only loader tests that require exact-branch CI before claiming them passed. No model fit, third-party data ingestion, WSL repair, Salad/HF job, signed receipt or production deployment.
