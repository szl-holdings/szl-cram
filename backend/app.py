"""FastAPI service: solve Sudoku with a classical energy + certificate.

Run:  uvicorn backend.app:app --host 127.0.0.1 --port 8000

POST /solve   {"puzzle": [81 ints, -1 for empty], "mode": "handset"|"learned",
               "model": "path or HF id (learned only)"}
GET  /health
Every answer carries the energy certificate {answer, E_final, threshold,
margin, sweeps}. No BPTT anywhere in this stack: training is pseudolikelihood
on solved boards; inference is relaxation.
"""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from szl_cram import CramSudoku, board_valid, make_certificate

app = FastAPI(title="szl-cram solver", version="0.1.0",
              description="Classical energy reasoning with certificates (audit of arXiv:2611.17309).")

_HAND = CramSudoku(9, 3, seed=0)
_HAND.set_hand(1.0)

_LEARNED_PATH = os.environ.get("SZL_CRAM_MODEL")
_LEARNED = None
_LEARNED_THR = None
if _LEARNED_PATH and os.path.exists(_LEARNED_PATH):
    z = np.load(_LEARNED_PATH)
    _LEARNED = CramSudoku(9, 3, seed=0)
    _LEARNED.W = z["W"].astype(np.float32)
    _LEARNED.B = z["B"].astype(np.float32)
    _LEARNED_THR = float(z["threshold"]) if "threshold" in z.files else None


class SolveRequest(BaseModel):
    puzzle: list
    mode: str = "handset"
    chains: int = 4
    max_sweeps: int = 200
    seed: int = 0

    @field_validator("puzzle")
    @classmethod
    def _check_puzzle(cls, v):
        if len(v) != 81:
            raise ValueError("puzzle must be exactly 81 cells")
        if not all(isinstance(x, int) and -1 <= x <= 8 for x in v):
            raise ValueError("cells must be ints in -1..8 (-1 = empty)")
        return v


@app.get("/health")
def health():
    return {"ok": True, "learned_model_loaded": _LEARNED is not None,
            "handset_ready": True}


@app.post("/solve")
def solve(req: SolveRequest):
    if req.mode == "learned" and _LEARNED is None:
        raise HTTPException(503, "no learned model configured (set SZL_CRAM_MODEL)")
    model = _HAND if req.mode != "learned" else _LEARNED
    thr = None if req.mode != "learned" else _LEARNED_THR
    P = np.asarray([req.puzzle], dtype=np.int64)
    answers, quorum, E = model.search(
        P, chains=max(1, req.chains), max_sweeps=max(10, min(1000, req.max_sweeps)),
        seed=req.seed, threshold=thr)
    ok = board_valid(answers[0], 9, 3)
    cert = make_certificate(
        model, answers[0], float(E.min()), thr, ok,
        model_id=("handset-1w" if req.mode != "learned" else "pl-fit"))
    return {"certificate": cert.to_dict(),
            "quorum_certificate": None if quorum is None else
            make_certificate(model, quorum[0], float(E.min()), thr,
                             board_valid(quorum[0], 9, 3),
                             model_id="handset-1w-quorum").to_dict()}
