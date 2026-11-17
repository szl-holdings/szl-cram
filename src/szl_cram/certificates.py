"""Energy certificates (the R3 pattern from the study).

Every answer a CRAM-style solver emits carries a signed-shape receipt:

    {answer, E_final, threshold, margin, sweeps}

``threshold`` is fitted from training-solution energies, so a margin > 0
means "at least as good as a training solution, probabilistically". The
certificate is cheap, tamper-evident in shape, and machine-checkable —
deny-by-default compatible confidence for governed systems.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from .core import peers_of


@dataclass
class EnergyCertificate:
    """Machine-checkable confidence receipt for one relaxation answer."""
    answer: list
    E_final: float
    threshold: Optional[float]
    margin: Optional[float]
    sweeps: int
    valid: bool
    model: str = "handset-1w"
    n: int = 9
    box: int = 3

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def board_valid(board, n: int, box: int) -> bool:
    """Check an n x n Sudoku board (values 0..n-1) against row/col/box rules."""
    board = list(board)
    N = n * n
    if len(board) != N:
        return False
    if any((v < 0) or (v >= n) for v in board):
        return False
    peers = peers_of(n, box)
    for i, v in enumerate(board):
        for j in peers[i]:
            if board[j] == v:
                return False
    return True


def make_certificate(model, answer, E_final, threshold, valid,
                     sweeps=None, model_id="cram") -> EnergyCertificate:
    """Build a certificate from a solver result.

    ``model`` is a CramSudoku instance; ``answer`` a single board (N ints);
    ``model_id`` a short honest label, e.g. "handset-1w" or "pl-fit".
    """
    ans = [int(v) for v in np.asarray(answer).reshape(-1)]
    margin = None if threshold is None else float(threshold - E_final)
    return EnergyCertificate(
        answer=ans,
        E_final=float(E_final),
        threshold=None if threshold is None else float(threshold),
        margin=margin,
        sweeps=int(model.last_sweeps if sweeps is None else sweeps),
        valid=bool(valid),
        model=model_id,
        n=model.n,
        box=model.box,
    )
