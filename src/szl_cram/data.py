"""Dataset generation and loading for Sudoku-class problems.

The study datasets (sudoku9.npz: 600 train / 150 extreme test dug to unique
solutions, 24-27 clues; sudoku16.npz: 250 train / 30 test) are mirrored at
https://huggingface.co/datasets/SZLHOLDINGS/cram-sudoku-extreme. This module
can regenerate equivalents from scratch with bounded, seeded randomness.
"""
from __future__ import annotations

import random

import numpy as np


def _peers(n: int, box: int) -> list[set[int]]:
    from .core import peers_of
    return [set(p) for p in peers_of(n, box)]


def _solve_count(board, n: int, box: int, limit: int = 2) -> int:
    """Count solutions of a puzzle (early exit at ``limit``)."""
    N = n * n
    peers = _peers(n, box)
    b = list(board)
    empties = [i for i in range(N) if b[i] < 0]

    def bt(k: int) -> int:
        if k == len(empties):
            return 1
        i = empties[k]
        used = {b[j] for j in peers[i] if b[j] >= 0}
        total = 0
        for v in range(n):
            if v not in used:
                b[i] = v
                total += bt(k + 1)
                b[i] = -1
                if total >= limit:
                    return total
        return 0

    return bt(0)


def generate_solutions(n: int, box: int, count: int, seed: int = 0) -> np.ndarray:
    """Generate ``count`` solved boards by randomized backtracking fill."""
    rng = random.Random(seed)
    N = n * n
    peers = _peers(n, box)
    out = []
    while len(out) < count:
        b = [-1] * N
        # Row-major fill: scattered random orders can hit catastrophic
        # backtracking blowups (bad early assignment, detected only deep).
        order = list(range(N))

        def place(k: int) -> bool:
            if k == N:
                return True
            i = order[k]
            used = {b[j] for j in peers[i] if b[j] >= 0}
            cand = [v for v in range(n) if v not in used]
            rng.shuffle(cand)
            for v in cand:
                b[i] = v
                if place(k + 1):
                    return True
                b[i] = -1
            return False

        if place(0):
            out.append(list(b))
    return np.asarray(out, dtype=np.int64)


def dig_puzzle(solution, n: int, box: int, target_clues: int, seed: int = 0) -> list:
    """Dig a puzzle down to ``target_clues`` clues, keeping the solution unique."""
    rng = random.Random(seed)
    puzzle = list(solution)
    order = list(range(n * n))
    rng.shuffle(order)
    for i in order:
        if sum(v >= 0 for v in puzzle) <= target_clues:
            break
        saved = puzzle[i]
        puzzle[i] = -1
        if _solve_count(puzzle, n, box, limit=2) != 1:
            puzzle[i] = saved
    return puzzle


def make_dataset(n: int = 9, box: int = 3, n_train: int = 600, n_test: int = 150,
                 target_clues: int = 25, seed: int = 0) -> dict:
    """Generate a train/test dataset of unique-solution puzzles."""
    sols = generate_solutions(n, box, n_train + n_test, seed=seed)
    train_sols = sols[:n_train]
    test_sols = sols[n_train:]
    test_puzzles = [dig_puzzle(s, n, box, target_clues, seed=seed + i)
                    for i, s in enumerate(test_sols)]
    return {
        "train_sol": train_sols,
        "test_puz": np.asarray(test_puzzles, dtype=np.int64),
        "test_sol": test_sols,
    }


def load_dataset(path: str) -> dict:
    """Load a .npz dataset produced by this module or mirrored from HF."""
    z = np.load(path)
    return {k: z[k] for k in z.files}
