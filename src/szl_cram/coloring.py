"""Designed energies for graph coloring (the R5 demo, generalized).

Zero training: the energy is hand-set ("same color on an edge costs 1"),
the solver is restart relaxation, and every answer carries the same
{answer, E_final, threshold, margin, sweeps} certificate as the Sudoku engine.
This is the feasibility demo of the certificate pattern from the study
(r5_receipt.json: 32 containers / 8 GPUs / 94 edges -> E=0 in 0.01 s).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class ColorResult:
    assignment: dict
    E_final: float
    threshold: float
    margin: float
    sweeps: int
    conflicts: int
    n_colors: int

    def to_dict(self) -> dict:
        return asdict(self)


def _conflicts(assignment: dict, edges: Iterable) -> int:
    return sum(1 for u, v in edges if assignment.get(u) == assignment.get(v))


def color_graph(nodes, edges, n_colors: int, restarts: int = 50,
                max_sweeps: int = 200, seed: int = 0) -> ColorResult:
    """Color ``nodes`` with ``n_colors`` colors minimizing edge conflicts.

    Hand-set energy: each monochromatic edge costs 1. Restart relaxation:
    conflicted nodes move to the color that minimizes their local cost.
    Returns the best assignment found with its certificate fields.
    """
    rng = random.Random(seed)
    nodes = list(nodes)
    edges = [(int(u), int(v)) for u, v in edges]
    adj: dict[int, list[int]] = {u: [] for u in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    best = None
    best_E = float("inf")
    sweeps_total = 0
    for _ in range(restarts):
        assignment = {u: rng.randrange(n_colors) for u in nodes}
        for sweep in range(max_sweeps):
            sweeps_total += 1
            conflicted = [u for u in nodes
                          if any(assignment[u] == assignment[v] for v in adj[u])]
            if not conflicted:
                break
            u = rng.choice(conflicted)
            counts = [0] * n_colors
            for v in adj[u]:
                counts[assignment[v]] += 1
            mn = min(counts)
            assignment[u] = rng.choice([c for c in range(n_colors) if counts[c] == mn])
        E = _conflicts(assignment, edges)
        if E < best_E:
            best_E = E
            best = dict(assignment)
        if best_E == 0:
            break

    conflicts = _conflicts(best, edges)
    return ColorResult(
        assignment=best,
        E_final=float(best_E),
        threshold=0.5,
        margin=float(0.5 - best_E),
        sweeps=sweeps_total,
        conflicts=conflicts,
        n_colors=n_colors,
    )
