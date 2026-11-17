"""szl-cram: classical energy reasoning with certificates.

A pure-numpy reimplementation and audit of CRAM (arXiv:2611.17309):
Hopfield-Tank-style pairwise constraint energies for Sudoku-class reasoning,
trained directly from solved boards by pseudolikelihood (no BPTT, no solver in
the training loop), solved at inference by relaxation search, and every answer
ships a machine-checkable energy certificate {answer, E_final, threshold,
margin, sweeps}.

Honest scope: completion-of-corrupted-pattern problems (Sudoku 9x9 and 16x16,
graph coloring by designed energy). Not a general reasoner. The study receipts
and the audit trail live in this repository (results/ and docs/PAPER.md).
"""
from .core import CramSudoku, peers_of
from .certificates import EnergyCertificate, make_certificate, board_valid
from .coloring import color_graph
from .data import generate_solutions, dig_puzzle, make_dataset, load_dataset

__version__ = "0.1.0"
__all__ = [
    "CramSudoku", "peers_of", "EnergyCertificate", "make_certificate",
    "board_valid", "color_graph", "generate_solutions", "dig_puzzle",
    "make_dataset", "load_dataset", "__version__",
]
