"""Quickstart: solve a puzzle with the hand-set energy and print its certificate.

Run:  python examples/quickstart.py
Requires: numpy  (pip install -e . from the repo root)
"""
import json

from szl_cram import CramSudoku, board_valid, make_certificate

PUZZLE = [
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
]  # 0 = empty; human digits 1-9, converted to model space 0-8 below

board = [v - 1 if v > 0 else -1 for v in PUZZLE]

m = CramSudoku(9, 3, seed=0)
m.set_hand()  # one unique weight value, zero trained parameters
print("model: hand-set energy (1 unique weight, 0 trained params)")

answers, quorum, E = m.search([board], chains=4, max_sweeps=200, seed=7)
answer = answers[0]
valid = board_valid(answer, 9, 3)

# hand-set threshold: a valid board has E=0; margin = threshold - E_final
THRESHOLD = 0.5
cert = make_certificate(m, answer, float(E.min()), THRESHOLD, valid,
                        sweeps=m.last_sweeps, model_id="handset-1w")
print(json.dumps(cert.to_dict(), indent=2))
print("valid:", cert.valid, "| margin:", round(cert.margin, 3))
