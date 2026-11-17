"""Fast, deterministic tests for the szl-cram package (no network, CPU only)."""
import json

import numpy as np

from szl_cram import (
    CramSudoku, board_valid, color_graph, dig_puzzle,
    generate_solutions, make_certificate,
)


def test_handset_solves_easy_puzzle():
    sol = generate_solutions(9, 3, 1, seed=7)[0]
    puzzle = dig_puzzle(sol, 9, 3, target_clues=40, seed=7)
    m = CramSudoku(9, 3, seed=0)
    m.set_hand(1.0)
    answers, quorum, E = m.search([puzzle], chains=4, max_sweeps=200, seed=0)
    assert board_valid(answers[0], 9, 3)
    assert np.array_equal(answers[0], sol)
    assert m.last_sweeps >= 1


def test_certificate_shape():
    sol = generate_solutions(9, 3, 1, seed=3)[0]
    m = CramSudoku(9, 3, seed=0)
    m.set_hand(1.0)
    answers, _, E = m.search([sol], chains=2, max_sweeps=10, seed=0)
    cert = make_certificate(m, answers[0], float(E.min()), 0.5,
                           board_valid(answers[0], 9, 3), model_id="handset-1w")
    d = json.loads(cert.to_json())
    for key in ("answer", "E_final", "threshold", "margin", "sweeps", "valid"):
        assert key in d
    assert len(d["answer"]) == 81
    assert d["model"] == "handset-1w"


def test_pl_fit_moves_weights():
    sols = generate_solutions(9, 3, 20, seed=11)
    m = CramSudoku(9, 3, seed=0)
    before = float(np.abs(m.W).sum())
    m.train_pl(sols, epochs=5, bs=8)
    after = float(np.abs(m.W).sum())
    assert after > before
    thr, mean, std = m.fit_stopping_threshold(sols)
    assert thr > mean


def test_board_valid_rejects_bad_board():
    sol = generate_solutions(9, 3, 1, seed=5)[0]
    assert board_valid(sol, 9, 3)
    bad = list(sol)
    bad[0] = bad[1]
    assert not board_valid(bad, 9, 3)


def test_coloring_finds_conflict_free():
    rng = np.random.default_rng(0)
    nodes = list(range(16))
    edges = set()
    while len(edges) < 20:
        u, v = rng.choice(nodes, 2, replace=False)
        edges.add((int(u), int(v)) if u < v else (int(v), int(u)))
    res = color_graph(nodes, sorted(edges), 4, seed=0)
    assert res.conflicts == 0
    assert res.E_final == 0.0
    assert res.margin == 0.5
    assert res.assignment  # non-empty
