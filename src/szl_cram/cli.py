"""Command line: ``szl-cram demo|solve|fit|color``.

Examples:
    szl-cram demo
    szl-cram solve --puzzle "...3.5.." (81 chars, 0-8 or . for empty)
    szl-cram fit --train 600 --epochs 40 --out model9.npz
    szl-cram color --nodes 32 --colors 8 --edges 94
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .certificates import board_valid, make_certificate
from .coloring import color_graph
from .core import CramSudoku
from .data import generate_solutions, dig_puzzle


def _print_board(board, n: int):
    box = int(n ** 0.5)
    for r in range(n):
        if box > 1 and r % box == 0 and r:
            print("-" * (2 * n + box - 1))
        print(" ".join("." if v < 0 else str(v) for v in board[r * n:(r + 1) * n]))


def cmd_demo(args) -> int:
    m = CramSudoku(9, 3, seed=args.seed)
    m.set_hand(1.0)
    sol = generate_solutions(9, 3, 1, seed=args.seed)[0]
    puzzle = dig_puzzle(sol, 9, 3, target_clues=40, seed=args.seed)
    print("Puzzle (40 clues):")
    _print_board(puzzle, 9)
    answers, quorum, E = m.search([puzzle], chains=4, max_sweeps=200, seed=args.seed)
    ok = board_valid(answers[0], 9, 3)
    cert = make_certificate(m, answers[0], E.min(), None, ok, model_id="handset-1w")
    print("\nAnswer (valid=%s):" % ok)
    _print_board(cert.answer, 9)
    print("\nEnergy certificate:")
    print(cert.to_json(indent=2))
    print("\nexact_match:", bool(np.array_equal(np.asarray(cert.answer), sol)))
    return 0 if ok else 1


def cmd_solve(args) -> int:
    raw = args.puzzle.strip().replace(".", "0") if "." in args.puzzle else args.puzzle
    puzzle = []
    for ch in args.puzzle.strip():
        if ch in ".0":
            puzzle.append(-1)
        elif ch.isdigit():
            puzzle.append(int(ch) - 1)
        else:
            print("bad character in puzzle: %r" % ch, file=sys.stderr)
            return 2
    if len(puzzle) != 81:
        print("puzzle must be 81 cells (got %d)" % len(puzzle), file=sys.stderr)
        return 2
    m = CramSudoku(9, 3, seed=0)
    m.set_hand(1.0)
    thr = None
    if args.model:
        z = np.load(args.model)
        m.W = z["W"].astype(np.float32)
        m.B = z["B"].astype(np.float32)
        thr = float(z["threshold"]) if "threshold" in z.files else None
    answers, quorum, E = m.search([puzzle], chains=4, max_sweeps=200, seed=args.seed,
                                   threshold=thr)
    ok = board_valid(answers[0], 9, 3)
    cert = make_certificate(m, answers[0], float(E.min()), thr, ok,
                           model_id=("pl-fit" if args.model else "handset-1w"))
    print(cert.to_json(indent=2))
    return 0 if ok else 1


def cmd_fit(args) -> int:
    m = CramSudoku(9, 3, seed=args.seed)
    sols = generate_solutions(9, 3, args.train, seed=args.seed)
    sec = m.train_pl(sols, epochs=args.epochs)
    thr, mean, std = m.fit_stopping_threshold(sols)
    np.savez(args.out, W=m.W, B=m.B, threshold=thr)
    print(json.dumps({
        "trained_params": m.n_params(),
        "fit_sec": round(sec, 2),
        "threshold": thr,
        "train_solution_E_mean": mean,
        "train_solution_E_std": std,
        "out": args.out,
    }, indent=2))
    return 0


def cmd_color(args) -> int:
    rng = np.random.default_rng(args.seed)
    nodes = list(range(args.nodes))
    edges = set()
    while len(edges) < args.edges:
        u, v = rng.choice(nodes, 2, replace=False)
        edges.add((int(u), int(v)) if u < v else (int(v), int(u)))
    res = color_graph(nodes, sorted(edges), args.colors, seed=args.seed)
    print(json.dumps(res.to_dict(), indent=2))
    return 0 if res.conflicts == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="szl-cram",
                                 description="Classical energy reasoning with certificates")
    ap.add_argument("--version", action="version", version="szl-cram " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="generate + solve an easy puzzle, print certificate")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("solve", help="solve an 81-char puzzle")
    p.add_argument("--puzzle", required=True)
    p.add_argument("--model", default=None, help="npz with learned W/B/threshold")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_solve)
    p = sub.add_parser("fit", help="fit the PL energy on generated boards")
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default="model9.npz")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("color", help="graph coloring demo (designed energy)")
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--colors", type=int, default=8)
    p.add_argument("--edges", type=int, default=94)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_color)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
