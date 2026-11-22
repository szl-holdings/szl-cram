"""Offline, standard-library regression for Sudoku certificate validity claims."""
import unittest
from types import SimpleNamespace

from szl_cram.certificates import board_valid, make_certificate


class CertificateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(n=9, box=3, last_sweeps=2)
        self.solved = [(row * 3 + row // 3 + col) % 9
                       for row in range(9) for col in range(9)]
        self.assertTrue(board_valid(self.solved, 9, 3))

    def certificate(self, answer, claimed_valid):
        return make_certificate(self.model, answer, -1.0, 0.0, claimed_valid)

    def test_rejects_caller_true_for_duplicate_sudoku_digit(self):
        corrupted = self.solved.copy()
        corrupted[0] = corrupted[1]
        self.assertFalse(board_valid(corrupted, 9, 3))
        self.assertFalse(self.certificate(corrupted, True).valid)

    def test_accepts_valid_board_when_caller_also_says_true(self):
        self.assertTrue(self.certificate(self.solved, True).valid)

    def test_does_not_promote_caller_false_to_true(self):
        self.assertFalse(self.certificate(self.solved, False).valid)


if __name__ == "__main__":
    unittest.main()
