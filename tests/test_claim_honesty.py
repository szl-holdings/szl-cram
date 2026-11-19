"""Offline safeguards against overstating CRAM reproduction or energy proofs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
FRONTEND = ROOT / "frontend" / "index.html"
STUDY = ROOT / "STUDY.md"


def test_readme_reports_unmatched_accuracy_without_claiming_speedup():
    body = README.read_text(encoding="utf-8")
    assert "1/150 (0.67%)" in body
    assert "13/150 (8.67%)" in body
    assert "did not reproduce" in body
    assert "not controlled matches" in body
    assert "cannot claim a SZL 1,000× speedup" in body
    assert "a source-backed full reproduction" not in body


def test_frontend_discloses_separate_corpora_and_failure():
    html = FRONTEND.read_text(encoding="utf-8")
    assert "1/150 (0.67%)" in html
    assert "13/150 (8.67%)" in html
    assert "did not reproduce" in html
    assert "not a matched head-to-head comparison" in html
    assert "not author-equivalent" in html
    assert "our audit confirms it" not in html
    assert "full audit and reproduction" not in html


def test_energy_margin_not_admitted_as_correctness():
    readme = README.read_text(encoding="utf-8").lower()
    frontend = FRONTEND.read_text(encoding="utf-8").lower()
    assert "positive margin is not an independently checked" in readme
    assert "correct sudoku solution" in readme
    assert "positive energy margin is" in frontend
    assert "not</b> proof of a correct solution" in frontend
    assert "independent sudoku constraint validation" in frontend


def test_retained_historical_failures_are_not_rewritten():
    study = STUDY.read_text(encoding="utf-8")
    assert "0/150" in study
    assert "1/150" in study
    assert "13/150" in study
    assert "99.996%" in study


def test_frontend_uses_source_and_not_undocumented_new_runtime():
    body = FRONTEND.read_text(encoding="utf-8")
    assert "https://github.com/szl-holdings/szl-cram" in body
    assert "independent Sudoku constraint validation" in body
    assert "experiment figures" in body
