# SPDX-License-Identifier: Apache-2.0
"""Source-level guard for the organization-owned canonical PyPI release gate.

A byte mismatch is a release hold even when the Python AST is identical.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SHA256 = "6a8e0846b73c917f422f155fa28b10abefb277f44d15b5e757e4f4780be735ad"
SOURCE = ROOT / ".github" / "scripts" / "szl_pypi_release_gate.py"


def test_pypi_release_gate_matches_reviewed_canonical_bytes():
    raw = SOURCE.read_bytes()
    assert b"\r" not in raw, "PyPI gate must retain canonical LF line endings"
    assert sha256(raw).hexdigest() == CANONICAL_SHA256, (
        "canonical PyPI release gate changed; reconcile with szl-org-health"
    )


def test_gitattributes_prevents_reintroducing_crlf():
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "/.github/scripts/szl_pypi_release_gate.py text eol=lf" in rules.splitlines()
