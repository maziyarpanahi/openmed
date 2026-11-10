"""Offline tests for ``scripts/docs/check_regulatory_claims.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "docs" / "check_regulatory_claims.py"


def _load_lint() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_regulatory_claims", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lint = _load_lint()


def _lint_page(root: Path, text: str) -> int:
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "page.md").write_text(text, encoding="utf-8")
    return lint.main(["--root", str(root)])


def test_fda_clearance_claim_fails_with_file_and_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "# Page\n\nOpenMed is FDA-cleared for diagnosis.\n"

    assert _lint_page(tmp_path, text) == 1
    assert "docs/page.md:3: FDA-cleared:" in capsys.readouterr().out


def test_negated_disclaimers_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "This software is not a medical device.\nIt is not FDA-cleared.\n"

    assert _lint_page(tmp_path, text) == 0
    assert "OK" in capsys.readouterr().out


def test_ce_marked_claim_survives_a_nearby_negation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "Not a medical device, but the kit is CE marked.\n"

    assert _lint_page(tmp_path, text) == 1
    assert "docs/page.md:1: CE-marked:" in capsys.readouterr().out


def test_negation_does_not_cover_a_separate_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "It is not a medical device and is FDA-cleared.\n"

    assert _lint_page(tmp_path, text) == 1
    assert "docs/page.md:1: FDA-cleared:" in capsys.readouterr().out


def test_current_repository_docs_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert lint.main(["--root", str(ROOT)]) == 0
    assert "OK" in capsys.readouterr().out
