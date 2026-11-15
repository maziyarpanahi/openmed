"""Offline reference and coverage gates for the multimodal threat model."""

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs/security/threat-model-multimodal-ambient.md"
SOURCE_BASE = "https://github.com/maziyarpanahi/openmed/blob/master/"
ISSUE_LINK = re.compile(r"https://github\.com/maziyarpanahi/openmed/issues/[1-9]\d*")


def _references(markdown: str) -> set[str]:
    inline = re.findall(r"`((?:openmed|swift|tests)/[^`\s]+)`", markdown)
    links = re.findall(re.escape(SOURCE_BASE) + r"([^\s)]+)", markdown)
    return {unquote(path).split("#", 1)[0] for path in (*inline, *links)}


def _validate_references(markdown: str, root: Path) -> None:
    references = _references(markdown)
    assert references, "threat model must reference enforcing code and tests"
    for reference in sorted(references):
        path = (root / reference).resolve()
        assert path.is_relative_to(root.resolve()), "reference escapes repository"
        assert path.is_file(), f"dangling threat-model reference: {reference}"


def _validate_rows(markdown: str) -> None:
    rows = re.findall(r"^\| MA-\d+ \|.*$", markdown, re.MULTILINE)
    assert rows, "threat matrix is empty"
    identifiers = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 4, "threat row must follow the documented schema"
        identifiers.append(cells[0])
        references = _references(cells[3])
        module = any(path.startswith(("openmed/", "swift/")) for path in references)
        test = any(path.startswith("tests/") for path in references)
        gap = ISSUE_LINK.search(cells[3]) is not None
        assert (module and test) or gap, f"unmapped threat: {cells[0]}"
    assert len(set(identifiers)) == len(identifiers), "duplicate threat identifier"


def test_multimodal_threat_model_references_and_rows() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    _validate_references(markdown, ROOT)
    _validate_rows(markdown)
    for category in (
        "Spoofing",
        "Tampering",
        "Repudiation",
        "Information disclosure",
        "Denial of service",
        "Elevation of privilege",
    ):
        assert re.search(rf"^\| MA-\d+ \|[^|]*{category}", markdown, re.MULTILINE)


@pytest.mark.parametrize("prefix", ["openmed", "swift", "tests/unit"])
@pytest.mark.parametrize("style", ["inline", "link"])
def test_dangling_module_or_test_reference_fails(
    tmp_path: Path, prefix: str, style: str
) -> None:
    reference = f"{prefix}/missing.py"
    markdown = (
        f"`{reference}`"
        if style == "inline"
        else f"[control]({SOURCE_BASE}{reference})"
    )
    with pytest.raises(AssertionError, match="dangling threat-model reference"):
        _validate_references(markdown, tmp_path)


@pytest.mark.parametrize(
    "control",
    [
        "No mitigation yet",
        "`openmed/example.py`",
        "`tests/unit/test_example.py`",
        "Residual risk accepted",
        "[#1](https://github.com/another/project/issues/1)",
    ],
)
def test_threat_row_without_pair_or_tracked_gap_fails(control: str) -> None:
    with pytest.raises(AssertionError, match="unmapped threat"):
        _validate_rows(f"| MA-01 | Tampering | Synthetic threat | {control} |")


@pytest.mark.parametrize(
    "control",
    [
        "`openmed/example.py` and `tests/unit/test_example.py`",
        "[#1](https://github.com/maziyarpanahi/openmed/issues/1)",
    ],
)
def test_synthetic_control_pair_or_gap_is_accepted(control: str) -> None:
    _validate_rows(f"| MA-01 | Tampering | Synthetic threat | {control} |")


def test_reference_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="reference escapes repository"):
        _validate_references("`openmed/../../outside.py`", tmp_path)


def test_empty_mapping_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="reference enforcing code"):
        _validate_references("No references", tmp_path)
    with pytest.raises(AssertionError, match="threat matrix is empty"):
        _validate_rows("No rows")
