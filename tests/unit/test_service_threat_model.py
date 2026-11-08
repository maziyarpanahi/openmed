"""Offline reference, coverage and sentinel gates for the service threat model."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from openmed.eval.service_release import load_service_p0_checklist, scan_for_sentinels

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs/security/service-threat-model.md"
GATE_PAGE = ROOT / "docs/release/v4.0-service-release-gate.md"
SOURCE_BASE = "https://github.com/maziyarpanahi/openmed/blob/master/"
ISSUE_LINK = re.compile(r"https://github\.com/maziyarpanahi/openmed/issues/([1-9]\d*)")
STRIDE = (
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Information disclosure",
    "Denial of service",
    "Elevation of privilege",
)
BOUNDARIES = tuple(f"SB{index}" for index in range(1, 9))


def _references(markdown: str) -> set[str]:
    inline = re.findall(r"`((?:openmed|tests|deploy|scripts|docs)/[^`\s]*)`", markdown)
    links = re.findall(re.escape(SOURCE_BASE) + r"([^\s)]+)", markdown)
    return {unquote(path).split("#", 1)[0] for path in (*inline, *links)}


def _validate_references(markdown: str, root: Path) -> None:
    references = _references(markdown)
    assert references, "threat model must reference enforcing code and tests"
    for reference in sorted(references):
        path = (root / reference).resolve()
        assert path.is_relative_to(root.resolve()), "reference escapes repository"
        exists = path.is_dir() if reference.endswith("/") else path.is_file()
        assert exists, f"dangling threat-model reference: {reference}"


def _rows(markdown: str) -> list[list[str]]:
    rows = re.findall(r"^\| SV-\d+ \|.*$", markdown, re.MULTILINE)
    assert rows, "threat matrix is empty"
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def _validate_rows(markdown: str) -> None:
    identifiers = []
    for cells in _rows(markdown):
        assert len(cells) == 4, "threat row must follow the documented schema"
        identifiers.append(cells[0])
        references = _references(cells[3])
        module = any(path.startswith(("openmed/", "deploy/")) for path in references)
        test = any(path.startswith("tests/") for path in references)
        gap = ISSUE_LINK.search(cells[3]) is not None
        assert (module and test) or gap, f"unmapped threat: {cells[0]}"
    assert len(set(identifiers)) == len(identifiers), "duplicate threat identifier"
    numbers = [int(identifier.split("-")[1]) for identifier in identifiers]
    assert numbers == list(range(1, len(numbers) + 1)), "threat ids must be dense"


def test_service_threat_model_references_resolve_and_rows_are_mapped() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    _validate_references(markdown, ROOT)
    _validate_rows(markdown)


def test_every_stride_category_and_boundary_is_covered() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    rows = _rows(markdown)
    for category in STRIDE:
        assert any(category in cells[1] for cells in rows), category
    for boundary in BOUNDARIES:
        assert re.search(rf"^\| {boundary}: ", markdown, re.MULTILINE), boundary
        assert any(boundary in _expand_ranges(cells[2]) for cells in rows), (
            f"no threat at {boundary}"
        )


def _expand_ranges(text: str) -> set[str]:
    found = set(re.findall(r"SB\d", text))
    for start, end in re.findall(r"SB(\d) to SB(\d)", text):
        found.update(f"SB{index}" for index in range(int(start), int(end) + 1))
    return found


def test_every_checklist_p0_is_mapped_to_a_threat_row() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    linked = {
        f"#{number}"
        for cells in _rows(markdown)
        for number in ISSUE_LINK.findall(cells[3])
    }
    # The release-evidence P0 is enforced by the gate rows rather than a gap.
    gate_rows = [cells for cells in _rows(markdown) if "service_release.py" in cells[3]]
    assert gate_rows
    for item in load_service_p0_checklist()["items"]:
        if item["issue"] == "#3603":
            continue
        assert item["issue"] in linked, f"P0 without a threat row: {item['issue']}"


def test_threat_model_and_gate_page_carry_no_prohibited_values() -> None:
    for page in (PAGE, GATE_PAGE):
        assert scan_for_sentinels(page.read_text(encoding="utf-8")) == (), page.name


def test_threat_model_states_the_v40_retarget_and_review_status() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    assert "folded into v4.0" in markdown
    assert "issues/4253" in markdown and "issues/3543" in markdown
    assert "re-reviewed independently" in markdown
    assert "../release/v4.0-service-release-gate.md" in markdown


@pytest.mark.parametrize("prefix", ["openmed", "deploy", "tests/unit"])
def test_dangling_reference_fails(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(AssertionError, match="dangling threat-model reference"):
        _validate_references(f"`{prefix}/missing.py`", tmp_path)


def test_reference_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="reference escapes repository"):
        _validate_references("`openmed/../../outside.py`", tmp_path)


@pytest.mark.parametrize(
    "control",
    [
        "No mitigation yet",
        "`openmed/example.py`",
        "`tests/unit/test_example.py`",
        "[#1](https://github.com/another/project/issues/1)",
    ],
)
def test_row_without_control_pair_or_tracked_gap_fails(control: str) -> None:
    with pytest.raises(AssertionError, match="unmapped threat"):
        _validate_rows(f"| SV-01 | Tampering | Synthetic threat at SB1 | {control} |")


def test_duplicate_or_sparse_ids_fail() -> None:
    row = "| SV-{0:02d} | Tampering | Threat | `openmed/a.py` `tests/b.py` |"
    with pytest.raises(AssertionError, match="duplicate threat identifier"):
        _validate_rows("\n".join([row.format(1), row.format(1)]))
    with pytest.raises(AssertionError, match="threat ids must be dense"):
        _validate_rows("\n".join([row.format(1), row.format(3)]))
