"""Offline checks for the OpenMedKit portable-kit threat model page."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "security" / "openmedkit-threat-model.md"
NAV_ENTRY = "security/openmedkit-threat-model.md"
RUNTIME_GUIDES = (
    "swift-openmedkit.md",
    "android-integration.md",
    "runtimes/onnxruntime-web.md",
)
REPO_TOP_LEVEL = (
    ".github",
    "android",
    "docs",
    "examples",
    "fixtures",
    "js",
    "openmed",
    "scripts",
    "swift",
    "tests",
    "web",
)
SOURCE_SUFFIXES = (".c", ".dart", ".h", ".kt", ".py", ".rs", ".swift", ".ts")
REQUIRED_OWNERS = {
    "3640",
    "3644",
    "3710",
    "3714",
    "3715",
    "3718",
    "3930",
    "3937",
    "3942",
    "3943",
    "3951",
}
REPOSITORY_LINK = re.compile(
    r"https://github\.com/maziyarpanahi/openmed/blob/master/(?P<path>[^\s)#]+)"
)
INLINE_CODE = re.compile(r"`([^`\n]+)`")
ISSUE_REF = re.compile(r"#(\d{3,5})\b")
PRIVATE_MARKERS = (
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"/private/"),
    re.compile(r"[A-Za-z]:\\\\?Users"),
    re.compile(r"~/"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*KEY-----"),
    re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
)


def _section(markdown: str, heading: str) -> str:
    assert heading in markdown, f"missing section: {heading.strip()}"
    return markdown.split(heading, 1)[1].split("\n## ", 1)[0]


def _table_rows(section: str, first_cell: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if re.match(rf"^\| {first_cell} \|", line):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _surfaces(markdown: str) -> set[str]:
    section = _section(markdown, "### Supported surfaces\n")
    rows = _table_rows(section, r"[A-Z][A-Za-z ]+")
    names = {row[0] for row in rows if row[0] != "Surface"}
    assert names, "surface table is missing"
    return names


def _boundaries(markdown: str) -> set[str]:
    section = _section(markdown, "## 3. Trust boundaries\n")
    ids = set(re.findall(r"(?m)^\| \*\*(TB\d+)\*\* \|", section))
    assert ids, "trust-boundary table is missing"
    return ids


def _owners(markdown: str) -> set[str]:
    section = _section(markdown, "## 7. Owning issues\n")
    issues = re.findall(r"(?m)^\| \[#(\d+)\]\(https://github\.com/", section)
    assert issues, "owning-issue table is missing"
    assert len(issues) == len(set(issues)), "owning issue listed twice"
    return set(issues)


def _abuse_rows(markdown: str) -> list[list[str]]:
    section = _section(markdown, "## 6. Abuse cases and controls\n")
    rows = _table_rows(section, r"K-\d{2}")
    assert rows, "abuse-case table is missing"
    return rows


def _repo_paths(text: str) -> set[str]:
    """Return inline-code tokens that name a repository path."""
    paths = set()
    for token in INLINE_CODE.findall(text):
        if any(token.startswith(f"{top}/") for top in REPO_TOP_LEVEL):
            paths.add(token.rstrip("/"))
        elif "/" not in token and (ROOT / token).is_file():
            paths.add(token)
    paths.update(match["path"] for match in REPOSITORY_LINK.finditer(text))
    return paths


def _assert_paths_exist(markdown: str, *, root: Path) -> None:
    paths = _repo_paths(markdown)
    assert paths, "page quotes no repository paths"
    for path in sorted(paths):
        relative = Path(path)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert (root / relative).exists(), f"missing referenced path: {path}"
    full_names = {Path(path).name for path in paths if "/" in path}
    for token in INLINE_CODE.findall(markdown):
        if "/" not in token and token.endswith(SOURCE_SUFFIXES):
            if (root / token).is_file():
                continue
            assert token in full_names, f"short file name without full path: {token}"


def _assert_abuse_rows(markdown: str, *, root: Path) -> None:
    surfaces = _surfaces(markdown)
    boundaries = _boundaries(markdown)
    owners = _owners(markdown)
    covered_surfaces: set[str] = set()
    covered_boundaries: set[str] = set()
    used_owners: set[str] = set()
    for row in _abuse_rows(markdown):
        assert len(row) == 7, f"abuse-case row has {len(row)} cells: {row[0]}"
        case_id, surface_cell, boundary_cell, _, control, owner, residual = row
        named = {item.strip() for item in surface_cell.split(",")}
        assert named <= surfaces | {"All"}, f"unknown surface in {case_id}"
        covered_surfaces |= named - {"All"}
        crossed = {item.strip() for item in boundary_cell.split(",")}
        assert crossed <= boundaries, f"unknown boundary in {case_id}"
        covered_boundaries |= crossed
        issues = set(ISSUE_REF.findall(owner))
        paths = {path for path in _repo_paths(control) if (root / path).exists()}
        assert issues or paths, f"{case_id} has no control path or owner"
        assert residual, f"{case_id} has no residual risk"
        assert issues <= owners, f"{case_id} owner missing from owner table"
        used_owners |= issues
    assert covered_surfaces == surfaces, "surface without an abuse case"
    assert covered_boundaries == boundaries, "trust boundary without an abuse case"
    assert used_owners == owners, "owner table lists an unused issue"


def _nav_entries(node: object) -> list[tuple[str, str]]:
    if isinstance(node, list):
        return [entry for child in node for entry in _nav_entries(child)]
    if isinstance(node, dict):
        entries = []
        for title, child in node.items():
            if isinstance(child, str):
                entries.append((title, child))
            else:
                entries.extend(_nav_entries(child))
        return entries
    return []


def test_page_is_navigated_and_registered() -> None:
    config = yaml.load(
        (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    entries = [entry for entry in _nav_entries(config["nav"]) if entry[1] == NAV_ENTRY]
    assert entries == [("OpenMedKit Portable-kit Threat Model", NAV_ENTRY)]
    publication = yaml.safe_load(
        (DOCS / "brand" / "system" / "publication.yml").read_text(encoding="utf-8")
    )
    assert NAV_ENTRY in publication["classification"]["navigated"]


def test_runtime_guides_link_the_threat_model(subtests) -> None:
    for guide in RUNTIME_GUIDES:
        with subtests.test(guide=guide):
            text = (DOCS / guide).read_text(encoding="utf-8")
            assert "openmedkit-threat-model.md" in text


def test_every_referenced_repository_path_exists() -> None:
    _assert_paths_exist(PAGE.read_text(encoding="utf-8"), root=ROOT)


def test_every_surface_and_boundary_has_a_mapped_abuse_case() -> None:
    _assert_abuse_rows(PAGE.read_text(encoding="utf-8"), root=ROOT)


def test_required_v40_owning_issues_are_mapped() -> None:
    assert REQUIRED_OWNERS <= _owners(PAGE.read_text(encoding="utf-8"))


def test_page_lists_unsupported_deployments_and_out_of_scope_threats() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    section = _section(markdown, "## 9. Unsupported deployments\n")
    assert "out of scope" in section
    assert section.count("\n- ") >= 8
    assert "### Controls without an owning issue" in markdown


def test_page_contains_no_private_paths_or_credentials() -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    for marker in PRIVATE_MARKERS:
        assert not marker.search(markdown), marker.pattern


_ROW = "| K-01 | Swift | TB1 | **Case.** | {control} | {owner} | {residual} |\n"
_SKELETON = (
    "### Supported surfaces\n\n| Surface | Package |\n| --- | --- |\n"
    "| Swift | `swift/OpenMedKit` |\n\n"
    "## 3. Trust boundaries\n\n| **TB1** | Model acquisition |\n\n"
    "## 6. Abuse cases and controls\n\n{rows}\n"
    "## 7. Owning issues\n\n"
    "| [#3640](https://github.com/maziyarpanahi/openmed/issues/3640) | x |\n"
)


def _synthetic(control: str, owner: str = "#3640", residual: str = "Gap.") -> str:
    row = _ROW.format(control=control, owner=owner, residual=residual)
    return _SKELETON.format(rows=row)


def test_path_check_rejects_a_missing_path() -> None:
    markdown = "See `swift/OpenMedKit/Sources/OpenMedKit/Missing.swift`.\n"
    with pytest.raises(AssertionError, match="missing referenced path"):
        _assert_paths_exist(markdown, root=ROOT)


def test_path_check_rejects_an_unanchored_short_file_name() -> None:
    markdown = "See `swift/OpenMedKit/Package.swift` and `Unanchored.swift`.\n"
    with pytest.raises(AssertionError, match="short file name"):
        _assert_paths_exist(markdown, root=ROOT)


def test_row_check_accepts_a_mapped_row() -> None:
    _assert_abuse_rows(_synthetic("`swift/OpenMedKit/Package.swift`"), root=ROOT)


@pytest.mark.parametrize(
    ("markdown", "reason"),
    [
        (_synthetic("No control.", owner="none"), "no control path or owner"),
        (_synthetic("`swift/OpenMedKit/Package.swift`", residual=""), "residual"),
        (_synthetic("No control.", owner="#9999"), "owner missing"),
        (_SKELETON.format(rows=""), "abuse-case table is missing"),
    ],
)
def test_row_check_rejects_unmapped_rows(markdown: str, reason: str) -> None:
    with pytest.raises(AssertionError, match=reason):
        _assert_abuse_rows(markdown, root=ROOT)
