"""Keep the shared parity fixture README in sync with the fixture directory."""

from __future__ import annotations

from pathlib import Path

PARITY_DIR = Path(__file__).parents[1] / "fixtures" / "parity"
README = PARITY_DIR / "README.md"


def _missing_rows(directory: Path, readme_text: str) -> list[str]:
    rows = [line for line in readme_text.splitlines() if line.startswith("| `")]
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and not any(row.startswith(f"| `{path.name}` |") for row in rows)
    )


def test_every_parity_fixture_has_a_readme_row() -> None:
    assert _missing_rows(PARITY_DIR, README.read_text(encoding="utf-8")) == []


def test_missing_row_is_reported(tmp_path: Path) -> None:
    (tmp_path / "listed.json").write_text("{}", encoding="utf-8")
    (tmp_path / "new_contract_v1.json").write_text("{}", encoding="utf-8")
    readme = "| Fixture |\n| --- |\n| `listed.json` | 1 |\nnew_contract_v1.json\n"

    assert _missing_rows(tmp_path, readme) == ["new_contract_v1.json"]
