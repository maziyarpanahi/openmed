"""Keep repository example links in the cookbook from going stale."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = ROOT / "docs" / "cookbook.md"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)\s]+)\)")
REPOSITORY_BLOB_PREFIX = "https://github.com/maziyarpanahi/openmed/blob/master/"


def _linked_example_paths(markdown: str) -> list[Path]:
    paths: list[Path] = []
    for target in MARKDOWN_LINK.findall(markdown):
        target_without_fragment = target.split("#", maxsplit=1)[0]
        if target_without_fragment.startswith(REPOSITORY_BLOB_PREFIX):
            relative_path = target_without_fragment.removeprefix(REPOSITORY_BLOB_PREFIX)
        elif target_without_fragment.startswith("../examples/"):
            relative_path = target_without_fragment.removeprefix("../")
        else:
            continue

        if relative_path.startswith("examples/"):
            paths.append(Path(relative_path))
    return paths


def test_every_cookbook_example_link_exists() -> None:
    linked_paths = _linked_example_paths(COOKBOOK.read_text(encoding="utf-8"))

    assert linked_paths, "cookbook must link to at least one repository example"
    missing = [str(path) for path in linked_paths if not (ROOT / path).is_file()]
    assert not missing, f"cookbook links to missing example files: {missing}"
