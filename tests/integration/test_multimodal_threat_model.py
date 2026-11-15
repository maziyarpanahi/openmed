"""Publication wiring and deletion regression for the real threat-model page."""

from pathlib import Path

import pytest
import yaml

from tests.unit.test_multimodal_threat_model import (
    PAGE,
    ROOT,
    _references,
    _validate_references,
    _validate_rows,
)

pytestmark = pytest.mark.integration


def test_published_threat_model_references_survive_only_while_files_exist(
    tmp_path: Path,
) -> None:
    markdown = PAGE.read_text(encoding="utf-8")
    relative = PAGE.relative_to(ROOT / "docs").as_posix()
    publication = yaml.safe_load(
        (ROOT / "docs/brand/system/publication.yml").read_text(encoding="utf-8")
    )
    assert relative in publication["classification"]["navigated"]
    assert relative in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    _validate_rows(markdown)

    # Materialize only reference paths in a synthetic tree. No media or source
    # payload is copied; this exercises the actual page's complete reference set.
    references = _references(markdown)
    for reference in references:
        path = tmp_path / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _validate_references(markdown, tmp_path)
    for prefix in ("openmed/", "tests/"):
        reference = next(path for path in sorted(references) if path.startswith(prefix))
        path = tmp_path / reference
        path.unlink()
        with pytest.raises(AssertionError, match="dangling threat-model reference"):
            _validate_references(markdown, tmp_path)
        path.touch()
