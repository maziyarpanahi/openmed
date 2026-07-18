"""Swahili coverage for the shared README translation drift guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.i18n.check_readme_drift import (
    DriftError,
    _relative_targets,
    build_manifest,
    check_repository,
    split_h2_sections,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "i18n" / "readme_section_hashes.json"


def _write_manifest(root: Path, manifest_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(
            build_manifest(root, manifest_path),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fixture_repo(root: Path) -> Path:
    (root / "docs" / "i18n").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "docs" / "i18n" / "glossary.md").write_text(
        "# Glossary\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        '<a href="README.sw.md">Kiswahili</a>\n'
        "\n## First\n\nEnglish body.\n"
        "\n## Second\n\n[Guide](docs/guide.md)\n",
        encoding="utf-8",
    )
    (root / "README.sw.md").write_text(
        '<a href="README.md">English</a>\n'
        "\n## Kwanza\n\nMaudhui ya Kiswahili.\n"
        "\n## Pili\n\n[Mwongozo](docs/guide.md)\n",
        encoding="utf-8",
    )
    manifest_path = root / "docs" / "i18n" / "readme_section_hashes.json"
    _write_manifest(root, manifest_path)
    return manifest_path


def test_source_heading_addition_fails_until_swahili_is_updated(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    source = tmp_path / "README.md"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n## Third\n\nNew source section.\n",
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="H2 sections"):
        check_repository(tmp_path, manifest_path)

    translation = tmp_path / "README.sw.md"
    translation.write_text(
        translation.read_text(encoding="utf-8")
        + "\n## Tatu\n\nSehemu mpya ya Kiswahili.\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, manifest_path)
    check_repository(tmp_path, manifest_path)


def test_source_body_edit_requires_swahili_manifest_refresh(tmp_path: Path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    source = tmp_path / "README.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "English body.", "Changed English body."
        ),
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="README.sw.md is stale"):
        check_repository(tmp_path, manifest_path)


def test_committed_swahili_readme_is_in_shared_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "README.sw.md" in manifest["translations"]
    check_repository(ROOT, MANIFEST)

    source_sections = split_h2_sections((ROOT / "README.md").read_text())
    translation_sections = split_h2_sections((ROOT / "README.sw.md").read_text())
    assert len(source_sections) == len(translation_sections)


def test_swahili_readme_preserves_all_link_and_image_targets() -> None:
    source_targets = _relative_targets((ROOT / "README.md").read_text())
    translation_targets = _relative_targets((ROOT / "README.sw.md").read_text())
    source_targets.remove("README.sw.md")
    translation_targets.remove("README.md")
    assert source_targets == translation_targets


def test_ci_uses_shared_drift_check_for_all_readmes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "readme-i18n-drift:" in workflow
    assert "'README*'" in workflow
    assert "python scripts/i18n/check_readme_drift.py" in workflow
