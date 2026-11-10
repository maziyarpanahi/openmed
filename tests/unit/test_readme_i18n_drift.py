"""Tests for the root README translation drift guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.i18n.check_readme_drift import (
    PREAMBLE,
    DriftError,
    _relative_targets,
    build_manifest,
    check_repository,
    split_h2_sections,
    validate_relative_links,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "i18n" / "readme_section_hashes.json"


def _write_fixture_repo(root: Path) -> Path:
    (root / "docs" / "i18n").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "docs" / "i18n" / "glossary.md").write_text(
        "# Glossary\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        '<a href="README.zh-CN.md">简体中文</a>\n'
        "\n## First\n\nEnglish body.\n"
        "\n## Second\n\n[Guide](docs/guide.md)\n",
        encoding="utf-8",
    )
    (root / "README.zh-CN.md").write_text(
        '<a href="README.md">English</a>\n'
        "\n## 第一节\n\n中文内容。\n"
        "\n## 第二节\n\n[指南](docs/guide.md)\n",
        encoding="utf-8",
    )
    manifest_path = root / "docs" / "i18n" / "readme_section_hashes.json"
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
    return manifest_path


def test_split_h2_sections_ignores_headings_inside_fences() -> None:
    sections = split_h2_sections(
        "intro\n\n## Real\n\n```markdown\n## Not a section\n```\n"
    )

    assert [section.heading for section in sections] == [PREAMBLE, "Real"]
    assert "## Not a section" in sections[1].content


def test_english_only_edit_fails_until_manifest_is_updated(tmp_path: Path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    source = tmp_path / "README.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace("English body.", "Changed body."),
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="README.zh-CN.md is stale"):
        check_repository(tmp_path, manifest_path)

    manifest_path.write_text(
        json.dumps(
            build_manifest(tmp_path, manifest_path),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    check_repository(tmp_path, manifest_path)


def test_translation_edit_requires_manifest_refresh(tmp_path: Path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    translation = tmp_path / "README.zh-CN.md"
    translation.write_text(
        translation.read_text(encoding="utf-8").replace("中文内容。", "更新内容。"),
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="manifest entry"):
        check_repository(tmp_path, manifest_path)


def test_relative_link_validation_reports_missing_target(tmp_path: Path) -> None:
    readme = tmp_path / "README.zh-CN.md"
    readme.write_text("[缺失文档](docs/missing.md)\n", encoding="utf-8")

    with pytest.raises(DriftError, match="docs/missing.md"):
        validate_relative_links(tmp_path, readme)


def test_committed_readmes_and_manifest_are_in_sync() -> None:
    check_repository(ROOT, MANIFEST)

    source_sections = split_h2_sections(
        (ROOT / "README.md").read_text(encoding="utf-8")
    )
    translation_sections = split_h2_sections(
        (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    )
    assert len(source_sections) == len(translation_sections)


def test_translated_readme_preserves_all_link_and_image_targets() -> None:
    source_targets = _relative_targets((ROOT / "README.md").read_text(encoding="utf-8"))
    translation_targets = _relative_targets(
        (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    )
    source_targets.remove("README.zh-CN.md")
    translation_targets.remove("README.md")

    assert source_targets == translation_targets


def test_ci_runs_drift_check_for_readme_changes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "readme-i18n-drift:" in workflow
    assert "'README*'" in workflow
    assert "python scripts/i18n/check_readme_drift.py" in workflow
    assert "docs/i18n/glossary.md" in workflow


def test_glossary_contains_canonical_required_terms() -> None:
    glossary = (ROOT / "docs" / "i18n" / "glossary.md").read_text(encoding="utf-8")

    for source, translation in {
        "de-identification": "去标识化",
        "on-device": "设备本地",
        "entity extraction": "实体抽取",
        "local-first": "本地优先",
        "Privacy Filter": "隐私过滤器",
    }.items():
        assert f"| {source} | {translation} |" in glossary
