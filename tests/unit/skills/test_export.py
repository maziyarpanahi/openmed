"""Focused tests for deterministic, offline Agent Skill bundle export."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.skills.export import ExportError, export_bundle, resolve_source_revision


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    skills_root = tmp_path / "skills"
    for name, body in (
        ("alpha-skill", "---\nname: alpha-skill\ndescription: Synthetic alpha\n---\n"),
        ("beta-skill", "---\nname: beta-skill\ndescription: Synthetic beta\n---\n"),
    ):
        skill_dir = skills_root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "example.md").write_text(
            "Synthetic offline guidance.\n", encoding="utf-8"
        )

    compatibility = {
        "format": "openmed-agent-skill-compatibility",
        "schema_version": 1,
        "hosts": {
            "codex": {
                "display_name": "Synthetic host",
                "skills_dir": "~/.synthetic/skills",
                "capabilities": {"archive_formats": ["zip", "tar.gz"]},
            },
            "other": {
                "display_name": "Another synthetic host",
                "skills_dir": "~/.other/skills",
                "capabilities": {"archive_formats": ["zip"]},
            },
        },
        "packs": {
            "small": {"description": "Synthetic pack", "skills": ["alpha-skill"]}
        },
    }
    compatibility_path = tmp_path / "compatibility.json"
    compatibility_path.write_text(
        json.dumps(compatibility, indent=2) + "\n", encoding="utf-8"
    )
    return skills_root, compatibility_path


def test_export_is_deterministic_and_records_file_checksums(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    first = export_bundle(
        tmp_path / "first.zip",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
    )
    second = export_bundle(
        tmp_path / "second.zip",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
    )

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.archive_sha256 == second.archive_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["source_revision"] == "synthetic-revision"
    assert first.manifest["skills"] == ["alpha-skill", "beta-skill"]
    assert all(
        not record["path"].startswith("/") and str(tmp_path) not in record["path"]
        for record in first.manifest["files"]
    )

    with zipfile.ZipFile(first.archive_path) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "compatibility.json",
            "skills/alpha-skill/SKILL.md",
            "skills/alpha-skill/references/example.md",
            "skills/beta-skill/SKILL.md",
            "skills/beta-skill/references/example.md",
        ]
        embedded_manifest = json.loads(archive.read("manifest.json"))
    assert embedded_manifest == json.loads(
        first.manifest_path.read_text(encoding="utf-8")
    )
    assert embedded_manifest["checksums"]["algorithm"] == "sha256"
    assert all(record["sha256"] for record in embedded_manifest["files"])


def test_pack_and_host_selection_are_data_driven(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    result = export_bundle(
        tmp_path / "small.zip",
        skills_root=skills_root,
        compatibility_path=compatibility,
        packs=["small"],
        hosts=["codex"],
        source_revision="synthetic-revision",
        bundle_name="synthetic-pack",
    )

    assert result.manifest["bundle_name"] == "synthetic-pack"
    assert result.manifest["skills"] == ["alpha-skill"]
    assert list(result.manifest["hosts"]) == ["codex"]
    with zipfile.ZipFile(result.archive_path) as archive:
        assert "skills/alpha-skill/SKILL.md" in archive.namelist()
        assert "skills/beta-skill/SKILL.md" not in archive.namelist()


def test_existing_outputs_are_preserved_without_force(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    archive_path = tmp_path / "bundle.zip"
    result = export_bundle(
        archive_path,
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
    )
    original_archive = archive_path.read_bytes()
    original_manifest = result.manifest_path.read_bytes()

    with pytest.raises(ExportError, match="refusing to overwrite"):
        export_bundle(
            archive_path,
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    assert archive_path.read_bytes() == original_archive
    assert result.manifest_path.read_bytes() == original_manifest
    forced = export_bundle(
        archive_path,
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
        force=True,
    )
    assert forced.archive_path.read_bytes() == original_archive


def test_tar_gz_output_has_the_same_manifest_contract(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    result = export_bundle(
        tmp_path / "bundle.tar.gz",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
    )

    with tarfile.open(result.archive_path, mode="r:gz") as archive:
        names = archive.getnames()
        embedded = json.loads(archive.extractfile("manifest.json").read())
    assert names[0] == "manifest.json"
    assert embedded == result.manifest


def test_source_revision_falls_back_without_git_metadata(tmp_path: Path) -> None:
    assert resolve_source_revision(tmp_path) == "unknown"
