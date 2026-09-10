"""Focused tests for the offline topical Agent Skill pack builder."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "scripts" / "skills" / "build_packs.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("openmed_build_packs", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_skill(skills_dir: Path, name: str, body: str = "Synthetic skill.") -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Synthetic test skill.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, packs: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"manifest_version": 1, "packs": packs}, indent=2) + "\n",
        encoding="utf-8",
    )


def _pack(pack_id: str, skills: list[str], *, max_bytes: int = 10000) -> dict:
    return {
        "id": pack_id,
        "version": "1.0.0",
        "description": "Synthetic test pack.",
        "skills": skills,
        "budget": {"max_skills": len(skills), "max_bytes": max_bytes},
    }


def test_committed_manifest_is_valid_and_topical() -> None:
    builder = _load_builder()

    manifest = builder.load_manifest()
    reports = builder.validate_manifest(manifest)

    assert manifest.manifest_version == 1
    assert {report.pack.identifier for report in reports} == {
        "coding",
        "evaluation",
        "interoperability",
        "privacy",
        "research",
    }
    memberships = [skill for report in reports for skill in report.pack.skills]
    assert len(memberships) == len(set(memberships))
    assert all(
        report.skill_count <= report.pack.budget.max_skills
        and report.size_bytes <= report.pack.budget.max_bytes
        for report in reports
    )


def test_missing_skill_is_rejected_without_echoing_skill_content(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "present-skill", "Synthetic body only.")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [_pack("privacy", ["present-skill", "missing-skill"])],
    )

    manifest = builder.load_manifest(manifest_path)
    with pytest.raises(builder.PackValidationError) as raised:
        builder.validate_manifest(manifest, skills_dir)

    message = "\n".join(raised.value.errors)
    assert "missing-skill" in message
    assert "Synthetic body only" not in message


def test_manifest_read_errors_do_not_expose_local_paths(tmp_path: Path) -> None:
    builder = _load_builder()
    sensitive = "PatientJaneDoe"
    missing_manifest = tmp_path / sensitive / "manifest.json"

    with pytest.raises(builder.PackValidationError) as raised:
        builder.load_manifest(missing_manifest)

    rendered = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert sensitive not in rendered


def test_duplicate_membership_is_rejected(tmp_path: Path) -> None:
    builder = _load_builder()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            _pack("privacy", ["shared-skill"]),
            _pack("research", ["shared-skill"]),
        ],
    )

    with pytest.raises(builder.PackValidationError) as raised:
        builder.load_manifest(manifest_path)

    assert any("assigned to both" in error for error in raised.value.errors)


def test_boolean_manifest_version_is_rejected(tmp_path: Path) -> None:
    builder = _load_builder()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": True,
                "packs": [_pack("privacy", ["privacy-skill"])],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(builder.PackValidationError) as raised:
        builder.load_manifest(manifest_path)

    assert any("manifest_version" in error for error in raised.value.errors)


def test_pack_byte_budget_is_enforced(tmp_path: Path) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill", "Synthetic body that exceeds one byte.")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [_pack("privacy", ["privacy-skill"], max_bytes=1)],
    )

    manifest = builder.load_manifest(manifest_path)
    with pytest.raises(builder.PackValidationError) as raised:
        builder.validate_manifest(manifest, skills_dir)

    assert any("byte budget" in error for error in raised.value.errors)


@pytest.mark.skipif(
    os.name == "nt",
    reason="directory symlink behavior differs without Windows developer mode",
)
def test_build_is_deterministic_and_keeps_one_source_of_skill_content(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "alpha-skill")
    _write_skill(skills_dir, "zeta-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [_pack("privacy", ["zeta-skill", "alpha-skill"])],
    )
    output_dir = tmp_path / "output"

    builder.build_from_files(manifest_path, skills_dir, output_dir)
    pack_dir = output_dir / "privacy"
    metadata_before = (pack_dir / "pack.json").read_bytes()
    links_before = {
        path.name: os.readlink(path) for path in sorted((pack_dir / "skills").iterdir())
    }

    assert (pack_dir / "skills" / "alpha-skill").is_symlink()
    assert (pack_dir / "skills" / "alpha-skill").resolve() == (
        skills_dir / "alpha-skill"
    ).resolve()
    assert not (pack_dir / "skills" / "alpha-skill" / "SKILL.md").is_symlink()

    builder.build_from_files(manifest_path, skills_dir, output_dir)
    assert (pack_dir / "pack.json").read_bytes() == metadata_before
    assert {
        path.name: os.readlink(path) for path in sorted((pack_dir / "skills").iterdir())
    } == links_before


def test_selection_only_writes_metadata_without_copying_skill_content(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_pack("privacy", ["privacy-skill"])])
    output_dir = tmp_path / "selection"

    builder.build_from_files(
        manifest_path,
        skills_dir,
        output_dir,
        selection_only=True,
    )

    metadata = json.loads((output_dir / "privacy" / "pack.json").read_text())
    assert metadata["skills"] == ["privacy-skill"]
    assert not (output_dir / "privacy" / "skills").exists()


def test_output_inside_skill_sources_is_rejected_before_writing(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_pack("privacy", ["privacy-skill"])])

    with pytest.raises(builder.PackBuildError, match="outside the skills"):
        builder.build_from_files(
            manifest_path,
            skills_dir,
            skills_dir / "generated-packs",
        )

    assert not (skills_dir / "generated-packs").exists()


def test_in_memory_manifest_cannot_escape_the_pack_output(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest = builder.PackManifest(
        manifest_version=1,
        packs=(
            builder.PackSpec(
                identifier="../escape",
                version="1.0.0",
                description="Synthetic test pack.",
                skills=("privacy-skill",),
                budget=builder.PackBudget(max_skills=1, max_bytes=10_000),
            ),
        ),
    )
    output_dir = tmp_path / "output"

    with pytest.raises(builder.PackValidationError, match="validation failed"):
        builder.build_packs(manifest, skills_dir, output_dir)

    assert not output_dir.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="directory symlink behavior differs without Windows developer mode",
)
def test_existing_links_must_be_exact_relative_generated_targets(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_pack("privacy", ["privacy-skill"])])
    output_dir = tmp_path / "output"
    links_dir = output_dir / "privacy" / "skills"
    links_dir.mkdir(parents=True)
    destination = links_dir / "privacy-skill"
    destination.symlink_to(
        (skills_dir / "privacy-skill").resolve(),
        target_is_directory=True,
    )

    with pytest.raises(builder.PackBuildError, match="unexpected skill link"):
        builder.build_from_files(manifest_path, skills_dir, output_dir)

    assert not (output_dir / "privacy" / "pack.json").exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="directory symlink behavior differs without Windows developer mode",
)
def test_rebuild_rejects_stale_skill_links_before_rewriting_metadata(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "alpha-skill")
    _write_skill(skills_dir, "zeta-skill")
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "output"
    _write_manifest(
        manifest_path,
        [_pack("privacy", ["alpha-skill", "zeta-skill"])],
    )
    builder.build_from_files(manifest_path, skills_dir, output_dir)
    metadata_path = output_dir / "privacy" / "pack.json"
    metadata_before = metadata_path.read_bytes()

    _write_manifest(manifest_path, [_pack("privacy", ["alpha-skill"])])
    with pytest.raises(builder.PackBuildError, match="stale skill output"):
        builder.build_from_files(manifest_path, skills_dir, output_dir)

    assert metadata_path.read_bytes() == metadata_before


def test_foreign_pack_metadata_is_not_overwritten(tmp_path: Path) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_pack("privacy", ["privacy-skill"])])
    pack_dir = tmp_path / "output" / "privacy"
    pack_dir.mkdir(parents=True)
    metadata_path = pack_dir / "pack.json"
    metadata_path.write_text("synthetic foreign content\n", encoding="utf-8")

    with pytest.raises(builder.PackBuildError, match="metadata is not safe"):
        builder.build_from_files(manifest_path, skills_dir, tmp_path / "output")

    assert metadata_path.read_text(encoding="utf-8") == "synthetic foreign content\n"


def test_metadata_replace_failure_preserves_the_previous_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "privacy-skill")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [_pack("privacy", ["privacy-skill"])])
    output_dir = tmp_path / "output"
    builder.build_from_files(manifest_path, skills_dir, output_dir)
    metadata_path = output_dir / "privacy" / "pack.json"
    metadata_before = metadata_path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(builder.os, "replace", fail_replace)

    with pytest.raises(builder.PackBuildError, match="could not write pack"):
        builder.build_from_files(manifest_path, skills_dir, output_dir)

    assert metadata_path.read_bytes() == metadata_before
    assert list(metadata_path.parent.glob(".pack-json-*.tmp")) == []
