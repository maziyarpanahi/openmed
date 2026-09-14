"""Focused tests for deterministic, offline Agent Skill bundle export."""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.skills import export as export_module
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
        hosts=["codex"],
    )

    with tarfile.open(result.archive_path, mode="r:gz") as archive:
        names = archive.getnames()
        embedded = json.loads(archive.extractfile("manifest.json").read())
    assert names[0] == "manifest.json"
    assert embedded == result.manifest


def test_source_revision_falls_back_without_git_metadata(tmp_path: Path) -> None:
    assert resolve_source_revision(tmp_path) == "unknown"


def test_outputs_cannot_alias_or_modify_skill_sources(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    archive = tmp_path / "bundle.zip"

    with pytest.raises(ExportError, match="paths must be different"):
        export_bundle(
            archive,
            manifest_path=tmp_path / "nested" / ".." / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    with pytest.raises(ExportError, match="outside the skills source tree"):
        export_bundle(
            skills_root / "alpha-skill" / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    with pytest.raises(ExportError, match="must not replace compatibility"):
        export_bundle(
            compatibility,
            manifest_path=tmp_path / "sidecar.json",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            force=True,
        )

    pack_manifest = tmp_path / "source-packs.json"
    pack_manifest.write_text("synthetic source metadata\n", encoding="utf-8")
    with pytest.raises(ExportError, match="must not replace source metadata"):
        export_bundle(
            pack_manifest,
            manifest_path=tmp_path / "sidecar.json",
            skills_root=skills_root,
            compatibility_path=compatibility,
            pack_manifest_path=pack_manifest,
            source_revision="synthetic-revision",
            force=True,
        )
    assert pack_manifest.read_text(encoding="utf-8") == "synthetic source metadata\n"


def test_nonportable_and_unexpected_skill_files_are_rejected(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    unexpected = skills_root / "alpha-skill" / "synthetic-sensitive-value.txt"
    unexpected.write_text("synthetic secret\n", encoding="utf-8")

    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    assert "unsupported root file" in str(captured.value)
    assert "synthetic-sensitive-value" not in str(captured.value)


def test_git_aware_export_rejects_untracked_skill_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    tracked = {path.resolve() for path in skills_root.rglob("*") if path.is_file()}
    secret_marker = "synthetic-sensitive-value"
    untracked = skills_root / "alpha-skill" / "references" / f"{secret_marker}.md"
    untracked.write_text("synthetic secret\n", encoding="utf-8")
    monkeypatch.setattr(
        export_module,
        "_tracked_skill_files",
        lambda _root, _names: tracked,
    )

    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    assert "untracked file" in str(captured.value)
    assert secret_marker not in str(captured.value)


def test_casefolding_member_collision_is_rejected(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    references = skills_root / "alpha-skill" / "references"
    upper = references / "CASE.md"
    lower = references / "case.md"
    upper.write_text("first\n", encoding="utf-8")
    lower.write_text("second\n", encoding="utf-8")
    if upper.samefile(lower):
        pytest.skip("filesystem does not permit case-distinct fixture paths")

    with pytest.raises(ExportError, match="portable path collision"):
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )


def test_unknown_selection_is_not_echoed(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    secret_marker = "synthetic-sensitive-value"

    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            skills=[secret_marker],
        )

    assert "unknown identifier" in str(captured.value)
    assert secret_marker not in str(captured.value)


def test_selection_iterator_failures_do_not_expose_values(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    secret_marker = "synthetic-sensitive-value"

    def failing_selection():
        yield "alpha-skill"
        raise RuntimeError(secret_marker)

    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            skills=failing_selection(),
        )

    assert "selection could not be read" in str(captured.value)
    assert secret_marker not in str(captured.value)


def test_archive_suffix_cannot_disagree_with_format(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)

    with pytest.raises(ExportError, match="conflicts"):
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            archive_format="tar.gz",
        )

    result = export_bundle(
        tmp_path / "bundle.TGZ",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
        hosts=["codex"],
    )
    assert result.manifest["archive"]["format"] == "tar.gz"


def test_selected_host_must_declare_archive_support(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)

    with pytest.raises(ExportError, match="host does not support"):
        export_bundle(
            tmp_path / "bundle.tar.gz",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            hosts=["other"],
        )


def test_finalize_failure_restores_both_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    archive = tmp_path / "bundle.zip"
    first = export_bundle(
        archive,
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="first-revision",
    )
    original_archive = archive.read_bytes()
    original_manifest = first.manifest_path.read_bytes()
    real_replace = export_module.os.replace
    replace_calls = 0

    def fail_second_install(source, target):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("synthetic finalize failure")
        return real_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", fail_second_install)

    with pytest.raises(ExportError, match="could not be finalized"):
        export_bundle(
            archive,
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="second-revision",
            force=True,
        )

    assert archive.read_bytes() == original_archive
    assert first.manifest_path.read_bytes() == original_manifest


def test_canonical_pack_manifest_is_embedded_and_controls_selection(
    tmp_path: Path,
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    packs_dir = skills_root / "packs"
    packs_dir.mkdir()
    pack_manifest = {
        "manifest_version": 1,
        "packs": [
            {
                "id": "small",
                "version": "1.0.0",
                "description": "Synthetic canonical pack.",
                "skills": ["alpha-skill"],
                "budget": {"max_skills": 2, "max_bytes": 10000},
            }
        ],
    }
    (packs_dir / "manifest.json").write_text(
        json.dumps(pack_manifest), encoding="utf-8"
    )

    result = export_bundle(
        tmp_path / "bundle.zip",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
        packs=["small"],
    )

    assert result.manifest["skills"] == ["alpha-skill"]
    with zipfile.ZipFile(result.archive_path) as archive:
        assert "skill-packs.json" in archive.namelist()
        embedded_packs = json.loads(archive.read("skill-packs.json"))
        embedded_compatibility = json.loads(archive.read("compatibility.json"))
    assert embedded_packs == pack_manifest
    assert embedded_compatibility["packs"]["small"]["version"] == "1.0.0"


def test_conflicting_duplicate_pack_declarations_fail_closed(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    packs_dir = skills_root / "packs"
    packs_dir.mkdir()
    (packs_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "packs": [
                    {
                        "id": "small",
                        "version": "1.0.0",
                        "description": "Synthetic conflicting pack.",
                        "skills": ["beta-skill"],
                        "budget": {"max_skills": 2, "max_bytes": 10000},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExportError, match="conflict with the canonical"):
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
            packs=["small"],
        )


@pytest.mark.parametrize("schema_version", [True, "1", 2])
def test_compatibility_schema_version_is_an_exact_integer(
    tmp_path: Path, schema_version: object
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    payload = json.loads(compatibility.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    compatibility.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportError, match="schema version"):
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )


def test_duplicate_or_unknown_metadata_fields_fail_without_echoing_values(
    tmp_path: Path,
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    secret_marker = "synthetic-sensitive-value"
    compatibility.write_text(
        """{
          "format": "openmed-agent-skill-compatibility",
          "schema_version": 1,
          "schema_version": 1,
          "hosts": {},
          "packs": {}
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ExportError, match="duplicate object keys"):
        export_bundle(
            tmp_path / "duplicate.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )

    _, compatibility = _write_fixture(tmp_path / "unknown")
    payload = json.loads(compatibility.read_text(encoding="utf-8"))
    payload[secret_marker] = secret_marker
    compatibility.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "unknown.zip",
            skills_root=tmp_path / "unknown" / "skills",
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )
    assert secret_marker not in str(captured.value)


def test_pack_manifest_is_canonicalized_before_embedding(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    packs_dir = skills_root / "packs"
    packs_dir.mkdir()
    secret_marker = "synthetic-sensitive-value"
    (packs_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "packs": [
                    {
                        "id": "small",
                        "version": "1.0.0",
                        "description": "Synthetic pack.",
                        "skills": ["alpha-skill"],
                        "budget": {"max_skills": 2, "max_bytes": 10000},
                        secret_marker: secret_marker,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )
    assert secret_marker not in str(captured.value)


def test_default_host_selection_enforces_archive_capabilities(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)

    with pytest.raises(ExportError, match="host does not support"):
        export_bundle(
            tmp_path / "bundle.tar.gz",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )


def test_public_string_inputs_are_bounded_and_type_checked(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    common = {
        "skills_root": skills_root,
        "compatibility_path": compatibility,
        "source_revision": "synthetic-revision",
    }

    with pytest.raises(ExportError, match="archive format"):
        export_bundle(tmp_path / "format.zip", archive_format=1, **common)
    with pytest.raises(ExportError, match="bundle name"):
        export_bundle(tmp_path / "name.zip", bundle_name="a" * 129, **common)
    with pytest.raises(ExportError, match="source revision"):
        export_bundle(
            tmp_path / "revision.zip",
            source_revision="a" * 257,
            skills_root=skills_root,
            compatibility_path=compatibility,
        )


def test_windows_reserved_and_trailing_dot_components_are_not_portable() -> None:
    assert not export_module._portable_component("CON.md")
    assert not export_module._portable_component("lpt9")
    assert not export_module._portable_component("example.")
    assert export_module._portable_component("example.md")


def test_skill_scripts_keep_a_deterministic_executable_mode(tmp_path: Path) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    scripts_dir = skills_root / "alpha-skill" / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    result = export_bundle(
        tmp_path / "bundle.zip",
        skills_root=skills_root,
        compatibility_path=compatibility,
        source_revision="synthetic-revision",
    )

    with zipfile.ZipFile(result.archive_path) as archive:
        script = archive.getinfo("skills/alpha-skill/scripts/run.py")
        skill = archive.getinfo("skills/alpha-skill/SKILL.md")
    assert (script.external_attr >> 16) & 0o777 == 0o755
    assert (skill.external_attr >> 16) & 0o777 == 0o644


def test_file_swap_to_symlink_cannot_escape_the_skill_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    source = skills_root / "alpha-skill" / "references" / "example.md"
    outside = tmp_path / "synthetic-sensitive-value.txt"
    outside.write_text("synthetic-sensitive-value\n", encoding="utf-8")
    real_reader = export_module._read_skill_file
    swapped = False

    def swap_before_open(root: Path, skill_name: str, relative: Path) -> bytes:
        nonlocal swapped
        if (
            not swapped
            and skill_name == "alpha-skill"
            and relative.name == "example.md"
        ):
            source.unlink()
            try:
                source.symlink_to(outside)
            except OSError:
                pytest.skip("filesystem does not permit symlink fixtures")
            swapped = True
        return real_reader(root, skill_name, relative)

    monkeypatch.setattr(export_module, "_read_skill_file", swap_before_open)
    with pytest.raises(ExportError) as captured:
        export_bundle(
            tmp_path / "bundle.zip",
            skills_root=skills_root,
            compatibility_path=compatibility,
            source_revision="synthetic-revision",
        )
    assert "synthetic-sensitive-value" not in str(captured.value)


def test_cli_does_not_print_local_output_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skills_root, compatibility = _write_fixture(tmp_path)
    output = tmp_path / "synthetic-private-directory" / "bundle.zip"

    status = export_module.main(
        [
            "--output",
            str(output),
            "--skills-root",
            str(skills_root),
            "--compatibility",
            str(compatibility),
            "--source-revision",
            "synthetic-revision",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert str(tmp_path) not in captured.out
    assert "Archive written." in captured.out
    assert output.name not in captured.out

    secret_marker = "synthetic-sensitive-value"
    status = export_module.main(
        [
            "--output",
            str(tmp_path / "invalid.zip"),
            "--skills-root",
            str(skills_root),
            "--compatibility",
            str(compatibility),
            "--source-revision",
            "synthetic-revision",
            "--format",
            secret_marker,
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert secret_marker not in captured.err
