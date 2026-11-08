"""Offline tests for the versioned air-gap lifecycle manifest contract."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from openmed.cli import airgap, main_module
from openmed.cli.airgap_lifecycle import (
    LifecycleError,
    artifact_identity,
    is_pure_wheel,
    set_digest,
    validate_lifecycle,
    version_key,
)
from tests.fixtures.airgap_lifecycle import (
    RELEASE_VERSION,
    UPGRADE_VERSION,
    build_lifecycle_bundle,
    write_wheel,
)

_DIGEST = r"^sha256:[0-9a-f]{64}$"


def _manifest(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / airgap.MANIFEST_NAME).read_text(encoding="utf-8"))


def _checks(manifest: dict[str, Any], **kwargs: Any) -> dict[str, tuple[str, str]]:
    validation = validate_lifecycle(manifest, **kwargs)
    return {
        check.check_id: (check.status, check.reason_code) for check in validation.checks
    }


def test_bundle_manifest_carries_versioned_lifecycle_sets_with_digests(
    tmp_path: Path,
) -> None:
    bundle = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")

    manifest = _manifest(bundle)
    lifecycle = manifest["lifecycle"]
    inventory = {item["path"]: item["sha256"] for item in manifest["artifacts"]}

    assert airgap.verify_bundle(bundle) == []
    assert lifecycle["schema_version"] == 1
    assert lifecycle["distribution"] == "openmed"
    assert set(lifecycle["sets"]) == {"release", "upgrade", "rollback", "recovery"}
    assert lifecycle["sets"]["release"]["version"] == RELEASE_VERSION
    assert lifecycle["sets"]["upgrade"]["version"] == UPGRADE_VERSION
    assert lifecycle["sets"]["upgrade"]["from_versions"] == [RELEASE_VERSION]
    assert lifecycle["sets"]["rollback"]["version"] == RELEASE_VERSION
    for name, section in lifecycle["sets"].items():
        assert (
            section["root"]
            == {
                "release": "wheels",
                "upgrade": "lifecycle/upgrade",
                "rollback": "lifecycle/rollback",
                "recovery": "lifecycle/recovery",
            }[name]
        )
        assert section["artifacts"]
        for item in section["artifacts"]:
            assert inventory[item["path"]] == item["sha256"]
        records = [(item["path"], item["sha256"]) for item in section["artifacts"]]
        assert section["digest"] == set_digest(section["root"], records)
    # Identical previous-release bytes share a digest wherever they are held.
    assert (
        lifecycle["sets"]["rollback"]["digest"]
        == lifecycle["sets"]["release"]["digest"]
    )
    assert lifecycle["migrations"] == [
        {
            "id": "0001",
            "sequence": 1,
            "from_version": RELEASE_VERSION,
            "to_version": UPGRADE_VERSION,
            "path": "lifecycle/migrations/0001-synthetic-schema.json",
            "sha256": inventory["lifecycle/migrations/0001-synthetic-schema.json"],
        }
    ]
    assert _checks(manifest) == {
        "lifecycle_contract": ("passed", "ok"),
        "compatibility": ("passed", "ok"),
        "validity_window": ("passed", "not_required"),
    }


def test_bundle_without_lifecycle_inputs_keeps_the_v1_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        airgap,
        "_download_wheels",
        lambda wheels_dir, **_: write_wheel(wheels_dir, RELEASE_VERSION),
    )
    monkeypatch.setattr(
        airgap,
        "prefetch_model",
        lambda model, *, cache_dir: str(_snapshot(Path(cache_dir))),
    )
    airgap.build_bundle(tmp_path / "kit")

    manifest = _manifest(tmp_path / "kit")

    assert manifest["schema_version"] == 1
    assert "lifecycle" not in manifest
    assert _checks(manifest)["lifecycle_contract"] == ("failed", "lifecycle_missing")


def _snapshot(cache: Path) -> Path:
    snapshot = cache / "models--synthetic--model" / "snapshots" / "synthetic"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"upgrade_version": RELEASE_VERSION}, "version_order_invalid"),
        ({"upgrade_version": "0.0.1"}, "version_order_invalid"),
        ({"rollback_version": "0.0.1"}, "rollback_target_mismatch"),
        ({"upgrade_from": ("0.0.1",)}, "upgrade_incompatible"),
        (
            {"not_after": datetime.now(timezone.utc) - timedelta(days=1)},
            "artifacts_expired",
        ),
        (
            {"not_before": datetime.now(timezone.utc) + timedelta(days=30)},
            "artifacts_not_yet_valid",
        ),
    ],
)
def test_build_refuses_kits_that_would_fail_preflight(
    tmp_path: Path, kwargs: dict[str, Any], code: str
) -> None:
    with pytest.raises(LifecycleError) as exc_info:
        build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs", **kwargs)

    assert exc_info.value.code == code
    assert str(exc_info.value) == code
    assert not (tmp_path / "kit").exists()


def _add_symlink(directory: Path) -> None:
    (directory / "link.whl").symlink_to(next(directory.glob("*.whl")))


def _add_nested(directory: Path) -> None:
    (directory / "nested").mkdir()
    (directory / "nested" / "openmed-9-py3-none-any.whl").write_bytes(b"x")


def _add_text(directory: Path) -> None:
    (directory / "notes.txt").write_text("synthetic")


@pytest.mark.parametrize("mutate", [_add_symlink, _add_nested, _add_text])
def test_lifecycle_wheel_sets_must_be_flat_regular_installables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[Path], None]
) -> None:
    if mutate is _add_symlink and os.name == "nt":
        pytest.skip("symlinks need privileges on Windows")
    original = write_wheel

    def _mutating_write(directory: Path, version: str, **kwargs: Any) -> Path:
        wheel = original(directory, version, **kwargs)
        if directory.name == "upgrade":
            mutate(directory)
        return wheel

    monkeypatch.setattr("tests.fixtures.airgap_lifecycle.write_wheel", _mutating_write)

    with pytest.raises(ValueError):
        build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    assert not (tmp_path / "kit").exists()


def test_migration_assets_need_ordered_names(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    (inputs / "migrations").mkdir(parents=True)
    (inputs / "migrations" / "add-index.json").write_text("{}")

    with pytest.raises(ValueError):
        build_lifecycle_bundle(tmp_path / "kit", inputs)


def test_version_ordering_follows_installer_semantics() -> None:
    ordered = ["1.0.dev1", "1.0a1", "1.0b2", "1.0rc1", "1.0", "1.0.post1", "1.1"]

    assert sorted(reversed(ordered), key=version_key) == ordered
    assert version_key("4.0") == version_key("4.0.0")
    with pytest.raises(LifecycleError):
        version_key("latest")


def test_artifact_names_are_parsed_without_trusting_metadata() -> None:
    assert artifact_identity("openmed-4.0.0-py3-none-any.whl") == ("openmed", "4.0.0")
    assert artifact_identity("Open_Med-4.0.0.tar.gz") == ("open-med", "4.0.0")
    assert artifact_identity("not-a-wheel.txt") is None
    assert is_pure_wheel("openmed-4.0.0-py3-none-any.whl")
    assert not is_pure_wheel("torch-2.0-cp311-cp311-manylinux2014_x86_64.whl")


def _mutate_manifest(
    bundle: Path, mutate: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    manifest = _manifest(bundle)
    mutate(manifest)
    return manifest


def _sets(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["lifecycle"]["sets"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda m: m["lifecycle"].update(schema_version=2), "lifecycle_invalid"),
        (lambda m: _sets(m).pop("upgrade"), "set_missing"),
        (
            lambda m: _sets(m)["upgrade"].update(digest="sha256:" + "0" * 64),
            "set_digest_mismatch",
        ),
        (
            lambda m: _sets(m)["upgrade"]["artifacts"][0].update(sha256="0" * 64),
            "set_digest_mismatch",
        ),
        (lambda m: _sets(m)["upgrade"].update(version="9.9.9"), "set_version_mismatch"),
        (lambda m: m.update(openmed_version="0.0.1"), "lifecycle_invalid"),
        (lambda m: m["lifecycle"]["migrations"].clear(), "set_digest_mismatch"),
        (
            lambda m: m["lifecycle"].update(distribution="other"),
            "lifecycle_invalid",
        ),
    ],
)
def test_contract_tampering_fails_with_controlled_codes(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    bundle = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    manifest = _mutate_manifest(bundle, mutate)

    checks = _checks(manifest)

    assert checks["lifecycle_contract"] == ("failed", code)
    assert checks["compatibility"] == ("not_run", "prior_check_failed")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda m: _sets(m)["upgrade"].update(from_versions=["0.0.1"]),
            "upgrade_incompatible",
        ),
        (
            lambda m: m["lifecycle"]["migrations"][0].update(sequence=2),
            "migration_order_invalid",
        ),
        (
            lambda m: m["lifecycle"]["migrations"][0].update(to_version="0.0.1"),
            "migration_order_invalid",
        ),
    ],
)
def test_compatibility_decisions_fail_with_controlled_codes(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    bundle = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    manifest = _mutate_manifest(bundle, mutate)

    assert _checks(manifest)["compatibility"] == ("failed", code)


def test_platform_wheels_need_the_target_interpreter(tmp_path: Path) -> None:
    bundle = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    manifest = _manifest(bundle)
    platform_wheel = "wheels/numpy-2.0-cp311-cp311-manylinux2014_x86_64.whl"
    manifest["artifacts"].append({"path": platform_wheel, "sha256": "1" * 64})
    release = _sets(manifest)["release"]
    release["artifacts"].append({"path": platform_wheel, "sha256": "1" * 64})
    records = [(item["path"], item["sha256"]) for item in release["artifacts"]]
    release["digest"] = set_digest("wheels", records)
    manifest["target"]["python_version"] = "3.11"

    assert _checks(manifest, python_version="3.11")["compatibility"] == ("passed", "ok")
    assert _checks(manifest, python_version="3.13")["compatibility"] == (
        "failed",
        "target_incompatible",
    )


def test_validity_window_uses_the_supplied_clock(tmp_path: Path) -> None:
    not_after = datetime(2030, 1, 1, tzinfo=timezone.utc)
    bundle = build_lifecycle_bundle(
        tmp_path / "kit", tmp_path / "inputs", not_after=not_after
    )
    manifest = _manifest(bundle)

    assert manifest["lifecycle"]["validity"] == {
        "not_before": None,
        "not_after": "2030-01-01T00:00:00Z",
    }
    assert _checks(manifest, now=not_after - timedelta(seconds=1))[
        "validity_window"
    ] == ("passed", "ok")
    assert _checks(manifest, now=not_after + timedelta(seconds=1))[
        "validity_window"
    ] == ("failed", "artifacts_expired")


def test_cli_builds_a_lifecycle_kit_from_caller_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        airgap,
        "_download_wheels",
        lambda wheels_dir, **_: write_wheel(wheels_dir, RELEASE_VERSION),
    )
    monkeypatch.setattr(
        airgap,
        "prefetch_model",
        lambda model, *, cache_dir: str(_snapshot(Path(cache_dir))),
    )
    write_wheel(tmp_path / "next", UPGRADE_VERSION)
    write_wheel(tmp_path / "previous", RELEASE_VERSION)

    code = main_module.main(
        [
            "airgap",
            "bundle",
            str(tmp_path / "kit"),
            "--upgrade-wheels",
            str(tmp_path / "next"),
            "--rollback-wheels",
            str(tmp_path / "previous"),
            "--valid-until",
            "2099-01-01T00:00:00Z",
        ]
    )

    manifest = _manifest(tmp_path / "kit")
    assert code == 0
    assert manifest["lifecycle"]["validity"]["not_after"] == "2099-01-01T00:00:00Z"
    assert "recovery" not in manifest["lifecycle"]["sets"]
    assert manifest["lifecycle"]["migrations"] == []


def test_cli_requires_both_upgrade_and_rollback_sets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_wheel(tmp_path / "next", UPGRADE_VERSION)

    code = main_module.main(
        [
            "airgap",
            "bundle",
            str(tmp_path / "kit"),
            "--upgrade-wheels",
            str(tmp_path / "next"),
        ]
    )

    assert code == 2
    assert "--rollback-wheels" in capsys.readouterr().err
    assert not (tmp_path / "kit").exists()
