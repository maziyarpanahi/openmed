"""Offline tests for the air-gap lifecycle rehearsal state machine and report."""

from __future__ import annotations

import copy
import errno
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from openmed.cli import airgap, airgap_rehearsal, main_module
from openmed.cli.airgap_rehearsal import (
    FAULT_POINTS,
    PHASES,
    RECOVERY_HOOKS,
    LifecycleInstallation,
    RehearsalError,
    RehearsalReportError,
    StagedSet,
    build_rehearsal_report,
    parse_hook_specs,
    rehearse,
    resolve_hooks,
    run_hook,
    run_rehearsal_inside,
    validate_rehearsal_report,
)
from openmed.core.attestation import (
    NO_EGRESS_NEGATIVE_CONTROLS,
    NO_EGRESS_POSITIVE_CONTROLS,
)
from openmed.core.audit import stable_hash
from tests.fixtures.airgap_lifecycle import (
    ALL_HOOKS,
    RELEASE_VERSION,
    UPGRADE_VERSION,
    FakeInstaller,
    build_lifecycle_bundle,
)

_GENERATED_AT = "2026-11-08T12:00:00Z"
_KEY = b"synthetic-rehearsal-signing-key-0001"
_FIXTURES = "tests.fixtures.airgap_lifecycle"


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    return build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")


def _evidence(**overrides: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "kernel": "linux",
        "interfaces": {"total": 1, "loopback_up": 1, "non_loopback_up": 0},
        "routes": {"ipv4_non_loopback": 0, "ipv6_non_loopback": 0},
        "negative_controls": [
            {"id": case, "status": "blocked", "reason_code": "network_unreachable"}
            for case in NO_EGRESS_NEGATIVE_CONTROLS
        ],
        "negative_controls_after": [
            {"id": case, "status": "blocked", "reason_code": "network_unreachable"}
            for case in NO_EGRESS_NEGATIVE_CONTROLS
        ],
        "positive_controls": [
            {"id": case, "status": "reachable", "reason_code": "loopback_reachable"}
            for case in NO_EGRESS_POSITIVE_CONTROLS
        ],
    }
    evidence.update(overrides)
    return evidence


def _inner(kit: Path, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("installer", FakeInstaller())
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return run_rehearsal_inside(kit, work_dir=work, **kwargs)


def _report(rehearsal: dict[str, Any], **kwargs: Any) -> Any:
    kwargs.setdefault("evidence", _evidence())
    return build_rehearsal_report(
        host_kind="linux",
        requested_boundary="auto",
        boundary_kind="netns",
        rehearsal=rehearsal,
        generated_at=_GENERATED_AT,
        **kwargs,
    )


def _phase(result: dict[str, Any], phase: str) -> dict[str, Any]:
    return next(item for item in result["phases"] if item["id"] == phase)


def _check(result: dict[str, Any], phase: str, check: str) -> tuple[str, str]:
    record = next(
        item for item in _phase(result, phase)["checks"] if item["id"] == check
    )
    return record["status"], record["reason_code"]


# -- state machine shape ----------------------------------------------------


def test_phases_follow_the_documented_lifecycle_order() -> None:
    assert list(PHASES) == [
        "preflight",
        "install",
        "smoke",
        "upgrade",
        "injected_failure",
        "rollback",
        "restore",
        "continuity",
    ]
    for hook, phase in RECOVERY_HOOKS.items():
        assert hook in PHASES[phase]


# -- full runs with the fast installer --------------------------------------


def test_every_phase_passes_when_all_recovery_features_are_present(
    kit: Path, tmp_path: Path
) -> None:
    installer = FakeInstaller()

    result = _inner(kit, tmp_path, installer=installer, hooks=ALL_HOOKS)

    assert [phase["status"] for phase in result["phases"]] == ["passed"] * 8
    assert installer.installs == ["install", "upgrade", "upgrade", "rollback"]
    assert result["final_state"] == {
        "version": RELEASE_VERSION,
        "set_id": "rollback",
        "set_digest": result["bundle"]["sets"][2]["digest"],
        "committed_transactions": 3,
        "duplicate_commits": 0,
        "journal_open": False,
    }
    assert result["fault"] == {
        "upgrade_interruption": "after_activate",
        "recovery": "rolled_back_to_checkpoint",
        "disk_fault": "installed_file_modified",
    }
    assert _check(result, "continuity", "admission") == ("passed", "admission_resumed")
    report = _report(result)
    assert report.status == "passed"
    assert report.reason_code == "rehearsal_passed"
    assert report.exit_code == 0
    providers = {
        check["id"]: check["provider"]
        for phase in report.to_dict()["phases"]
        for check in phase["checks"]
        if check["id"] in RECOVERY_HOOKS
    }
    assert set(providers.values()) == {"caller"}


def test_missing_recovery_features_are_reported_and_never_pass(
    kit: Path, tmp_path: Path
) -> None:
    result = _inner(kit, tmp_path)

    statuses = {phase["id"]: phase["status"] for phase in result["phases"]}
    assert statuses == {
        "preflight": "passed",
        "install": "passed",
        "smoke": "passed",
        "upgrade": "incomplete",
        "injected_failure": "passed",
        "rollback": "passed",
        "restore": "incomplete",
        "continuity": "incomplete",
    }
    for hook, phase in RECOVERY_HOOKS.items():
        assert _check(result, phase, hook) == ("not_available", "feature_not_present")
    assert _check(result, "continuity", "admission") == (
        "not_available",
        "recovery_feature_unavailable",
    )
    report = _report(result)
    assert report.status == "incomplete"
    assert report.reason_code == "recovery_feature_unavailable"
    assert report.exit_code == 4
    assert not report.passed


def test_no_declared_migrations_need_no_migration_feature(tmp_path: Path) -> None:
    kit = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "in", migrations=False)

    result = _inner(kit, tmp_path)

    assert _check(result, "upgrade", "migration") == ("passed", "not_required")
    assert _phase(result, "upgrade")["status"] == "passed"


@pytest.mark.parametrize(
    ("fault_point", "recovery"),
    [
        ("before_activate", "rolled_back_to_checkpoint"),
        ("after_activate", "rolled_back_to_checkpoint"),
        ("after_checkpoint", "resumed_committed"),
    ],
)
def test_interrupted_upgrade_lands_on_exactly_one_declared_state(
    kit: Path, tmp_path: Path, fault_point: str, recovery: str
) -> None:
    installer = FakeInstaller()

    result = _inner(
        kit, tmp_path, installer=installer, hooks=ALL_HOOKS, fault_point=fault_point
    )

    assert _check(result, "upgrade", "interrupted_attempt") == ("passed", recovery)
    assert _check(result, "upgrade", "resume_upgrade") == ("passed", "ok")
    assert result["fault"]["recovery"] == recovery
    assert result["final_state"]["committed_transactions"] == 3
    assert result["final_state"]["duplicate_commits"] == 0
    # A committed upgrade is resumed, not installed a second time.
    expected_upgrades = 1 if recovery == "resumed_committed" else 2
    assert installer.installs.count("upgrade") == expected_upgrades


def test_archive_kits_rehearse_like_directories(tmp_path: Path) -> None:
    archive = build_lifecycle_bundle(tmp_path / "kit.tar.gz", tmp_path / "inputs")

    result = _inner(archive, tmp_path, hooks=ALL_HOOKS)

    assert all(phase["status"] == "passed" for phase in result["phases"])


def test_unreadable_bundle_is_a_controlled_preflight_failure(tmp_path: Path) -> None:
    garbage = tmp_path / "kit.tar.gz"
    garbage.write_bytes(b"not an archive")

    result = _inner(garbage, tmp_path)

    assert _check(result, "preflight", "bundle_integrity") == (
        "failed",
        "bundle_unreadable",
    )
    assert all(phase["status"] == "not_run" for phase in result["phases"][1:])


# -- preflight refuses before any mutation ----------------------------------


def _wheel(kit: Path, root: str) -> Path:
    return next((kit / root).glob("*.whl"))


def _rewrite(kit: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = kit / airgap.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _missing(kit: Path) -> None:
    _wheel(kit, "wheels").unlink()


def _modified(kit: Path) -> None:
    with _wheel(kit, "lifecycle/upgrade").open("ab") as handle:
        handle.write(b"\x00")


def _unexpected(kit: Path) -> None:
    (kit / "lifecycle" / "upgrade" / "openmed-9.9.9-py3-none-any.whl").write_bytes(
        b"substituted"
    )


def _incompatible(kit: Path) -> None:
    _rewrite(
        kit,
        lambda m: m["lifecycle"]["sets"]["upgrade"].update(from_versions=["0.0.1"]),
    )


def _misordered(kit: Path) -> None:
    _rewrite(
        kit,
        lambda m: m["lifecycle"]["migrations"][0].update(sequence=7, id="0007"),
    )


def _expired(kit: Path) -> None:
    _rewrite(
        kit,
        lambda m: m["lifecycle"]["validity"].update(not_after="2026-01-01T00:00:00Z"),
    )


@pytest.mark.parametrize(
    ("mutate", "check", "code"),
    [
        (_missing, "bundle_integrity", "artifact_missing"),
        (_modified, "bundle_integrity", "artifact_modified"),
        (_unexpected, "bundle_integrity", "artifact_unexpected"),
        (_incompatible, "compatibility", "upgrade_incompatible"),
        (_misordered, "compatibility", "migration_order_invalid"),
        (_expired, "validity_window", "artifacts_expired"),
    ],
)
def test_bad_artifacts_fail_preflight_before_any_mutation(
    kit: Path,
    tmp_path: Path,
    mutate: Callable[[Path], None],
    check: str,
    code: str,
) -> None:
    mutate(kit)
    installer = FakeInstaller()

    result = _inner(kit, tmp_path, installer=installer, hooks=ALL_HOOKS)

    assert _check(result, "preflight", check) == ("failed", code)
    assert installer.installs == []
    assert result["final_state"] is None
    assert all(phase["status"] == "not_run" for phase in result["phases"][1:])
    report = _report(result)
    assert (report.status, report.reason_code) == ("failed", code)
    assert report.to_dict()["failed_phase"] == "preflight"


def test_incorrectly_ordered_upgrade_is_refused_even_if_built_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(airgap, "_require_valid_lifecycle", lambda manifest: None)
    kit = build_lifecycle_bundle(
        tmp_path / "kit", tmp_path / "inputs", upgrade_version="0.0.1"
    )
    installer = FakeInstaller()

    result = _inner(kit, tmp_path, installer=installer)

    assert _check(result, "preflight", "compatibility") == (
        "failed",
        "version_order_invalid",
    )
    assert installer.installs == []


def test_validity_is_checked_against_the_rehearsal_clock(
    kit: Path, tmp_path: Path
) -> None:
    _rewrite(
        kit,
        lambda m: m["lifecycle"]["validity"].update(not_before="2030-01-01T00:00:00Z"),
    )

    early = _inner(kit, tmp_path, now=datetime(2029, 1, 1, tzinfo=timezone.utc))
    later = _inner(
        kit, tmp_path, now=datetime(2030, 6, 1, tzinfo=timezone.utc), hooks=ALL_HOOKS
    )

    assert _check(early, "preflight", "validity_window") == (
        "failed",
        "artifacts_not_yet_valid",
    )
    assert _check(later, "preflight", "validity_window") == ("passed", "ok")


def test_artifact_changed_after_preflight_is_caught_before_install(
    kit: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = airgap_rehearsal._Rehearsal._phase_preflight

    def _tamper_after_preflight(self: Any) -> None:
        original(self)
        with _wheel(kit, "wheels").open("ab") as handle:
            handle.write(b"\x00")

    monkeypatch.setattr(
        airgap_rehearsal._Rehearsal, "_phase_preflight", _tamper_after_preflight
    )
    installer = FakeInstaller()

    result = _inner(kit, tmp_path, installer=installer)

    assert _check(result, "install", "set_verified") == ("failed", "artifact_modified")
    assert installer.installs == []


# -- transactional installation ----------------------------------------------


def _staged(set_id: str, version: str, wheel: Path) -> StagedSet:
    digest = stable_hash([[wheel.name, set_id]])
    return StagedSet(set_id, version, digest, ((wheel, "0" * 64),))


@pytest.fixture
def wheels(tmp_path: Path) -> dict[str, Path]:
    directory = tmp_path / "wheels"
    directory.mkdir()
    paths = {}
    for name, version in (("release", RELEASE_VERSION), ("upgrade", UPGRADE_VERSION)):
        paths[name] = directory / f"openmed-{version}-py3-none-any.whl"
        paths[name].write_bytes(b"synthetic")
    return paths


def _transition(
    installation: LifecycleInstallation, txn: str, staged: StagedSet
) -> Any:
    return installation.transition(
        txn, staged, distribution="openmed", import_name="openmed"
    )


@pytest.mark.parametrize("fault_point", FAULT_POINTS)
def test_transition_recovery_never_leaves_two_states(
    tmp_path: Path, wheels: dict[str, Path], fault_point: str
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    upgrade = _staged("upgrade", UPGRADE_VERSION, wheels["upgrade"])
    root = tmp_path / "installation"
    installer = FakeInstaller()
    _transition(LifecycleInstallation(root, installer), "install", release)

    armed = LifecycleInstallation(root, installer, fault_point=fault_point)
    with pytest.raises(airgap_rehearsal.InjectedFault):
        _transition(armed, "upgrade", upgrade)
    restarted = LifecycleInstallation(root, installer)
    recovery = restarted.recover()

    expected = upgrade if fault_point == "after_checkpoint" else release
    assert recovery.declared is not None
    assert recovery.declared.set_digest == expected.digest
    assert restarted.active() == recovery.declared == restarted.declared()
    assert restarted.journal() is None
    if fault_point == "after_checkpoint":
        assert recovery.outcome == "resumed_committed"
        assert restarted.slot_path(recovery.declared).is_dir()
    else:
        assert recovery.outcome == "rolled_back_to_checkpoint"
        assert not (root / "releases" / "upgrade").exists()
    resumed = _transition(restarted, "upgrade", upgrade)
    assert resumed.outcome == (
        "already_committed" if fault_point == "after_checkpoint" else "committed"
    )
    assert restarted.checkpoint()["committed"] == ["install", "upgrade"]
    assert restarted.recover().outcome == "clean"


def test_disk_full_during_checkpoint_write_rolls_back(
    tmp_path: Path, wheels: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    upgrade = _staged("upgrade", UPGRADE_VERSION, wheels["upgrade"])
    root = tmp_path / "installation"
    installation = LifecycleInstallation(root, FakeInstaller())
    _transition(installation, "install", release)
    checkpoint_before = installation.checkpoint_path.read_bytes()
    original = airgap_rehearsal._write_json_atomic

    def _disk_full(path: Path, payload: Any) -> None:
        if path.name == "checkpoint.json":
            raise OSError(errno.ENOSPC, "No space left on device")
        original(path, payload)

    monkeypatch.setattr(airgap_rehearsal, "_write_json_atomic", _disk_full)
    with pytest.raises(RehearsalError) as exc_info:
        _transition(installation, "upgrade", upgrade)
    monkeypatch.setattr(airgap_rehearsal, "_write_json_atomic", original)

    assert exc_info.value.code == "checkpoint_failed"
    assert installation.checkpoint_path.read_bytes() == checkpoint_before
    recovery = LifecycleInstallation(root, FakeInstaller()).recover()
    assert recovery.outcome == "rolled_back_to_checkpoint"
    assert recovery.declared is not None
    assert recovery.declared.set_id == "release"
    assert not list(installation.state_dir.glob("*.tmp"))


def test_torn_checkpoint_replace_leaves_the_last_green_checkpoint(
    tmp_path: Path, wheels: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    upgrade = _staged("upgrade", UPGRADE_VERSION, wheels["upgrade"])
    root = tmp_path / "installation"
    installation = LifecycleInstallation(root, FakeInstaller())
    _transition(installation, "install", release)
    checkpoint_before = installation.checkpoint_path.read_bytes()
    original_replace = airgap_rehearsal.os.replace

    def _replace(source: Any, target: Any) -> None:
        if Path(target).name == "checkpoint.json":
            raise OSError(errno.EIO, "Input/output error")
        original_replace(source, target)

    monkeypatch.setattr(airgap_rehearsal.os, "replace", _replace)
    with pytest.raises(RehearsalError):
        _transition(installation, "upgrade", upgrade)
    monkeypatch.undo()

    assert installation.checkpoint_path.read_bytes() == checkpoint_before
    assert not list(installation.state_dir.glob(".*.tmp"))
    recovery = LifecycleInstallation(root, FakeInstaller()).recover()
    assert recovery.outcome == "rolled_back_to_checkpoint"
    assert recovery.declared is not None
    assert recovery.declared.set_id == "release"


def test_failed_staged_smoke_keeps_the_prior_release_active(
    tmp_path: Path, wheels: dict[str, Path]
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    upgrade = _staged("upgrade", UPGRADE_VERSION, wheels["upgrade"])
    installer = FakeInstaller(smoke_versions={"upgrade": "0.0.0"})
    installation = LifecycleInstallation(tmp_path / "installation", installer)
    _transition(installation, "install", release)

    with pytest.raises(RehearsalError) as exc_info:
        _transition(installation, "upgrade", upgrade)

    assert exc_info.value.code == "version_mismatch"
    assert installation.active() == installation.declared()
    assert installation.declared() is not None
    assert installation.declared().set_id == "release"
    assert installation.journal() is None
    assert not (tmp_path / "installation" / "releases" / "upgrade").exists()


def test_open_journal_blocks_a_new_transition(
    tmp_path: Path, wheels: dict[str, Path]
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    upgrade = _staged("upgrade", UPGRADE_VERSION, wheels["upgrade"])
    root = tmp_path / "installation"
    _transition(LifecycleInstallation(root, FakeInstaller()), "install", release)
    armed = LifecycleInstallation(root, FakeInstaller(), fault_point="before_activate")
    with pytest.raises(airgap_rehearsal.InjectedFault):
        _transition(armed, "upgrade", upgrade)

    with pytest.raises(RehearsalError) as exc_info:
        _transition(LifecycleInstallation(root, FakeInstaller()), "rollback", release)

    assert exc_info.value.code == "journal_open"


@pytest.mark.parametrize("name", ["checkpoint.json", "journal.json", "active.json"])
def test_torn_state_is_ambiguous_rather_than_guessed(
    tmp_path: Path, wheels: dict[str, Path], name: str
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    installation = LifecycleInstallation(tmp_path / "installation", FakeInstaller())
    _transition(installation, "install", release)
    (installation.state_dir / name).write_text("{torn", encoding="utf-8")

    with pytest.raises(RehearsalError) as exc_info:
        installation.recover()

    assert exc_info.value.code == "state_ambiguous"


def test_duplicate_commits_are_detected(
    tmp_path: Path, wheels: dict[str, Path]
) -> None:
    release = _staged("release", RELEASE_VERSION, wheels["release"])
    installation = LifecycleInstallation(tmp_path / "installation", FakeInstaller())
    _transition(installation, "install", release)
    checkpoint = installation.checkpoint()
    checkpoint["committed"] = ["install", "install"]
    installation.checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RehearsalError) as exc_info:
        installation.recover()

    assert exc_info.value.code == "duplicate_commit"


# -- hooks ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (["backup_restore"], "hook_spec_invalid"),
        (["backup_restore=not a module"], "hook_spec_invalid"),
        (["telemetry=pkg.mod:fn"], "hook_unknown"),
        (
            ["audit_continuity=pkg:fn", "audit_continuity=pkg:other"],
            "hook_spec_invalid",
        ),
    ],
)
def test_hook_specs_are_validated(values: list[str], code: str) -> None:
    with pytest.raises(RehearsalError) as exc_info:
        parse_hook_specs(values)

    assert exc_info.value.code == code


def _context(tmp_path: Path) -> Any:
    return airgap_rehearsal.HookContext(
        hook="audit_continuity",
        phase="continuity",
        active_version=RELEASE_VERSION,
        active_python=tmp_path / "python",
        state_dir=tmp_path,
        recovery_dir=None,
        migrations=(),
        from_version=RELEASE_VERSION,
        to_version=UPGRADE_VERSION,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (f"{_FIXTURES}:passing_hook", ("passed", "ok", "caller")),
        (f"{_FIXTURES}:failing_hook", ("failed", "continuity_gap", "caller")),
        (f"{_FIXTURES}:leaky_hook", ("failed", "hook_result_invalid", "caller")),
        (f"{_FIXTURES}:raising_hook", ("failed", "hook_error", "caller")),
        (f"{_FIXTURES}:missing_function", ("failed", "hook_error", "caller")),
        ("openmed_no_such_module:hook", ("failed", "hook_error", "caller")),
    ],
)
def test_hook_results_are_normalized_to_controlled_codes(
    tmp_path: Path, spec: str, expected: tuple[str, str, str]
) -> None:
    hooks = resolve_hooks({"audit_continuity": spec})

    record = run_hook(hooks["audit_continuity"], _context(tmp_path))

    assert (record["status"], record["reason_code"], record["provider"]) == expected
    assert str(tmp_path) not in json.dumps(record)


def test_unregistered_features_resolve_to_not_available(tmp_path: Path) -> None:
    hooks = resolve_hooks()

    assert {hook.provider for hook in hooks.values()} == {"none"}
    record = run_hook(hooks["backup_restore"], _context(tmp_path))
    assert record == {
        "status": "not_available",
        "reason_code": "feature_not_present",
        "digest": None,
        "provider": "none",
    }


@pytest.mark.parametrize(
    "result",
    [
        {"status": "passed", "reason_code": "continuity_gap"},
        {"status": "not_available", "reason_code": "ok"},
        {"status": "failed", "reason_code": "ok"},
        {"status": "passed", "reason_code": "ok", "digest": "md5:abc"},
        {"status": "maybe"},
        ["passed"],
    ],
)
def test_inconsistent_hook_results_are_rejected(tmp_path: Path, result: Any) -> None:
    hook = airgap_rehearsal.ResolvedHook("audit_continuity", "caller", lambda _: result)

    record = run_hook(hook, _context(tmp_path))

    assert (record["status"], record["reason_code"]) == (
        "failed",
        "hook_result_invalid",
    )


def test_failing_continuity_hook_fails_the_rehearsal(kit: Path, tmp_path: Path) -> None:
    hooks = dict(ALL_HOOKS, audit_continuity=f"{_FIXTURES}:failing_hook")

    result = _inner(kit, tmp_path, hooks=hooks)

    assert _check(result, "continuity", "audit_continuity") == (
        "failed",
        "continuity_gap",
    )
    assert _check(result, "continuity", "admission") == (
        "not_run",
        "prior_check_failed",
    )
    report = _report(result)
    assert (report.status, report.reason_code) == ("failed", "continuity_gap")
    assert report.to_dict()["failed_phase"] == "continuity"


# -- report -----------------------------------------------------------------


def test_report_is_value_free_bounded_and_reproducible(
    kit: Path, tmp_path: Path
) -> None:
    hooks = dict(ALL_HOOKS, credential_reload=f"{_FIXTURES}:raising_hook")
    result = _inner(kit, tmp_path, hooks=hooks)

    first = _report(result)
    second = build_rehearsal_report(
        host_kind="linux",
        requested_boundary="auto",
        boundary_kind="netns",
        evidence=_evidence(),
        rehearsal=result,
        generated_at="2026-11-09T08:30:00Z",
    )
    text = first.to_json()

    assert len(text.encode()) <= airgap_rehearsal.MAX_REPORT_BYTES
    for forbidden in (str(tmp_path), "secret", "patient", ".whl", "/airgap"):
        assert forbidden not in text
    assert first.to_dict()["repro_hash"] == second.to_dict()["repro_hash"]
    assert first.to_dict()["integrity_hash"] != second.to_dict()["integrity_hash"]
    assert _check(result, "restore", "credential_reload") == ("failed", "hook_error")
    markdown = first.to_markdown()
    assert "credential_reload" in markdown and str(tmp_path) not in markdown


def test_signed_report_verifies_offline_only_with_its_key(
    kit: Path, tmp_path: Path
) -> None:
    report = _report(_inner(kit, tmp_path, hooks=ALL_HOOKS), key=_KEY, key_id="site-a")
    payload = report.to_dict()

    assert payload["signature"]["algorithm"] == "HMAC-SHA256"
    assert payload["signature"]["key_id"] == "site-a"
    assert _KEY.decode() not in report.to_json()
    validate_rehearsal_report(payload, key=_KEY, require_signature=True)
    with pytest.raises(RehearsalReportError) as wrong:
        validate_rehearsal_report(payload, key=b"another-synthetic-key-000")
    assert wrong.value.code == "signature_mismatch"
    with pytest.raises(RehearsalReportError) as keyless:
        validate_rehearsal_report(payload, require_signature=True)
    assert keyless.value.code == "signature_key_required"
    unsigned = _report(_inner(kit, tmp_path, hooks=ALL_HOOKS)).to_dict()
    with pytest.raises(RehearsalReportError) as missing:
        validate_rehearsal_report(unsigned, key=_KEY, require_signature=True)
    assert missing.value.code == "signature_missing"


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(payload)
    payload["signature"] = None
    payload.pop("repro_hash")
    payload.pop("integrity_hash")
    stable = {
        k: v for k, v in payload.items() if k not in {"generated_at", "signature"}
    }
    payload["repro_hash"] = stable_hash(stable)
    unsigned = {k: v for k, v in payload.items() if k != "signature"}
    payload["integrity_hash"] = stable_hash(unsigned)
    return payload


def _flip_status(payload: dict[str, Any]) -> None:
    payload["status"] = "passed"
    payload["reason_code"] = "rehearsal_passed"


def _hide_hook(payload: dict[str, Any]) -> None:
    for check in payload["phases"][6]["checks"]:
        check.update(status="passed", reason_code="ok")


def _add_value(payload: dict[str, Any]) -> None:
    payload["phases"][0]["checks"][0]["path"] = "/srv/kit"


@pytest.mark.parametrize(
    ("tamper", "rehash", "code"),
    [
        (_flip_status, False, "status_mismatch"),
        (_flip_status, True, "status_mismatch"),
        (_hide_hook, True, "report_schema_invalid"),
        (_add_value, True, "report_schema_invalid"),
        (
            lambda p: p["bundle"].update(manifest_digest="sha256:" + "0" * 64),
            False,
            "repro_hash_mismatch",
        ),
        (
            lambda p: p.update(generated_at="2026-11-10T00:00:00Z"),
            False,
            "integrity_hash_mismatch",
        ),
        (lambda p: p.update(extra=1), True, "report_schema_invalid"),
    ],
)
def test_edited_reports_do_not_validate(
    kit: Path,
    tmp_path: Path,
    tamper: Callable[[dict[str, Any]], None],
    rehash: bool,
    code: str,
) -> None:
    payload = _report(_inner(kit, tmp_path)).to_dict()
    tamper(payload)
    if rehash:
        payload = _rehash(payload)

    with pytest.raises(RehearsalReportError) as exc_info:
        validate_rehearsal_report(payload)

    assert exc_info.value.code == code


def _not_blocked(key: str) -> dict[str, Any]:
    records = copy.deepcopy(_evidence()[key])
    records[0].update(status="not_blocked", reason_code="connection_established")
    return {key: records}


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (_not_blocked("negative_controls"), "egress_not_blocked"),
        (_not_blocked("negative_controls_after"), "egress_not_blocked"),
        ({"negative_controls_after": []}, "negative_control_incomplete"),
        (
            {"interfaces": {"total": 2, "loopback_up": 1, "non_loopback_up": 1}},
            "boundary_not_isolated",
        ),
        ({"kernel": "unknown"}, "boundary_not_isolated"),
        ({"positive_controls": []}, "positive_control_failed"),
    ],
)
def test_a_leaky_or_broken_boundary_fails_the_rehearsal(
    kit: Path, tmp_path: Path, overrides: dict[str, Any], code: str
) -> None:
    report = _report(
        _inner(kit, tmp_path, hooks=ALL_HOOKS), evidence=_evidence(**overrides)
    )

    assert (report.status, report.reason_code) == ("failed", code)
    assert report.exit_code == 1


def test_in_boundary_results_with_values_are_rejected(
    kit: Path, tmp_path: Path
) -> None:
    result = _inner(kit, tmp_path)
    result["phases"][1]["checks"][0]["reason_code"] = "copied /srv/kit/wheels"

    with pytest.raises(RehearsalReportError) as exc_info:
        _report(result)

    assert exc_info.value.code == "rehearsal_result_invalid"


# -- host orchestration -----------------------------------------------------


_LOAD_REAL_GATE = airgap_rehearsal.load_no_egress_gate


def _gate_module() -> Any:
    gate = _LOAD_REAL_GATE()
    assert gate is not None
    return gate


def _fake_gate(
    tmp_path: Path,
    kit: Path,
    *,
    kind: str = "netns",
    skip_reason: str | None = None,
    stdout: str | None = None,
    failure: str | None = None,
    hooks: dict[str, str] | None = None,
    commands: list[list[str]] | None = None,
) -> Any:
    real = _gate_module()
    plan = real.BoundaryPlan(
        kind if skip_reason is None else "none",
        skip_reason=skip_reason,
        executable="/usr/bin/unshare" if kind == "netns" else "/usr/bin/docker",
        image="openmed-airgap:local" if kind == "container" else None,
    )

    def _run_in_boundary(plan: Any, command: list[str], **kwargs: Any) -> Any:
        if commands is not None:
            commands.append(list(command))
        if failure is not None:
            return real.BoundaryRun("", -1, failure)
        if stdout is not None:
            return real.BoundaryRun(stdout, 0)
        result = _evidence()
        result["rehearsal"] = _inner(kit, tmp_path, hooks=hooks)
        return real.BoundaryRun(real.RESULT_MARKER + json.dumps(result) + "\n", 0)

    return SimpleNamespace(
        IMAGE_ENV_VAR=real.IMAGE_ENV_VAR,
        host_kind=lambda: "linux",
        select_boundary=lambda *args, **kwargs: plan,
        boundary_command=real.boundary_command,
        run_in_boundary=_run_in_boundary,
        parse_inner_output=real.parse_inner_output,
    )


@pytest.mark.parametrize("kind", ["netns", "container"])
def test_every_phase_runs_inside_the_selected_boundary(
    kit: Path, tmp_path: Path, kind: str
) -> None:
    commands: list[list[str]] = []
    gate = _fake_gate(tmp_path, kit, kind=kind, hooks=ALL_HOOKS, commands=commands)

    report = rehearse(
        kit,
        gate=gate,
        hooks=ALL_HOOKS,
        generated_at=_GENERATED_AT,
        container_tmpfs_size="2g",
        key=_KEY,
    )

    assert report.status == "passed"
    assert report.to_dict()["boundary"]["kind"] == kind
    (command,) = commands
    script = str(airgap_rehearsal.INNER_SCRIPT)
    if kind == "netns":
        assert command[:5] == [
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--",
        ]
        assert command[6] == script
        assert command[command.index("--bundle") + 1] == str(kit.resolve())
    else:
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--pull") + 1] == "never"
        assert f"type=bind,src={kit.resolve()},dst=/airgap/bundle,readonly" in command
        assert "/tmp:rw,nosuid,nodev,size=2g" in command
        assert command[command.index("--bundle") + 1] == "/airgap/bundle"
        assert (
            command[command.index("python") + 1]
            == "/workspace/scripts/airgap/rehearse_inside.py"
        )
    hook_args = [command[i + 1] for i, arg in enumerate(command) if arg == "--hook"]
    assert len(hook_args) == len(ALL_HOOKS)


@pytest.mark.parametrize("reason", ["unsupported_host", "boundary_unavailable"])
def test_hosts_without_a_boundary_are_skipped_never_passed(
    kit: Path, tmp_path: Path, reason: str
) -> None:
    gate = _fake_gate(tmp_path, kit, skip_reason=reason)

    report = rehearse(kit, gate=gate, generated_at=_GENERATED_AT)

    assert (report.status, report.reason_code, report.exit_code) == (
        "skipped",
        reason,
        3,
    )
    assert report.to_dict()["phases"] == []
    validate_rehearsal_report(report.to_dict())


def test_real_gate_reports_macos_as_unsupported(
    kit: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(gate, "host_kind", lambda: "darwin")
    monkeypatch.delenv(gate.IMAGE_ENV_VAR, raising=False)

    report = rehearse(kit, gate=gate, boundary="netns", generated_at=_GENERATED_AT)

    assert (report.status, report.reason_code) == ("skipped", "unsupported_host")


def test_missing_gate_checkout_is_skipped(kit: Path, tmp_path: Path) -> None:
    report = rehearse(
        kit,
        generated_at=_GENERATED_AT,
        inner_script=tmp_path / "absent.py",
    )

    assert (report.status, report.reason_code) == ("skipped", "gate_unavailable")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"failure": "runner_timeout"}, "runner_timeout"),
        ({"stdout": "no marker\n"}, "runner_failed"),
        ({"stdout": "OPENMED_NO_EGRESS_RESULT {}\n"}, "runner_result_invalid"),
    ],
)
def test_runner_problems_fail_closed(
    kit: Path, tmp_path: Path, kwargs: dict[str, Any], code: str
) -> None:
    report = rehearse(kit, gate=_fake_gate(tmp_path, kit, **kwargs))

    assert (report.status, report.reason_code) == ("failed", code)
    assert report.to_dict()["phases"] == []


def test_invalid_in_boundary_result_becomes_a_runner_failure(
    kit: Path, tmp_path: Path
) -> None:
    gate = _gate_module()
    result = _evidence()
    result["rehearsal"] = {"phases": "everything passed"}
    stdout = gate.RESULT_MARKER + json.dumps(result) + "\n"

    report = rehearse(kit, gate=_fake_gate(tmp_path, kit, stdout=stdout))

    assert (report.status, report.reason_code) == ("failed", "runner_result_invalid")


def test_rehearse_rejects_invalid_arguments(kit: Path) -> None:
    for kwargs, code in (
        ({"boundary": "host"}, "boundary_invalid"),
        ({"fault_point": "never"}, "fault_point_invalid"),
        ({"hooks": {"backup_restore": "not valid"}}, "hook_spec_invalid"),
        ({"generated_at": "yesterday"}, "generated_at_invalid"),
    ):
        with pytest.raises(RehearsalError) as exc_info:
            rehearse(kit, **kwargs)
        assert exc_info.value.code == code


# -- command line -----------------------------------------------------------


def test_cli_rehearse_writes_a_signed_report_and_verifies_it(
    kit: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _fake_gate(tmp_path, kit, hooks=ALL_HOOKS)
    monkeypatch.setattr(airgap_rehearsal, "load_no_egress_gate", lambda: gate)
    key_file = tmp_path / "rehearsal.key"
    key_file.write_bytes(_KEY + b"\n")
    output = tmp_path / "evidence" / "rehearsal.json"
    markdown = tmp_path / "evidence" / "rehearsal.md"
    hook_args = [
        arg for name, spec in ALL_HOOKS.items() for arg in ("--hook", f"{name}={spec}")
    ]

    code = main_module.main(
        [
            "airgap",
            "rehearse",
            str(kit),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--key-file",
            str(key_file),
            "--generated-at",
            _GENERATED_AT,
            *hook_args,
        ]
    )
    err = capsys.readouterr().err

    assert code == 0
    assert "status=passed" in err and "signed=yes" in err
    assert str(kit) not in err and str(tmp_path) not in err
    assert "rehearsal_passed" in markdown.read_text(encoding="utf-8")
    assert main_module.main(["airgap", "verify-report", str(output)]) == 0
    assert (
        main_module.main(
            [
                "airgap",
                "verify-report",
                str(output),
                "--key-file",
                str(key_file),
                "--require-signature",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["phases"][3]["status"] = "failed"
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert main_module.main(["airgap", "verify-report", str(output)]) == 1
    assert "report_schema_invalid" in capsys.readouterr().err


def test_cli_rehearse_exit_codes_distinguish_incomplete_and_skipped(
    kit: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = _fake_gate(tmp_path, kit)
    monkeypatch.setattr(airgap_rehearsal, "load_no_egress_gate", lambda: incomplete)
    assert (
        main_module.main(
            ["airgap", "rehearse", str(kit), "--output", str(tmp_path / "a.json")]
        )
        == 4
    )
    skipped = _fake_gate(tmp_path, kit, skip_reason="unsupported_host")
    monkeypatch.setattr(airgap_rehearsal, "load_no_egress_gate", lambda: skipped)
    assert (
        main_module.main(
            ["airgap", "rehearse", str(kit), "--output", str(tmp_path / "b.json")]
        )
        == 3
    )
    assert main_module.main(["airgap", "verify-report", str(tmp_path / "b.json")]) == 3


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--hook", "telemetry=pkg:fn"], "hook_unknown"),
        (["--generated-at", "yesterday"], "generated_at_invalid"),
        (["--key-id", "bad id!"], "signing_key_id_invalid"),
    ],
)
def test_cli_rehearse_usage_errors_exit_two(
    kit: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    assert main_module.main(["airgap", "rehearse", str(kit), *extra]) == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_short_keys_and_missing_inputs(
    tmp_path: Path, kit: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = tmp_path / "short.key"
    short.write_bytes(b"too-short")

    assert (
        main_module.main(["airgap", "rehearse", str(kit), "--key-file", str(short)])
        == 2
    )
    assert "signing_key_too_short" in capsys.readouterr().err
    assert main_module.main(["airgap", "rehearse", str(tmp_path / "absent")]) == 2
    assert "bundle_unreadable" in capsys.readouterr().err
    assert (
        main_module.main(
            [
                "airgap",
                "verify-report",
                str(tmp_path / "absent.json"),
                "--require-signature",
            ]
        )
        == 1
    )
    assert (
        main_module.main(["airgap", "verify-report", str(tmp_path / "absent.json")])
        == 1
    )
    assert "report_unreadable" in capsys.readouterr().err


def test_archives_are_mounted_like_directories(tmp_path: Path) -> None:
    archive = build_lifecycle_bundle(tmp_path / "kit.tar.gz", tmp_path / "inputs")
    commands: list[list[str]] = []
    gate = _fake_gate(tmp_path, archive, kind="container", commands=commands)

    rehearse(archive, gate=gate)

    assert tarfile.is_tarfile(archive)
    assert (
        f"type=bind,src={archive.resolve()},dst=/airgap/bundle,readonly" in commands[0]
    )


def test_report_timestamps_must_be_canonical(kit: Path, tmp_path: Path) -> None:
    payload = _report(_inner(kit, tmp_path)).to_dict()
    payload["generated_at"] = datetime(
        2026, 11, 8, tzinfo=timezone(timedelta(hours=2))
    ).isoformat()

    with pytest.raises(RehearsalReportError) as exc_info:
        validate_rehearsal_report(_rehash(payload))

    assert exc_info.value.code == "report_schema_invalid"
