"""End-to-end v4.0 service release gate: drill, inventory, build and verify."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.eval import service_recovery_drill as drill
from openmed.eval.service_release import (
    SERVICE_RELEASE_NOT_READY,
    SERVICE_RELEASE_READY,
    evaluate_service_release,
    main,
    verify_service_release_packet,
)
from tests.fixtures import service_release as fx

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "release" / "service_release_gate.py"
DRILL = ROOT / "scripts" / "release" / "service_recovery_drill.py"

pytestmark = pytest.mark.integration


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


def _candidate_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "README.md").write_text("synthetic candidate\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "OpenMed Test")
    _git(root, "config", "user.email", "test@openmed.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", "synthetic candidate")
    return root, _git(root, "rev-parse", "HEAD")


def _run(script: Path, *arguments: str, cwd: Path) -> subprocess.CompletedProcess:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENMED_SERVICE_RELEASE")
    }
    environment["OPENMED_SERVICE_RELEASE_KEY"] = fx.TEST_ONLY_SIGNING_KEY.decode()
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def test_real_checkout_binding_reaches_ready_with_stand_in_producers(
    tmp_path: Path,
) -> None:
    root, commit = _candidate_checkout(tmp_path)
    _root, manifest = fx.make_service_release(tmp_path, candidate=commit, root=root)

    packet = evaluate_service_release(
        manifest,
        repo_root=root,
        signing_key=fx.TEST_ONLY_SIGNING_KEY,
        key_id=fx.TEST_ONLY_KEY_ID,
        loaders=fx.stand_in_loaders(),
    )
    assert packet.decision == SERVICE_RELEASE_READY, packet.blocking_codes
    assert packet.repository_binding == {
        "blocking_codes": [],
        "head_commit": commit,
        "verified": True,
    }
    assert verify_service_release_packet(
        packet.to_dict(),
        fx.TEST_ONLY_SIGNING_KEY,
        expected_candidate=commit,
        require_ready=True,
    ).valid

    (root / "CHANGE.md").write_text("later commit\n", encoding="utf-8")
    _git(root, "add", "CHANGE.md")
    _git(root, "commit", "-q", "-m", "later")
    moved = evaluate_service_release(
        manifest,
        repo_root=root,
        signing_key=fx.TEST_ONLY_SIGNING_KEY,
        loaders=fx.stand_in_loaders(),
    )
    assert moved.blocking_codes == ("checkout_commit_mismatch",)


def test_master_cli_flow_signs_not_ready_and_verifies_offline(tmp_path: Path) -> None:
    root, commit = _candidate_checkout(tmp_path)
    _root, manifest = fx.make_service_release(
        tmp_path, candidate=commit, root=root, omit=("recovery-drill", "failure-codes")
    )

    drilled = _run(
        DRILL,
        "--output",
        "evidence/drill.json",
        "--generated-at",
        fx.GENERATED_AT,
        cwd=root,
    )
    assert drilled.returncode == 1, drilled.stderr
    report = json.loads((root / "evidence" / "drill.json").read_text())
    assert report["candidate_commit"] == commit
    assert report["state"] == drill.DRILL_INCOMPLETE

    inventory = _run(
        GATE,
        "--write-failure-codes",
        "evidence/failure-codes.json",
        "--candidate",
        commit,
        cwd=root,
    )
    assert inventory.returncode == 0, inventory.stderr
    for entry_id, kind, relative in (
        ("recovery-drill", "recovery_drill_report", "evidence/drill.json"),
        ("failure-codes", "failure_code_inventory", "evidence/failure-codes.json"),
    ):
        manifest["evidence"].append(
            {
                "candidate_commit": commit,
                "generated_at": fx.GENERATED_AT,
                "id": entry_id,
                "kind": kind,
                "path": relative,
                "sha256": fx.digest((root / relative).read_bytes()),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    built = _run(
        GATE,
        "--manifest",
        "manifest.json",
        "--output",
        "packet.json",
        "--repo-root",
        str(root),
        cwd=root,
    )
    assert built.returncode == 1, built.stderr
    assert "NOT_READY" in built.stdout
    assert "typed_loader_unavailable:security-profile" in built.stdout
    assert "drill_step_not_available:recovery-drill" in built.stdout
    for leaked in (str(tmp_path), fx.TEST_ONLY_SIGNING_KEY.decode()):
        assert leaked not in built.stdout + built.stderr

    verified = _run(
        GATE, "--verify", "packet.json", "--expected-candidate", commit, cwd=root
    )
    assert verified.returncode == 0, verified.stdout
    assert "VALID; decision=NOT_READY" in verified.stdout

    required = _run(GATE, "--verify", "packet.json", "--require-ready", cwd=root)
    assert required.returncode == 1
    assert "decision_not_ready" in required.stdout

    packet = json.loads((root / "packet.json").read_text())
    assert packet["decision"] == SERVICE_RELEASE_NOT_READY
    packet["evidence"].pop()
    (root / "tampered.json").write_text(json.dumps(packet), encoding="utf-8")
    tampered = _run(GATE, "--verify", "tampered.json", cwd=root)
    assert tampered.returncode == 1
    assert "signature_invalid" in tampered.stdout

    again = _run(
        GATE, "--manifest", "manifest.json", "--output", "packet.json", cwd=root
    )
    assert again.returncode == 2
    assert "output_exists" in again.stderr


def test_cli_refuses_missing_key_and_bad_arguments(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENMED_SERVICE_RELEASE_KEY", raising=False)
    assert main(["--verify", str(tmp_path / "packet.json")]) == 2
    assert "signing_key_missing" in capsys.readouterr().err

    key_file = tmp_path / "key"
    key_file.write_bytes(fx.TEST_ONLY_SIGNING_KEY)
    assert (
        main(
            [
                "--verify",
                str(tmp_path / "missing.json"),
                "--signing-key-file",
                str(key_file),
            ]
        )
        == 2
    )
    assert "packet_unreadable" in capsys.readouterr().err
    assert (
        main(
            [
                "--verify",
                str(tmp_path / "missing.json"),
                "--signing-key-file",
                str(key_file),
                "--expected-candidate",
                "HEAD",
            ]
        )
        == 2
    )
    assert "expected_candidate_invalid" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--manifest", "manifest.json"])
    with pytest.raises(SystemExit):
        main(["--write-failure-codes", "codes.json"])
