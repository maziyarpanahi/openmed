"""End-to-end proof of the v4.0 kit release gate in a real Git checkout.

The script entry point builds the packet from synthetic evidence in a
throwaway repository, then verifies it offline in a separate process. The
signing key is the committed TEST-ONLY fixture key.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.eval.kit_release import (
    KIT_RELEASE_NOT_READY,
    KIT_RELEASE_READY,
    main,
    verify_kit_release_packet,
)
from tests.fixtures.kit_release import (
    OTHER_COMMIT,
    TEST_ONLY_SIGNING_KEY,
    git_checkout,
    retarget,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release" / "kit_release_gate.py"

pytestmark = pytest.mark.integration


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env=_environment(),
    )


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("OPENMED_KIT_RELEASE_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_script_builds_and_verifies_a_signed_packet_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "candidate"
    head = git_checkout(checkout)
    manifest = write_evidence(checkout, prefix="build/kit-release")
    retarget(manifest, checkout, head)
    manifest_path = tmp_path / "kit-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    key_file = tmp_path / "test-only-key.txt"
    key_file.write_bytes(TEST_ONLY_SIGNING_KEY)
    packet_path = tmp_path / "kit-release-packet.json"

    built = _run(
        "--manifest",
        str(manifest_path),
        "--repo-root",
        str(checkout),
        "--signing-key-file",
        str(key_file),
        "--key-id",
        "test-only-kit-release",
        "--output",
        str(packet_path),
    )
    assert built.returncode == 0, built.stdout + built.stderr
    assert f"-> {KIT_RELEASE_READY}" in built.stdout

    verified = _run(
        "--verify",
        str(packet_path),
        "--signing-key-file",
        str(key_file),
        "--expected-candidate",
        head,
        "--require-ready",
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    assert payload["repository_binding"] == {
        "blocking_codes": [],
        "head_commit": head,
        "verified": True,
    }
    assert verify_kit_release_packet(payload, TEST_ONLY_SIGNING_KEY).valid

    # Further checks run in-process against the same real checkout.
    assert (
        main(
            [
                "--verify",
                str(packet_path),
                "--signing-key-file",
                str(key_file),
                "--expected-candidate",
                OTHER_COMMIT,
            ]
        )
        == 1
    )
    assert "candidate_mismatch" in capsys.readouterr().out
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(checkout),
                "--signing-key-file",
                str(key_file),
                "--output",
                str(packet_path),
            ]
        )
        == 2
    )
    assert "output_exists" in capsys.readouterr().err
    for output in (built, verified):
        assert str(tmp_path) not in output.stdout + output.stderr


def test_evidence_from_another_commit_yields_a_signed_not_ready_packet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "candidate"
    head = git_checkout(checkout)
    manifest = write_evidence(checkout)
    retarget(manifest, checkout, head)
    manifest["release"]["candidate_commit"] = OTHER_COMMIT
    manifest_path = tmp_path / "kit-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    key_file = tmp_path / "test-only-key.txt"
    key_file.write_bytes(TEST_ONLY_SIGNING_KEY)
    packet_path = tmp_path / "kit-release-packet.json"

    status = main(
        [
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(checkout),
            "--signing-key-file",
            str(key_file),
            "--output",
            str(packet_path),
            "--json",
        ]
    )

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == KIT_RELEASE_NOT_READY
    assert "checkout_commit_mismatch" in payload["blocking_codes"]
    assert "wrong_candidate:support-matrix" in payload["blocking_codes"]
    assert "wrong_candidate:lane-ci-swift-macos" in payload["blocking_codes"]
    assert verify_kit_release_packet(payload, TEST_ONLY_SIGNING_KEY).valid
