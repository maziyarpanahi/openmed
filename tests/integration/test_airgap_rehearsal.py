"""Rehearse the air-gap lifecycle with real virtual environments and pip.

The boundary test needs a Linux host that permits unprivileged user and network
namespaces (``unshare --user --map-root-user --net``), or a Docker daemon plus a
local image named by ``OPENMED_NO_EGRESS_IMAGE`` with Python 3.11+, ``venv``,
and OpenMed's core dependencies. Hosts without either report a controlled skip;
that skip is never a pass.

Every artifact is synthetic: tiny pure-Python ``openmed`` wheels and a
placeholder model snapshot built offline in a temporary directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.cli.airgap_rehearsal import (
    PHASES,
    PipVenvInstaller,
    run_rehearsal_inside,
    validate_rehearsal_report,
)
from openmed.core.attestation import NO_EGRESS_NEGATIVE_CONTROLS
from tests.fixtures.airgap_lifecycle import (
    ALL_HOOKS,
    RELEASE_VERSION,
    build_lifecycle_bundle,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="the rehearsal's in-boundary runner targets POSIX hosts",
    ),
]

_KEY = b"synthetic-integration-rehearsal-key"


def _world_readable(root: Path) -> None:
    """Let the container's unprivileged user read a bind-mounted kit."""

    for path in [root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def test_lifecycle_rehearsal_runs_inside_the_os_no_egress_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kit = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    _world_readable(kit)
    key_file = tmp_path / "rehearsal.key"
    key_file.write_bytes(_KEY)
    report_path = tmp_path / "rehearsal.json"
    hook_args = [
        arg for name, spec in ALL_HOOKS.items() for arg in ("--hook", f"{name}={spec}")
    ]

    code = main_module.main(
        [
            "airgap",
            "rehearse",
            str(kit),
            "--output",
            str(report_path),
            "--key-file",
            str(key_file),
            "--generated-at",
            "2026-11-08T12:00:00Z",
            *hook_args,
        ]
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if code == 3:
        assert payload["status"] == "skipped"
        image = os.environ.get("OPENMED_NO_EGRESS_IMAGE")
        pytest.skip(
            "no OS-enforced no-egress boundary on this host "
            f"(reason_code={payload['reason_code']}, host={payload['host']['kind']}); "
            "needs unprivileged user+network namespaces on Linux or Docker with "
            f"OPENMED_NO_EGRESS_IMAGE set (currently {'set' if image else 'unset'})"
        )

    assert code == 0, capsys.readouterr().err
    report = validate_rehearsal_report(payload, key=_KEY, require_signature=True)
    assert report.passed
    assert payload["boundary"]["kind"] in {"netns", "container"}
    assert payload["boundary"]["isolated"] is True
    for key in ("negative_controls", "negative_controls_after"):
        assert [case["id"] for case in payload[key]] == list(
            NO_EGRESS_NEGATIVE_CONTROLS
        )
        assert all(case["status"] == "blocked" for case in payload[key])
    assert [phase["id"] for phase in payload["phases"]] == list(PHASES)
    assert all(phase["status"] == "passed" for phase in payload["phases"])
    assert payload["final_state"]["set_id"] == "rollback"
    assert payload["final_state"]["version"] == RELEASE_VERSION
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")


def test_lifecycle_phases_pass_with_real_pip_on_loopback_without_the_boundary(
    tmp_path: Path,
) -> None:
    """Exercise the venv + hash-locked pip path itself on any POSIX host."""

    kit = build_lifecycle_bundle(tmp_path / "kit", tmp_path / "inputs")
    work = tmp_path / "work"
    work.mkdir()

    result = run_rehearsal_inside(
        kit, hooks=ALL_HOOKS, installer=PipVenvInstaller(), work_dir=work
    )

    failed = {
        check["id"]: check["reason_code"]
        for phase in result["phases"]
        for check in phase["checks"]
        if check["status"] != "passed"
    }
    assert failed == {}
    checks = {
        (phase["id"], check["id"]): check
        for phase in result["phases"]
        for check in phase["checks"]
    }
    # The same release bytes install to the same payload digest every time.
    assert (
        checks[("install", "install_release")]["digest"]
        == checks[("rollback", "rollback_release")]["digest"]
    )
    assert (
        checks[("upgrade", "smoke_upgraded")]["digest"]
        != checks[("smoke", "import_release")]["digest"]
    )
    assert result["fault"]["disk_fault"] == "installed_file_modified"
    assert list(work.iterdir()) == []
