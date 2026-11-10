from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.labels import DIRECT_IDENTIFIER
from openmed.core.policy import load_policy


def _profile_payload(name: str = "hipaa_safe_harbor") -> dict[str, object]:
    return copy.deepcopy(load_policy(name).to_dict())


def _write_profile(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stdout_report(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_policy_lint_cli_accepts_bundled_policy_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(["policy", "lint", "hipaa_safe_harbor"])
    report = _stdout_report(capsys)

    assert result == 0
    assert report["error_count"] == 0
    assert report["warning_count"] == 0


def test_policy_lint_cli_exits_zero_on_clean_profile_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_profile(tmp_path, _profile_payload())

    result = main_module.main(["policy", "lint", str(path)])
    report = _stdout_report(capsys)

    assert result == 0
    assert report["error_count"] == 0


def test_policy_lint_cli_exits_nonzero_on_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _profile_payload()
    payload["threshold_profile"] = "missing_threshold_profile"
    path = _write_profile(tmp_path, payload)

    result = main_module.main(["policy", "lint", str(path)])
    report = _stdout_report(capsys)

    assert result == 1
    assert report["error_count"] == 1
    assert report["errors"][0]["code"] == "POLICY_UNKNOWN_THRESHOLD_PROFILE"


def test_policy_lint_cli_strict_exits_nonzero_on_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _profile_payload()
    policy_label_actions = payload["policy_label_actions"]
    assert isinstance(policy_label_actions, dict)
    policy_label_actions[DIRECT_IDENTIFIER] = "keep"
    path = _write_profile(tmp_path, payload)

    non_strict = main_module.main(["policy", "lint", str(path)])
    non_strict_report = _stdout_report(capsys)
    strict = main_module.main(["policy", "lint", "--strict", str(path)])
    strict_report = _stdout_report(capsys)

    assert non_strict == 0
    assert non_strict_report["error_count"] == 0
    assert non_strict_report["warning_count"] > 0
    assert strict == 1
    assert strict_report["error_count"] == 0
    assert strict_report["warning_count"] > 0
