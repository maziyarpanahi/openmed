import json
import socket
from pathlib import Path

import pytest

from scripts.privacy_scan import main


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scan_is_deterministic_and_report_contains_counts_only(tmp_path, capsys):
    source = tmp_path / "trace.txt"
    report = tmp_path / "report.json"
    synthetic_email = "casey@example.test"
    source.write_text(f"email: {synthetic_email}\n", encoding="utf-8")

    first_status = main(
        ["--paths", str(source), "--output", str(report), "--policy", "default"]
    )
    first_output = capsys.readouterr()
    first_report = _read_report(report)

    second_status = main(
        ["--paths", str(source), "--output", str(report), "--policy", "default"]
    )
    second_output = capsys.readouterr()
    second_report = _read_report(report)

    assert first_status == second_status == 1
    assert first_report == second_report
    assert first_report["status"] == "failed"
    assert first_report["findings"] == 1
    assert first_report["findings_by_rule"] == {"email": 1}
    assert synthetic_email not in first_output.out + first_output.err
    assert synthetic_email not in second_output.out + second_output.err
    assert synthetic_email not in report.read_text(encoding="utf-8")


def test_synthetic_fixture_allowlist_skips_only_the_selected_file(tmp_path, capsys):
    allowlisted = tmp_path / "synthetic.txt"
    scanned = tmp_path / "trace.txt"
    report = tmp_path / "report.json"
    allowlisted.write_text("patient_id: synthetic-record-123\n", encoding="utf-8")
    scanned.write_text("phone: +1 555-010-1234\n", encoding="utf-8")

    status = main(
        [
            "--paths",
            f"{allowlisted}\n{scanned}",
            "--synthetic-fixture-allowlist",
            str(allowlisted),
            "--output",
            str(report),
        ]
    )
    output = capsys.readouterr()
    payload = _read_report(report)

    assert status == 1
    assert payload["allowlisted_files"] == 1
    assert payload["scanned_files"] == 1
    assert payload["findings"] == 1
    assert "synthetic-record-123" not in output.out + output.err


def test_fixture_contents_cannot_expand_the_allowlist(tmp_path, capsys):
    allowlisted = tmp_path / "synthetic.txt"
    scanned = tmp_path / "trace.txt"
    report = tmp_path / "report.json"
    allowlisted.write_text("**\n", encoding="utf-8")
    scanned.write_text("email: casey@example.test\n", encoding="utf-8")

    status = main(
        [
            "--paths",
            f"{allowlisted}\n{scanned}",
            "--synthetic-fixture-allowlist",
            str(allowlisted),
            "--output",
            str(report),
        ]
    )
    capsys.readouterr()
    payload = _read_report(report)

    assert status == 1
    assert payload["allowlisted_files"] == 1
    assert payload["scanned_files"] == 1
    assert payload["findings_by_rule"] == {"email": 1}


def test_report_output_replaces_a_symlink_without_overwriting_its_target(
    tmp_path, capsys
):
    source = tmp_path / "safe.txt"
    source.write_text("synthetic fixture\n", encoding="utf-8")
    target = tmp_path / "target.txt"
    target_content = "email: casey@example.test\n"
    target.write_text(target_content, encoding="utf-8")
    report = tmp_path / "report.json"
    try:
        report.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    assert main(["--paths", str(tmp_path), "--output", str(report)]) == 1
    capsys.readouterr()

    assert not report.is_symlink()
    assert _read_report(report)["findings_by_rule"] == {"email": 1}
    assert target.read_text(encoding="utf-8") == target_content


def test_custom_policy_enables_only_requested_rules(tmp_path):
    source = tmp_path / "trace.txt"
    policy = tmp_path / "policy.json"
    report = tmp_path / "report.json"
    synthetic_credential = "AKIA" + "ABCDEFGHIJKLMNOP"
    source.write_text(
        f"email: casey@example.test\nkey: {synthetic_credential}\n",
        encoding="utf-8",
    )
    policy.write_text(
        json.dumps({"name": "email-only", "rules": ["email"]}),
        encoding="utf-8",
    )

    status = main(
        [
            "--paths",
            str(source),
            "--policy",
            str(policy),
            "--output",
            str(report),
        ]
    )
    payload = _read_report(report)

    assert status == 1
    assert payload["policy"] == "email-only"
    assert payload["rules"] == ["email"]
    assert payload["findings_by_rule"] == {"email": 1}


def test_configuration_errors_do_not_echo_sensitive_path_values(tmp_path, capsys):
    sensitive_name = "not-a-secret@example.test"
    missing_path = tmp_path / sensitive_name
    report = tmp_path / "report.json"

    status = main(["--paths", str(missing_path), "--output", str(report)])
    output = capsys.readouterr()

    assert status == 2
    assert sensitive_name not in output.out + output.err
    assert sensitive_name not in report.read_text(encoding="utf-8")
    assert _read_report(report) == {
        "error_category": "configuration",
        "schema_version": 1,
        "status": "error",
    }


def test_scanner_makes_no_network_calls(tmp_path, monkeypatch):
    source = tmp_path / "safe.txt"
    report = tmp_path / "report.json"
    source.write_text("synthetic fixture\n", encoding="utf-8")

    def fail_if_network_is_used(*args, **kwargs):
        raise AssertionError("network access is not part of a privacy scan")

    monkeypatch.setattr(socket, "socket", fail_if_network_is_used)

    assert main(["--paths", str(source), "--output", str(report)]) == 0


def test_composite_action_declares_inputs_scan_and_upload():
    action = Path(".github/actions/privacy-scan/action.yml").read_text(encoding="utf-8")

    assert "using: composite" in action
    assert "paths:" in action
    assert "policy:" in action
    assert "synthetic-fixture-allowlist:" in action
    assert "scripts/privacy_scan.py" in action
    assert "::" not in action.split("run: |", 1)[0]
    assert "actions/upload-artifact@v7" in action
    assert "if: always()" in action
