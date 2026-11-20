import json

import pytest

from openmed.cli.main import main
from openmed.compliance.ehds_spe import check_ehds_spe

PROFILE = dict(
    outbound_egress_denied=True,
    authentication_enabled=True,
    audit_recording_enabled=True,
    encryption_at_rest=True,
    export_classes=["anonymised", "aggregate"],
    access_logging_enabled=True,
    minimum_necessary_configured=True,
)


def test_declared_profile_from_cli(tmp_path, capsys):
    path = tmp_path / "private-profile.json"
    path.write_text(
        json.dumps(
            {**PROFILE, "hostname": "private-host", "password": "private-secret"}
        )
    )
    assert main(["compliance", "ehds-spe", "check", "--profile", str(path)]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["status"] == "pass" and len(report["controls"]) == 7
    assert all(c["status"] == "pass" for c in report["controls"].values())
    assert "legal determination" in report["notice"]
    assert "private-" not in output


def test_false_missing_and_bad_types_have_distinct_outcomes():
    profile = {
        **PROFILE,
        "outbound_egress_denied": False,
        "audit_recording_enabled": False,
        "access_logging_enabled": "private-credential",
        "export_classes": ["aggregate", "private-host"],
    }
    del profile["encryption_at_rest"]
    report = check_ehds_spe(profile)
    assert report["status"] == "fail"
    controls = report["controls"]
    assert controls["encryption_at_rest"]["status"] == "not_assessed"
    for key in (
        "outbound_egress_denied",
        "audit_recording_enabled",
        "export_classes",
        "access_logging_enabled",
    ):
        assert controls[key]["status"] == "fail"
    assert "private-" not in json.dumps(report)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"authentication_enabled":true,"authentication_enabled":false}',
        "malformed",
    ],
)
def test_incomplete_or_invalid_input_returns_nonzero(tmp_path, capsys, payload):
    path = tmp_path / "private-path"
    path.write_text(payload)
    assert main(["compliance", "ehds-spe", "check", "--profile", str(path)]) != 0
    output = capsys.readouterr()
    assert "private-path" not in output.err + output.out
