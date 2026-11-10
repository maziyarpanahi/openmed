from __future__ import annotations

import subprocess
import sys

import pytest

from openmed.interop.bridges import medcat

FAKE_MEDCAT_JSON = (
    '[{"cui": "C0011860", "name": "Diabetes Mellitus", "score": 0.98}, '
    '{"cui": "73211009", "name": "Diabetes mellitus", "score": 0.91, '
    '"ontology": "SNOMED"}]'
)


def _ack_env() -> dict[str, str]:
    return {medcat.LICENSE_ENV_VAR: "1"}


def _fake_runner(output: str):
    def runner(command, *, text, timeout, cwd, extra_env):
        return output

    return runner


# ---------------------------------------------------------------------------
# License gate
# ---------------------------------------------------------------------------


def test_license_acknowledged_reads_env_var():
    assert medcat.license_acknowledged({medcat.LICENSE_ENV_VAR: "1"}) is True
    assert medcat.license_acknowledged({medcat.LICENSE_ENV_VAR: "true"}) is True
    assert medcat.license_acknowledged({medcat.LICENSE_ENV_VAR: "0"}) is False
    assert medcat.license_acknowledged({}) is False


def test_ensure_license_acknowledged_prints_notice_and_passes_when_set(capsys):
    medcat.ensure_license_acknowledged(env=_ack_env())

    captured = capsys.readouterr()
    assert "Elastic License 2.0" in captured.err


def test_ensure_license_acknowledged_blocks_without_ack_or_tty(monkeypatch):
    monkeypatch.setattr(medcat, "_prompt_for_acknowledgement", lambda: False)

    with pytest.raises(
        medcat.MedCATLicenseNotAcknowledgedError, match=medcat.LICENSE_ENV_VAR
    ):
        medcat.ensure_license_acknowledged(env={}, allow_interactive_prompt=True)


def test_ensure_license_acknowledged_blocks_when_prompt_disabled():
    with pytest.raises(medcat.MedCATLicenseNotAcknowledgedError):
        medcat.ensure_license_acknowledged(env={}, allow_interactive_prompt=False)


def test_run_medcat_refuses_without_license_ack():
    with pytest.raises(medcat.MedCATLicenseNotAcknowledgedError):
        medcat.run_medcat(
            "patient has diabetes",
            command=["medcat-cli"],
            env={},
            allow_interactive_prompt=False,
            runner=_fake_runner(FAKE_MEDCAT_JSON),
        )


# ---------------------------------------------------------------------------
# Stubbed-subprocess mapping path
# ---------------------------------------------------------------------------


def test_run_medcat_maps_concepts_with_stubbed_subprocess():
    calls = []

    def runner(command, *, text, timeout, cwd, extra_env):
        calls.append((tuple(command), text))
        return FAKE_MEDCAT_JSON

    results = medcat.run_medcat(
        "patient has diabetes",
        command=["medcat-cli", "--model", "umls-full"],
        env=_ack_env(),
        allow_interactive_prompt=False,
        runner=runner,
    )

    assert calls == [(("medcat-cli", "--model", "umls-full"), "patient has diabetes")]
    assert results == [
        {
            "system": "UMLS",
            "code": "C0011860",
            "score": 0.98,
            "name": "Diabetes Mellitus",
        },
        {
            "system": "SNOMED",
            "code": "73211009",
            "score": 0.91,
            "name": "Diabetes mellitus",
        },
    ]


def test_run_medcat_never_invokes_subprocess_when_license_refused(monkeypatch):
    def unexpected_runner(*args, **kwargs):
        raise AssertionError("subprocess runner must not be called without license ack")

    with pytest.raises(medcat.MedCATLicenseNotAcknowledgedError):
        medcat.run_medcat(
            "patient has diabetes",
            command=["medcat-cli"],
            env={},
            allow_interactive_prompt=False,
            runner=unexpected_runner,
        )


def test_default_runner_is_used_when_no_runner_supplied(monkeypatch):
    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = FAKE_MEDCAT_JSON
        stderr = ""

    def fake_run(
        command,
        *,
        input,
        stdout,
        stderr,
        text,
        encoding,
        errors,
        timeout,
        cwd,
        env,
        check,
    ):
        captured["command"] = command
        captured["input"] = input
        assert stdout is subprocess.PIPE
        assert stderr is subprocess.DEVNULL
        assert encoding == "utf-8"
        assert errors == "strict"
        return FakeCompletedProcess()

    monkeypatch.setattr(medcat.subprocess, "run", fake_run)

    results = medcat.run_medcat(
        "patient has diabetes",
        command=["medcat-cli"],
        env=_ack_env(),
        allow_interactive_prompt=False,
    )

    assert captured["command"] == ["medcat-cli"]
    assert captured["input"] == "patient has diabetes"
    assert results[0]["code"] == "C0011860"


def test_default_runner_raises_on_nonzero_exit(monkeypatch):
    class FakeFailedProcess:
        returncode = 1
        stdout = ""
        stderr = "patient Jane Doe: model not found"

    monkeypatch.setattr(medcat.subprocess, "run", lambda *a, **k: FakeFailedProcess())

    with pytest.raises(medcat.MedCATBridgeError) as raised:
        medcat.run_medcat(
            "patient has diabetes",
            command=["medcat-cli"],
            env=_ack_env(),
            allow_interactive_prompt=False,
        )
    assert "Jane Doe" not in str(raised.value)
    assert "stderr was suppressed" in str(raised.value)


def test_default_runner_reports_missing_executable_without_echoing_input(monkeypatch):
    def missing_process(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(medcat.subprocess, "run", missing_process)

    with pytest.raises(medcat.MedCATBridgeError) as raised:
        medcat.run_medcat(
            "patient Jane Doe",
            command=["missing-medcat"],
            env=_ack_env(),
            allow_interactive_prompt=False,
        )
    assert "Jane Doe" not in str(raised.value)
    assert "could not be started" in str(raised.value)


def test_run_medcat_rejects_invalid_command_before_invocation():
    with pytest.raises(ValueError, match="non-empty sequence"):
        medcat.run_medcat(
            "patient has diabetes",
            command=[],
            env=_ack_env(),
            allow_interactive_prompt=False,
        )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def test_parse_medcat_output_handles_json_array():
    results = medcat.parse_medcat_output(FAKE_MEDCAT_JSON)

    assert results == [
        {
            "system": "UMLS",
            "code": "C0011860",
            "score": 0.98,
            "name": "Diabetes Mellitus",
        },
        {
            "system": "SNOMED",
            "code": "73211009",
            "score": 0.91,
            "name": "Diabetes mellitus",
        },
    ]


def test_parse_medcat_output_handles_ndjson():
    ndjson = (
        '{"cui": "C0011860", "name": "Diabetes Mellitus", "score": 0.98}\n'
        '{"cui": "C0020538", "name": "Hypertension", "score": 0.87}\n'
    )

    results = medcat.parse_medcat_output(ndjson)

    assert [r["code"] for r in results] == ["C0011860", "C0020538"]
    assert all(r["system"] == "UMLS" for r in results)


def test_parse_medcat_output_handles_wrapped_concepts_object():
    wrapped = '{"concepts": [{"cui": "C0011860", "score": 0.5}]}'

    results = medcat.parse_medcat_output(wrapped)

    assert results == [{"system": "UMLS", "code": "C0011860", "score": 0.5}]


def test_parse_medcat_output_handles_empty_wrapped_concepts_object():
    assert medcat.parse_medcat_output('{"concepts": []}') == []


def test_parse_medcat_output_empty_output_returns_empty_list():
    assert medcat.parse_medcat_output("") == []
    assert medcat.parse_medcat_output("   ") == []


def test_parse_medcat_output_unknown_system_falls_back_to_default():
    raw = '[{"cui": "C0011860", "system": "ICD10"}]'

    results = medcat.parse_medcat_output(raw, default_system="SNOMED")

    assert results == [{"system": "SNOMED", "code": "C0011860"}]


def test_parse_medcat_output_rejects_invalid_default_system():
    with pytest.raises(ValueError, match="default_system"):
        medcat.parse_medcat_output("[]", default_system="ICD10")


def test_parse_medcat_output_missing_cui_raises():
    with pytest.raises(medcat.MedCATBridgeError, match="cui"):
        medcat.parse_medcat_output('[{"name": "Diabetes"}]')


def test_parse_medcat_output_rejects_nonfinite_score():
    with pytest.raises(medcat.MedCATBridgeError, match="invalid score"):
        medcat.parse_medcat_output('[{"cui": "C0011860", "score": "NaN"}]')


def test_parse_medcat_output_unparsable_raises():
    with pytest.raises(medcat.MedCATBridgeError):
        medcat.parse_medcat_output("not json at all {")


def test_parse_medcat_output_non_list_json_raises():
    with pytest.raises(medcat.MedCATBridgeError):
        medcat.parse_medcat_output('"just a string"')


# ---------------------------------------------------------------------------
# Guard: core import never pulls in the MedCAT bridge
# ---------------------------------------------------------------------------


def test_import_openmed_does_not_import_medcat_bridge():
    for name in list(sys.modules):
        if name == "openmed.interop.bridges.medcat" or name.startswith(
            "openmed.interop.bridges.medcat."
        ):
            del sys.modules[name]

    import openmed  # noqa: F401

    assert "openmed.interop.bridges.medcat" not in sys.modules


def test_fresh_import_openmed_does_not_import_medcat_bridge():
    code = """
import sys
import openmed
assert 'openmed.interop.bridges.medcat' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_interop_bridges_package_does_not_import_medcat_module():
    for name in list(sys.modules):
        if name == "openmed.interop.bridges.medcat" or name.startswith(
            "openmed.interop.bridges.medcat."
        ):
            del sys.modules[name]

    from openmed.interop import bridges  # noqa: F401

    assert "openmed.interop.bridges.medcat" not in sys.modules
