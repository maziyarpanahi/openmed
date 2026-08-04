from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.interop.bridges import sdcmicro


def _stub_result() -> dict[str, object]:
    return {
        "package_version": "5.8.2",
        "row_count": 4,
        "global_risk": 0.25,
        "global_risk_pct": 25.0,
        "expected_reidentifications": 1.0,
        "individual_risk": {
            "max": 0.5,
            "mean": 0.25,
            "median": 0.25,
            "p95": 0.5,
        },
        "k_anonymity": {
            "achieved_k": 2,
            "target_k": 2,
            "class_count": 2,
            "class_size_distribution": [{"size": 2, "class_count": 2}],
            "singleton_class_count": 0,
            "singleton_record_count": 0,
            "violating_class_count": 0,
            "violating_record_count": 0,
        },
        "l_diversity": [
            {
                "attribute": "diagnosis_group",
                "achieved_distinct": 2,
                "achieved_entropy": 1.0,
                "achieved_recursive": 1.0,
                "target": 2,
                "violating_class_count": 0,
            }
        ],
    }


@pytest.fixture
def accepted_license(monkeypatch):
    monkeypatch.setenv(sdcmicro.LICENSE_ACKNOWLEDGEMENT_ENV, "1")


def test_stubbed_rscript_maps_aggregate_measures_and_uses_private_csv(
    monkeypatch,
    accepted_license,
    capsys,
):
    del accepted_license
    observed: dict[str, object] = {}
    monkeypatch.setattr(sdcmicro.shutil, "which", lambda candidate: f"/opt/{candidate}")
    expected_rscript = os.path.abspath("/opt/approved-Rscript")

    def fake_run(command, **kwargs):
        input_path = Path(command[3])
        output_path = Path(command[4])
        config_path = Path(command[5])
        observed["command"] = command
        observed["kwargs"] = kwargs
        observed["input_mode"] = input_path.stat().st_mode & 0o777
        with input_path.open(encoding="utf-8") as handle:
            observed["rows"] = list(csv.DictReader(handle))
        observed["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        output_path.write_text(json.dumps(_stub_result()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sdcmicro.subprocess, "run", fake_run)

    result = sdcmicro.run_sdcmicro(
        [
            {"age_band": "40-49", "region": "north", "diagnosis_group": "A"},
            {"age_band": "40-49", "region": "north", "diagnosis_group": "B"},
            {"age_band": "50-59", "region": "south", "diagnosis_group": "A"},
            {"age_band": "50-59", "region": "south", "diagnosis_group": "B"},
        ],
        quasi_identifiers=("age_band", "region"),
        sensitive_attributes=("diagnosis_group",),
        rscript="approved-Rscript",
    )

    assert observed["command"][0:2] == [expected_rscript, "--vanilla"]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 60.0,
        "cwd": observed["kwargs"]["cwd"],
    }
    assert observed["input_mode"] == 0o600
    assert observed["rows"][0] == {
        "age_band": "40-49",
        "region": "north",
        "diagnosis_group": "A",
    }
    assert observed["config"]["quasi_identifiers"] == ["age_band", "region"]
    assert result["engine"] == {
        "name": "sdcMicro",
        "version": "5.8.2",
        "execution": "subprocess",
        "license": "GPL-2.0",
    }
    assert result["reid_rate"] == 0.25
    assert result["k_min"] == 2
    assert result["sample_identity_risk"]["global_percent"] == 25.0
    assert result["k_anonymity"] == {
        "achieved_k": 2,
        "target_k": 2,
        "class_count": 2,
        "class_size_distribution": [{"size": 2, "class_count": 2}],
        "singleton_class_count": 0,
        "singleton_record_count": 0,
        "k_violating_class_count": 0,
        "violating_record_count": 0,
        "meets_target": True,
    }
    assert result["attribute_disclosure"] == [
        {
            "attribute": "diagnosis_group",
            "l_diversity": {
                "metric": "distinct",
                "achieved": 2.0,
                "threshold": 2,
                "violating_classes": 0,
                "meets_target": True,
                "entropy_achieved": 1.0,
                "recursive_achieved": 1.0,
            },
        }
    ]
    assert "GPL-2.0" in capsys.readouterr().err


def test_license_refusal_prints_notice_before_rscript_lookup(monkeypatch, capsys):
    monkeypatch.delenv(sdcmicro.LICENSE_ACKNOWLEDGEMENT_ENV, raising=False)

    def fail_lookup(_candidate):
        raise AssertionError("Rscript lookup must remain behind the license gate")

    monkeypatch.setattr(sdcmicro.shutil, "which", fail_lookup)

    with pytest.raises(
        sdcmicro.SDCMicroLicenseError,
        match=sdcmicro.LICENSE_ACKNOWLEDGEMENT_ENV,
    ):
        sdcmicro.run_sdcmicro(
            [{"age_band": "40-49"}],
            quasi_identifiers=("age_band",),
        )

    assert "GPL-2.0" in capsys.readouterr().err


def test_missing_rscript_and_sdcmicro_fail_closed(monkeypatch, accepted_license):
    del accepted_license
    monkeypatch.setattr(sdcmicro.shutil, "which", lambda _candidate: None)
    with pytest.raises(sdcmicro.SDCMicroUnavailableError, match="Rscript executable"):
        sdcmicro.run_sdcmicro(
            [{"age_band": "40-49"}],
            quasi_identifiers=("age_band",),
        )

    monkeypatch.setattr(sdcmicro.shutil, "which", lambda _candidate: "/opt/Rscript")
    monkeypatch.setattr(
        sdcmicro.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            69,
            stdout="",
            stderr=f"{sdcmicro._SDCMICRO_UNAVAILABLE_SENTINEL}\n",
        ),
    )
    with pytest.raises(
        sdcmicro.SDCMicroUnavailableError, match="sdcMicro is unavailable"
    ):
        sdcmicro.run_sdcmicro(
            [{"age_band": "40-49"}],
            quasi_identifiers=("age_band",),
        )


def test_import_openmed_core_never_imports_sdcmicro_bridge_or_rpy2():
    code = """
import json
import sys
import openmed.core

blocked = [
    name for name in sys.modules
    if name == "rpy2"
    or name.startswith("rpy2.")
    or name == "openmed.interop.bridges.sdcmicro"
]
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_interop_gpl_extra_has_no_bundled_dependency():
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - exercised on the supported Python 3.10 lane
        import tomli as tomllib

    project_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    with project_path.open("rb") as handle:
        optional = tomllib.load(handle)["project"]["optional-dependencies"]

    assert optional["interop-gpl"] == []
