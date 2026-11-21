"""Value-free configuration warnings for the 4.3 migration."""

import json
import os
import subprocess
import sys

from fastapi.testclient import TestClient

from openmed.cli.main import main
from openmed.core.settings_registry import service_setting_findings


def test_preflight_and_startup_share_safe_findings(monkeypatch, caplog, capsys):
    from openmed.service.app import create_app

    key = "OPENMED_SERVICE_" + "UNRECOGNIZED_OPTION"
    for name in list(os.environ):
        if name.startswith("OPENMED_SERVICE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv(key, "sensitive-example-never-print")
    assert main(["config", "service-check"]) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "findings": [{"name": key, "code": "unknown_setting"}]
    }
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        assert client.get("/livez").status_code == 200
    assert sum(key in record.getMessage() for record in caplog.records) == 1
    assert "sensitive-example-never-print" not in caplog.text + output


def test_valid_formats_ranges_and_invalid_values():
    assert (
        service_setting_findings(
            {
                "OPENMED_SERVICE_AUTH_ENABLED": "true",
                "OPENMED_SERVICE_KEEP_ALIVE": "1h30m",
                "OPENMED_SERVICE_SERVED_MODELS": "a,b",
                "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES": "2GiB",
                "OPENMED_SERVICE_LOG_FORMAT": "json",
                "OPENMED_SERVICE_AUTH_API_KEYS": "[]",
                "OPENMED_SERVICE_BATCH_MAX_SIZE": "10",
            }
        )
        == []
    )
    invalid = {
        "OPENMED_SERVICE_AUTH_ENABLED": "private",
        "OPENMED_SERVICE_BATCH_MAX_SIZE": "101",
        "OPENMED_SERVICE_AUTH_API_KEYS": "private",
        "OPENMED_SERVICE_LOG_FORMAT": "private",
        "OPENMED_SERVICE_RATE_LIMIT_RPS": "nan",
    }
    assert service_setting_findings(invalid) == [
        {"name": name, "code": "invalid_value"} for name in sorted(invalid)
    ]


def test_legacy_call_warns_once_per_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import warnings
from openmed.service.logging import service_log_config_from_env
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always", DeprecationWarning)
    assert service_log_config_from_env() == service_log_config_from_env()
    assert len([w for w in seen if "service_log_config_from_env is deprecated" in str(w.message)]) == 1
""",
        ],
        check=True,
    )
