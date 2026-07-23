from __future__ import annotations

import argparse
import json

from openmed.cli import main_module
from openmed.cli.main import _handle_doctor
from openmed.core import doctor as doctor_module
from openmed.core.doctor import run_diagnostics


def test_run_diagnostics_returns_list():
    results = run_diagnostics()

    assert isinstance(results, list)
    assert len(results) > 0


def test_required_checks_present():
    results = run_diagnostics()

    names = {item["name"] for item in results}

    assert "python_version" in names
    assert "python_arch" in names
    assert "openmed_version" in names
    assert "low_resource_memory" in names
    assert "hf_token" in names
    assert "hf_endpoint" in names
    assert "hf_cache" in names
    assert "http_proxy" in names
    assert "https_proxy" in names
    assert "all_proxy" in names
    assert "no_proxy" in names
    assert "openmed_offline" in names
    assert "manifest_exists" in names
    assert "manifest_rows" in names


def test_hf_token_not_exposed(monkeypatch):
    monkeypatch.setenv(
        "HF_TOKEN",
        "SUPER_SECRET_TOKEN_VALUE",
    )

    results = run_diagnostics()

    serialized = str(results)

    assert "SUPER_SECRET_TOKEN_VALUE" not in serialized

    token_check = next(item for item in results if item["name"] == "hf_token")

    assert token_check["status"] == "PASS"
    assert token_check["present"] is True


def test_hf_token_absence_reports_boolean(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    results = run_diagnostics()

    token_check = next(item for item in results if item["name"] == "hf_token")

    assert token_check["status"] == "WARN"
    assert token_check["present"] is False


def test_legacy_hf_token_presence_is_reported_without_exposure(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "LEGACY_SECRET_TOKEN_VALUE")

    results = run_diagnostics()
    token_check = next(item for item in results if item["name"] == "hf_token")

    assert token_check["present"] is True
    assert "LEGACY_SECRET_TOKEN_VALUE" not in str(results)


def test_network_environment_is_reported_without_proxy_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.org")
    monkeypatch.setenv(
        "HTTP_PROXY",
        "http://network-user:SUPER_SECRET_PROXY_PASSWORD@proxy.example.org:8080",
    )
    monkeypatch.setenv(
        "http_proxy",
        "http://network-user:SUPER_SECRET_PROXY_PASSWORD@proxy.example.org:8080",
    )
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.org:8443")
    monkeypatch.setenv("https_proxy", "http://proxy.example.org:8443")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example.org:1080")
    monkeypatch.setenv("all_proxy", "socks5://proxy.example.org:1080")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,.example.org")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,.example.org")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    results = run_diagnostics()
    checks = {item["name"]: item for item in results}

    assert checks["hf_endpoint"]["details"] == "https://mirror.example.org"
    assert checks["hf_endpoint"]["source"] == "HF_ENDPOINT"
    assert checks["http_proxy"]["details"] == ("http://***:***@proxy.example.org:8080")
    assert checks["http_proxy"]["source"].lower() == "http_proxy"
    assert checks["https_proxy"]["present"] is True
    assert checks["all_proxy"]["details"] == "socks5://proxy.example.org:1080"
    assert checks["no_proxy"]["details"] == "localhost,127.0.0.1,.example.org"
    assert checks["hf_cache"]["details"] == str(tmp_path / "hf-home" / "hub")
    assert checks["hf_cache"]["source"] == "HF_HOME"
    assert checks["openmed_offline"]["enabled"] is True
    assert "SUPER_SECRET_PROXY_PASSWORD" not in str(results)


def test_network_environment_reports_defaults(monkeypatch):
    for name in (
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "XDG_CACHE_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "OPENMED_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)

    results = run_diagnostics()
    checks = {item["name"]: item for item in results}

    assert checks["hf_endpoint"]["details"] == doctor_module.DEFAULT_HF_ENDPOINT
    assert checks["hf_endpoint"]["source"] == "default"
    assert checks["http_proxy"]["details"] == "not set"
    assert checks["http_proxy"]["present"] is False
    assert checks["hf_cache"]["details"] == str(
        doctor_module.Path.home() / ".cache" / "huggingface" / "hub"
    )
    assert checks["openmed_offline"]["enabled"] is False


def test_network_environment_redacts_schemeless_credentials_and_url_secrets(
    monkeypatch,
):
    monkeypatch.setenv(
        "HF_ENDPOINT",
        "https://mirror.example.org/models?access_token=endpoint-secret#private",
    )
    monkeypatch.setenv("HTTP_PROXY", "http://ignored.example.org:8080")
    monkeypatch.setenv(
        "http_proxy",
        "network-user:proxy-secret@proxy.example.org:8080/path?token=hidden#private",
    )

    results = run_diagnostics()
    checks = {item["name"]: item for item in results}
    serialized = str(results)

    assert checks["hf_endpoint"]["details"] == "https://mirror.example.org/models"
    assert checks["http_proxy"]["details"] == ("***:***@proxy.example.org:8080/path")
    assert checks["http_proxy"]["source"] == "http_proxy"
    assert "endpoint-secret" not in serialized
    assert "proxy-secret" not in serialized
    assert "token=hidden" not in serialized


def test_network_environment_warns_for_plaintext_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://mirror.example.org")

    results = run_diagnostics()
    endpoint = next(item for item in results if item["name"] == "hf_endpoint")

    assert endpoint["status"] == "WARN"
    assert "HTTPS" in endpoint["hint"]


def test_network_environment_escapes_control_characters(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.org\nforged-output")
    monkeypatch.setenv("https_proxy", "http://proxy.example.org\nforged-output")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home\nforged-output"))

    results = run_diagnostics()
    checks = {item["name"]: item for item in results}

    assert checks["hf_endpoint"]["status"] == "WARN"
    assert "\n" not in checks["hf_endpoint"]["details"]
    assert "\n" not in checks["https_proxy"]["details"]
    assert "\\u000a" in checks["https_proxy"]["details"]
    assert "\n" not in checks["hf_cache"]["details"]
    assert "\\u000a" in checks["hf_cache"]["details"]


def test_offline_mode_escapes_control_characters(monkeypatch):
    monkeypatch.setenv("OPENMED_OFFLINE", "0\nPASS forged: enabled")

    results = run_diagnostics()
    offline = next(item for item in results if item["name"] == "openmed_offline")

    assert "\n" not in offline["details"]
    assert "\\u000a" in offline["details"]


def test_unsupported_python_version_fails(monkeypatch):
    monkeypatch.setattr(doctor_module.sys, "version_info", (3, 9, 18))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.9.18")

    results = run_diagnostics()

    python_check = next(item for item in results if item["name"] == "python_version")

    assert python_check["status"] == "FAIL"


def test_unsupported_python_architecture_fails(monkeypatch):
    monkeypatch.setattr(doctor_module.platform, "machine", lambda: "i386")

    results = run_diagnostics()

    arch_check = next(item for item in results if item["name"] == "python_arch")

    assert arch_check["status"] == "FAIL"


def test_manifest_check_uses_canonical_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    results = run_diagnostics()

    manifest_check = next(item for item in results if item["name"] == "manifest_exists")

    assert manifest_check["status"] == "PASS"
    assert manifest_check["details"].endswith("models.jsonl")


def test_missing_manifest_reports_row_check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        doctor_module,
        "MANIFEST_PATH",
        tmp_path / "missing-models.jsonl",
    )

    results = run_diagnostics()

    manifest_exists = next(
        item for item in results if item["name"] == "manifest_exists"
    )
    manifest_rows = next(item for item in results if item["name"] == "manifest_rows")

    assert manifest_exists["status"] == "WARN"
    assert manifest_rows["status"] == "WARN"
    assert "missing" in manifest_rows["details"]


def test_optional_dependencies_are_warn_or_pass():
    results = run_diagnostics()

    optional_checks = [
        item
        for item in results
        if item["name"]
        in {
            "mlx",
            "coreml",
            "onnx",
            "hf",
            "multimodal",
        }
    ]

    for check in optional_checks:
        assert check["status"] in {
            "PASS",
            "WARN",
        }


def test_doctor_suggests_low_resource_profile_below_8gb(monkeypatch):
    monkeypatch.setattr(doctor_module, "_effective_memory_bytes", lambda: 6 * 1024**3)

    results = run_diagnostics()

    memory = next(item for item in results if item["name"] == "low_resource_memory")
    assert memory["status"] == "PASS"
    assert memory["fits_low_resource"] is True
    assert memory["profile_suggested"] is True
    assert "OPENMED_PROFILE=low_resource" in memory["hint"]


def test_doctor_warns_when_below_4gb(monkeypatch):
    monkeypatch.setattr(doctor_module, "_effective_memory_bytes", lambda: 3 * 1024**3)

    results = run_diagnostics()

    memory = next(item for item in results if item["name"] == "low_resource_memory")
    assert memory["status"] == "WARN"
    assert memory["fits_low_resource"] is False


def test_doctor_warns_when_optional_dependencies_are_missing(monkeypatch, capsys):
    def missing_dependency(module_name):
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(
        doctor_module.importlib,
        "import_module",
        missing_dependency,
    )

    exit_code = _handle_doctor(argparse.Namespace(json=False))

    captured = capsys.readouterr()
    assert exit_code == 0

    for name in doctor_module.OPTIONAL_EXTRAS:
        assert f"WARN {name}:" in captured.out


def test_doctor_returns_zero_when_no_fail():
    args = argparse.Namespace(
        json=False,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 0


def test_doctor_text_output_includes_network_environment(monkeypatch, capsys):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.org")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.org:8080")
    monkeypatch.setenv("http_proxy", "http://proxy.example.org:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.org:8443")
    monkeypatch.setenv("https_proxy", "http://proxy.example.org:8443")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    exit_code = _handle_doctor(argparse.Namespace(json=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS hf_endpoint: https://mirror.example.org" in captured.out
    assert "PASS http_proxy: http://proxy.example.org:8080" in captured.out
    assert "PASS https_proxy: http://proxy.example.org:8443" in captured.out
    assert "PASS no_proxy: localhost,127.0.0.1" in captured.out
    assert "PASS hf_cache:" in captured.out
    assert "PASS openmed_offline: enabled (OPENMED_OFFLINE=1)" in captured.out


def test_fail_returns_nonzero(monkeypatch):
    def fake_run_diagnostics():
        return [
            {
                "name": "fake_fail",
                "status": "FAIL",
                "details": "testing",
            }
        ]

    monkeypatch.setattr(
        "openmed.core.doctor.run_diagnostics",
        fake_run_diagnostics,
    )

    args = argparse.Namespace(
        json=False,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 1


def test_json_mode_returns_zero():
    args = argparse.Namespace(
        json=True,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 0


def test_doctor_json_command_is_registered(monkeypatch, capsys):
    diagnostics = [
        {
            "name": "python_version",
            "status": "PASS",
            "details": "3.11.10",
        }
    ]

    monkeypatch.setattr(
        "openmed.core.doctor.run_diagnostics",
        lambda: diagnostics,
    )

    exit_code = main_module.main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["command"] == "doctor"
    assert payload["data"]["checks"] == diagnostics
