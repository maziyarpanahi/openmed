"""Optional dependency diagnostics must survive ordinary import failures."""

import json

import pytest

from openmed.core import doctor


@pytest.mark.parametrize("failure", [OSError, RuntimeError, ValueError])
def test_broken_native_import_is_reported_and_remaining_checks_run(
    monkeypatch, failure
):
    seen = []

    def load(name):
        seen.append(name)
        if name == "onnxruntime":
            raise failure("SYNTHETIC_ERROR_VALUE_MUST_NOT_BE_ECHOED")
        return object()

    monkeypatch.setattr(doctor.importlib, "import_module", load)
    checks = []
    doctor._check_optional_dependencies(checks)
    assert seen == list(doctor.OPTIONAL_EXTRAS.values())
    by_name = {c["name"]: c for c in checks}
    assert by_name["onnx"]["status"] == "WARN"
    assert failure.__name__ in by_name["onnx"]["details"]
    assert "SYNTHETIC_ERROR_VALUE" not in json.dumps(checks)
    assert by_name["hf"]["status"] == "PASS"
    assert by_name["multimodal"]["details"] == "Pillow installed"


@pytest.mark.parametrize("failure", [ImportError, ModuleNotFoundError])
def test_missing_dependency_behavior_is_preserved(monkeypatch, failure):
    def missing(name):
        raise failure("synthetic missing dependency")

    monkeypatch.setattr(doctor.importlib, "import_module", missing)
    checks = []
    doctor._check_optional_dependencies(checks)
    assert len(checks) == len(doctor.OPTIONAL_EXTRAS)
    assert all(
        c["status"] == "WARN" and "not installed" in c["details"] for c in checks
    )
    assert all("Install with:" in c["hint"] for c in checks)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(monkeypatch, interrupt):
    def stop(name):
        raise interrupt()

    monkeypatch.setattr(doctor.importlib, "import_module", stop)
    with pytest.raises(interrupt):
        doctor._check_optional_dependencies([])


def test_healthy_dependencies_remain_passes(monkeypatch):
    monkeypatch.setattr(doctor.importlib, "import_module", lambda _name: object())
    checks = [{"name": "earlier-check"}]
    doctor._check_optional_dependencies(checks)
    assert checks[0] == {"name": "earlier-check"}
    assert all(c["status"] == "PASS" for c in checks[1:])
