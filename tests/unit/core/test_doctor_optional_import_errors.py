"""Optional dependency diagnostics must survive ordinary import failures."""

import json

import pytest

from openmed.core import doctor


@pytest.mark.parametrize("failure", [OSError, RuntimeError, ValueError, TypeError])
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


@pytest.mark.parametrize(
    "parent", [Exception, OSError, RuntimeError, ValueError, TypeError]
)
def test_custom_exception_names_never_enter_diagnostics(monkeypatch, parent):
    private_name = "SYNTHETIC_PRIVATE_EXCEPTION_CLASS"
    failure = type(private_name, (parent,), {})

    def load(name):
        if name == "onnxruntime":
            raise failure("SYNTHETIC_PRIVATE_MESSAGE")
        return object()

    monkeypatch.setattr(doctor.importlib, "import_module", load)
    checks = []
    doctor._check_optional_dependencies(checks)
    assert "SYNTHETIC_PRIVATE" not in json.dumps(checks)
    result = next(check for check in checks if check["name"] == "onnx")
    assert result["details"] == f"onnxruntime import failed ({parent.__name__})"


@pytest.mark.parametrize(
    "parent", [Exception, OSError, RuntimeError, ValueError, TypeError]
)
def test_exception_class_hooks_are_not_invoked(monkeypatch, parent):
    hook_calls = []

    def private_class_hook(_self):
        hook_calls.append(True)
        raise RuntimeError("SYNTHETIC_PRIVATE_CLASS_HOOK")

    failure = type(
        "SYNTHETIC_PRIVATE_EXCEPTION_CLASS",
        (parent,),
        {"__class__": property(private_class_hook)},
    )
    seen = []

    def load(name):
        seen.append(name)
        if name == "onnxruntime":
            raise failure("SYNTHETIC_PRIVATE_MESSAGE")
        return object()

    monkeypatch.setattr(doctor.importlib, "import_module", load)
    checks = []
    doctor._check_optional_dependencies(checks)
    assert hook_calls == []
    assert seen == list(doctor.OPTIONAL_EXTRAS.values())
    by_name = {check["name"]: check for check in checks}
    assert by_name["onnx"]["details"] == (
        f"onnxruntime import failed ({parent.__name__})"
    )
    assert by_name["hf"]["status"] == "PASS"
    assert by_name["multimodal"]["status"] == "PASS"
    assert "SYNTHETIC_PRIVATE" not in json.dumps(checks)


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
