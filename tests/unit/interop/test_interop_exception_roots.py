"""Rerooted interop exceptions keep their pre-taxonomy behavior.

``tests/fixtures/error_roots/rerooted/interop.json`` was recorded with
``scripts/release/check_error_roots.py --record openmed.interop`` before the
classes gained ``LegacyRootedError``. Each class must now be an
``OpenMedError`` with its recorded stable code, keep every builtin base in the
recorded MRO order, and produce identical ``args``, ``str()`` and pickling
results for the recorded synthetic constructor calls.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openmed.core.errors import LegacyRootedError, OpenMedError

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "tests" / "fixtures" / "error_roots" / "rerooted" / "interop.json"
ALLOWLIST = ROOT / "tests" / "fixtures" / "error_roots" / "unrooted_allowlist.txt"
PACKAGE = "openmed.interop"

# The snapshot was recorded without the optional Airflow extra, where this
# class derives from the module's RuntimeError fallback rather than
# ``airflow.exceptions.AirflowException``.
_AIRFLOW_DEPENDENT = "openmed.interop.airflow:RedactionOperatorError"


def _load_gate():
    name = "check_error_roots"
    if name in sys.modules:
        return sys.modules[name]
    script = ROOT / "scripts" / "release" / "check_error_roots.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
_DOCUMENT = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
_CLASSES = _DOCUMENT["classes"]


def _builtin_bases(recorded: dict) -> list[type]:
    return [
        getattr(builtins, name.partition(":")[2])
        for name in recorded["mro"]
        if name.startswith("builtins:")
    ]


def test_snapshot_covers_the_package() -> None:
    assert _DOCUMENT["schema_version"] == gate.SNAPSHOT_SCHEMA_VERSION
    assert _DOCUMENT["sample_calls"] == [list(call) for call in gate.SAMPLE_CALLS]
    assert len(_CLASSES) == 99
    assert all(name.startswith(PACKAGE + ".") for name in _CLASSES)


def test_package_has_no_unrooted_allowlist_entries() -> None:
    entries = gate.read_allowlist(ALLOWLIST)
    assert [entry for entry in entries if entry.startswith(PACKAGE + ".")] == []


@pytest.mark.parametrize("name", sorted(_CLASSES))
def test_rerooted_exception_keeps_behavior(name: str) -> None:
    recorded = _CLASSES[name]
    if name == _AIRFLOW_DEPENDENT:
        from openmed.interop import airflow

        if airflow._AIRFLOW_AVAILABLE:
            pytest.skip("snapshot recorded without the optional airflow extra")
    error_class = gate.load_class(name)

    assert issubclass(error_class, OpenMedError)
    assert issubclass(error_class, LegacyRootedError)
    bases = _builtin_bases(recorded)
    assert bases, name
    for base in bases:
        assert issubclass(error_class, base), base
    assert isinstance(error_class.code, str) and error_class.code
    assert error_class.code == recorded["code"]
    assert gate.replay_snapshot(name, recorded) == []

    for sample in recorded["samples"]:
        if "args" not in sample:
            continue
        error = error_class(*gate.SAMPLE_CALLS[sample["call"]])
        assert isinstance(error, OpenMedError)
        assert all(isinstance(error, base) for base in bases)
        assert repr(error.args) == sample["args"]
        if type(error).to_dict is LegacyRootedError.to_dict:
            # Classes that already had their own to_dict() keep it unchanged.
            payload = error.to_dict()
            assert isinstance(payload["code"], str) and payload["code"]
            assert isinstance(payload["message"], str)
