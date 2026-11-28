"""Exceptions outside the rerooted packages that inherit a root keep behaving.

Rerooting ``openmed.agent`` and ``openmed.structured`` also roots two
subclasses defined elsewhere: ``ConsentReceiptStoreError`` (through
``ApprovalNonceStoreError``) and ``DecisionQualificationError`` (through
``DecisionError``). The snapshot was recorded with
``scripts/release/check_error_roots.py --record openmed.mcp.consent_receipts
--record openmed.eval.decision_qualification`` before the rerooting, so it also
covers the consent-receipt classes that stay unrooted. Every class must still
match it, and only the two inheriting classes may be rooted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openmed.core.errors import LegacyRootedError

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = (
    ROOT
    / "tests"
    / "fixtures"
    / "error_roots"
    / "rerooted"
    / "service_structured_agent_dependents.json"
)
INHERITS_A_ROOT = {
    "openmed.eval.decision_qualification:DecisionQualificationError",
    "openmed.mcp.consent_receipts:ConsentReceiptStoreError",
}


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
_CLASSES = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["classes"]


def test_only_the_inheriting_classes_are_rooted() -> None:
    rooted = {
        name
        for name in _CLASSES
        if issubclass(gate.load_class(name), LegacyRootedError)
    }
    assert rooted == INHERITS_A_ROOT


@pytest.mark.parametrize("name", sorted(_CLASSES))
def test_dependent_exception_behavior_is_unchanged(name: str) -> None:
    recorded = _CLASSES[name]
    error_class = gate.load_class(name)

    assert gate.behavior_snapshot(error_class) == recorded
    if name in INHERITS_A_ROOT:
        assert gate.replay_snapshot(name, recorded) == []


def test_mcp_keeps_the_consent_mapping_for_the_rooted_store_error() -> None:
    from openmed.mcp.consent_receipts import ConsentReceiptStoreError
    from openmed.mcp.server import _error_envelope

    envelope = _error_envelope(ConsentReceiptStoreError("synthetic_reason"))
    assert envelope == {
        "error": {
            "code": "consent_denied",
            "message": (
                "The consent receipt does not authorize this state-changing tool."
            ),
        },
        "is_error": True,
    }
