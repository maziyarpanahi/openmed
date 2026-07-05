"""Tests for the structured error taxonomy (OM-824).

Covers:

* the rooted exception hierarchy and its backwards-compatible builtin bases,
* actionable, PHI-free messages asserted against fixtures,
* the robustness gate (no bare ``except`` / ``except Exception`` swallow on the
  enumerated public modules),
* malformed input yielding :class:`InputError` end to end,
* the service and MCP layers mapping each taxonomy class to a stable error code.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import openmed
from openmed.core.errors import (
    BudgetExceededError,
    CapabilityError,
    ConfigurationError,
    InferenceError,
    InputError,
    InternalError,
    MissingExtraError,
    ModelLoadError,
    OpenMedError,
    PolicyError,
    redact_detail,
)

# Repository root: tests/unit/core/test_error_taxonomy.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Public modules the taxonomy governs (from the OM-824 issue "Files" section).
_PUBLIC_MODULES = [
    "openmed/core/pii.py",
    "openmed/service/app.py",
    "openmed/mcp/server.py",
    "openmed/processing/outputs.py",
    "openmed/ner/exceptions.py",
    "openmed/core/errors.py",
]


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #


def test_everything_descends_from_openmed_error():
    for klass in (
        InputError,
        ConfigurationError,
        CapabilityError,
        MissingExtraError,
        ModelLoadError,
        PolicyError,
        BudgetExceededError,
        InternalError,
        InferenceError,
    ):
        assert issubclass(klass, OpenMedError)


@pytest.mark.parametrize(
    "klass,builtin",
    [
        (InputError, ValueError),
        (ConfigurationError, ValueError),
        (PolicyError, ValueError),
        (CapabilityError, ImportError),
        (MissingExtraError, ImportError),
        (ModelLoadError, ImportError),
        (BudgetExceededError, RuntimeError),
        (InternalError, RuntimeError),
        (InferenceError, RuntimeError),
    ],
)
def test_backcompat_builtin_bases(klass, builtin):
    """Each typed error keeps the builtin base its predecessor used."""
    assert issubclass(klass, builtin)


def test_capability_subclasses():
    assert issubclass(MissingExtraError, CapabilityError)
    assert issubclass(ModelLoadError, CapabilityError)
    assert issubclass(InferenceError, InternalError)


def test_exports_from_top_level_and_core():
    for name in (
        "OpenMedError",
        "InputError",
        "ConfigurationError",
        "CapabilityError",
        "MissingExtraError",
        "ModelLoadError",
        "PolicyError",
        "BudgetExceededError",
        "InternalError",
        "InferenceError",
        "redact_detail",
    ):
        assert hasattr(openmed, name), f"openmed.{name} is not exported"
        assert name in openmed.__all__


def test_legacy_missing_dependency_aliases_join_taxonomy():
    """Pre-existing optional-dependency errors participate in the taxonomy."""
    from openmed.core.pii import MissingOptionalDependencyError
    from openmed.ner.exceptions import MissingDependencyError

    assert issubclass(MissingOptionalDependencyError, MissingExtraError)
    assert issubclass(MissingDependencyError, MissingExtraError)
    # Legacy ImportError-based catches still work.
    assert issubclass(MissingOptionalDependencyError, ImportError)
    assert issubclass(MissingDependencyError, ImportError)


# --------------------------------------------------------------------------- #
# Structured attributes / codes
# --------------------------------------------------------------------------- #


def test_error_carries_code_message_and_details():
    exc = InputError("bad thing", details={"argument": "text"})
    assert exc.code == "input_error"
    assert exc.message == "bad thing"
    assert exc.details == {"argument": "text"}
    assert str(exc) == "bad thing"


def test_each_class_has_distinct_stable_code():
    codes = {
        OpenMedError("x").code,
        InputError("x").code,
        ConfigurationError("x").code,
        CapabilityError("x").code,
        MissingExtraError("x").code,
        ModelLoadError("x").code,
        PolicyError("x").code,
        BudgetExceededError("x").code,
        InternalError("x").code,
        InferenceError("x").code,
    }
    # Ten classes must map to ten distinct codes.
    assert len(codes) == 10


def test_missing_extra_error_records_package_and_extra():
    exc = MissingExtraError(
        "needs the widgets extra",
        package="widgets",
        extra="widgets",
        feature="widget things",
    )
    assert exc.package == "widgets"
    assert exc.extra == "widgets"
    assert exc.details["package"] == "widgets"
    assert exc.details["extra"] == "widgets"


def test_model_load_error_records_model_name():
    exc = ModelLoadError("could not load", model_name="acme/model")
    assert exc.model_name == "acme/model"
    assert exc.details["model_name"] == "acme/model"


# --------------------------------------------------------------------------- #
# Actionable, PHI-free messages (asserted against fixtures)
# --------------------------------------------------------------------------- #


def test_redact_detail_hides_content_but_is_stable():
    secret = "John Doe, SSN 123-45-6789"
    descriptor = redact_detail(secret)
    assert secret not in descriptor
    assert "123-45-6789" not in descriptor
    assert "len=" in descriptor and "sha256=" in descriptor
    # Stable / reproducible for the same input.
    assert redact_detail(secret) == descriptor


def test_unknown_language_message_is_actionable_and_phi_free():
    from openmed.core.pii import _resolve_effective_pii_model

    raw_phi = "Patient Jane Roe, MRN 0042"
    with pytest.raises(InputError) as excinfo:
        # ``lang`` is a caller-supplied language selector, never PHI, but the
        # surrounding text (raw_phi) must never leak into the message.
        _resolve_effective_pii_model(raw_phi, "not-a-lang")
    message = str(excinfo.value)
    assert "not-a-lang" in message  # names the offending selector
    assert "supported" in message.lower()  # remediation
    assert raw_phi not in message  # no raw text/PHI


def test_method_conflict_message_never_leaks_secret_values():
    from openmed.core.pii import _resolve_deidentification_method

    secret = b"top-secret-hmac-key-material"
    with pytest.raises(InputError) as excinfo:
        _resolve_deidentification_method(
            "mask",
            None,
            None,
            date_shift_secret=secret,
        )
    message = str(excinfo.value)
    # Names the parameter and the fix, but not the secret's value.
    assert "date_shift_secret" in message
    assert "top-secret" not in message
    assert "method='shift_dates'" in message

    # A patient_key value must likewise never appear in its conflict message.
    with pytest.raises(InputError) as excinfo2:
        _resolve_deidentification_method(
            "mask",
            None,
            None,
            patient_key="patient-XYZ",
        )
    message2 = str(excinfo2.value)
    assert "patient_key" in message2
    assert "patient-XYZ" not in message2


def test_output_format_message_is_actionable():
    from openmed.processing.outputs import format_predictions

    with pytest.raises(InputError) as excinfo:
        format_predictions([], "text", output_format="xml")
    message = str(excinfo.value)
    assert "xml" in message
    for fmt in ("dict", "json", "html", "csv"):
        assert fmt in message


# --------------------------------------------------------------------------- #
# Malformed input yields InputError (not a generic ValueError) end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda: openmed.extract_pii(12345),
        lambda: openmed.deidentify(None),
        lambda: openmed.reidentify(object(), {}),
        lambda: openmed.reidentify("safe", ["not", "a", "mapping"]),
    ],
)
def test_malformed_input_raises_input_error_end_to_end(call):
    with pytest.raises(InputError) as excinfo:
        call()
    # Back-compat: existing ``except ValueError`` callers still catch it.
    assert isinstance(excinfo.value, ValueError)
    # No raw input value echoed into the message.
    assert "12345" not in str(excinfo.value)


def test_input_error_is_catchable_as_openmed_error():
    with pytest.raises(OpenMedError):
        openmed.extract_pii(3.14)


# --------------------------------------------------------------------------- #
# Robustness gate: no bare-except / except-Exception swallow on public modules
# --------------------------------------------------------------------------- #


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    """Return True if a broad handler silently swallows the exception.

    A broad ``except Exception`` / bare ``except:`` is a *swallow* only when its
    body does nothing meaningful with the error: it neither re-raises, nor
    references the bound exception, nor performs any recovery work. In other
    words the body is a pure no-op (``pass`` / ``...``). Handlers that re-raise
    (typically converting to a typed taxonomy error), capture the exception into
    a result, or run a deliberate fallback are all acceptable.
    """
    body = handler.body

    # Re-raising anywhere in the body is always acceptable.
    if any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
        return False

    # Referencing the bound exception name means it is not discarded.
    if handler.name is not None:
        for node in ast.walk(handler):
            if isinstance(node, ast.Name) and node.id == handler.name:
                return False  # exception is used -> not a swallow

    # A pure no-op body (only ``pass`` / ``...``) is a swallow.
    def _is_noop(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return stmt.value.value is Ellipsis
        return False

    return all(_is_noop(stmt) for stmt in body)


def test_no_bare_except_or_swallowed_exception_on_public_path():
    """Enumerated public modules must not swallow broad exceptions silently.

    A ``except Exception`` or bare ``except:`` is only allowed when its body
    re-raises (typically converting to a typed taxonomy error), captures the
    exception, or performs a deliberate recovery. A pure no-op swallow on the
    public path hides failures and is forbidden by OM-824.
    """
    offenders: list[str] = []
    for rel in _PUBLIC_MODULES:
        path = _REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            is_bare = handler.type is None
            is_broad = isinstance(handler.type, ast.Name) and handler.type.id in {
                "Exception",
                "BaseException",
            }
            if (is_bare or is_broad) and _handler_swallows(handler):
                offenders.append(f"{rel}:{handler.lineno}")

    assert not offenders, (
        "Broad except handlers that silently swallow were found on the public "
        "path:\n" + "\n".join(offenders)
    )


def test_public_modules_exist():
    for rel in _PUBLIC_MODULES:
        assert (_REPO_ROOT / rel).is_file(), f"missing enumerated module {rel}"


# --------------------------------------------------------------------------- #
# Service and MCP layers map each taxonomy class to a stable error code
# --------------------------------------------------------------------------- #


def test_mcp_error_codes_cover_taxonomy_and_are_stable():
    from openmed.mcp.server import MCP_ERROR_CODES, mcp_error_payload

    expected = {
        "OpenMedError": "openmed_error",
        "InputError": "input_error",
        "ConfigurationError": "configuration_error",
        "CapabilityError": "capability_error",
        "MissingExtraError": "missing_extra",
        "ModelLoadError": "model_load_error",
        "PolicyError": "policy_error",
        "BudgetExceededError": "budget_exceeded",
        "InternalError": "internal_error",
        "InferenceError": "inference_error",
    }
    assert MCP_ERROR_CODES == expected

    payload = mcp_error_payload(InputError("bad language", details={"lang": "zz"}))
    assert payload["error"]["code"] == "input_error"
    assert payload["error"]["message"] == "bad language"
    assert payload["error"]["details"] == {"lang": "zz"}


def test_mcp_error_codes_match_class_default_codes():
    from openmed.mcp.server import MCP_ERROR_CODES

    classes = {
        "OpenMedError": OpenMedError,
        "InputError": InputError,
        "ConfigurationError": ConfigurationError,
        "CapabilityError": CapabilityError,
        "MissingExtraError": MissingExtraError,
        "ModelLoadError": ModelLoadError,
        "PolicyError": PolicyError,
        "BudgetExceededError": BudgetExceededError,
        "InternalError": InternalError,
        "InferenceError": InferenceError,
    }
    for name, klass in classes.items():
        assert MCP_ERROR_CODES[name] == klass.code


@pytest.mark.parametrize(
    "exc,status,code",
    [
        (InputError("bad", details={"a": 1}), 400, "input_error"),
        (ConfigurationError("cfg"), 400, "configuration_error"),
        (PolicyError("policy"), 400, "policy_error"),
        (CapabilityError("cap"), 503, "capability_error"),
        (MissingExtraError("extra", package="x"), 503, "missing_extra"),
        (ModelLoadError("load", model_name="m"), 503, "model_load_error"),
        (BudgetExceededError("budget"), 503, "budget_exceeded"),
        (InternalError("boom"), 500, "internal_error"),
        (InferenceError("infer"), 500, "inference_error"),
    ],
)
def test_service_maps_taxonomy_to_stable_http_status_and_code(exc, status, code):
    from openmed.service.app import _openmed_error_response, _taxonomy_http_status

    assert _taxonomy_http_status(exc) == status
    response = _openmed_error_response(exc)
    assert response.status_code == status
    body = response.body.decode("utf-8")
    assert f'"code":"{code}"' in body or f'"code": "{code}"' in body


def test_service_5xx_omits_details_but_4xx_includes_them():
    from openmed.service.app import _openmed_error_response

    client = _openmed_error_response(InputError("bad", details={"argument": "text"}))
    server = _openmed_error_response(InternalError("boom", details={"trace": "x"}))
    assert b'"argument"' in client.body  # 4xx keeps PHI-free remediation details
    assert b'"trace"' not in server.body  # 5xx never leaks internal context
