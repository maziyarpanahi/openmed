"""Contract tests for OpenMed's structured public error taxonomy."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Callable

import pytest

import openmed
from openmed import core
from openmed.core.budget import BudgetExceededError as BudgetModuleError
from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.errors import (
    ERROR_CODES,
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
from openmed.core.pii import _resolve_deidentification_method
from openmed.multimodal.exceptions import MissingDependencyError as MultimediaMissing
from openmed.ner.exceptions import MissingDependencyError as NERMissing
from openmed.processing.text import InputError as ProcessingInputError
from openmed.utils.gateway import InputValidationError, validate_language
from openmed.utils.validation import validate_output_format

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = _ROOT / "tests" / "fixtures" / "error_taxonomy.json"
_PUBLIC_MODULES = (
    "openmed/core/errors.py",
    "openmed/core/pii.py",
    "openmed/mcp/server.py",
    "openmed/ner/exceptions.py",
    "openmed/processing/outputs.py",
    "openmed/service/app.py",
)

_CLASSES = {
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


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_taxonomy_hierarchy_and_stable_codes() -> None:
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

    assert dict(ERROR_CODES) == expected
    assert len(set(ERROR_CODES.values())) == len(ERROR_CODES)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in ERROR_CODES.values())
    for name, error_class in _CLASSES.items():
        assert issubclass(error_class, OpenMedError)
        assert error_class.code == expected[name]


@pytest.mark.parametrize(
    ("error_class", "legacy_base"),
    [
        (InputError, ValueError),
        (InputError, TypeError),
        (ConfigurationError, ValueError),
        (ConfigurationError, TypeError),
        (ConfigurationError, KeyError),
        (CapabilityError, ImportError),
        (MissingExtraError, ImportError),
        (ModelLoadError, ImportError),
        (ModelLoadError, ValueError),
        (PolicyError, ValueError),
        (PolicyError, TypeError),
        (BudgetExceededError, RuntimeError),
        (InternalError, RuntimeError),
        (InferenceError, RuntimeError),
    ],
)
def test_builtin_exception_compatibility(
    error_class: type[OpenMedError],
    legacy_base: type[BaseException],
) -> None:
    assert issubclass(error_class, legacy_base)


def test_public_exports_and_existing_error_aliases_remain_compatible() -> None:
    for name, error_class in _CLASSES.items():
        assert getattr(openmed, name) is error_class
        assert getattr(core, name) is error_class
        assert name in openmed.__all__
        assert name in core.__all__

    assert openmed.ERROR_CODES is ERROR_CODES
    assert ProcessingInputError is InputError
    assert BudgetModuleError is BudgetExceededError
    assert issubclass(InputValidationError, InputError)
    assert issubclass(MissingOptionalDependencyError, MissingExtraError)
    assert isinstance(NERMissing("synthetic", "Install the extra."), MissingExtraError)
    assert isinstance(
        MultimediaMissing("synthetic", "Install the extra."),
        MissingExtraError,
    )


def test_actionable_synthetic_fixture_covers_every_error_class() -> None:
    fixture = _load_fixture()
    assert fixture["schema_version"] == 1
    assert fixture["synthetic"] is True
    forbidden = str(fixture["forbidden_raw_text"])
    cases = fixture["cases"]
    assert isinstance(cases, list)
    assert {case["class"] for case in cases} == set(_CLASSES)

    action_words = (
        "choose",
        "configure",
        "install",
        "pass",
        "reduce",
        "report",
        "retry",
        "verify",
    )
    for case in cases:
        error_class = _CLASSES[case["class"]]
        error = error_class(case["message"], details=case["details"])
        payload = error.to_dict()

        assert error.code == case["code"]
        assert payload == {
            "code": case["code"],
            "message": case["message"],
            "details": case["details"],
        }
        assert any(word in error.message.lower() for word in action_words)
        assert forbidden not in json.dumps(payload, sort_keys=True)
        assert error.to_dict(include_details=False) == {
            "code": case["code"],
            "message": case["message"],
        }


def test_redact_detail_is_stable_and_never_echoes_raw_text() -> None:
    raw_phi = "SYNTHETIC_RAW_PHI_SENTINEL_1354"
    first = redact_detail(raw_phi)

    assert first == redact_detail(raw_phi)
    assert raw_phi not in first
    assert re.fullmatch(r"<redacted bytes=31 sha256=[0-9a-f]{64}>", first)


@pytest.mark.parametrize(
    "call",
    [
        lambda: openmed.extract_pii(12345),
        lambda: openmed.deidentify(None),
        lambda: openmed.reidentify(object(), {}),
        lambda: openmed.reidentify("synthetic", ["not", "a", "mapping"]),
        lambda: openmed.reidentify("synthetic", {"[NAME]": object()}),
    ],
)
def test_malformed_public_pii_input_raises_input_error_end_to_end(
    call: Callable[[], object],
) -> None:
    with pytest.raises(InputError) as raised:
        call()

    assert isinstance(raised.value, (TypeError, ValueError))
    assert raised.value.code == "input_error"


@pytest.mark.parametrize(
    "call",
    [
        lambda secret: _resolve_deidentification_method(secret, None, None),
        lambda secret: validate_language(secret),
        lambda secret: validate_output_format(secret),
    ],
)
def test_invalid_selectors_do_not_echo_untrusted_values(
    call: Callable[[str], object],
) -> None:
    raw_phi = "SYNTHETIC_RAW_PHI_SENTINEL_1354"
    with pytest.raises(InputError) as raised:
        call(raw_phi)

    serialized = json.dumps(raised.value.to_dict(), sort_keys=True)
    assert raw_phi not in serialized


def test_budget_failure_is_rooted_and_carries_only_safe_counts() -> None:
    budget = openmed.RequestBudget(max_input_chars=4)
    with pytest.raises(BudgetExceededError) as raised:
        budget.check_input_length(42, checkpoint="synthetic_guard")

    assert raised.value.code == "budget_exceeded"
    assert raised.value.details == {
        "kind": "input_chars",
        "limit": 4,
        "observed": 42,
        "checkpoint": "synthetic_guard",
    }


def test_service_maps_the_complete_taxonomy_to_documented_http_contract() -> None:
    from openmed.service.app import (
        _openmed_error_response,
        _taxonomy_http_status,
        create_app,
    )

    expected_statuses = {
        "OpenMedError": 500,
        "InputError": 400,
        "ConfigurationError": 400,
        "CapabilityError": 503,
        "MissingExtraError": 503,
        "ModelLoadError": 503,
        "PolicyError": 400,
        "BudgetExceededError": 503,
        "InternalError": 500,
        "InferenceError": 500,
    }
    cases = _load_fixture()["cases"]
    for case in cases:
        error = _CLASSES[case["class"]](
            case["message"],
            details=case["details"],
        )
        status = expected_statuses[case["class"]]
        response = _openmed_error_response(error)
        body = json.loads(response.body)

        assert _taxonomy_http_status(error) == status
        assert response.status_code == status
        assert body["error"]["code"] == case["code"]
        assert body["error"]["message"] == case["message"]
        assert body["error"]["details"] == (case["details"] if status < 500 else None)

    assert OpenMedError in create_app().exception_handlers


def test_mcp_maps_taxonomy_and_specialized_input_codes() -> None:
    from openmed.mcp.server import MCP_ERROR_CODES, _error_envelope, mcp_error_payload

    assert MCP_ERROR_CODES == dict(ERROR_CODES)
    error = InputError(
        "Correct the malformed request and retry.",
        details={"offset": 4},
    )
    assert mcp_error_payload(error) == {
        "error": {
            "code": "input_error",
            "message": error.message,
            "details": {"offset": 4},
        },
        "is_error": True,
    }

    specialized = InputValidationError(
        "Pass text in the documented type and retry.",
        code="text_type",
        metadata={"type": "int"},
    )
    assert _error_envelope(specialized)["error"]["code"] == "text_type"


def _is_broad_exception_handler(handler: ast.ExceptHandler) -> bool:
    return isinstance(handler.type, ast.Name) and handler.type.id in {
        "BaseException",
        "Exception",
    }


def _is_noop_handler(handler: ast.ExceptHandler) -> bool:
    return all(
        isinstance(node, ast.Pass)
        or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
        )
        for node in handler.body
    )


def test_public_error_paths_have_no_bare_or_broad_silent_swallows() -> None:
    offenders: list[str] = []
    for relative_path in _PUBLIC_MODULES:
        path = _ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        for handler in (
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ):
            if handler.type is None or (
                _is_broad_exception_handler(handler) and _is_noop_handler(handler)
            ):
                offenders.append(f"{relative_path}:{handler.lineno}")

    assert not offenders, "silent broad exception handlers:\n" + "\n".join(offenders)


def test_public_error_documentation_covers_codes_and_transports() -> None:
    documentation = (_ROOT / "docs" / "api" / "errors.md").read_text(encoding="utf-8")
    for name, code in ERROR_CODES.items():
        assert name in documentation
        assert f"`{code}`" in documentation
    assert "REST" in documentation
    assert "MCP" in documentation
