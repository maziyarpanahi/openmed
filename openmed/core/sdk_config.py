"""Portable SDK configuration subset shared by Python and the OpenMedKit ports.

The Python, Swift, Kotlin and TypeScript surfaces spell the same intent
(offline only, pinned model, confidence threshold, policy profile, ONNX
variant) differently. This module is the Python reference for one portable
document with six keys, described by ``sdk_config.schema.json``:

``confidence_threshold``, ``local_only``, ``model_id``, ``model_revision``,
``onnx_variant`` and ``policy_profile``.

Validation fails closed. Unknown keys are rejected, and every problem is
reported as an :class:`SdkConfigIssue` holding a controlled code and, where
it is a known key, the field name. Issues never contain input values or
caller-supplied key names.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .errors import LegacyRootedError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openmed.core.config import OpenMedConfig

SDK_CONFIG_SCHEMA_ID: Final = "https://openmed.life/schemas/sdk_config.schema.json"
SDK_CONFIG_VERSION: Final = 1
SDK_CONFIG_SCHEMA_FILE: Final = (
    Path(__file__).with_name("schemas") / "json" / "sdk_config.schema.json"
)

#: The portable keys, in the order issues are reported.
SDK_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "confidence_threshold",
    "local_only",
    "model_id",
    "model_revision",
    "onnx_variant",
    "policy_profile",
)

ONNX_VARIANTS: Final[tuple[str, ...]] = ("auto", "int8", "fp16", "fp32")
#: Mirrors :data:`openmed.core.portable_policy.PORTABLE_PROFILES`.
POLICY_PROFILES: Final[tuple[str, ...]] = (
    "australia_privacy_act",
    "canada_pipeda",
    "clinical_minimal_redaction",
    "gdpr_art9_health",
    "gdpr_pseudonymization",
    "hipaa_expert_review_assist",
    "hipaa_safe_harbor",
    "research_limited_dataset",
    "strict_no_leak",
)

#: Values a kit uses for absent keys. ``local_only`` is on by default.
SDK_CONFIG_DEFAULTS: Final[Mapping[str, Any]] = {
    "confidence_threshold": 0.5,
    "local_only": True,
    "model_id": None,
    "model_revision": None,
    "onnx_variant": "auto",
    "policy_profile": "hipaa_safe_harbor",
}

#: ``OpenMedConfig`` attribute for each key, or ``None`` when Python takes the
#: setting per call instead of from ``OpenMedConfig``.
OPENMED_CONFIG_FIELDS: Final[Mapping[str, str | None]] = {
    "confidence_threshold": None,
    "local_only": "local_only",
    "model_id": "pii_model",
    "model_revision": "pii_model_revision",
    "onnx_variant": "onnx_variant",
    "policy_profile": None,
}

CODE_CONFIG_NOT_OBJECT: Final = "config_not_object"
CODE_UNKNOWN_FIELD: Final = "unknown_field"
CODE_WRONG_TYPE: Final = "wrong_type"
CODE_OUT_OF_RANGE: Final = "out_of_range"
CODE_NOT_ALLOWED_VALUE: Final = "not_allowed_value"
CODE_INVALID_FORMAT: Final = "invalid_format"
CODE_MISSING_DEPENDENCY: Final = "missing_dependency"

#: The closed set of issue codes.
ISSUE_CODES: Final[tuple[str, ...]] = (
    CODE_CONFIG_NOT_OBJECT,
    CODE_INVALID_FORMAT,
    CODE_MISSING_DEPENDENCY,
    CODE_NOT_ALLOWED_VALUE,
    CODE_OUT_OF_RANGE,
    CODE_UNKNOWN_FIELD,
    CODE_WRONG_TYPE,
)

_MODEL_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?"
)
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True, slots=True)
class SdkConfigIssue:
    """One value-free validation problem.

    Attributes:
        code: A member of :data:`ISSUE_CODES`.
        field: The portable key the issue concerns, or ``None`` for
            document-level issues (not an object, unknown keys).
    """

    code: str
    field: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return the JSON-ready issue."""

        return {"code": self.code, "field": self.field}


class SdkConfigError(ValueError, LegacyRootedError):
    """Raised when an SDK configuration document fails validation."""

    def __init__(self, issues: tuple[SdkConfigIssue, ...]) -> None:
        self.issues = issues
        summary = ", ".join(
            issue.code if issue.field is None else f"{issue.field}:{issue.code}"
            for issue in issues
        )
        super().__init__(f"invalid SDK configuration: {summary}")


@lru_cache(maxsize=1)
def load_sdk_config_schema() -> dict[str, Any]:
    """Load the bundled ``sdk_config.schema.json``."""

    return json.loads(SDK_CONFIG_SCHEMA_FILE.read_text(encoding="utf-8"))


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _field_issue(name: str, value: object) -> str | None:
    if name == "confidence_threshold":
        if not _is_number(value):
            return CODE_WRONG_TYPE
        # Compare integers exactly: float() overflows on a huge JSON integer.
        if isinstance(value, int):
            return None if 0 <= value <= 1 else CODE_OUT_OF_RANGE
        number = float(value)  # type: ignore[arg-type]
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return CODE_OUT_OF_RANGE
        return None
    if name == "local_only":
        return None if isinstance(value, bool) else CODE_WRONG_TYPE
    if name in {"model_id", "model_revision"}:
        if value is None:
            return None
        if not _is_string(value):
            return CODE_WRONG_TYPE
        pattern = _MODEL_ID_RE if name == "model_id" else _REVISION_RE
        if pattern.fullmatch(value) is None or ".." in value:  # type: ignore[arg-type]
            return CODE_INVALID_FORMAT
        return None
    allowed = ONNX_VARIANTS if name == "onnx_variant" else POLICY_PROFILES
    if not _is_string(value):
        return CODE_WRONG_TYPE
    return None if value in allowed else CODE_NOT_ALLOWED_VALUE


def validate_sdk_config(document: object) -> tuple[SdkConfigIssue, ...]:
    """Return every issue in a portable SDK configuration document.

    Issues are ordered deterministically: ``config_not_object`` alone when the
    document is not an object; otherwise at most one ``unknown_field``, then
    at most one issue per known key in :data:`SDK_CONFIG_FIELDS` order.
    ``missing_dependency`` is reported on ``model_revision`` when it is a
    valid string while ``model_id`` is absent or null.

    Args:
        document: A parsed JSON value.

    Returns:
        The issues; empty when the document is valid.
    """

    if not isinstance(document, Mapping):
        return (SdkConfigIssue(CODE_CONFIG_NOT_OBJECT, None),)
    issues: list[SdkConfigIssue] = []
    if any(key not in SDK_CONFIG_FIELDS for key in document):
        issues.append(SdkConfigIssue(CODE_UNKNOWN_FIELD, None))
    for name in SDK_CONFIG_FIELDS:
        if name not in document:
            continue
        code = _field_issue(name, document[name])
        if (
            code is None
            and name == "model_revision"
            and document[name] is not None
            and document.get("model_id") is None
        ):
            code = CODE_MISSING_DEPENDENCY
        if code is not None:
            issues.append(SdkConfigIssue(code, name))
    return tuple(issues)


def resolve_sdk_config(document: object) -> dict[str, Any]:
    """Validate a document and fill absent keys with the portable defaults.

    Args:
        document: A parsed JSON value.

    Returns:
        All six keys in :data:`SDK_CONFIG_FIELDS` order. An integer
        ``confidence_threshold`` is returned as a float.

    Raises:
        SdkConfigError: When :func:`validate_sdk_config` reports any issue.
    """

    issues = validate_sdk_config(document)
    if issues:
        raise SdkConfigError(issues)
    assert isinstance(document, Mapping)
    resolved = {
        name: document.get(name, SDK_CONFIG_DEFAULTS[name])
        for name in SDK_CONFIG_FIELDS
    }
    resolved["confidence_threshold"] = float(resolved["confidence_threshold"])
    return resolved


def sdk_config_from_openmed_config(
    config: OpenMedConfig,
    *,
    confidence_threshold: float = SDK_CONFIG_DEFAULTS["confidence_threshold"],
    policy_profile: str = SDK_CONFIG_DEFAULTS["policy_profile"],
) -> dict[str, Any]:
    """Project an ``OpenMedConfig`` onto the portable subset.

    ``confidence_threshold`` and ``policy_profile`` are not ``OpenMedConfig``
    fields: Python takes them per call (``extract_pii(confidence_threshold=)``
    and ``deidentify(policy=)``), so they are passed here explicitly.

    Args:
        config: The configuration to project.
        confidence_threshold: Threshold to record, in ``[0, 1]``.
        policy_profile: Portable policy profile name.

    Returns:
        A validated document with all six keys.

    Raises:
        SdkConfigError: When a projected value is outside the portable subset,
            for example a ``pii_model`` that is a filesystem path.
    """

    document = {
        "confidence_threshold": confidence_threshold,
        "local_only": config.local_only,
        "model_id": config.pii_model,
        "model_revision": config.pii_model_revision,
        "onnx_variant": config.onnx_variant,
        "policy_profile": policy_profile,
    }
    return resolve_sdk_config(document)


__all__ = [
    "CODE_CONFIG_NOT_OBJECT",
    "CODE_INVALID_FORMAT",
    "CODE_MISSING_DEPENDENCY",
    "CODE_NOT_ALLOWED_VALUE",
    "CODE_OUT_OF_RANGE",
    "CODE_UNKNOWN_FIELD",
    "CODE_WRONG_TYPE",
    "ISSUE_CODES",
    "ONNX_VARIANTS",
    "OPENMED_CONFIG_FIELDS",
    "POLICY_PROFILES",
    "SDK_CONFIG_DEFAULTS",
    "SDK_CONFIG_FIELDS",
    "SDK_CONFIG_SCHEMA_ID",
    "SDK_CONFIG_VERSION",
    "SdkConfigError",
    "SdkConfigIssue",
    "load_sdk_config_schema",
    "resolve_sdk_config",
    "sdk_config_from_openmed_config",
    "validate_sdk_config",
]
