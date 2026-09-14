"""Offline capability checks for local clinical small-language-model packages.

The probe reads only a package's JSON metadata.  It does not construct a
tokenizer or model, read model weights, inspect clinical input, open a socket,
or provide a remote fallback.  This makes it suitable for a deployment gate
that runs before a clinical request enters the local inference path.

The public report contains only bounded metadata, fixed reason codes, counts,
booleans, and a non-reversible fingerprint.  In particular, it never echoes a
model path, identifier, arbitrary manifest value, prompt, or patient data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

CAPABILITY_SCHEMA_VERSION: Final = "openmed.clinical-slm-capabilities.v1"
CLINICAL_SLM_CAPABILITIES_SCHEMA_VERSION: Final = CAPABILITY_SCHEMA_VERSION
SCHEMA_VERSION: Final = CAPABILITY_SCHEMA_VERSION

BOUNDED_SUMMARIZATION: Final = "bounded_summarization"
NLI: Final = "nli"
SUPPORTED_CAPABILITIES: Final = (BOUNDED_SUMMARIZATION, NLI)
DEFAULT_CAPABILITIES: Final = SUPPORTED_CAPABILITIES

MANIFEST_FILENAMES: Final = (
    "clinical-slm-capabilities.json",
    "clinical-slm-manifest.json",
    "manifest.json",
)

_MAX_IDENTIFIER_LENGTH: Final = 128
_MAX_CONTEXT_TOKENS: Final = 2**31 - 1
_KNOWN_QUANTIZATION_SCHEMES: Final = frozenset(
    {
        "bf16",
        "fp16",
        "fp32",
        "int2",
        "int3",
        "int4",
        "int8",
        "q4_0",
        "q4_k_m",
        "q5_k_m",
        "q8_0",
        "none",
    }
)
_QUANTIZATION_ALIASES: Final = {
    "float32": "fp32",
    "float16": "fp16",
    "float16e": "fp16",
    "bfloat16": "bf16",
    "full_precision": "fp32",
    "fullprecision": "fp32",
    "unquantized": "none",
    "int_2": "int2",
    "int_3": "int3",
    "int_4": "int4",
    "int_8": "int8",
    "q4-k-m": "q4_k_m",
    "q5-k-m": "q5_k_m",
}

_TASK_ALIASES: Final[dict[str, str]] = {
    "bounded_summary": BOUNDED_SUMMARIZATION,
    "bounded_summarization": BOUNDED_SUMMARIZATION,
    "summarize": BOUNDED_SUMMARIZATION,
    "clinical_summary": BOUNDED_SUMMARIZATION,
    "clinical_summarization": BOUNDED_SUMMARIZATION,
    "summary": BOUNDED_SUMMARIZATION,
    "summarization": BOUNDED_SUMMARIZATION,
    "bounded_summarization_task": BOUNDED_SUMMARIZATION,
    "clinical_nli": NLI,
    "natural_language_inference": NLI,
    "nli": NLI,
}

_CAPABILITY_ALIASES: Final[dict[str, str]] = {
    **_TASK_ALIASES,
    "bounded_summary": BOUNDED_SUMMARIZATION,
    "summary": BOUNDED_SUMMARIZATION,
    "summarize": BOUNDED_SUMMARIZATION,
    "natural_language_inference": NLI,
}

_RUNTIME_FEATURE_ALIASES: Final[dict[str, str]] = {
    "core_ml": "coreml",
    "hf": "transformers",
    "onnx_runtime": "onnxruntime",
    "pytorch": "torch",
    "sentence_piece": "sentencepiece",
}
_RUNTIME_FEATURE_MODULES: Final[dict[str, tuple[str, ...]]] = {
    "coreml": ("coremltools",),
    "mlx": ("mlx",),
    "onnxruntime": ("onnxruntime",),
    "sentencepiece": ("sentencepiece",),
    "tokenizers": ("tokenizers",),
    "torch": ("torch",),
    "transformers": ("transformers",),
}

_REASON_ORDER: Final = (
    "manifest_invalid",
    "offline_required",
    "offline_metadata_invalid",
    "human_review_required",
    "human_review_metadata_invalid",
    "cloud_fallback_forbidden",
    "network_required",
    "task_not_declared",
    "context_limit_missing",
    "context_limit_invalid",
    "context_limit_too_small",
    "output_limit_missing",
    "quantization_missing",
    "quantization_invalid",
    "quantization_unsupported",
    "runtime_feature_unknown",
    "runtime_feature_invalid",
    "runtime_feature_missing",
)
_REASON_INDEX: Final = {reason: index for index, reason in enumerate(_REASON_ORDER)}

_ERROR_MESSAGES: Final = {
    "invalid_manifest": "clinical SLM capability manifest is invalid",
    "manifest_missing": "clinical SLM capability manifest is missing",
    "manifest_unreadable": "clinical SLM capability manifest cannot be read",
    "manifest_invalid_json": "clinical SLM capability manifest is not valid JSON",
    "duplicate_manifest_field": "clinical SLM capability manifest has duplicate fields",
    "ambiguous_manifest_field": "clinical SLM capability manifest has conflicting aliases",
    "invalid_requirement": "clinical SLM capability requirement is invalid",
    "unknown_capability": "clinical SLM capability is not supported by this probe",
    "cloud_fallback_disabled": "cloud fallback is disabled for clinical SLM probes",
}

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.+-]{0,127}$")


class ClinicalSLMCapabilityError(ValueError):
    """Raised for an invalid local capability probe input.

    Only a stable error code and a static message are retained.  Caller
    values, paths, and manifest contents are intentionally excluded.
    """

    code = "clinical_slm_capability_error"

    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code if reason_code in _ERROR_MESSAGES else "invalid_manifest"
        )
        self.code = self.reason_code
        super().__init__(_ERROR_MESSAGES[self.reason_code])

    def to_dict(self) -> dict[str, str]:
        """Return a machine-readable, value-free error object."""

        return {"code": self.reason_code, "message": str(self)}


def _fail(reason_code: str) -> None:
    raise ClinicalSLMCapabilityError(reason_code) from None


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("invalid_manifest")
    raise AssertionError("unreachable")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_manifest_field")
        result[key] = value
    return result


def _copy_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        _fail("invalid_manifest")
    try:
        copied = dict(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_manifest")
    if any(type(key) is not str for key in copied):
        _fail("invalid_manifest")
    return copied


def _parse_json(payload: str | bytes | bytearray) -> dict[str, Any]:
    try:
        decoded = json.loads(payload, object_pairs_hook=_strict_json_object)
    except ClinicalSLMCapabilityError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, RecursionError):
        _fail("manifest_invalid_json")
    if not isinstance(decoded, Mapping):
        _fail("invalid_manifest")
    return _copy_mapping(decoded)


def _read_manifest_file(source: str | Path) -> dict[str, Any]:
    try:
        path = Path(source)
        if path.is_dir():
            candidates = tuple(path / name for name in MANIFEST_FILENAMES)
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        if path.is_symlink() or not path.is_file():
            _fail("manifest_missing")
        payload = path.read_bytes()
    except ClinicalSLMCapabilityError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("manifest_unreadable")
    return _parse_json(payload)


def load_clinical_slm_capability_manifest(
    source: Mapping[str, Any] | str | Path | bytes | bytearray,
) -> dict[str, Any]:
    """Load one local capability manifest without importing a model runtime.

    ``source`` may be a JSON-like mapping, a JSON document, or a path to a
    manifest file/directory.  Directory reads consider only the three fixed
    manifest filenames in :data:`MANIFEST_FILENAMES`; no directory scan or
    network lookup is performed.
    """

    if isinstance(source, Mapping):
        return _copy_mapping(source)
    if isinstance(source, (bytes, bytearray)):
        return _parse_json(source)
    if isinstance(source, str) and source.lstrip().startswith(("{", "[")):
        return _parse_json(source)
    if isinstance(source, (str, Path)):
        return _read_manifest_file(source)
    try:
        return _read_manifest_file(Path(source))  # type: ignore[arg-type]
    except ClinicalSLMCapabilityError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("manifest_unreadable")
    raise AssertionError("unreachable")


load_capability_manifest = load_clinical_slm_capability_manifest
load_clinical_slm_manifest = load_clinical_slm_capability_manifest


def _alias_value(
    payload: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: Any = None,
) -> tuple[bool, Any]:
    present = tuple(name for name in names if name in payload)
    if len(present) > 1:
        _fail("ambiguous_manifest_field")
    if present:
        return True, payload[present[0]]
    return False, default


def _normalise_token(value: Any) -> str | None:
    if type(value) is not str:
        return None
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    if (
        not token
        or len(token) > _MAX_IDENTIFIER_LENGTH
        or _IDENTIFIER_RE.fullmatch(token) is None
    ):
        return None
    return token


def _normalise_task(value: Any) -> str | None:
    token = _normalise_token(value)
    if token is None:
        return None
    return _TASK_ALIASES.get(token)


def _normalise_capability(value: Any) -> str | None:
    token = _normalise_token(value)
    if token is None:
        return None
    return _CAPABILITY_ALIASES.get(token)


def _normalise_runtime_feature(value: Any) -> str | None:
    token = _normalise_token(value)
    if token is None:
        return None
    return _RUNTIME_FEATURE_ALIASES.get(token, token)


def _sequence_values(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, Mapping):
        values: list[Any] = []
        try:
            for key, enabled in value.items():
                if type(enabled) is not bool:
                    return None
                if enabled:
                    values.append(key)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return None
        return tuple(values)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return None


def _normalise_tasks(value: Any) -> tuple[tuple[str, ...], int, bool]:
    if value is None:
        return (), 0, False
    values = _sequence_values(value)
    if values is None:
        return (), 1, True
    tasks: set[str] = set()
    unknown_count = 0
    for item in values:
        task = _normalise_task(item)
        if task is None:
            unknown_count += 1
        else:
            tasks.add(task)
    return tuple(sorted(tasks)), unknown_count, True


def _positive_int(value: Any) -> int | None:
    if type(value) is not int or not 0 < value <= _MAX_CONTEXT_TOKENS:
        return None
    return value


@dataclass(frozen=True, slots=True, repr=False)
class ContextLimits:
    """Safe context-budget metadata extracted from a package manifest."""

    max_context_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    declared: bool = False

    def __repr__(self) -> str:
        """Return a bounded metadata-only representation."""

        return "ContextLimits(<bounded metadata>)"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic context-budget metadata."""

        return {
            "declared": self.declared,
            "max_context_tokens": self.max_context_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }


def _normalise_context_limits(
    payload: Mapping[str, Any],
) -> tuple[ContextLimits, tuple[str, ...]]:
    present, raw_context = _alias_value(
        payload,
        ("context_limits", "context_limit", "context_window"),
        default=None,
    )
    context: dict[str, Any] = {}
    invalid = False
    if present:
        if isinstance(raw_context, Mapping):
            try:
                context = dict(raw_context)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                invalid = True
        elif type(raw_context) is int:
            context = {"max_context_tokens": raw_context}
        else:
            invalid = True

    for key in (
        "max_context_tokens",
        "context_length",
        "max_tokens",
        "context_tokens",
        "max_position_embeddings",
        "max_input_tokens",
        "input_tokens",
        "max_output_tokens",
        "output_tokens",
        "max_new_tokens",
    ):
        if key in payload and key not in context:
            context[key] = payload[key]
            present = True

    def first(keys: Sequence[str]) -> tuple[bool, Any]:
        for key in keys:
            if key in context:
                return True, context[key]
        return False, None

    context_present, raw_total = first(
        (
            "max_context_tokens",
            "context_length",
            "max_tokens",
            "context_tokens",
            "max_position_embeddings",
            "total_tokens",
            "max_total_tokens",
        )
    )
    input_present, raw_input = first(("max_input_tokens", "input_tokens"))
    output_present, raw_output = first(
        ("max_output_tokens", "output_tokens", "max_new_tokens")
    )

    total = _positive_int(raw_total) if context_present else None
    input_limit = _positive_int(raw_input) if input_present else None
    output_limit = _positive_int(raw_output) if output_present else None
    if (
        (context_present and total is None)
        or (input_present and input_limit is None)
        or (output_present and output_limit is None)
    ):
        invalid = True

    if context_present and total is not None and not input_present:
        input_limit = total
    limits = ContextLimits(
        max_context_tokens=total,
        max_input_tokens=input_limit,
        max_output_tokens=output_limit,
        declared=bool(present),
    )
    return limits, ("context_limit_invalid",) if invalid else ()


@dataclass(frozen=True, slots=True, repr=False)
class QuantizationMetadata:
    """Safe quantization metadata extracted from a package manifest."""

    scheme: str | None = None
    bits: int | None = None
    declared: bool = False

    def __repr__(self) -> str:
        """Return a bounded metadata-only representation."""

        return "QuantizationMetadata(<bounded metadata>)"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic quantization metadata."""

        return {
            "declared": self.declared,
            "scheme": self.scheme,
            "bits": self.bits,
        }


def _normalise_quantization(
    payload: Mapping[str, Any],
) -> tuple[QuantizationMetadata, tuple[str, ...]]:
    present, raw_quantization = _alias_value(
        payload,
        ("quantization", "quantisation"),
        default=None,
    )
    if not present:
        return QuantizationMetadata(), ("quantization_missing",)

    quantization: dict[str, Any]
    if isinstance(raw_quantization, Mapping):
        try:
            quantization = dict(raw_quantization)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return QuantizationMetadata(declared=True), ("quantization_invalid",)
    elif type(raw_quantization) is str:
        quantization = {"scheme": raw_quantization}
    else:
        return QuantizationMetadata(declared=True), ("quantization_invalid",)

    raw_scheme = None
    for key in ("scheme", "method", "type", "format", "name"):
        if key in quantization:
            raw_scheme = quantization[key]
            break
    token = _normalise_token(raw_scheme)
    if token is not None:
        token = _QUANTIZATION_ALIASES.get(token, token)
    raw_bits = quantization.get("bits", quantization.get("precision"))
    bits = (
        raw_bits if type(raw_bits) is int and raw_bits in {2, 3, 4, 8, 16, 32} else None
    )
    invalid = raw_scheme is None or token not in _KNOWN_QUANTIZATION_SCHEMES
    if raw_bits is not None and bits is None:
        invalid = True
    if token is not None and token.startswith("int") and bits is not None:
        try:
            invalid = invalid or int(token[3:]) != bits
        except ValueError:
            invalid = True
    if bits is None and token is not None and token.startswith("int"):
        try:
            inferred_bits = int(token[3:])
        except ValueError:
            inferred_bits = None
        if inferred_bits in {2, 3, 4, 8}:
            bits = inferred_bits
    metadata = QuantizationMetadata(
        scheme=token if token in _KNOWN_QUANTIZATION_SCHEMES else None,
        bits=bits,
        declared=True,
    )
    return metadata, ("quantization_invalid",) if invalid else ()


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeFeatureMetadata:
    """Safe runtime feature metadata and local availability observations."""

    required: tuple[str, ...] = ()
    available: tuple[str, ...] = ()
    unknown_count: int = 0
    invalid_count: int = 0

    def __repr__(self) -> str:
        """Return a bounded metadata-only representation."""

        return "RuntimeFeatureMetadata(<bounded metadata>)"

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic runtime feature metadata."""

        return {
            "required": list(self.required),
            "available": list(self.available),
            "unknown_count": self.unknown_count,
            "invalid_count": self.invalid_count,
        }


def _normalise_runtime_requirements(
    payload: Mapping[str, Any],
    available_runtime_features: Iterable[str] | Mapping[str, bool] | None,
) -> tuple[RuntimeFeatureMetadata, tuple[str, ...]]:
    present, raw_required = _alias_value(
        payload,
        ("required_runtime_features", "runtime_features", "runtime_requirements"),
        default=None,
    )
    required_values = _sequence_values(raw_required) if present else ()
    invalid_count = int(present and required_values is None)
    unknown_count = 0
    required: set[str] = set()
    if required_values is not None:
        for value in required_values:
            feature = _normalise_runtime_feature(value)
            if feature is None:
                unknown_count += 1
            else:
                required.add(feature)

    available: set[str] = set()
    if available_runtime_features is None:
        for feature in required:
            modules = _RUNTIME_FEATURE_MODULES.get(feature, ())
            if not modules:
                continue
            try:
                if all(
                    importlib.util.find_spec(module) is not None for module in modules
                ):
                    available.add(feature)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                continue
    else:
        if isinstance(available_runtime_features, Mapping):
            try:
                for key, value in available_runtime_features.items():
                    if type(value) is not bool:
                        invalid_count += 1
                        continue
                    feature = _normalise_runtime_feature(key)
                    if feature is None:
                        unknown_count += 1
                    elif value:
                        available.add(feature)
                    else:
                        available.discard(feature)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                invalid_count += 1
        else:
            values = _sequence_values(available_runtime_features)
            if values is None:
                invalid_count += 1
            else:
                for value in values:
                    feature = _normalise_runtime_feature(value)
                    if feature is None:
                        unknown_count += 1
                    else:
                        available.add(feature)

    reasons: list[str] = []
    if unknown_count:
        reasons.append("runtime_feature_unknown")
    if invalid_count:
        reasons.append("runtime_feature_invalid")
    if any(feature not in available for feature in required):
        reasons.append("runtime_feature_missing")
    return (
        RuntimeFeatureMetadata(
            required=tuple(sorted(required)),
            available=tuple(sorted(available & required)),
            unknown_count=unknown_count,
            invalid_count=invalid_count,
        ),
        tuple(reasons),
    )


def _policy_bool(
    payload: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: bool,
) -> tuple[bool, bool]:
    present, value = _alias_value(payload, names, default=default)
    if not present:
        return default, False
    return value if type(value) is bool else default, type(value) is not bool


def _manifest_fingerprint(
    *,
    tasks: tuple[str, ...],
    context_limits: ContextLimits,
    quantization: QuantizationMetadata,
    runtime: RuntimeFeatureMetadata,
    offline: bool,
    human_review_required: bool,
) -> str:
    safe_payload = {
        "context_limits": context_limits.to_dict(),
        "human_review_required": human_review_required,
        "offline": offline,
        "quantization": quantization.to_dict(),
        "required_runtime_features": list(runtime.required),
        "supported_tasks": list(tasks),
    }
    digest = hashlib.sha256(_canonical_json(safe_payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True, repr=False)
class _NormalisedManifest:
    tasks: tuple[str, ...]
    unknown_task_count: int
    tasks_declared: bool
    context_limits: ContextLimits
    context_reasons: tuple[str, ...]
    quantization: QuantizationMetadata
    quantization_reasons: tuple[str, ...]
    runtime: RuntimeFeatureMetadata
    runtime_reasons: tuple[str, ...]
    offline: bool
    offline_invalid: bool
    human_review_required: bool
    human_review_invalid: bool
    cloud_fallback_forbidden: bool
    network_required: bool
    manifest_fingerprint: str

    def __repr__(self) -> str:
        """Return a value-free representation."""

        return "_NormalisedManifest(<bounded metadata>)"


def _normalise_manifest(
    payload: Mapping[str, Any],
    available_runtime_features: Iterable[str] | Mapping[str, bool] | None,
) -> _NormalisedManifest:
    present, raw_tasks = _alias_value(
        payload,
        ("supported_tasks", "tasks", "task", "task_capabilities"),
        default=None,
    )
    tasks, unknown_task_count, tasks_declared = _normalise_tasks(
        raw_tasks if present else None
    )
    context_limits, context_reasons = _normalise_context_limits(payload)
    quantization, quantization_reasons = _normalise_quantization(payload)
    runtime, runtime_reasons = _normalise_runtime_requirements(
        payload,
        available_runtime_features,
    )
    offline, offline_invalid = _policy_bool(
        payload,
        ("offline", "local_only"),
        default=False,
    )
    human_review_required, human_review_invalid = _policy_bool(
        payload,
        ("human_review_required", "review_required"),
        default=False,
    )
    fallback_value, fallback_present = _alias_value(
        payload,
        ("cloud_fallback", "allow_cloud_fallback", "remote_fallback"),
        default=False,
    )
    cloud_fallback_forbidden = fallback_present and fallback_value is not False
    network_value, network_present = _alias_value(
        payload,
        ("network_required", "requires_network"),
        default=False,
    )
    network_required = network_present and network_value is not False
    return _NormalisedManifest(
        tasks=tasks,
        unknown_task_count=unknown_task_count,
        tasks_declared=tasks_declared,
        context_limits=context_limits,
        context_reasons=context_reasons,
        quantization=quantization,
        quantization_reasons=quantization_reasons,
        runtime=runtime,
        runtime_reasons=runtime_reasons,
        offline=offline,
        offline_invalid=offline_invalid,
        human_review_required=human_review_required,
        human_review_invalid=human_review_invalid,
        cloud_fallback_forbidden=cloud_fallback_forbidden,
        network_required=network_required,
        manifest_fingerprint=_manifest_fingerprint(
            tasks=tasks,
            context_limits=context_limits,
            quantization=quantization,
            runtime=runtime,
            offline=offline,
            human_review_required=human_review_required,
        ),
    )


def _ordered_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    unique = {reason for reason in reasons if reason in _REASON_INDEX}
    return tuple(sorted(unique, key=lambda reason: _REASON_INDEX[reason]))


@dataclass(frozen=True, slots=True, repr=False)
class UnsupportedCapabilityReason:
    """One machine-readable reason a requested capability is unavailable."""

    capability: str
    code: str

    def __post_init__(self) -> None:
        if self.capability not in SUPPORTED_CAPABILITIES:
            object.__setattr__(self, "capability", "package")
        if self.code not in _REASON_INDEX:
            object.__setattr__(self, "code", "manifest_invalid")

    def __repr__(self) -> str:
        """Return a safe reason representation."""

        return f"UnsupportedCapabilityReason({self.capability!r}, {self.code!r})"

    def to_dict(self) -> dict[str, str]:
        """Return the reason as a JSON-compatible mapping."""

        return {"capability": self.capability, "code": self.code}


CapabilityReason = UnsupportedCapabilityReason


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMCapabilityCheck:
    """Result for one requested clinical SLM capability."""

    name: str
    supported: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_CAPABILITIES:
            object.__setattr__(self, "name", NLI)
        object.__setattr__(self, "reason_codes", _ordered_reasons(self.reason_codes))
        derived = not self.reason_codes
        if self.supported is not derived:
            object.__setattr__(self, "supported", derived)

    @property
    def available(self) -> bool:
        """Return whether this capability can be selected locally."""

        return self.supported

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return stable unsupported reason codes."""

        return self.reason_codes

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, machine-readable check."""

        return {
            "supported": self.supported,
            "reason_codes": list(self.reason_codes),
        }


CapabilityCheck = ClinicalSLMCapabilityCheck


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMCapabilityReport(Mapping[str, Any]):
    """Deterministic, PHI-safe report from an offline capability probe."""

    _checks: tuple[ClinicalSLMCapabilityCheck, ...]
    tasks: tuple[str, ...]
    unknown_task_count: int
    tasks_declared: bool
    context_limits: ContextLimits
    quantization: QuantizationMetadata
    runtime_features: RuntimeFeatureMetadata
    offline: bool
    human_review_required: bool
    manifest_fingerprint: str
    schema_version: str = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        names = tuple(check.name for check in self._checks)
        if names != tuple(sorted(names, key=SUPPORTED_CAPABILITIES.index)):
            raise ValueError("clinical SLM capability checks must be ordered")
        if len(set(names)) != len(names):
            raise ValueError("clinical SLM capability checks must be unique")
        object.__setattr__(self, "_checks", tuple(self._checks))
        object.__setattr__(self, "tasks", tuple(sorted(set(self.tasks))))
        object.__setattr__(self, "runtime_features", self.runtime_features)

    @property
    def capabilities(self) -> Mapping[str, ClinicalSLMCapabilityCheck]:
        """Return checks keyed by their stable capability name."""

        return MappingProxyType({check.name: check for check in self._checks})

    @property
    def checks(self) -> tuple[ClinicalSLMCapabilityCheck, ...]:
        """Return checks in deterministic evaluation order."""

        return self._checks

    @property
    def supported(self) -> bool:
        """Return whether every requested capability is locally supported."""

        return all(check.supported for check in self._checks)

    @property
    def ready(self) -> bool:
        """Alias for :attr:`supported` used by deployment gates."""

        return self.supported

    @property
    def compatible(self) -> bool:
        """Alias for :attr:`supported`."""

        return self.supported

    @property
    def is_supported(self) -> bool:
        """Return whether every requested capability is locally supported."""

        return self.supported

    @property
    def supported_tasks(self) -> tuple[str, ...]:
        """Return the recognized tasks declared by the package."""

        return self.tasks

    @property
    def context_limit(self) -> ContextLimits:
        """Return the normalized context-budget metadata."""

        return self.context_limits

    @property
    def required_runtime_features(self) -> tuple[str, ...]:
        """Return the normalized runtime features required by the package."""

        return self.runtime_features.required

    @property
    def available_runtime_features(self) -> tuple[str, ...]:
        """Return the runtime features available to this probe."""

        return self.runtime_features.available

    @property
    def fingerprint(self) -> str:
        """Return the deterministic capability metadata fingerprint."""

        return self.manifest_fingerprint

    @property
    def manifest_digest(self) -> str:
        """Alias for :attr:`manifest_fingerprint`."""

        return self.manifest_fingerprint

    @property
    def decision(self) -> str:
        """Return the stable report decision."""

        return "supported" if self.supported else "unsupported"

    @property
    def unsupported_reasons(self) -> tuple[UnsupportedCapabilityReason, ...]:
        """Return all unsupported reasons in stable capability/code order."""

        return tuple(
            UnsupportedCapabilityReason(check.name, code)
            for check in self._checks
            for code in check.reason_codes
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return unique reason codes in stable order."""

        return _ordered_reasons(reason.code for reason in self.unsupported_reasons)

    @property
    def unsupported_reason_codes(self) -> tuple[str, ...]:
        """Alias for :attr:`reason_codes`."""

        return self.reason_codes

    @property
    def network_required(self) -> bool:
        """Return whether this probe requires network access (always false)."""

        return False

    @property
    def cloud_fallback_allowed(self) -> bool:
        """Return whether a cloud fallback is permitted (always false)."""

        return False

    def capability(self, name: str) -> ClinicalSLMCapabilityCheck:
        """Return one check without echoing an unknown caller value."""

        canonical = _normalise_capability(name)
        if canonical is None or canonical not in self.capabilities:
            _fail("unknown_capability")
        return self.capabilities[canonical]

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate-only report payload."""

        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "supported": self.supported,
            "requested_capabilities": [check.name for check in self._checks],
            "capabilities": {check.name: check.to_dict() for check in self._checks},
            "unsupported_reasons": [
                reason.to_dict() for reason in self.unsupported_reasons
            ],
            "reason_codes": list(self.reason_codes),
            "inspected": {
                "supported_tasks": list(self.tasks),
                "unknown_task_count": self.unknown_task_count,
                "tasks_declared": self.tasks_declared,
                "context_limits": self.context_limits.to_dict(),
                "quantization": self.quantization.to_dict(),
                "runtime_features": self.runtime_features.to_dict(),
                "offline": self.offline,
                "human_review_required": self.human_review_required,
                "manifest_fingerprint": self.manifest_fingerprint,
            },
            "network": {
                "mandatory": False,
                "cloud_fallback": "disabled",
            },
        }

    as_dict = to_dict

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON without paths or raw manifest values."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def __getitem__(self, key: str) -> Any:
        """Allow mapping-style access to the serialized report."""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over stable report field names."""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of serialized report fields."""

        return len(self.to_dict())

    def __bool__(self) -> bool:
        """Use the report as its support decision in boolean contexts."""

        return self.supported

    def __repr__(self) -> str:
        """Return a content-free diagnostic representation."""

        return "ClinicalSLMCapabilityReport(<aggregate>)"


CapabilityReport = ClinicalSLMCapabilityReport


def _normalise_requested_capabilities(
    required_capabilities: Iterable[str] | None,
    required_tasks: Iterable[str] | None,
) -> tuple[str, ...]:
    if required_capabilities is not None and required_tasks is not None:
        _fail("invalid_requirement")
    requested = required_capabilities
    if requested is None:
        requested = required_tasks
    if requested is None:
        return SUPPORTED_CAPABILITIES
    if isinstance(requested, str):
        values: Iterable[Any] = (requested,)
    else:
        try:
            values = tuple(requested)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_requirement")
    result: set[str] = set()
    for value in values:
        capability = _normalise_capability(value)
        if capability is None:
            _fail("unknown_capability")
        result.add(capability)
    if not result:
        _fail("invalid_requirement")
    return tuple(
        capability for capability in SUPPORTED_CAPABILITIES if capability in result
    )


def _normalise_min_context(
    min_context_tokens: int | None,
    required_context_tokens: int | None,
) -> int | None:
    if min_context_tokens is not None and required_context_tokens is not None:
        _fail("invalid_requirement")
    value = (
        min_context_tokens
        if min_context_tokens is not None
        else required_context_tokens
    )
    if value is None:
        return None
    if _positive_int(value) is None:
        _fail("invalid_requirement")
    return value


def _normalise_allowed_quantizations(
    allowed_quantizations: Iterable[str] | None,
) -> frozenset[str] | None:
    if allowed_quantizations is None:
        return None
    if isinstance(allowed_quantizations, str):
        values: Iterable[Any] = (allowed_quantizations,)
    else:
        try:
            values = tuple(allowed_quantizations)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_requirement")
    result: set[str] = set()
    for value in values:
        token = _normalise_token(value)
        if token is None:
            _fail("invalid_requirement")
        token = _QUANTIZATION_ALIASES.get(token, token)
        if token not in _KNOWN_QUANTIZATION_SCHEMES:
            _fail("invalid_requirement")
        result.add(token)
    if not result:
        _fail("invalid_requirement")
    return frozenset(result)


def probe_clinical_slm_capabilities(
    manifest: Mapping[str, Any] | str | Path | bytes | bytearray,
    *,
    required_capabilities: Iterable[str] | None = None,
    required_tasks: Iterable[str] | None = None,
    available_runtime_features: Iterable[str] | Mapping[str, bool] | None = None,
    runtime_features: Iterable[str] | Mapping[str, bool] | None = None,
    min_context_tokens: int | None = None,
    required_context_tokens: int | None = None,
    allowed_quantizations: Iterable[str] | None = None,
    require_offline: bool = True,
    require_human_review: bool = True,
) -> ClinicalSLMCapabilityReport:
    """Probe local clinical SLM capabilities from metadata only.

    Args:
        manifest: JSON-like package metadata, a JSON document, or a local path
            to one of the fixed manifest filenames.
        required_capabilities: Capability names to check.  The default checks
            both ``bounded_summarization`` and ``nli``.
        required_tasks: Alias for ``required_capabilities`` for callers that
            express the deployment requirement as task names.
        available_runtime_features: Explicit local runtime feature profile.
            When omitted, known Python-backed features are checked with
            :func:`importlib.util.find_spec` without importing them.
        runtime_features: Alias for ``available_runtime_features``.
        min_context_tokens: Optional lower bound for each requested input.
        required_context_tokens: Alias for ``min_context_tokens``.
        allowed_quantizations: Optional allow-list for the declared package
            quantization scheme.
        require_offline: Require an explicit true offline/local-only flag.
        require_human_review: Require an explicit true human-review flag.

    Returns:
        A deterministic :class:`ClinicalSLMCapabilityReport`.  Unsupported
        capabilities carry fixed machine-readable reason codes; no raw
        manifest values are copied into the report.

    Raises:
        ClinicalSLMCapabilityError: If the source or probe requirements are
            malformed.  Its message is always content-free.
    """

    if runtime_features is not None and available_runtime_features is not None:
        _fail("invalid_requirement")
    selected_runtime_features = (
        runtime_features if runtime_features is not None else available_runtime_features
    )
    if type(require_offline) is not bool or type(require_human_review) is not bool:
        _fail("invalid_requirement")
    requested = _normalise_requested_capabilities(
        required_capabilities,
        required_tasks,
    )
    minimum_context = _normalise_min_context(
        min_context_tokens,
        required_context_tokens,
    )
    allowed_quantization = _normalise_allowed_quantizations(allowed_quantizations)
    payload = load_clinical_slm_capability_manifest(manifest)
    normalised = _normalise_manifest(payload, selected_runtime_features)

    global_reasons: list[str] = []
    if require_offline:
        if normalised.offline_invalid:
            global_reasons.append("offline_metadata_invalid")
        elif not normalised.offline:
            global_reasons.append("offline_required")
    if require_human_review:
        if normalised.human_review_invalid:
            global_reasons.append("human_review_metadata_invalid")
        elif not normalised.human_review_required:
            global_reasons.append("human_review_required")
    if normalised.cloud_fallback_forbidden:
        global_reasons.append("cloud_fallback_forbidden")
    if normalised.network_required:
        global_reasons.append("network_required")
    global_reasons.extend(normalised.quantization_reasons)
    global_reasons.extend(normalised.context_reasons)
    global_reasons.extend(normalised.runtime_reasons)

    checks: list[ClinicalSLMCapabilityCheck] = []
    for capability in requested:
        reasons = list(global_reasons)
        if capability not in normalised.tasks:
            reasons.append("task_not_declared")
        effective_input = (
            normalised.context_limits.max_input_tokens
            or normalised.context_limits.max_context_tokens
        )
        if effective_input is None:
            reasons.append("context_limit_missing")
        if capability == BOUNDED_SUMMARIZATION:
            if normalised.context_limits.max_output_tokens is None:
                reasons.append("output_limit_missing")
        if minimum_context is not None and (
            effective_input is None or effective_input < minimum_context
        ):
            reasons.append("context_limit_too_small")
        if (
            allowed_quantization is not None
            and normalised.quantization.scheme is not None
            and normalised.quantization.scheme not in allowed_quantization
        ):
            reasons.append("quantization_unsupported")
        ordered = _ordered_reasons(reasons)
        checks.append(
            ClinicalSLMCapabilityCheck(
                name=capability,
                supported=not ordered,
                reason_codes=ordered,
            )
        )

    return ClinicalSLMCapabilityReport(
        _checks=tuple(checks),
        tasks=normalised.tasks,
        unknown_task_count=normalised.unknown_task_count,
        tasks_declared=normalised.tasks_declared,
        context_limits=normalised.context_limits,
        quantization=normalised.quantization,
        runtime_features=normalised.runtime,
        offline=normalised.offline,
        human_review_required=normalised.human_review_required,
        manifest_fingerprint=normalised.manifest_fingerprint,
    )


def probe_clinical_slm_package(
    *args: Any, **kwargs: Any
) -> ClinicalSLMCapabilityReport:
    """Compatibility name for :func:`probe_clinical_slm_capabilities`."""

    return probe_clinical_slm_capabilities(*args, **kwargs)


inspect_clinical_slm_capabilities = probe_clinical_slm_capabilities
check_clinical_slm_capabilities = probe_clinical_slm_capabilities
probe = probe_clinical_slm_capabilities


__all__ = [
    "BOUNDED_SUMMARIZATION",
    "CAPABILITY_SCHEMA_VERSION",
    "CapabilityCheck",
    "CapabilityReason",
    "CapabilityReport",
    "ClinicalSLMCapabilityCheck",
    "ClinicalSLMCapabilityError",
    "ClinicalSLMCapabilityReport",
    "CLINICAL_SLM_CAPABILITIES_SCHEMA_VERSION",
    "ContextLimits",
    "DEFAULT_CAPABILITIES",
    "MANIFEST_FILENAMES",
    "NLI",
    "QuantizationMetadata",
    "RuntimeFeatureMetadata",
    "SCHEMA_VERSION",
    "SUPPORTED_CAPABILITIES",
    "UnsupportedCapabilityReason",
    "check_clinical_slm_capabilities",
    "inspect_clinical_slm_capabilities",
    "load_capability_manifest",
    "load_clinical_slm_capability_manifest",
    "load_clinical_slm_manifest",
    "probe",
    "probe_clinical_slm_capabilities",
    "probe_clinical_slm_package",
]
