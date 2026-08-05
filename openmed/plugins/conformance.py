"""Offline conformance checks for third-party OpenMed plugin packages.

The kit validates static component metadata and exercises public component
methods only with deterministic synthetic values. It does not enumerate entry
points, install packages, open sockets, or persist source text.
"""

from __future__ import annotations

import argparse
import importlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from inspect import getattr_static
from typing import Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

REASON_COMPONENT_CONTRACT = "component_contract"
REASON_DUPLICATE_COMPONENT = "duplicate_component"
REASON_INVALID_LABEL = "invalid_label"
REASON_INVALID_METADATA = "invalid_metadata"
REASON_MISSING_LABELS = "missing_labels"
REASON_NETWORK_EGRESS = "network_egress_not_local_first"
REASON_NON_PERMISSIVE_LICENSE = "non_permissive_license"
REASON_PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
REASON_RUNTIME_CONTRACT = "runtime_contract"
REASON_UNKNOWN_COMPONENT_KIND = "unknown_component_kind"

_FALLBACK_SDK_MAJOR = 1
_FALLBACK_COMPONENT_KINDS = frozenset(
    {
        "anonymizer_provider",
        "exporter",
        "interop_adapter",
        "language_pack",
        "recognizer",
    }
)
_FALLBACK_PERMISSIVE_LICENSES = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Unlicense",
        "Zlib",
    }
)
_COMPONENT_METHODS = {
    "anonymizer_provider": ("replacement_for",),
    "exporter": ("export",),
    "interop_adapter": ("to_openmed_spans", "from_openmed_spans"),
    "language_pack": ("language_code", "canonical_labels"),
    "recognizer": ("recognize",),
}
_LICENSE_EXPRESSION_RE = re.compile(r"[A-Za-z0-9.+()\-\s]+")
_LICENSE_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_PROBE_SURFACE = "OPENMED_SYNTHETIC_PERSON"
_PROBE_TEXT = f"conformance {_PROBE_SURFACE} fixture"
_PROBE_START = _PROBE_TEXT.index(_PROBE_SURFACE)
_PROBE_SPAN = OpenMedSpan(
    doc_id="openmed-plugin-conformance",
    start=_PROBE_START,
    end=_PROBE_START + len(_PROBE_SURFACE),
    text_hash=hmac_text_hash(_PROBE_SURFACE, "openmed-plugin-conformance"),
    entity_type="person",
    canonical_label="PERSON",
    score=1.0,
    detector="openmed-plugin-conformance",
    evidence={"source": "synthetic_fixture"},
)


@dataclass(frozen=True)
class PluginConformanceFailure:
    """One deterministic plugin conformance failure.

    Args:
        reason: Stable machine-readable failure reason.
        message: Safe explanation that never includes probe source text.
        component_id: Qualified component id when valid metadata exposed one.
    """

    reason: str
    message: str
    component_id: str = ""

    def format(self) -> str:
        """Return a concise human-readable failure line."""

        prefix = f"{self.component_id}: " if self.component_id else ""
        return f"{self.reason}: {prefix}{self.message}"


@dataclass(frozen=True)
class PluginConformanceReport:
    """Result of checking one plugin package's component objects."""

    components_checked: int
    failures: tuple[PluginConformanceFailure, ...]

    @property
    def passed(self) -> bool:
        """Return whether every supplied component conformed."""

        return not self.failures

    def require_pass(self) -> None:
        """Raise :class:`PluginConformanceError` when checks failed."""

        if not self.passed:
            raise PluginConformanceError(self)


class PluginConformanceError(AssertionError):
    """Raised by :func:`assert_plugin_conforms` for a failed report."""

    def __init__(self, report: PluginConformanceReport) -> None:
        self.report = report
        details = "\n".join(f"- {failure.format()}" for failure in report.failures)
        super().__init__(f"OpenMed plugin conformance failed:\n{details}")


@dataclass(frozen=True)
class _ComponentMetadata:
    plugin_id: str
    component_id: str
    kind: str
    sdk_version: str
    license: str
    network_egress: bool
    labels: tuple[str, ...]
    languages: tuple[str, ...]

    @property
    def qualified_id(self) -> str:
        return f"{self.plugin_id}:{self.component_id}"


def check_plugin_conformance(components: object) -> PluginConformanceReport:
    """Check plugin components using only synthetic, offline probes.

    Args:
        components: A component, component iterable, or zero-argument factory.

    Returns:
        A deterministic report containing safe, specific failure reasons.
    """

    failures: list[PluginConformanceFailure] = []
    component_values = _coerce_components(components, failures)
    if not component_values:
        if not failures:
            failures.append(
                PluginConformanceFailure(
                    REASON_INVALID_METADATA,
                    "plugin package must provide at least one component",
                )
            )
        return PluginConformanceReport(0, tuple(failures))

    seen_ids: set[str] = set()
    for component in component_values:
        metadata, metadata_failures = _validate_metadata(component)
        failures.extend(metadata_failures)
        if metadata is None:
            continue

        if metadata.qualified_id in seen_ids:
            failures.append(
                PluginConformanceFailure(
                    REASON_DUPLICATE_COMPONENT,
                    "qualified component id must be unique within the package",
                    metadata.qualified_id,
                )
            )
        else:
            seen_ids.add(metadata.qualified_id)

        failures.extend(_validate_component_contract(component, metadata))

    return PluginConformanceReport(len(component_values), tuple(failures))


def assert_plugin_conforms(components: object) -> PluginConformanceReport:
    """Require plugin components to pass the offline conformance kit.

    Args:
        components: A component, component iterable, or zero-argument factory.

    Returns:
        The passing conformance report.

    Raises:
        PluginConformanceError: If metadata or a component contract fails.
    """

    report = check_plugin_conformance(components)
    report.require_pass()
    return report


def _coerce_components(
    value: object,
    failures: list[PluginConformanceFailure],
) -> tuple[object, ...]:
    if _looks_like_component(value):
        return (value,)
    if callable(value):
        try:
            return _coerce_components(value(), failures)
        except Exception as exc:
            failures.append(
                PluginConformanceFailure(
                    REASON_COMPONENT_CONTRACT,
                    f"component factory raised {exc.__class__.__name__}",
                )
            )
            return ()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        components: list[object] = []
        try:
            for item in value:
                components.extend(_coerce_components(item, failures))
        except Exception as exc:
            failures.append(
                PluginConformanceFailure(
                    REASON_COMPONENT_CONTRACT,
                    f"component iterable raised {exc.__class__.__name__}",
                )
            )
        return tuple(components)
    return (value,)


def _looks_like_component(value: object) -> bool:
    if value is None:
        return False
    try:
        getattr_static(value, "metadata")
    except AttributeError:
        return False
    return True


def _validate_metadata(
    component: object,
) -> tuple[_ComponentMetadata | None, tuple[PluginConformanceFailure, ...]]:
    try:
        raw_metadata = getattr(component, "metadata")
    except Exception as exc:
        return None, (
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                f"component metadata raised {exc.__class__.__name__}",
            ),
        )
    if callable(raw_metadata):
        try:
            raw_metadata = raw_metadata()
        except Exception as exc:
            return None, (
                PluginConformanceFailure(
                    REASON_INVALID_METADATA,
                    f"metadata factory raised {exc.__class__.__name__}",
                ),
            )

    try:
        payload = _metadata_mapping(raw_metadata)
    except Exception as exc:
        return None, (
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                f"metadata fields raised {exc.__class__.__name__}",
            ),
        )
    if payload is None:
        return None, (
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                "plugin component must expose metadata as a mapping or SDK object",
            ),
        )

    field_failure = _metadata_field_failure(payload)
    if field_failure is not None:
        return None, (field_failure,)

    metadata = _ComponentMetadata(
        plugin_id=payload["plugin_id"].strip(),
        component_id=payload["component_id"].strip(),
        kind=payload["kind"].strip().lower(),
        sdk_version=payload["sdk_version"].strip(),
        license=payload["license"].strip(),
        network_egress=payload["network_egress"],
        labels=tuple(label.strip() for label in payload["labels"] if label.strip()),
        languages=tuple(
            language.strip().lower().replace("_", "-")
            for language in payload["languages"]
            if language.strip()
        ),
    )
    failures = _metadata_contract_failures(metadata)
    return metadata, failures


def _metadata_mapping(raw_metadata: object) -> dict[str, Any] | None:
    if isinstance(raw_metadata, Mapping):
        payload = dict(raw_metadata)
    else:
        fields = (
            "plugin_id",
            "component_id",
            "kind",
            "sdk_version",
            "license",
            "network_egress",
            "labels",
            "languages",
        )
        try:
            payload = {field: getattr(raw_metadata, field) for field in fields}
        except (AttributeError, TypeError):
            return None

    payload.setdefault("sdk_version", "1.0.0")
    payload.setdefault("license", "")
    payload.setdefault("network_egress", False)
    payload.setdefault("labels", ())
    payload.setdefault("languages", ("*",))
    return payload


def _metadata_field_failure(
    payload: Mapping[str, Any],
) -> PluginConformanceFailure | None:
    for field_name in (
        "plugin_id",
        "component_id",
        "kind",
        "sdk_version",
        "license",
    ):
        if not isinstance(payload.get(field_name), str):
            return PluginConformanceFailure(
                REASON_INVALID_METADATA,
                f"{field_name} must be a string",
            )
    if not isinstance(payload.get("network_egress"), bool):
        return PluginConformanceFailure(
            REASON_INVALID_METADATA,
            "network_egress must be a boolean",
        )
    for field_name in ("labels", "languages"):
        value = payload.get(field_name)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return PluginConformanceFailure(
                REASON_INVALID_METADATA,
                f"{field_name} must be a sequence of strings",
            )
        if any(not isinstance(item, str) for item in value):
            return PluginConformanceFailure(
                REASON_INVALID_METADATA,
                f"{field_name} must contain only strings",
            )
    return None


def _metadata_contract_failures(
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    failures: list[PluginConformanceFailure] = []
    component_id = metadata.qualified_id
    if not metadata.plugin_id:
        failures.append(
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                "plugin_id must be non-empty",
            )
        )
    if not metadata.component_id:
        failures.append(
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                "component_id must be non-empty",
            )
        )
    if ":" in metadata.plugin_id or ":" in metadata.component_id:
        failures.append(
            PluginConformanceFailure(
                REASON_INVALID_METADATA,
                "plugin_id and component_id must not contain ':'",
                component_id,
            )
        )

    sdk_major, component_kinds, permissive_licenses = _sdk_contract()
    if metadata.kind not in component_kinds:
        failures.append(
            PluginConformanceFailure(
                REASON_UNKNOWN_COMPONENT_KIND,
                "component kind must be one of " + ", ".join(sorted(component_kinds)),
                component_id,
            )
        )
    if _semver_major(metadata.sdk_version) != sdk_major:
        failures.append(
            PluginConformanceFailure(
                REASON_PROTOCOL_VERSION_MISMATCH,
                f"sdk_version must target supported major {sdk_major}",
                component_id,
            )
        )
    if metadata.kind == "recognizer" and not metadata.labels:
        failures.append(
            PluginConformanceFailure(
                REASON_MISSING_LABELS,
                "recognizer plugins must declare at least one canonical label",
                component_id,
            )
        )
    for label in metadata.labels:
        if label not in CANONICAL_LABELS:
            failures.append(
                PluginConformanceFailure(
                    REASON_INVALID_LABEL,
                    f"label {label!r} is not in the canonical OpenMed label schema",
                    component_id,
                )
            )
            break
    if metadata.network_egress:
        failures.append(
            PluginConformanceFailure(
                REASON_NETWORK_EGRESS,
                "self-certified plugins must declare network_egress=false",
                component_id,
            )
        )
    if not _is_permissive_license(metadata.license, permissive_licenses):
        failures.append(
            PluginConformanceFailure(
                REASON_NON_PERMISSIVE_LICENSE,
                "license must be a permissive SPDX expression",
                component_id,
            )
        )
    return tuple(failures)


def _validate_component_contract(
    component: object,
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    failures: list[PluginConformanceFailure] = []
    for method_name in _COMPONENT_METHODS.get(metadata.kind, ()):
        try:
            method = getattr(component, method_name, None)
        except Exception as exc:
            failures.append(
                PluginConformanceFailure(
                    REASON_COMPONENT_CONTRACT,
                    f"{method_name} attribute raised {exc.__class__.__name__}",
                    metadata.qualified_id,
                )
            )
            continue
        if not callable(method):
            failures.append(
                PluginConformanceFailure(
                    REASON_COMPONENT_CONTRACT,
                    f"{metadata.kind} component must define {method_name}()",
                    metadata.qualified_id,
                )
            )
    if failures:
        return tuple(failures)

    if metadata.kind == "recognizer":
        failures.extend(_probe_recognizer(component, metadata))
    elif metadata.kind == "exporter":
        failures.extend(_probe_exporter(component, metadata))
    elif metadata.kind == "anonymizer_provider":
        failures.extend(_probe_anonymizer(component, metadata))
    elif metadata.kind == "language_pack":
        failures.extend(_probe_language_pack(component, metadata))
    return tuple(failures)


def _probe_recognizer(
    component: object,
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    try:
        spans = component.recognize(_PROBE_TEXT)
    except Exception as exc:
        return (_runtime_exception(metadata, "recognize", exc),)
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        return (_runtime_failure(metadata, "recognize() must return a sequence"),)

    for span in spans:
        if not isinstance(span, OpenMedSpan):
            return (
                _runtime_failure(
                    metadata,
                    "recognize() must return only OpenMedSpan objects",
                ),
            )
        if span.start < 0 or span.end > len(_PROBE_TEXT):
            return (
                _runtime_failure(
                    metadata,
                    "recognize() returned offsets outside its source text",
                ),
            )
        if span.canonical_label not in metadata.labels:
            return (
                _runtime_failure(
                    metadata,
                    "recognize() returned a label absent from declared labels",
                ),
            )
        surface = _PROBE_TEXT[span.start : span.end]
        if surface and (
            _contains_surface(span.evidence, surface)
            or _contains_surface(span.metadata, surface)
        ):
            return (
                _runtime_failure(
                    metadata,
                    "recognize() copied source text into span metadata or evidence",
                ),
            )
    return ()


def _probe_exporter(
    component: object,
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    try:
        exported = component.export((_PROBE_SPAN,))
    except Exception as exc:
        return (_runtime_exception(metadata, "export", exc),)
    valid_output = isinstance(exported, (str, bytes, Mapping)) or (
        isinstance(exported, Sequence)
        and not isinstance(exported, (str, bytes))
        and all(isinstance(item, Mapping) for item in exported)
    )
    if not valid_output:
        return (
            _runtime_failure(
                metadata,
                "export() returned an unsupported output type",
            ),
        )
    if _contains_surface(exported, _PROBE_SURFACE):
        return (
            _runtime_failure(
                metadata,
                "export() included raw source text in its artifact",
            ),
        )
    return ()


def _probe_anonymizer(
    component: object,
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    try:
        replacement = component.replacement_for(_PROBE_SPAN, _PROBE_SURFACE)
    except Exception as exc:
        return (_runtime_exception(metadata, "replacement_for", exc),)
    if not isinstance(replacement, str) or not replacement:
        return (
            _runtime_failure(
                metadata,
                "replacement_for() must return a non-empty string",
            ),
        )
    if _PROBE_SURFACE in replacement:
        return (
            _runtime_failure(
                metadata,
                "replacement_for() must not preserve the source surface",
            ),
        )
    return ()


def _probe_language_pack(
    component: object,
    metadata: _ComponentMetadata,
) -> tuple[PluginConformanceFailure, ...]:
    try:
        language = component.language_code()
        labels = component.canonical_labels()
    except Exception as exc:
        return (_runtime_exception(metadata, "language pack probe", exc),)
    if not isinstance(language, str) or not language.strip():
        return (
            _runtime_failure(
                metadata,
                "language_code() must return a non-empty string",
            ),
        )
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        return (
            _runtime_failure(
                metadata,
                "canonical_labels() must return a sequence of strings",
            ),
        )
    if any(
        not isinstance(label, str) or label not in CANONICAL_LABELS for label in labels
    ):
        return (
            _runtime_failure(
                metadata,
                "canonical_labels() returned an unsupported label",
            ),
        )
    return ()


def _runtime_exception(
    metadata: _ComponentMetadata,
    method_name: str,
    exc: Exception,
) -> PluginConformanceFailure:
    return _runtime_failure(
        metadata,
        f"{method_name}() raised {exc.__class__.__name__} on the synthetic probe",
    )


def _runtime_failure(
    metadata: _ComponentMetadata,
    message: str,
) -> PluginConformanceFailure:
    return PluginConformanceFailure(
        REASON_RUNTIME_CONTRACT,
        message,
        metadata.qualified_id,
    )


def _contains_surface(value: object, surface: str) -> bool:
    if isinstance(value, str):
        return surface in value
    if isinstance(value, bytes):
        return surface.encode("utf-8") in value
    if isinstance(value, Mapping):
        return any(
            _contains_surface(key, surface) or _contains_surface(item, surface)
            for key, item in value.items()
        )
    if isinstance(value, Sequence):
        return any(_contains_surface(item, surface) for item in value)
    return False


def _sdk_contract() -> tuple[int, frozenset[str], frozenset[str]]:
    try:
        from .protocols import PLUGIN_COMPONENT_KINDS, PLUGIN_SDK_MAJOR
        from .registry import PERMISSIVE_LICENSES
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "openmed.plugins.protocols",
            "openmed.plugins.registry",
        }:
            raise
        return (
            _FALLBACK_SDK_MAJOR,
            _FALLBACK_COMPONENT_KINDS,
            _FALLBACK_PERMISSIVE_LICENSES,
        )
    return (
        PLUGIN_SDK_MAJOR,
        frozenset(PLUGIN_COMPONENT_KINDS),
        frozenset(PERMISSIVE_LICENSES),
    )


def _semver_major(version: str) -> int:
    match = _SEMVER_RE.fullmatch(version)
    return int(match.group(1)) if match is not None else -1


def _is_permissive_license(
    expression: str,
    permissive_licenses: frozenset[str],
) -> bool:
    if _LICENSE_EXPRESSION_RE.fullmatch(expression) is None:
        return False
    tokens = tuple(_LICENSE_TOKEN_RE.findall(expression))
    license_tokens = tuple(
        token for token in tokens if token.upper() not in {"AND", "OR", "WITH"}
    )
    return bool(license_tokens) and all(
        token in permissive_licenses for token in license_tokens
    )


def _load_target(target: str) -> object:
    module_name, separator, attribute = target.partition(":")
    attribute = attribute if separator else "plugin_components"
    if not module_name or not attribute:
        raise ValueError("target must use module[:attribute] syntax")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conformance kit for one importable plugin target.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero for conformance and one for a load or validation failure.
    """

    parser = argparse.ArgumentParser(
        description="Run offline OpenMed plugin conformance checks."
    )
    parser.add_argument(
        "target",
        help="Import target as module[:attribute] (default: plugin_components)",
    )
    args = parser.parse_args(argv)
    try:
        target = _load_target(args.target)
    except Exception as exc:
        print(f"FAIL: invalid_target: target raised {exc.__class__.__name__}")
        return 1

    report = check_plugin_conformance(target)
    if report.passed:
        print(f"PASS: {report.components_checked} component(s) conform")
        return 0
    print(f"FAIL: {len(report.failures)} conformance error(s)")
    for failure in report.failures:
        print(f"- {failure.format()}")
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())


__all__ = [
    "REASON_COMPONENT_CONTRACT",
    "REASON_DUPLICATE_COMPONENT",
    "REASON_INVALID_LABEL",
    "REASON_INVALID_METADATA",
    "REASON_MISSING_LABELS",
    "REASON_NETWORK_EGRESS",
    "REASON_NON_PERMISSIVE_LICENSE",
    "REASON_PROTOCOL_VERSION_MISMATCH",
    "REASON_RUNTIME_CONTRACT",
    "REASON_UNKNOWN_COMPONENT_KIND",
    "PluginConformanceError",
    "PluginConformanceFailure",
    "PluginConformanceReport",
    "assert_plugin_conforms",
    "check_plugin_conformance",
    "main",
]
