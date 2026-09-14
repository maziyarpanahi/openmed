"""Deterministic, value-free prompt provenance for clinical SLM runs.

Clinical small-language-model runs have three independently changeable prompt
surfaces: the system instruction, the task instruction, and the output-format
contract.  :class:`ClinicalSLMTemplateSet` canonicalizes those surfaces and
binds each one to a SHA-256 digest before a caller renders runtime values.

The template text is retained only by the in-memory object that performs a
render.  Its provenance and representations contain digests, placeholder
names, and schema metadata; they never contain template text or substitution
values.  The module uses only the Python standard library and never performs
network I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, NoReturn

__all__ = [
    "CLINICAL_SLM_TEMPLATE_DIGEST_SCHEMA_VERSION",
    "CLINICAL_SLM_TEMPLATE_SCHEMA_VERSION",
    "CLINICAL_SLM_TEMPLATES_SCHEMA_VERSION",
    "ClinicalSLMRenderedTemplates",
    "ClinicalSLMTemplateDigestMismatchError",
    "ClinicalSLMTemplateError",
    "ClinicalSLMTemplateProvenance",
    "ClinicalSLMTemplateSet",
    "ClinicalSLMTemplateSubstitutionError",
    "ClinicalSLMTemplateValidationError",
    "MAX_TEMPLATE_BYTES",
    "PromptTemplateDigest",
    "PromptTemplateManifest",
    "PromptTemplateSet",
    "RenderedClinicalSLMTemplates",
    "SCHEMA_VERSION",
    "TEMPLATE_DIGEST_SCHEMA_VERSION",
    "TEMPLATE_NAMES",
    "TemplateDigest",
    "TemplateManifest",
    "TemplateSet",
    "UndeclaredTemplateSubstitutionError",
    "build_clinical_slm_template_set",
    "build_template_provenance",
    "build_template_set",
    "canonicalize_template",
    "compute_template_digest",
    "digest_template",
    "digest_templates",
    "render_clinical_slm_templates",
    "render_template_provenance",
    "render_templates",
    "validate_runtime_substitutions",
    "verify_runtime_substitutions",
    "verify_template_digests",
    "verify_template_provenance",
]


TEMPLATE_DIGEST_SCHEMA_VERSION: Final = "openmed.clinical-slm-template-digests.v1"
CLINICAL_SLM_TEMPLATE_DIGEST_SCHEMA_VERSION: Final = TEMPLATE_DIGEST_SCHEMA_VERSION
CLINICAL_SLM_TEMPLATE_SCHEMA_VERSION: Final = TEMPLATE_DIGEST_SCHEMA_VERSION
CLINICAL_SLM_TEMPLATES_SCHEMA_VERSION: Final = TEMPLATE_DIGEST_SCHEMA_VERSION
SCHEMA_VERSION: Final = TEMPLATE_DIGEST_SCHEMA_VERSION

TEMPLATE_NAMES: Final[tuple[str, ...]] = ("system", "task", "output_format")
MAX_TEMPLATE_BYTES: Final = 1 << 20
MAX_PLACEHOLDER_LENGTH: Final = 64

_TEMPLATE_NAME_ALIASES: Final[dict[str, str]] = {
    "system": "system",
    "system_prompt": "system",
    "system_template": "system",
    "task": "task",
    "task_prompt": "task",
    "task_template": "task",
    "output": "output_format",
    "output_format": "output_format",
    "output_format_template": "output_format",
    "output_template": "output_format",
    "format_template": "output_format",
}
_PLACEHOLDER_RE: Final = re.compile(
    rf"^[A-Za-z_][A-Za-z0-9_]{{0,{MAX_PLACEHOLDER_LENGTH - 1}}}$"
)
_DIGEST_RE: Final = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_ERROR_MESSAGES: Final[dict[str, str]] = {
    "invalid_input": "clinical SLM template input is invalid",
    "missing_template": "clinical SLM template is missing",
    "ambiguous_template": "clinical SLM template has conflicting aliases",
    "invalid_template": "clinical SLM template content is invalid",
    "invalid_placeholder": "clinical SLM template placeholder is invalid",
    "declaration_mismatch": (
        "clinical SLM template substitutions do not match declared placeholders"
    ),
    "invalid_substitution": "clinical SLM runtime substitution is invalid",
    "missing_substitution": "clinical SLM runtime substitution is missing",
    "undeclared_substitution": (
        "clinical SLM runtime substitution is not declared by the templates"
    ),
    "invalid_provenance": "clinical SLM template provenance is invalid",
    "template_digest_mismatch": "clinical SLM template digest does not match",
    "invalid_format": "clinical SLM template report format is invalid",
}
_SAFE_FIELDS: Final[frozenset[str]] = frozenset((*TEMPLATE_NAMES, "provenance"))

_MISSING = object()


class ClinicalSLMTemplateError(ValueError):
    """Base error with a stable, value-free reason code.

    Caller-provided template contents, paths, substitutions, and digest
    values are intentionally excluded from the exception message.
    """

    def __init__(self, reason_code: str, field_name: str | None = None) -> None:
        self.reason_code = (
            reason_code if reason_code in _ERROR_MESSAGES else "invalid_input"
        )
        self.code = self.reason_code
        self.field_name = field_name if field_name in _SAFE_FIELDS else None
        message = _ERROR_MESSAGES[self.reason_code]
        if self.field_name is not None:
            message = f"{message} ({self.field_name})"
        super().__init__(message)

    def to_dict(self) -> dict[str, str]:
        """Return a machine-readable error without caller values."""

        payload = {"code": self.reason_code, "message": str(self)}
        if self.field_name is not None:
            payload["field"] = self.field_name
        return payload


class ClinicalSLMTemplateValidationError(ClinicalSLMTemplateError):
    """Raised when a template or provenance record is malformed."""


class ClinicalSLMTemplateSubstitutionError(ClinicalSLMTemplateError):
    """Raised when runtime values do not satisfy the template contract."""


class UndeclaredTemplateSubstitutionError(ClinicalSLMTemplateSubstitutionError):
    """Raised when a caller supplies a runtime variable not in the contract."""

    def __init__(self, field_name: str | None = None) -> None:
        super().__init__("undeclared_substitution", field_name)


class ClinicalSLMTemplateDigestMismatchError(ClinicalSLMTemplateError):
    """Raised when a run's declared template provenance is stale or altered."""

    def __init__(self) -> None:
        super().__init__("template_digest_mismatch")


def _fail(
    reason_code: str,
    field_name: str | None = None,
    *,
    error_type: type[ClinicalSLMTemplateError] = ClinicalSLMTemplateValidationError,
) -> NoReturn:
    """Raise a static error without retaining the rejected value."""

    raise error_type(reason_code, field_name) from None


def _canonical_json(value: Any) -> str:
    """Serialize internal metadata using one deterministic JSON contract."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("invalid_input")
    raise AssertionError("unreachable")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _normalize_digest(value: Any) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail("invalid_provenance", "provenance")
    normalized = value.lower()
    return normalized if normalized.startswith("sha256:") else f"sha256:{normalized}"


def _normalize_template_name(value: Any) -> str:
    if type(value) is not str:
        _fail("invalid_input")
    name = _TEMPLATE_NAME_ALIASES.get(value.strip().lower())
    if name is None:
        _fail("invalid_input")
    return name


def _normalize_placeholder(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or _PLACEHOLDER_RE.fullmatch(value) is None
        or len(value) > MAX_PLACEHOLDER_LENGTH
    ):
        _fail("invalid_placeholder", field_name)
    return value


def _normalize_template(value: Any, field_name: str) -> str:
    """Normalize one template while keeping its content out of errors."""

    if type(value) is not str:
        _fail("invalid_template", field_name)
    try:
        normalized = unicodedata.normalize("NFC", value)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        encoded = normalized.encode("utf-8")
    except (UnicodeError, ValueError):
        _fail("invalid_template", field_name)
    if (
        not normalized.strip()
        or len(encoded) > MAX_TEMPLATE_BYTES
        or _CONTROL_RE.search(normalized) is not None
    ):
        _fail("invalid_template", field_name)
    return normalized


def canonicalize_template(template: str) -> str:
    """Return the canonical form used for a template digest.

    Unicode is normalized to NFC and CRLF/CR line endings become LF.  Other
    meaningful whitespace is preserved because it can affect a model prompt.
    """

    return _normalize_template(template, "template")


def _placeholders(template: str, field_name: str) -> tuple[str, ...]:
    try:
        parsed = tuple(string.Formatter().parse(template))
    except (TypeError, ValueError, RecursionError):
        _fail("invalid_placeholder", field_name)

    names: set[str] = set()
    for _literal, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if (
            conversion is not None
            or format_spec
            or _PLACEHOLDER_RE.fullmatch(field) is None
        ):
            _fail("invalid_placeholder", field_name)
        names.add(_normalize_placeholder(field, field_name))
    return tuple(sorted(names))


def _mapping_copy(
    value: Any,
    *,
    reason_code: str,
    error_type: type[ClinicalSLMTemplateError] = ClinicalSLMTemplateValidationError,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        _fail(reason_code, error_type=error_type)
    try:
        copied = dict(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail(reason_code, error_type=error_type)
    if any(type(key) is not str for key in copied):
        _fail(reason_code, error_type=error_type)
    return copied


def _first_alias(
    payload: Mapping[str, Any],
    aliases: Sequence[str],
    field_name: str,
    *,
    required: bool = True,
) -> Any:
    present = tuple(alias for alias in aliases if alias in payload)
    if len(present) > 1:
        _fail("ambiguous_template", field_name)
    if present:
        return payload[present[0]]
    if required:
        _fail("missing_template", field_name)
    return None


def _iter_placeholder_names(value: Any, field_name: str) -> tuple[str, ...]:
    if type(value) is str:
        values: Iterable[Any] = (value,)
    elif isinstance(value, (str, bytes, bytearray)):
        _fail("invalid_input", field_name)
    else:
        try:
            values = tuple(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_input", field_name)
    names = {_normalize_placeholder(item, field_name) for item in values}
    return tuple(sorted(names))


def _normalize_declarations(
    declarations: Any,
    placeholders: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    if declarations is None:
        return MappingProxyType(dict(placeholders))

    if isinstance(declarations, Mapping):
        payload = _mapping_copy(declarations, reason_code="invalid_input")
        normalized: dict[str, tuple[str, ...]] = {name: () for name in TEMPLATE_NAMES}
        seen_names: set[str] = set()
        for raw_name, raw_values in payload.items():
            name = _normalize_template_name(raw_name)
            if name in seen_names:
                _fail("ambiguous_template", name)
            seen_names.add(name)
            normalized[name] = _iter_placeholder_names(raw_values, name)
    else:
        names = _iter_placeholder_names(declarations, "provenance")
        normalized = {name: names for name in TEMPLATE_NAMES}

    for name in TEMPLATE_NAMES:
        if normalized[name] != placeholders[name]:
            _fail("declaration_mismatch", name)
    return MappingProxyType(normalized)


def _normalize_substitution_mapping(
    substitutions: Any,
) -> dict[str, str]:
    if substitutions is None:
        return {}
    payload = _mapping_copy(
        substitutions,
        reason_code="invalid_substitution",
        error_type=ClinicalSLMTemplateSubstitutionError,
    )
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if _PLACEHOLDER_RE.fullmatch(key) is None or type(value) is not str:
            _fail(
                "invalid_substitution",
                error_type=ClinicalSLMTemplateSubstitutionError,
            )
        normalized[key] = value
    return normalized


@dataclass(frozen=True, slots=True, repr=False)
class TemplateDigest:
    """Digest metadata for one named prompt template."""

    name: str
    digest: str
    placeholders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _normalize_template_name(self.name)
        if type(self.digest) is not str or _DIGEST_RE.fullmatch(self.digest) is None:
            _fail("invalid_provenance", "provenance")
        placeholders = tuple(
            sorted({_normalize_placeholder(value, name) for value in self.placeholders})
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "digest", _normalize_digest(self.digest))
        object.__setattr__(self, "placeholders", placeholders)

    def to_dict(self) -> dict[str, Any]:
        """Return metadata without the canonical template text."""

        return {
            "digest": self.digest,
            "name": self.name,
            "placeholders": list(self.placeholders),
        }

    @property
    def sha256(self) -> str:
        """Return the digest using the model-artifact naming convention."""

        return self.digest

    def __repr__(self) -> str:
        return (
            f"TemplateDigest(name={self.name!r}, digest={self.digest!r}, "
            f"placeholders={self.placeholders!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMTemplateProvenance(Mapping[str, Any]):
    """Value-free provenance binding all templates used by one SLM run."""

    template_digests: Mapping[str, str]
    template_set_digest: str
    declared_substitutions: Mapping[str, tuple[str, ...]]
    schema_version: str = TEMPLATE_DIGEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TEMPLATE_DIGEST_SCHEMA_VERSION:
            _fail("invalid_provenance", "provenance")
        digest_payload = _mapping_copy(
            self.template_digests,
            reason_code="invalid_provenance",
        )
        if set(digest_payload) != set(TEMPLATE_NAMES):
            _fail("invalid_provenance", "provenance")
        digests = {
            name: _normalize_digest(digest_payload[name]) for name in TEMPLATE_NAMES
        }
        declaration_payload = _mapping_copy(
            self.declared_substitutions,
            reason_code="invalid_provenance",
        )
        if set(declaration_payload) - set(TEMPLATE_NAMES):
            _fail("invalid_provenance", "provenance")
        declarations = {
            name: _iter_placeholder_names(declaration_payload.get(name, ()), name)
            for name in TEMPLATE_NAMES
        }
        object.__setattr__(self, "template_digests", MappingProxyType(digests))
        object.__setattr__(
            self,
            "declared_substitutions",
            MappingProxyType(declarations),
        )
        object.__setattr__(
            self,
            "template_set_digest",
            _normalize_digest(self.template_set_digest),
        )

    @property
    def system_digest(self) -> str:
        """Return the system-template digest."""

        return self.template_digests["system"]

    @property
    def task_digest(self) -> str:
        """Return the task-template digest."""

        return self.template_digests["task"]

    @property
    def output_format_digest(self) -> str:
        """Return the output-format-template digest."""

        return self.template_digests["output_format"]

    @property
    def digest(self) -> str:
        """Return the aggregate digest for the complete template contract."""

        return self.template_set_digest

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic provenance containing no prompt values."""

        return {
            "declared_substitutions": {
                name: list(self.declared_substitutions[name]) for name in TEMPLATE_NAMES
            },
            "schema_version": self.schema_version,
            "template_digests": {
                name: self.template_digests[name] for name in TEMPLATE_NAMES
            },
            "template_set_digest": self.template_set_digest,
        }

    def to_json(self) -> str:
        """Return compact, byte-stable JSON provenance."""

        return _canonical_json(self.to_dict())

    def to_markdown(self) -> str:
        """Return a value-free human-readable provenance report."""

        lines = [
            "# Clinical SLM Template Provenance",
            "",
            "Template text and runtime substitution values are omitted.",
            "",
            f"Schema: `{self.schema_version}`",
            f"Template-set digest: `{self.template_set_digest}`",
            "",
            "| Template | Digest | Declared substitutions |",
            "| --- | --- | --- |",
        ]
        for name in TEMPLATE_NAMES:
            declared = ", ".join(self.declared_substitutions[name]) or "none"
            lines.append(f"| `{name}` | `{self.template_digests[name]}` | {declared} |")
        return "\n".join(lines) + "\n"

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __repr__(self) -> str:
        return (
            "ClinicalSLMTemplateProvenance("
            f"template_set_digest={self.template_set_digest!r})"
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "ClinicalSLMTemplateProvenance":
        """Parse a previously emitted value-free provenance mapping."""

        copied = _mapping_copy(payload, reason_code="invalid_provenance")
        nested = copied.get("template_digests", _MISSING)
        if nested is _MISSING:
            nested = {
                "system": copied.get("system_template_digest", _MISSING),
                "task": copied.get("task_template_digest", _MISSING),
                "output_format": copied.get("output_format_template_digest", _MISSING),
            }
        digests = _mapping_copy(nested, reason_code="invalid_provenance")
        if any(digests.get(name, _MISSING) is _MISSING for name in TEMPLATE_NAMES):
            _fail("invalid_provenance", "provenance")
        return cls(
            template_digests=digests,
            template_set_digest=copied.get("template_set_digest", _MISSING),
            declared_substitutions=copied.get("declared_substitutions", {}),
            schema_version=copied.get("schema_version", TEMPLATE_DIGEST_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ClinicalSLMTemplateSet:
    """Canonical system, task, and output-format templates for one run.

    Placeholders must be simple names such as ``{clinical_note}``.  Runtime
    substitutions must exactly match the names present in the templates;
    extra values are rejected before any prompt is rendered.
    """

    system: str = field(repr=False)
    task: str = field(repr=False)
    output_format: str = field(repr=False)
    declared_substitutions: Mapping[str, tuple[str, ...]] = field(repr=False)
    _placeholders: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _digests: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _template_set_digest: str = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        system: str | None = None,
        task: str | None = None,
        output_format: str | None = None,
        *,
        system_template: str | None = None,
        task_template: str | None = None,
        output_format_template: str | None = None,
        format_template: str | None = None,
        output_template: str | None = None,
        declared_substitutions: Any = None,
        allowed_substitutions: Any = None,
        declarations: Any = None,
    ) -> None:
        system_value = _select_alias(
            system,
            (system_template,),
            "system",
        )
        task_value = _select_alias(task, (task_template,), "task")
        output_value = _select_alias(
            output_format,
            (output_format_template, format_template, output_template),
            "output_format",
        )
        declaration_value = _select_alias(
            declared_substitutions,
            (allowed_substitutions, declarations),
            "provenance",
            missing_is_none=True,
        )

        canonical = {
            "system": _normalize_template(system_value, "system"),
            "task": _normalize_template(task_value, "task"),
            "output_format": _normalize_template(output_value, "output_format"),
        }
        placeholders = MappingProxyType(
            {name: _placeholders(canonical[name], name) for name in TEMPLATE_NAMES}
        )
        normalized_declarations = _normalize_declarations(
            declaration_value,
            placeholders,
        )
        digests = MappingProxyType(
            {name: compute_template_digest(canonical[name]) for name in TEMPLATE_NAMES}
        )
        digest_payload = {
            "declared_substitutions": {
                name: list(normalized_declarations[name]) for name in TEMPLATE_NAMES
            },
            "schema_version": TEMPLATE_DIGEST_SCHEMA_VERSION,
            "templates": {
                name: {
                    "digest": digests[name],
                    "placeholders": list(placeholders[name]),
                }
                for name in TEMPLATE_NAMES
            },
        }
        template_set_digest = _sha256(_canonical_json(digest_payload).encode("utf-8"))

        object.__setattr__(self, "system", canonical["system"])
        object.__setattr__(self, "task", canonical["task"])
        object.__setattr__(self, "output_format", canonical["output_format"])
        object.__setattr__(self, "declared_substitutions", normalized_declarations)
        object.__setattr__(self, "_placeholders", placeholders)
        object.__setattr__(self, "_digests", digests)
        object.__setattr__(self, "_template_set_digest", template_set_digest)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClinicalSLMTemplateSet":
        """Build a template set from JSON-like template metadata."""

        copied = _mapping_copy(payload, reason_code="invalid_input")
        return cls(
            system=_first_alias(
                copied,
                ("system", "system_template", "system_prompt"),
                "system",
            ),
            task=_first_alias(
                copied,
                ("task", "task_template", "task_prompt"),
                "task",
            ),
            output_format=_first_alias(
                copied,
                (
                    "output_format",
                    "output_format_template",
                    "output_template",
                    "format_template",
                ),
                "output_format",
            ),
            declared_substitutions=_first_alias(
                copied,
                ("declared_substitutions", "allowed_substitutions", "declarations"),
                "provenance",
                required=False,
            ),
        )

    @property
    def system_template(self) -> str:
        """Return the canonical system template for local rendering."""

        return self.system

    @property
    def task_template(self) -> str:
        """Return the canonical task template for local rendering."""

        return self.task

    @property
    def output_format_template(self) -> str:
        """Return the canonical output-format template for local rendering."""

        return self.output_format

    @property
    def placeholders(self) -> Mapping[str, tuple[str, ...]]:
        """Return placeholder names without returning template contents."""

        return self._placeholders

    @property
    def digests(self) -> Mapping[str, str]:
        """Return the three individual prompt-template digests."""

        return self._digests

    @property
    def template_digests(self) -> Mapping[str, str]:
        """Alias for :attr:`digests`."""

        return self._digests

    @property
    def system_digest(self) -> str:
        """Return the system-template digest."""

        return self._digests["system"]

    @property
    def task_digest(self) -> str:
        """Return the task-template digest."""

        return self._digests["task"]

    @property
    def output_format_digest(self) -> str:
        """Return the output-format-template digest."""

        return self._digests["output_format"]

    @property
    def template_set_digest(self) -> str:
        """Return the aggregate digest for the complete prompt contract."""

        return self._template_set_digest

    @property
    def digest(self) -> str:
        """Return the aggregate template-set digest."""

        return self._template_set_digest

    @property
    def declared_variables(self) -> tuple[str, ...]:
        """Return all permitted runtime substitution names in stable order."""

        return tuple(
            sorted(
                {
                    value
                    for name in TEMPLATE_NAMES
                    for value in self.declared_substitutions[name]
                }
            )
        )

    @property
    def provenance(self) -> ClinicalSLMTemplateProvenance:
        """Return value-free provenance suitable for a run record."""

        return ClinicalSLMTemplateProvenance(
            template_digests=self._digests,
            template_set_digest=self._template_set_digest,
            declared_substitutions=self.declared_substitutions,
        )

    @property
    def provenance_report(self) -> dict[str, Any]:
        """Return a fresh, value-free provenance mapping."""

        return self.provenance.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Serialize only template provenance; raw prompt text is omitted."""

        return self.provenance.to_dict()

    def to_json(self) -> str:
        """Serialize only template provenance as deterministic JSON."""

        return self.provenance.to_json()

    def to_markdown(self) -> str:
        """Render only value-free template provenance."""

        return self.provenance.to_markdown()

    def validate_runtime_substitutions(
        self,
        substitutions: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Validate and copy runtime values without retaining them."""

        return verify_runtime_substitutions(self, substitutions)

    def render(
        self,
        substitutions: Mapping[str, str] | None = None,
        *,
        runtime_substitutions: Mapping[str, str] | None = None,
        **named_substitutions: str,
    ) -> "RenderedClinicalSLMTemplates":
        """Render all three templates after strict substitution validation."""

        return render_clinical_slm_templates(
            self,
            substitutions,
            runtime_substitutions=runtime_substitutions,
            **named_substitutions,
        )

    def __repr__(self) -> str:
        """Avoid exposing system, task, or output-format prompt text."""

        return (
            f"ClinicalSLMTemplateSet(template_set_digest={self._template_set_digest!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class RenderedClinicalSLMTemplates(Mapping[str, str]):
    """Rendered prompts paired with value-free template provenance.

    The three rendered strings are intentionally available for the local model
    call, but are excluded from ``repr`` and all report serializers.
    """

    system: str = field(repr=False)
    task: str = field(repr=False)
    output_format: str = field(repr=False)
    provenance: ClinicalSLMTemplateProvenance

    @property
    def system_prompt(self) -> str:
        """Return the rendered system prompt for the local model call."""

        return self.system

    @property
    def task_prompt(self) -> str:
        """Return the rendered task prompt for the local model call."""

        return self.task

    @property
    def output_format_prompt(self) -> str:
        """Return the rendered output-format prompt for the local model call."""

        return self.output_format

    @property
    def provenance_report(self) -> dict[str, Any]:
        """Return a value-free report for the corresponding run record."""

        return self.provenance.to_dict()

    def messages(self) -> tuple[dict[str, str], ...]:
        """Return local chat messages; callers must keep them out of reports."""

        return (
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.task},
            {"role": "system", "content": self.output_format},
        )

    def __getitem__(self, key: str) -> str:
        if key == "system":
            return self.system
        if key == "task":
            return self.task
        if key == "output_format":
            return self.output_format
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(TEMPLATE_NAMES)

    def __len__(self) -> int:
        return len(TEMPLATE_NAMES)

    def __repr__(self) -> str:
        return (
            "RenderedClinicalSLMTemplates("
            f"template_set_digest={self.provenance.template_set_digest!r})"
        )


def _select_alias(
    primary: Any,
    aliases: Sequence[Any],
    field_name: str,
    *,
    missing_is_none: bool = False,
) -> Any:
    values = [value for value in (primary, *aliases) if value is not None]
    if len(values) > 1:
        _fail("ambiguous_template", field_name)
    if values:
        return values[0]
    if missing_is_none:
        return None
    _fail("missing_template", field_name)


def _template_set_from_arguments(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any] | str | None,
    task: str | None,
    output_format: str | None,
    *,
    system: str | None,
    system_template: str | None,
    task_template: str | None,
    output_format_template: str | None,
    format_template: str | None,
    output_template: str | None,
    declared_substitutions: Any,
    allowed_substitutions: Any,
    declarations: Any,
) -> ClinicalSLMTemplateSet:
    aliases = (
        system,
        system_template,
        task_template,
        output_format_template,
        format_template,
        output_template,
        declared_substitutions,
        allowed_substitutions,
        declarations,
    )
    if (
        templates is not None
        and not isinstance(templates, str)
        and any(value is not None for value in aliases)
    ):
        _fail("ambiguous_template")

    if isinstance(templates, ClinicalSLMTemplateSet):
        if task is not None or output_format is not None:
            _fail("ambiguous_template")
        return templates
    if isinstance(templates, Mapping):
        if task is not None or output_format is not None:
            _fail("ambiguous_template")
        return ClinicalSLMTemplateSet.from_mapping(templates)

    primary_system = templates if isinstance(templates, str) else system
    if isinstance(templates, str) and system is not None:
        _fail("ambiguous_template", "system")
    if isinstance(templates, str) and system_template is not None:
        _fail("ambiguous_template", "system")
    return ClinicalSLMTemplateSet(
        system=primary_system,
        task=task if task is not None else task_template,
        output_format=output_format,
        system_template=system_template if primary_system is None else None,
        output_format_template=output_format_template,
        format_template=format_template,
        output_template=output_template,
        declared_substitutions=declared_substitutions,
        allowed_substitutions=allowed_substitutions,
        declarations=declarations,
    )


def build_template_set(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any] | str | None = None,
    task: str | None = None,
    output_format: str | None = None,
    *,
    system: str | None = None,
    system_template: str | None = None,
    task_template: str | None = None,
    output_format_template: str | None = None,
    format_template: str | None = None,
    output_template: str | None = None,
    declared_substitutions: Any = None,
    allowed_substitutions: Any = None,
    declarations: Any = None,
) -> ClinicalSLMTemplateSet:
    """Construct a validated template set from positional or named fields."""

    return _template_set_from_arguments(
        templates,
        task,
        output_format,
        system=system,
        system_template=system_template,
        task_template=task_template,
        output_format_template=output_format_template,
        format_template=format_template,
        output_template=output_template,
        declared_substitutions=declared_substitutions,
        allowed_substitutions=allowed_substitutions,
        declarations=declarations,
    )


build_clinical_slm_template_set = build_template_set


def build_template_provenance(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any] | str | None = None,
    task: str | None = None,
    output_format: str | None = None,
    *,
    system: str | None = None,
    system_template: str | None = None,
    task_template: str | None = None,
    output_format_template: str | None = None,
    format_template: str | None = None,
    output_template: str | None = None,
    declared_substitutions: Any = None,
    allowed_substitutions: Any = None,
    declarations: Any = None,
) -> ClinicalSLMTemplateProvenance:
    """Build deterministic, value-free provenance for three templates."""

    return build_template_set(
        templates,
        task,
        output_format,
        system=system,
        system_template=system_template,
        task_template=task_template,
        output_format_template=output_format_template,
        format_template=format_template,
        output_template=output_template,
        declared_substitutions=declared_substitutions,
        allowed_substitutions=allowed_substitutions,
        declarations=declarations,
    ).provenance


def compute_template_digest(template: str) -> str:
    """Return ``sha256:<hex>`` for a canonicalized template."""

    canonical = canonicalize_template(template)
    return _sha256(canonical.encode("utf-8"))


digest_template = compute_template_digest


def digest_templates(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any] | str | None = None,
    task: str | None = None,
    output_format: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Return the three named template digests without template contents."""

    return dict(
        build_template_provenance(
            templates,
            task,
            output_format,
            **kwargs,
        ).template_digests
    )


def verify_runtime_substitutions(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any],
    substitutions: Mapping[str, str] | None = None,
    *,
    runtime_substitutions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate runtime values against the exact declared placeholder set.

    Missing and extra names are both rejected.  The returned copy is intended
    only for the immediate local render and is not retained in provenance.
    """

    template_set = (
        templates
        if isinstance(templates, ClinicalSLMTemplateSet)
        else ClinicalSLMTemplateSet.from_mapping(templates)
    )
    if substitutions is not None and runtime_substitutions is not None:
        raise ClinicalSLMTemplateSubstitutionError("invalid_substitution") from None
    normalized = _normalize_substitution_mapping(
        runtime_substitutions if runtime_substitutions is not None else substitutions
    )
    expected = set(template_set.declared_variables)
    actual = set(normalized)
    if actual - expected:
        raise UndeclaredTemplateSubstitutionError() from None
    if expected - actual:
        raise ClinicalSLMTemplateSubstitutionError("missing_substitution") from None
    return normalized


validate_runtime_substitutions = verify_runtime_substitutions


def render_clinical_slm_templates(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any],
    substitutions: Mapping[str, str] | None = None,
    *,
    runtime_substitutions: Mapping[str, str] | None = None,
    **named_substitutions: str,
) -> RenderedClinicalSLMTemplates:
    """Render a template set only after rejecting undeclared substitutions."""

    template_set = (
        templates
        if isinstance(templates, ClinicalSLMTemplateSet)
        else ClinicalSLMTemplateSet.from_mapping(templates)
    )
    if runtime_substitutions is not None:
        if substitutions is not None or named_substitutions:
            _fail(
                "invalid_substitution",
                error_type=ClinicalSLMTemplateSubstitutionError,
            )
        substitutions = runtime_substitutions
    elif named_substitutions:
        if substitutions is not None:
            _fail(
                "invalid_substitution",
                error_type=ClinicalSLMTemplateSubstitutionError,
            )
        substitutions = named_substitutions
    values = verify_runtime_substitutions(template_set, substitutions)
    try:
        rendered = {
            name: getattr(template_set, name).format_map(values)
            for name in TEMPLATE_NAMES
        }
    except (KeyError, IndexError, ValueError, RecursionError):
        raise ClinicalSLMTemplateSubstitutionError("invalid_substitution") from None
    return RenderedClinicalSLMTemplates(
        system=rendered["system"],
        task=rendered["task"],
        output_format=rendered["output_format"],
        provenance=template_set.provenance,
    )


render_templates = render_clinical_slm_templates


def verify_template_provenance(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any],
    provenance: ClinicalSLMTemplateProvenance | Mapping[str, Any],
) -> ClinicalSLMTemplateProvenance:
    """Verify that recorded digests and declarations match current templates."""

    template_set = (
        templates
        if isinstance(templates, ClinicalSLMTemplateSet)
        else ClinicalSLMTemplateSet.from_mapping(templates)
    )
    if isinstance(provenance, ClinicalSLMTemplateProvenance):
        recorded = provenance
    else:
        source = provenance
        if isinstance(source, Mapping):
            nested = source.get("prompt_templates", source.get("template_provenance"))
            if isinstance(nested, Mapping):
                source = nested
        try:
            recorded = ClinicalSLMTemplateProvenance.from_mapping(source)
        except ClinicalSLMTemplateError:
            raise
        except Exception:
            _fail("invalid_provenance", "provenance")
    if recorded.to_dict() != template_set.provenance.to_dict():
        raise ClinicalSLMTemplateDigestMismatchError() from None
    return recorded


verify_template_digests = verify_template_provenance


def render_template_provenance(
    templates: ClinicalSLMTemplateSet | Mapping[str, Any],
    *,
    format: str = "json",
) -> str:
    """Render value-free template provenance as JSON or Markdown."""

    template_set = (
        templates
        if isinstance(templates, ClinicalSLMTemplateSet)
        else ClinicalSLMTemplateSet.from_mapping(templates)
    )
    if type(format) is not str:
        _fail("invalid_format")
    normalized = format.strip().lower()
    if normalized == "json":
        return template_set.provenance.to_json()
    if normalized in {"markdown", "md"}:
        return template_set.provenance.to_markdown()
    _fail("invalid_format")


# Compatibility names keep the contract discoverable to callers that use
# "prompt template" rather than "clinical SLM template" terminology.
PromptTemplateDigest = TemplateDigest
PromptTemplateManifest = ClinicalSLMTemplateProvenance
PromptTemplateSet = ClinicalSLMTemplateSet
TemplateManifest = ClinicalSLMTemplateProvenance
TemplateSet = ClinicalSLMTemplateSet
ClinicalSLMRenderedTemplates = RenderedClinicalSLMTemplates
