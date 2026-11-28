"""Declared OpenMed environment settings, loaded from ``settings.schema.json``.

The settings schema lists every environment variable OpenMed declares, grouped
into named sections.  Each declaration records the variable name, the JSON type
of its parsed value, its default, its bounds, whether it holds a secret, and
the module that owns it.  Entries in the ``core`` section reference the
matching ``OpenMedConfig`` property in ``config.schema.json`` for their type
and bounds.

This module only describes settings.  It never reads the process environment
and never returns a configured value, so its output is safe to render in
documentation, provenance reports and logs.  Secret settings declare no
default, and neither do user-supplied settings such as evaluation dataset
paths.  ``x-excluded-identifiers`` lists ``OPENMED_*`` identifiers that only
look like settings, such as FHIR system names and error codes, each with its
reason.  Existing loaders keep parsing their own variables; this registry does
not replace any of them.

Example:
    >>> from openmed.core.settings_registry import get_setting_declaration
    >>> get_setting_declaration("OPENMED_SERVICE_AUTH_API_KEYS").secret
    True
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import LegacyRootedError

SETTINGS_SCHEMA_VERSION = 1
SETTINGS_SCHEMA_FILENAME = "settings.schema.json"

SECTION_CORE = "core"
SECTION_SERVICE = "service"

#: ``x-env-format`` values: how the environment string encodes the value.
ENV_FORMATS = frozenset({"byte-size", "csv", "duration", "json", "path", "url"})

#: JSON Schema keywords reported as bounds of a declaration.
BOUND_KEYWORDS = (
    "enum",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
)

_JSON_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_CONFIG_SCHEMA_FILENAME = "config.schema.json"
_CONFIG_REF_PREFIX = f"{_CONFIG_SCHEMA_FILENAME}#/properties/"
_SCHEMA_VERSION_KEY = "schema_version"
_EXCLUDED_KEY = "x-excluded-identifiers"
_IDENTIFIER_RE = re.compile(r"OPENMED_[A-Z0-9_]*[A-Z0-9]")


class SettingsSchemaError(ValueError, LegacyRootedError):
    """Raised when the settings schema has a malformed declaration.

    Messages name the setting and the broken rule only; they never include a
    default or any other value from the document.
    """


@dataclass(frozen=True, slots=True)
class SettingDeclaration:
    """One declared environment setting.

    Attributes:
        name: Environment variable name, such as ``OPENMED_SERVICE_LOG_LEVEL``.
        section: Name of the schema section that declares the setting.
        description: One-line description of what the setting controls.
        types: JSON types of the parsed value, in schema order.
        module: Dotted name of the module that owns the setting.
        secret: Whether the setting holds a credential.  Secret settings never
            declare a default.
        has_default: Whether the schema declares a default.
        default: The declared default, frozen into tuples and read-only
            mappings; ``None`` when there is no default or the default is
            ``null``.  Check :attr:`has_default` to tell the two apart.
        bounds: Bound keywords such as ``minimum`` or ``enum``.
        env_format: How the environment string encodes the value (one of
            :data:`ENV_FORMATS`), or ``None`` for a plain scalar.
        default_note: Lower-case phrase that completes "Default: ..." for a
            default that depends on other settings or on the active profile.
        config_key: ``OpenMedConfig`` field the setting maps onto, if any.
        secret_file_for: For a ``NAME_FILE`` companion, the secret setting
            whose value the file holds.
        user_supplied: Whether the value points at data the user supplies,
            such as an evaluation dataset path.  User-supplied settings never
            declare a default, and OpenMed never bundles the data.
    """

    name: str
    section: str
    description: str
    types: tuple[str, ...]
    module: str
    secret: bool
    has_default: bool
    default: Any
    bounds: Mapping[str, Any]
    env_format: str | None = None
    default_note: str | None = None
    config_key: str | None = None
    secret_file_for: str | None = None
    user_supplied: bool = False

    @property
    def nullable(self) -> bool:
        """Return whether ``null`` (unset) is a valid parsed value."""

        return "null" in self.types

    def json_schema(self) -> dict[str, Any]:
        """Return a plain JSON Schema for the parsed value of this setting.

        Returns:
            A schema with the type and bounds only, suitable for validating a
            candidate value; it carries no default.
        """

        schema: dict[str, Any] = {
            "type": self.types[0] if len(self.types) == 1 else list(self.types)
        }
        for keyword, value in self.bounds.items():
            schema[keyword] = _thaw(value)
        return schema

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the declaration.

        Returns:
            The declaration fields.  A ``default`` key is present only when
            the schema declares one, which is never the case for a secret.
        """

        result: dict[str, Any] = {
            "name": self.name,
            "section": self.section,
            "description": self.description,
            "types": list(self.types),
            "module": self.module,
            "secret": self.secret,
            "has_default": self.has_default,
        }
        if self.has_default:
            result["default"] = _thaw(self.default)
        result["bounds"] = {key: _thaw(value) for key, value in self.bounds.items()}
        for key, value in (
            ("env_format", self.env_format),
            ("default_note", self.default_note),
            ("config_key", self.config_key),
            ("secret_file_for", self.secret_file_for),
        ):
            if value is not None:
                result[key] = value
        if self.user_supplied:
            result["user_supplied"] = True
        return result


@dataclass(frozen=True, slots=True)
class SettingsSection:
    """A named group of settings in the schema.

    Attributes:
        name: Section key, such as ``core`` or ``service``.
        title: Human-readable section title.
        description: What the section covers.
        settings: Declarations in the section, sorted by name.
    """

    name: str
    title: str
    description: str
    settings: tuple[SettingDeclaration, ...]

    def names(self) -> frozenset[str]:
        """Return the environment variable names declared in this section."""

        return frozenset(setting.name for setting in self.settings)


def settings_schema_path() -> Path:
    """Return the installed Draft 2020-12 settings schema path."""

    return Path(__file__).with_name(SETTINGS_SCHEMA_FILENAME)


def load_settings_schema() -> dict[str, Any]:
    """Return a fresh copy of the parsed settings schema document.

    Returns:
        The schema as a mutable dictionary owned by the caller.

    Raises:
        SettingsSchemaError: If the document repeats a key.
    """

    return copy.deepcopy(_cached_document())


def parse_settings_schema(
    document: Mapping[str, Any],
    *,
    config_schema: Mapping[str, Any] | None = None,
) -> tuple[SettingsSection, ...]:
    """Build section declarations from a settings schema document.

    Args:
        document: Parsed settings schema.
        config_schema: Parsed ``config.schema.json`` used to resolve ``core``
            references; defaults to the installed configuration schema.

    Returns:
        Sections in document order, each with declarations sorted by name.

    Raises:
        SettingsSchemaError: If a section or declaration is malformed, a name
            is declared twice or both declared and excluded, a secret or a
            user-supplied setting declares a default, or a secret-file
            companion names an undeclared secret.
    """

    if config_schema is None:
        config_schema = _cached_config_schema()
    if document.get(_SCHEMA_VERSION_KEY) != SETTINGS_SCHEMA_VERSION:
        raise SettingsSchemaError("settings schema has an unsupported version")
    properties = document.get("properties")
    if not isinstance(properties, Mapping):
        raise SettingsSchemaError("settings schema has no sections")

    sections: list[SettingsSection] = []
    seen: dict[str, str] = {}
    for section_name, section in properties.items():
        if section_name == _SCHEMA_VERSION_KEY:
            continue
        if not isinstance(section, Mapping) or not isinstance(
            section.get("properties"), Mapping
        ):
            raise SettingsSchemaError(f"section {section_name!r} has no settings")
        declarations = []
        for name, entry in sorted(section["properties"].items()):
            if name in seen:
                raise SettingsSchemaError(
                    f"{name} is declared in both {seen[name]!r} and {section_name!r}"
                )
            seen[name] = section_name
            declarations.append(_declaration(name, section_name, entry, config_schema))
        sections.append(
            SettingsSection(
                name=section_name,
                title=str(section.get("title") or section_name),
                description=str(section.get("description") or ""),
                settings=tuple(declarations),
            )
        )

    by_name = {item.name: item for part in sections for item in part.settings}
    for declaration in by_name.values():
        target = declaration.secret_file_for
        if target is None:
            continue
        if target not in by_name or not by_name[target].secret:
            raise SettingsSchemaError(
                f"{declaration.name} is a secret file for an undeclared secret"
            )
        if declaration.secret:
            raise SettingsSchemaError(
                f"{declaration.name} holds a path and must not be secret"
            )
    for name in _parse_excluded(document):
        if name in by_name:
            raise SettingsSchemaError(f"{name} is both declared and excluded")
    return tuple(sections)


def parse_excluded_identifiers(document: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the ``x-excluded-identifiers`` list of a settings schema.

    Args:
        document: Parsed settings schema.

    Returns:
        Read-only mapping of each excluded ``OPENMED_*`` identifier to the
        reason it is not a setting.

    Raises:
        SettingsSchemaError: If an entry is not an ``OPENMED_*`` name with a
            non-empty reason.
    """

    return MappingProxyType(_parse_excluded(document))


def excluded_identifiers() -> Mapping[str, str]:
    """Return ``OPENMED_*`` identifiers that only look like settings.

    These are constants such as FHIR system names or error codes, and names
    OpenMed exports to a subprocess without reading them.  Each carries the
    reason it is excluded.

    Returns:
        Read-only mapping of identifier to reason, sorted by identifier.
    """

    return _cached_excluded()


def settings_sections() -> tuple[SettingsSection, ...]:
    """Return every section of the installed settings schema.

    Returns:
        Sections in schema order; the result is cached and immutable.
    """

    return _cached_sections()


def setting_declarations(section: str | None = None) -> tuple[SettingDeclaration, ...]:
    """Return declarations, optionally limited to one section.

    Args:
        section: Section name; ``None`` returns every section.

    Returns:
        Declarations sorted by section order, then name.

    Raises:
        KeyError: If ``section`` is not a declared section.
    """

    sections = _cached_sections()
    if section is None:
        return tuple(item for part in sections for item in part.settings)
    for part in sections:
        if part.name == section:
            return part.settings
    raise KeyError(f"unknown settings section: {section!r}")


def get_setting_declaration(name: str) -> SettingDeclaration:
    """Return the declaration for one environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The matching declaration.

    Raises:
        KeyError: If the name is not declared.
    """

    try:
        return _cached_index()[name]
    except KeyError:
        raise KeyError(f"undeclared setting: {name!r}") from None


def declared_setting_names(section: str | None = None) -> frozenset[str]:
    """Return declared environment variable names.

    Args:
        section: Section name; ``None`` returns names from every section.

    Returns:
        The declared names.

    Raises:
        KeyError: If ``section`` is not a declared section.
    """

    return frozenset(item.name for item in setting_declarations(section))


def secret_setting_names() -> frozenset[str]:
    """Return the names of every setting marked ``x-secret``."""

    return frozenset(item.name for item in setting_declarations() if item.secret)


def _declaration(
    name: str,
    section: str,
    entry: Any,
    config_schema: Mapping[str, Any],
) -> SettingDeclaration:
    if not isinstance(entry, Mapping):
        raise SettingsSchemaError(f"{name} must be declared as an object")
    resolved = dict(_resolve_reference(name, entry, config_schema))
    resolved.update({key: value for key, value in entry.items() if key != "$ref"})

    description = resolved.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SettingsSchemaError(f"{name} has no description")
    raw_types = resolved.get("type")
    types = (raw_types,) if isinstance(raw_types, str) else tuple(raw_types or ())
    if not types or any(item not in _JSON_TYPES for item in types):
        raise SettingsSchemaError(f"{name} has an invalid JSON type")
    module = resolved.get("x-module")
    if not isinstance(module, str) or not module.startswith("openmed"):
        raise SettingsSchemaError(f"{name} has no owning module")
    secret = resolved.get("x-secret")
    if not isinstance(secret, bool):
        raise SettingsSchemaError(f"{name} must set x-secret to true or false")
    has_default = "default" in resolved
    if secret and has_default:
        raise SettingsSchemaError(f"{name} is secret and must not declare a default")
    env_format = _optional_text(name, resolved, "x-env-format")
    if env_format is not None and env_format not in ENV_FORMATS:
        raise SettingsSchemaError(f"{name} has an unknown x-env-format")
    user_supplied = resolved.get("x-user-supplied", False)
    if not isinstance(user_supplied, bool):
        raise SettingsSchemaError(f"{name} has an invalid x-user-supplied")
    if user_supplied and has_default:
        raise SettingsSchemaError(
            f"{name} is user-supplied and must not declare a default"
        )

    return SettingDeclaration(
        name=name,
        section=section,
        description=description.strip(),
        types=types,
        module=module,
        secret=secret,
        has_default=has_default,
        default=_freeze(resolved["default"]) if has_default else None,
        bounds=MappingProxyType(
            {
                keyword: _freeze(resolved[keyword])
                for keyword in BOUND_KEYWORDS
                if keyword in resolved
            }
        ),
        env_format=env_format,
        default_note=_optional_text(name, resolved, "x-default-note"),
        config_key=_optional_text(name, resolved, "x-config-key"),
        secret_file_for=_optional_text(name, resolved, "x-secret-file-for"),
        user_supplied=user_supplied,
    )


def _parse_excluded(document: Mapping[str, Any]) -> dict[str, str]:
    raw = document.get(_EXCLUDED_KEY, {})
    if not isinstance(raw, Mapping):
        raise SettingsSchemaError(f"{_EXCLUDED_KEY} must be an object")
    result: dict[str, str] = {}
    for name, reason in sorted(raw.items()):
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            raise SettingsSchemaError(f"{_EXCLUDED_KEY} has a malformed identifier")
        if not isinstance(reason, str) or not reason.strip():
            raise SettingsSchemaError(f"{name} is excluded without a reason")
        result[name] = reason.strip()
    return result


def _resolve_reference(
    name: str, entry: Mapping[str, Any], config_schema: Mapping[str, Any]
) -> Mapping[str, Any]:
    reference = entry.get("$ref")
    if reference is None:
        return {}
    if not isinstance(reference, str) or not reference.startswith(_CONFIG_REF_PREFIX):
        raise SettingsSchemaError(f"{name} references an unsupported schema")
    key = reference[len(_CONFIG_REF_PREFIX) :]
    target = (config_schema.get("properties") or {}).get(key)
    if not isinstance(target, Mapping):
        raise SettingsSchemaError(f"{name} references an unknown config property")
    if entry.get("x-config-key", key) != key:
        raise SettingsSchemaError(f"{name} references a different config key")
    return target


def _optional_text(name: str, entry: Mapping[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SettingsSchemaError(f"{name} has an invalid {key}")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SettingsSchemaError(f"settings schema repeats the key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


@lru_cache(maxsize=1)
def _cached_document() -> dict[str, Any]:
    return _read_json(settings_schema_path())


@lru_cache(maxsize=1)
def _cached_config_schema() -> dict[str, Any]:
    return _read_json(settings_schema_path().with_name(_CONFIG_SCHEMA_FILENAME))


@lru_cache(maxsize=1)
def _cached_sections() -> tuple[SettingsSection, ...]:
    return parse_settings_schema(_cached_document())


@lru_cache(maxsize=1)
def _cached_excluded() -> Mapping[str, str]:
    _cached_sections()  # validates that no excluded name is also declared
    return parse_excluded_identifiers(_cached_document())


@lru_cache(maxsize=1)
def _cached_index() -> Mapping[str, SettingDeclaration]:
    return MappingProxyType(
        {item.name: item for part in _cached_sections() for item in part.settings}
    )


__all__ = [
    "BOUND_KEYWORDS",
    "ENV_FORMATS",
    "SECTION_CORE",
    "SECTION_SERVICE",
    "SETTINGS_SCHEMA_FILENAME",
    "SETTINGS_SCHEMA_VERSION",
    "SettingDeclaration",
    "SettingsSchemaError",
    "SettingsSection",
    "declared_setting_names",
    "excluded_identifiers",
    "get_setting_declaration",
    "load_settings_schema",
    "parse_excluded_identifiers",
    "parse_settings_schema",
    "secret_setting_names",
    "setting_declarations",
    "settings_schema_path",
    "settings_sections",
]
