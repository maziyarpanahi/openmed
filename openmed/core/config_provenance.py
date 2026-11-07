"""Deterministic configuration resolution and value-free provenance.

The resolver in this module is deliberately independent of model loading and
network services.  It combines local defaults, a local file mapping, an
environment snapshot, and command-line values in one documented order while
keeping the selected values separate from the audit report.  The report never
contains configuration values; it records only stable keys, source classes,
and conflict categories.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Literal

SourceClass = Literal["default", "file", "environment", "cli"]

SCHEMA_VERSION = 1

DEFAULT_SOURCE: SourceClass = "default"
FILE_SOURCE: SourceClass = "file"
ENVIRONMENT_SOURCE: SourceClass = "environment"
CLI_SOURCE: SourceClass = "cli"

# Later entries win.  Keep this tuple immutable and use it for every merge and
# every report so input mapping order cannot affect the audit.
CONFIG_PRECEDENCE: tuple[SourceClass, ...] = (
    DEFAULT_SOURCE,
    FILE_SOURCE,
    ENVIRONMENT_SOURCE,
    CLI_SOURCE,
)

CONFLICT_NONE = "none"
CONFLICT_SAME_VALUE = "same_value"
CONFLICT_OVERRIDDEN = "overridden"
CONFLICT_CATEGORIES = (
    CONFLICT_NONE,
    CONFLICT_SAME_VALUE,
    CONFLICT_OVERRIDDEN,
)

ConfigInput = Mapping[str, Any] | str | Path | None

_UNSET = object()

# Names that are aliases in the existing configuration surface rather than a
# direct OPENMED_<field> spelling.  The canonical spelling wins when more than
# one alias is present in the same environment snapshot.
_ENVIRONMENT_ALIASES = {
    "HF_TOKEN": "hf_token",
    "OPENMED_DEVICE": "device",
    "OPENMED_TORCH_DEVICE": "device",
    "OPENMED_OFFLINE": "local_only",
    "OPENMED_CHINESE_USER_DICT": "chinese_user_dict_path",
}

_BOOLEAN_KEYS = frozenset(
    {
        "bnb_4bit_use_double_quant",
        "clinical_protect_enabled",
        "clinical_protect_use_builtin",
        "lazy_model_loading",
        "local_only",
        "load_in_4bit",
        "transliteration_aware_name_matching",
        "use_medical_tokenizer",
    }
)
_INTEGER_KEYS = frozenset(
    {"batch_size", "num_workers", "onnx_intra_op_num_threads", "timeout"}
)
_FLOAT_KEYS = frozenset({"indic_name_similarity_threshold"})
_LIST_KEYS = frozenset({"clinical_protect_terms", "medical_tokenizer_exceptions"})


class ConfigurationResolutionError(ValueError):
    """Base error for invalid configuration resolution inputs."""


class ConfigurationSourceError(ConfigurationResolutionError):
    """Raised when a local configuration source cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    """Value-free provenance for one resolved configuration key."""

    key: str
    source_class: SourceClass
    conflict_category: str
    sources: tuple[SourceClass, ...]
    overridden_sources: tuple[SourceClass, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entry without the selected value."""

        return {
            "key": self.key,
            "source_class": self.source_class,
            "conflict_category": self.conflict_category,
            "sources": list(self.sources),
            "overridden_sources": list(self.overridden_sources),
        }


@dataclass(frozen=True, slots=True)
class ConfigurationResolution:
    """Resolved values paired with a value-free provenance report.

    ``values`` is the effective configuration and may contain secrets supplied
    by the caller.  ``provenance_report`` is safe to serialize because it
    intentionally contains no values, paths, or source payloads.
    """

    values: Mapping[str, Any]
    entries: tuple[ProvenanceEntry, ...]

    @property
    def effective_values(self) -> Mapping[str, Any]:
        """Return the effective values selected by the precedence order."""

        return self.values

    @property
    def provenance_report(self) -> dict[str, Any]:
        """Return a stable, value-free report for this resolution."""

        return {
            "schema_version": SCHEMA_VERSION,
            "precedence": list(CONFIG_PRECEDENCE),
            "keys": {
                entry.key: {
                    "source_class": entry.source_class,
                    "conflict_category": entry.conflict_category,
                    "sources": list(entry.sources),
                    "overridden_sources": list(entry.overridden_sources),
                }
                for entry in self.entries
            },
        }

    @property
    def report(self) -> dict[str, Any]:
        """Alias for :attr:`provenance_report`."""

        return self.provenance_report

    def to_report(self) -> dict[str, Any]:
        """Return the value-free report as a fresh dictionary."""

        return self.provenance_report

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free report as a fresh dictionary.

        Effective values remain available through :attr:`values` and
        :attr:`effective_values`.  Keeping the conventional serialization
        helper value-free prevents an accidental log or artifact write from
        exposing credentials or other sensitive configuration values.
        """

        return self.provenance_report

    def __repr__(self) -> str:
        """Avoid exposing selected values through an accidental repr/log."""

        keys = tuple(self.values)
        return f"ConfigurationResolution(keys={keys!r}, entries={len(self.entries)})"


def default_config_values() -> dict[str, Any]:
    """Return OpenMed dataclass defaults without applying environment values.

    ``OpenMedConfig()`` applies environment overrides during construction.  A
    precedence audit needs an uncontaminated default layer, so this helper
    reads the dataclass field defaults directly instead.
    """

    # Import lazily so callers that only use explicit mappings do not pay for
    # configuration initialization until they request OpenMed defaults.
    from .config import OpenMedConfig

    result: dict[str, Any] = {}
    for config_field in fields(OpenMedConfig):
        if config_field.default is not MISSING:
            result[config_field.name] = copy.deepcopy(config_field.default)
        elif config_field.default_factory is not MISSING:  # pragma: no cover
            result[config_field.name] = copy.deepcopy(config_field.default_factory())
    return result


openmed_default_values = default_config_values


def resolve_configuration(
    defaults: Mapping[str, Any] | None = None,
    file_config: ConfigInput = None,
    environment: Mapping[str, Any] | None = None,
    cli: Mapping[str, Any] | Any | None = None,
    *,
    file_values: ConfigInput = None,
    env: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | Any | None = None,
    known_keys: Iterable[str] | None = None,
    env_prefix: str = "OPENMED_",
) -> ConfigurationResolution:
    """Resolve configuration sources and produce value-free provenance.

    The deterministic precedence order is ``default < file < environment <
    cli``.  ``file_config`` can be a mapping or a local TOML path.  The
    environment defaults to a snapshot of ``os.environ``; pass ``env={}`` or
    ``environment={}`` to audit without process environment values.  Environment
    names use ``OPENMED_<KEY>`` and known legacy aliases such as ``HF_TOKEN``.

    Args:
        defaults: Base values.  If omitted, OpenMed dataclass defaults are used
            without applying environment overrides.
        file_config: Mapping or local TOML file containing file-level values.
        environment: Environment snapshot.  ``None`` uses ``os.environ``.
        cli: Parsed command-line mapping or ``argparse.Namespace``.
        file_values: Alias for ``file_config``.
        env: Alias for ``environment``.
        cli_values: Alias for ``cli``.
        known_keys: Optional allow-list for normalized configuration keys.
        env_prefix: Prefix used for generic environment names.

    Returns:
        A :class:`ConfigurationResolution` whose ``values`` are effective
        settings and whose ``provenance_report`` contains no values.

    Raises:
        ConfigurationResolutionError: If a source is not a mapping or contains
            an invalid key/value shape.
        ConfigurationSourceError: If a TOML source cannot be loaded.
    """

    file_config = _select_alias(file_config, file_values, "file_config")
    environment = _select_alias(environment, env, "environment")
    cli = _select_alias(cli, cli_values, "cli")

    if defaults is None:
        defaults = default_config_values()
    default_layer = _normalize_mapping(defaults, DEFAULT_SOURCE)
    file_layer = _normalize_mapping(_load_file_config(file_config), FILE_SOURCE)
    cli_layer = _normalize_cli(cli)

    normalized_known_keys = _normalize_known_keys(known_keys)
    if normalized_known_keys is None:
        normalized_known_keys = set(default_layer)
        normalized_known_keys.update(file_layer)
        normalized_known_keys.update(cli_layer)

    if environment is None:
        environment = os.environ
    environment_layer = _normalize_environment(
        environment,
        known_keys=normalized_known_keys,
        env_prefix=env_prefix,
        references=(default_layer, file_layer, cli_layer),
    )

    layers: dict[SourceClass, dict[str, Any]] = {
        DEFAULT_SOURCE: default_layer,
        FILE_SOURCE: file_layer,
        ENVIRONMENT_SOURCE: environment_layer,
        CLI_SOURCE: cli_layer,
    }
    all_keys = sorted({key for layer in layers.values() for key in layer})

    resolved: dict[str, Any] = {}
    entries: list[ProvenanceEntry] = []
    for key in all_keys:
        sources = tuple(source for source in CONFIG_PRECEDENCE if key in layers[source])
        winner = sources[-1]
        winner_value = layers[winner][key]
        resolved[key] = winner_value

        differing_sources = tuple(
            source
            for source in sources[:-1]
            if not _values_equal(layers[source][key], winner_value)
        )
        if len(sources) == 1:
            conflict_category = CONFLICT_NONE
        elif differing_sources:
            conflict_category = CONFLICT_OVERRIDDEN
        else:
            conflict_category = CONFLICT_SAME_VALUE

        entries.append(
            ProvenanceEntry(
                key=key,
                source_class=winner,
                conflict_category=conflict_category,
                sources=sources,
                overridden_sources=differing_sources,
            )
        )

    return ConfigurationResolution(values=resolved, entries=tuple(entries))


def audit_config_precedence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return only the value-free provenance report for a resolution."""

    return resolve_configuration(*args, **kwargs).provenance_report


def audit_configuration(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`audit_config_precedence`."""

    return audit_config_precedence(*args, **kwargs)


resolve_config_precedence = resolve_configuration
resolve_config = resolve_configuration


def _select_alias(primary: Any, alias: Any, name: str) -> Any:
    if primary is not None and alias is not None:
        raise ConfigurationResolutionError(f"pass only one {name} source")
    return primary if primary is not None else alias


def _load_file_config(source: ConfigInput) -> Mapping[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, (str, Path)):
        raise ConfigurationSourceError("file configuration must be a mapping or path")

    path = Path(source).expanduser()
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10 with the optional dev extra
            import tomli as tomllib  # type: ignore[no-redef]
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, ValueError, ModuleNotFoundError):
        # Do not surface paths, parser details, or file content in an audit
        # exception.  The source class is sufficient for callers to diagnose
        # which input needs attention.
        raise ConfigurationSourceError(
            "file configuration could not be loaded"
        ) from None

    if not isinstance(loaded, Mapping):  # pragma: no cover - tomllib contract
        raise ConfigurationSourceError("file configuration must contain a mapping")
    return loaded


def _normalize_mapping(
    source: Mapping[str, Any], source_class: SourceClass
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ConfigurationResolutionError(
            f"{source_class} configuration must be a mapping"
        )

    normalized: dict[str, Any] = {}
    # Sorting makes duplicate spellings deterministic even for custom mapping
    # implementations whose iteration order is not stable.
    items = sorted(source.items(), key=lambda item: str(item[0]))
    for raw_key, value in items:
        key = _normalize_key(raw_key)
        normalized[key] = value
    return normalized


def _normalize_cli(source: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return _normalize_mapping(source, CLI_SOURCE)
    if hasattr(source, "__dict__"):
        # argparse.Namespace uses None for options not supplied on the command
        # line.  Omit those values while preserving explicit mapping None.
        values = {
            key: value for key, value in vars(source).items() if value is not None
        }
        return _normalize_mapping(values, CLI_SOURCE)
    raise ConfigurationResolutionError("cli configuration must be a mapping")


def _normalize_environment(
    source: Mapping[str, Any],
    *,
    known_keys: set[str],
    env_prefix: str,
    references: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ConfigurationResolutionError(
            "environment configuration must be a mapping"
        )
    if not isinstance(env_prefix, str) or not env_prefix:
        raise ConfigurationResolutionError("env_prefix must be a non-empty string")

    candidates: dict[str, list[tuple[int, str, Any]]] = {}
    for raw_key, raw_value in sorted(source.items(), key=lambda item: str(item[0])):
        key = _environment_key(raw_key, env_prefix)
        if key is None or key not in known_keys:
            continue
        priority = _environment_priority(raw_key, key, env_prefix)
        candidates.setdefault(key, []).append((priority, str(raw_key), raw_value))

    normalized: dict[str, Any] = {}
    for key in sorted(candidates):
        _, _, raw_value = min(candidates[key], key=lambda item: (item[0], item[1]))
        reference = _reference_value(key, references)
        normalized[key] = _coerce_value(key, raw_value, reference)
    return normalized


def _environment_key(raw_key: Any, env_prefix: str) -> str | None:
    if not isinstance(raw_key, str):
        raise ConfigurationResolutionError("environment keys must be strings")
    if raw_key in _ENVIRONMENT_ALIASES:
        return _ENVIRONMENT_ALIASES[raw_key]
    if raw_key.startswith(env_prefix):
        suffix = raw_key[len(env_prefix) :]
        if not suffix or suffix.lower() == "config":
            return None
        return _normalize_key(suffix)
    # Ignore unrelated ambient variables.  Accepting unprefixed names such as
    # ``DEVICE`` or ``PROFILE`` would let a host environment silently override
    # OpenMed configuration even though the documented contract requires the
    # OPENMED_ prefix (apart from the explicit compatibility aliases above).
    return None


def _environment_priority(raw_key: Any, key: str, env_prefix: str) -> int:
    if raw_key == "OPENMED_TORCH_DEVICE" and key == "device":
        return 0
    if raw_key == "OPENMED_DEVICE" and key == "device":
        return 1
    if raw_key == key:
        return 0
    canonical = f"{env_prefix}{key.upper()}"
    if raw_key == canonical:
        return 0
    if raw_key == "OPENMED_OFFLINE" and key == "local_only":
        return 1
    if raw_key == "HF_TOKEN" and key == "hf_token":
        return 1
    return 3


def _reference_value(key: str, references: tuple[Mapping[str, Any], ...]) -> Any:
    for reference in references:
        if key in reference:
            return reference[key]
    return _UNSET


def _coerce_value(key: str, value: Any, reference: Any) -> Any:
    if not isinstance(value, str):
        return value

    if key in _BOOLEAN_KEYS or isinstance(reference, bool):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigurationResolutionError(
            f"invalid environment value for {key!r}; expected a boolean"
        )

    if key in _INTEGER_KEYS or (
        isinstance(reference, int) and not isinstance(reference, bool)
    ):
        try:
            return int(value.strip())
        except ValueError:
            raise ConfigurationResolutionError(
                f"invalid environment value for {key!r}; expected an integer"
            ) from None

    if key in _FLOAT_KEYS or isinstance(reference, float):
        try:
            result = float(value.strip())
        except ValueError:
            raise ConfigurationResolutionError(
                f"invalid environment value for {key!r}; expected a number"
            ) from None
        if not math.isfinite(result):
            raise ConfigurationResolutionError(
                f"invalid environment value for {key!r}; expected a finite number"
            )
        return result

    if key in _LIST_KEYS or isinstance(reference, (list, tuple)):
        return [item.strip() for item in value.split(",") if item.strip()]

    return value


def _normalize_known_keys(known_keys: Iterable[str] | None) -> set[str] | None:
    if known_keys is None:
        return None
    normalized: set[str] = set()
    for key in known_keys:
        normalized.add(_normalize_key(key))
    return normalized


def _normalize_key(raw_key: Any) -> str:
    if not isinstance(raw_key, str):
        raise ConfigurationResolutionError("configuration keys must be strings")
    key = raw_key.strip().lstrip("-").replace("-", "_").lower()
    if not key:
        raise ConfigurationResolutionError("configuration keys must not be empty")
    return key


def _values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


__all__ = [
    "CLI_SOURCE",
    "CONFIG_PRECEDENCE",
    "CONFLICT_CATEGORIES",
    "CONFLICT_NONE",
    "CONFLICT_OVERRIDDEN",
    "CONFLICT_SAME_VALUE",
    "ConfigurationResolution",
    "ConfigurationResolutionError",
    "ConfigurationSourceError",
    "DEFAULT_SOURCE",
    "ENVIRONMENT_SOURCE",
    "FILE_SOURCE",
    "ProvenanceEntry",
    "audit_config_precedence",
    "audit_configuration",
    "default_config_values",
    "openmed_default_values",
    "resolve_config",
    "resolve_config_precedence",
    "resolve_configuration",
]
