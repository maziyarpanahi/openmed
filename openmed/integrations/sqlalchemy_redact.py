"""SQLAlchemy write-time de-identification helpers.

This optional integration provides a :class:`RedactedText` type for declarative
column-level redaction and event installers for applications that need to keep
ordinary SQLAlchemy ``Text`` columns. Redaction happens before a value reaches
the database; result processing never attempts to re-identify or transform
stored values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, Callable

try:
    from sqlalchemy import Text, event, inspect
    from sqlalchemy.types import TypeDecorator
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "SQLAlchemy redaction requires the 'sqlalchemy' extra. "
        "Install with `pip install openmed[sqlalchemy]`."
    ) from exc

DEFAULT_SQLALCHEMY_POLICY = "hipaa_safe_harbor"

PipelineFactory = Callable[["SQLAlchemyRedactionConfig"], Any]


@dataclass(frozen=True)
class SQLAlchemyRedactionConfig:
    """Configuration shared by SQLAlchemy write-time redaction helpers.

    Args:
        policy_profile: OpenMed policy profile applied to every configured value.
        method: De-identification method passed to the cached pipeline.
        model_name: Optional model registry key or model identifier. ``None`` uses
            OpenMed's default PII model.
        confidence_threshold: Minimum model confidence for redaction.
        keep_year: Preserve the year when redacting date entities.
        use_smart_merging: Enable semantic span merging.
        lang: ISO 639-1 language hint.
        normalize_accents: Optional accent-normalization override.
        use_safety_sweep: Enable deterministic structured-identifier detection.
    """

    policy_profile: str = DEFAULT_SQLALCHEMY_POLICY
    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    use_smart_merging: bool = True
    lang: str = "en"
    normalize_accents: bool | None = None
    use_safety_sweep: bool = True

    def __post_init__(self) -> None:
        if not self.policy_profile.strip():
            raise ValueError("policy_profile must not be empty")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not self.lang.strip():
            raise ValueError("lang must not be empty")


class RedactionPipelineRegistry:
    """Own one lazily constructed pipeline for a redaction configuration.

    A registry can be shared by multiple mapped columns and event hooks. Pipeline
    creation and execution are synchronized so the same warmed pipeline is reused
    safely across concurrent ORM writes instead of being rebuilt per row.

    Args:
        config: Redaction configuration for this registry.
        pipeline_factory: Optional pipeline constructor, primarily useful for
            offline adapters and tests. It receives ``config`` once.
    """

    def __init__(
        self,
        config: SQLAlchemyRedactionConfig | None = None,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config = config or SQLAlchemyRedactionConfig()
        self._pipeline_factory = pipeline_factory or _create_pipeline
        self._pipeline: Any | None = None
        self._creation_lock = RLock()
        self._run_lock = RLock()

    @property
    def pipeline(self) -> Any:
        """Return the registry pipeline, constructing it at most once."""

        if self._pipeline is None:
            with self._creation_lock:
                if self._pipeline is None:
                    self._pipeline = self._pipeline_factory(self.config)
        return self._pipeline

    def redact(self, value: str | None) -> str | None:
        """Return a write-safe redacted value while preserving ``None``."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("redacted SQLAlchemy columns accept only str or None")
        if value == "":
            return value

        with self._run_lock:
            result = self.pipeline.run(
                value,
                method=self.config.method,
                keep_year=self.config.keep_year,
            )
        return _deidentified_text(result)


class RedactedText(TypeDecorator[str]):
    """SQLAlchemy text type that de-identifies non-null values on bind.

    Reads return the already-redacted database value unchanged. Pass the same
    :class:`RedactionPipelineRegistry` to multiple columns or hooks to share one
    pipeline explicitly.

    Args:
        length: Optional SQLAlchemy text length hint.
        policy_profile: OpenMed policy profile name. ``policy`` is an alias.
        policy: Alias for ``policy_profile``.
        config: Complete redaction configuration.
        registry: Explicit pipeline registry to share across columns or hooks.
    """

    impl = Text
    cache_ok = True

    def __init__(
        self,
        length: int | None = None,
        *,
        policy_profile: str | None = None,
        policy: str | None = None,
        config: SQLAlchemyRedactionConfig | None = None,
        registry: RedactionPipelineRegistry | None = None,
    ) -> None:
        self.registry = _resolve_registry(
            policy_profile=policy_profile,
            policy=policy,
            config=config,
            registry=registry,
        )
        super().__init__(length=length)

    def process_bind_param(self, value: str | None, dialect: Any) -> str | None:
        """De-identify ``value`` immediately before SQL parameter binding."""

        del dialect
        return self.registry.redact(value)

    def process_result_value(self, value: str | None, dialect: Any) -> str | None:
        """Return the already-redacted stored value without transformation."""

        del dialect
        return value


@dataclass
class RedactionEventRegistration:
    """Handle for removing an installed SQLAlchemy redaction event."""

    target: Any
    event_name: str
    listener: Callable[..., None]
    _removed: bool = False

    def remove(self) -> None:
        """Remove the event listener. Repeated calls are harmless."""

        if self._removed:
            return
        event.remove(self.target, self.event_name, self.listener)
        self._removed = True


def install_session_redaction(
    session_target: Any,
    columns: Mapping[type[Any], Sequence[str] | str],
    *,
    policy_profile: str | None = None,
    policy: str | None = None,
    config: SQLAlchemyRedactionConfig | None = None,
    registry: RedactionPipelineRegistry | None = None,
) -> RedactionEventRegistration:
    """Install an opt-in ``before_flush`` redactor on a session target.

    ``session_target`` may be a SQLAlchemy ``Session`` instance, ``Session``
    subclass, or session factory accepted by SQLAlchemy's event API. ``columns``
    maps each model class to one column name or a sequence of column names.

    New objects are always inspected. Dirty objects are redacted only when the
    configured attribute changed during the current unit of work.

    Args:
        session_target: SQLAlchemy session event target.
        columns: Model-to-column mapping for write-time redaction.
        policy_profile: OpenMed policy profile name. ``policy`` is an alias.
        policy: Alias for ``policy_profile``.
        config: Complete redaction configuration.
        registry: Explicit pipeline registry to share with other hooks or types.

    Returns:
        A removable event-registration handle.
    """

    normalized = _normalize_model_columns(columns)
    resolved_registry = _resolve_registry(
        policy_profile=policy_profile,
        policy=policy,
        config=config,
        registry=registry,
    )

    def before_flush(session: Any, flush_context: Any, instances: Any) -> None:
        del flush_context, instances
        new_object_ids = {id(instance) for instance in session.new}
        for instance in (*session.new, *session.dirty):
            is_new = id(instance) in new_object_ids
            for model, column_names in normalized:
                if isinstance(instance, model):
                    _redact_instance(
                        instance,
                        column_names,
                        resolved_registry,
                        changed_only=not is_new,
                    )

    event.listen(session_target, "before_flush", before_flush)
    return RedactionEventRegistration(session_target, "before_flush", before_flush)


def install_mapper_redaction(
    model: type[Any],
    columns: Sequence[str] | str,
    *,
    policy_profile: str | None = None,
    policy: str | None = None,
    config: SQLAlchemyRedactionConfig | None = None,
    registry: RedactionPipelineRegistry | None = None,
    propagate: bool = True,
) -> RedactionEventRegistration:
    """Install an opt-in ``before_insert`` redactor for one mapped model.

    Args:
        model: Declarative mapped class that owns the configured attributes.
        columns: One column name or a sequence of column names to redact.
        policy_profile: OpenMed policy profile name. ``policy`` is an alias.
        policy: Alias for ``policy_profile``.
        config: Complete redaction configuration.
        registry: Explicit pipeline registry to share with other hooks or types.
        propagate: Apply the mapper event to mapped subclasses as well.

    Returns:
        A removable event-registration handle.
    """

    column_names = _normalize_column_names(columns)
    resolved_registry = _resolve_registry(
        policy_profile=policy_profile,
        policy=policy,
        config=config,
        registry=registry,
    )

    def before_insert(mapper: Any, connection: Any, instance: Any) -> None:
        del mapper, connection
        _redact_instance(
            instance,
            column_names,
            resolved_registry,
            changed_only=False,
        )

    event.listen(model, "before_insert", before_insert, propagate=propagate)
    return RedactionEventRegistration(model, "before_insert", before_insert)


def _create_pipeline(config: SQLAlchemyRedactionConfig) -> Any:
    from openmed.core.pipeline import Pipeline

    kwargs: dict[str, Any] = {
        "confidence_threshold": config.confidence_threshold,
        "use_smart_merging": config.use_smart_merging,
        "lang": config.lang,
        "normalize_accents": config.normalize_accents,
        "use_safety_sweep": config.use_safety_sweep,
        "policy": config.policy_profile,
    }
    if config.model_name is not None:
        kwargs["model_name"] = config.model_name
    return Pipeline(**kwargs)


@lru_cache(maxsize=32)
def _default_registry(
    config: SQLAlchemyRedactionConfig,
) -> RedactionPipelineRegistry:
    return RedactionPipelineRegistry(config)


def _resolve_registry(
    *,
    policy_profile: str | None,
    policy: str | None,
    config: SQLAlchemyRedactionConfig | None,
    registry: RedactionPipelineRegistry | None,
) -> RedactionPipelineRegistry:
    requested_policy = _resolve_policy(policy_profile, policy)

    if config is not None and requested_policy is not None:
        if config.policy_profile != requested_policy:
            raise ValueError("config and policy_profile must select the same policy")
    resolved_config = config or SQLAlchemyRedactionConfig(
        policy_profile=requested_policy or DEFAULT_SQLALCHEMY_POLICY
    )

    if registry is not None:
        if config is not None and registry.config != config:
            raise ValueError("registry and config must use the same configuration")
        if (
            requested_policy is not None
            and registry.config.policy_profile != requested_policy
        ):
            raise ValueError("registry and policy_profile must select the same policy")
        return registry

    return _default_registry(resolved_config)


def _resolve_policy(
    policy_profile: str | None,
    policy: str | None,
) -> str | None:
    if policy_profile is not None and policy is not None and policy_profile != policy:
        raise ValueError("policy and policy_profile must match when both are provided")
    resolved = policy_profile if policy_profile is not None else policy
    if resolved is not None and not resolved.strip():
        raise ValueError("policy_profile must not be empty")
    return resolved


def _normalize_model_columns(
    columns: Mapping[type[Any], Sequence[str] | str],
) -> tuple[tuple[type[Any], tuple[str, ...]], ...]:
    if not columns:
        raise ValueError("columns must configure at least one mapped model")
    normalized: list[tuple[type[Any], tuple[str, ...]]] = []
    for model, names in columns.items():
        if not isinstance(model, type):
            raise TypeError("columns keys must be mapped model classes")
        normalized.append((model, _normalize_column_names(names)))
    return tuple(normalized)


def _normalize_column_names(columns: Sequence[str] | str) -> tuple[str, ...]:
    values = (columns,) if isinstance(columns, str) else tuple(columns)
    if not values:
        raise ValueError("columns must not be empty")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("column names must be non-empty strings")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _redact_instance(
    instance: Any,
    column_names: tuple[str, ...],
    registry: RedactionPipelineRegistry,
    *,
    changed_only: bool,
) -> None:
    state = inspect(instance)
    for column_name in column_names:
        if column_name not in state.attrs:
            raise AttributeError(
                f"{type(instance).__name__} has no mapped attribute {column_name!r}"
            )
        attribute = state.attrs[column_name]
        if changed_only and not attribute.history.has_changes():
            continue
        value = getattr(instance, column_name)
        redacted = registry.redact(value)
        if redacted != value:
            setattr(instance, column_name, redacted)


def _deidentified_text(result: Any) -> str:
    candidate = getattr(result, "deidentification_result", result)
    if isinstance(candidate, str):
        return candidate
    value = getattr(candidate, "deidentified_text", None)
    if not isinstance(value, str):
        raise TypeError("redaction pipeline must return de-identified text as a string")
    return value


__all__ = [
    "DEFAULT_SQLALCHEMY_POLICY",
    "RedactedText",
    "RedactionEventRegistration",
    "RedactionPipelineRegistry",
    "SQLAlchemyRedactionConfig",
    "install_mapper_redaction",
    "install_session_redaction",
]
