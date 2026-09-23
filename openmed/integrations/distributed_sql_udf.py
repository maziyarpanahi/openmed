"""Vectorized Python UDF entrypoint for distributed SQL engines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from openmed.core.policy import canonical_policy_name
from openmed.utils.validation import validate_batch_size

DEFAULT_DISTRIBUTED_SQL_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
DEFAULT_DISTRIBUTED_SQL_PROFILE = "hipaa_safe_harbor"
DEFAULT_DISTRIBUTED_SQL_BATCH_SIZE = 64

_PROFILE_ALIASES = {
    "hipaa": DEFAULT_DISTRIBUTED_SQL_PROFILE,
    "safe_harbor": DEFAULT_DISTRIBUTED_SQL_PROFILE,
}
_RESERVED_PROCESS_BATCH_KWARGS = frozenset(
    {
        "batch_size",
        "continue_on_error",
        "loader",
        "method",
        "model_name",
        "operation",
        "policy",
    }
)

ProcessBatch = Callable[..., Any]
LoaderFactory = Callable[[], Any | None]


@dataclass(frozen=True)
class DistributedSQLUDFConfig:
    """Runtime settings for a distributed SQL de-identification worker."""

    model_name: str = DEFAULT_DISTRIBUTED_SQL_MODEL
    default_profile: str = DEFAULT_DISTRIBUTED_SQL_PROFILE
    batch_size: int = DEFAULT_DISTRIBUTED_SQL_BATCH_SIZE
    method: str = "mask"
    confidence_threshold: float = 0.7
    process_batch_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_name = _non_empty_string(self.model_name, "model_name")
        method = _non_empty_string(self.method, "method")
        default_profile = _canonical_profile(self.default_profile)
        batch_size = validate_batch_size(self.batch_size)
        confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        process_batch_kwargs = dict(self.process_batch_kwargs)
        reserved = sorted(_RESERVED_PROCESS_BATCH_KWARGS & process_batch_kwargs.keys())
        if reserved:
            names = ", ".join(reserved)
            raise ValueError(
                "process_batch_kwargs must not override reserved settings: " + names
            )

        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "default_profile", default_profile)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "confidence_threshold", confidence_threshold)
        object.__setattr__(self, "process_batch_kwargs", process_batch_kwargs)


class DistributedSQLDeidentifyUDF:
    """Process vector windows for one distributed SQL Python worker.

    The object is intended to be constructed once in each worker process. Its
    model loader is created only when the first non-empty batch arrives and is
    then forwarded to every :func:`openmed.processing.process_batch` call.
    """

    def __init__(
        self,
        *,
        config: DistributedSQLUDFConfig | None = None,
        process_batch_fn: ProcessBatch | None = None,
        loader_factory: LoaderFactory | None = None,
    ) -> None:
        self.config = config or DistributedSQLUDFConfig()
        self._process_batch_fn = process_batch_fn
        self._loader_factory = loader_factory
        self._loader: Any | None = None
        self._loader_initialized = False

    def __call__(
        self,
        texts: Sequence[str | None],
        profiles: Sequence[str | None] | str | None = None,
    ) -> list[str | None]:
        """Return a vector of de-identified SQL cell values."""

        return self.deidentify_batch(texts, profiles)

    def deidentify(
        self,
        text: str | None,
        profile: str | None = None,
    ) -> str | None:
        """Return one de-identified value for scalar compatibility."""

        return self.deidentify_batch([text], profile)[0]

    def deidentify_batch(
        self,
        texts: Sequence[str | None],
        profiles: Sequence[str | None] | str | None = None,
    ) -> list[str | None]:
        """De-identify a vector while preserving nulls, empties, and row order."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of SQL cell values")

        values = list(texts)
        resolved_profiles = self._resolve_profiles(profiles, len(values))
        output: list[str | None] = [None] * len(values)

        for start in range(0, len(values), self.config.batch_size):
            stop = min(start + self.config.batch_size, len(values))
            self._process_window(
                values,
                resolved_profiles,
                output,
                start=start,
                stop=stop,
            )

        return output

    def _process_window(
        self,
        values: Sequence[str | None],
        profiles: Sequence[str | None],
        output: list[str | None],
        *,
        start: int,
        stop: int,
    ) -> None:
        grouped_positions: dict[str, list[int]] = {}
        for position in range(start, stop):
            value = values[position]
            if value is None:
                output[position] = None
                continue
            if not isinstance(value, str):
                raise TypeError(
                    "distributed SQL text values must be strings or None; "
                    f"row {position} is {type(value).__name__}"
                )
            if value == "":
                output[position] = ""
                continue
            profile = _canonical_profile(
                profiles[position] or self.config.default_profile
            )
            grouped_positions.setdefault(profile, []).append(position)

        for profile, positions in grouped_positions.items():
            batch_values = []
            for position in positions:
                value = values[position]
                if not isinstance(value, str) or not value:
                    raise RuntimeError("invalid internal distributed SQL row state")
                batch_values.append(value)
            batch_output = self._run_process_batch(batch_values, profile=profile)
            for position, redacted in zip(positions, batch_output, strict=True):
                output[position] = redacted

    def _run_process_batch(
        self,
        texts: Sequence[str | None],
        *,
        profile: str,
    ) -> list[str]:
        process_batch = self._get_process_batch()
        batch_result = process_batch(
            list(texts),
            model_name=self.config.model_name,
            operation="deidentify",
            batch_size=self.config.batch_size,
            loader=self._get_loader(),
            continue_on_error=False,
            policy=profile,
            method=self.config.method,
            confidence_threshold=self.config.confidence_threshold,
            **dict(self.config.process_batch_kwargs),
        )
        items = list(getattr(batch_result, "items", ()))
        if len(items) != len(texts):
            raise RuntimeError(
                "process_batch returned "
                f"{len(items)} results for {len(texts)} distributed SQL rows"
            )

        redacted: list[str] = []
        for index, item in enumerate(items):
            if getattr(item, "error", None) is not None:
                raise RuntimeError(
                    f"process_batch failed for distributed SQL row {index}"
                )
            redacted.append(_result_text(getattr(item, "result", None), index=index))
        return redacted

    def _resolve_profiles(
        self,
        profiles: Sequence[str | None] | str | None,
        expected: int,
    ) -> list[str | None]:
        if profiles is None or isinstance(profiles, str):
            return [profiles] * expected
        if isinstance(profiles, bytes):
            raise TypeError("profiles must be a string or a sequence of profile names")

        values = list(profiles)
        if len(values) != expected:
            raise ValueError(
                f"profiles contains {len(values)} values for {expected} text rows"
            )
        return values

    def _get_process_batch(self) -> ProcessBatch:
        if self._process_batch_fn is None:
            from openmed.processing import process_batch

            self._process_batch_fn = process_batch
        return self._process_batch_fn

    def _get_loader(self) -> Any | None:
        if self._loader_initialized:
            return self._loader

        self._loader_initialized = True
        factory = self._loader_factory or _default_loader_factory
        self._loader = factory()
        return self._loader


OPENMED_DEIDENTIFY_DESCRIPTOR: dict[str, Any] = {
    "name": "openmed_deidentify",
    "language": "python",
    "entrypoint": ("openmed.integrations.distributed_sql_udf:deidentify_batch"),
    "arguments": [
        {"name": "text", "sql_type": "VARCHAR", "python_batch": "texts"},
        {"name": "profile", "sql_type": "VARCHAR", "python_batch": "profiles"},
    ],
    "return_type": "VARCHAR",
    "vectorized": True,
    "null_handling": "called_on_null_input",
    "default_batch_size": DEFAULT_DISTRIBUTED_SQL_BATCH_SIZE,
}


@lru_cache(maxsize=1)
def _default_worker() -> DistributedSQLDeidentifyUDF:
    """Return the process-local worker used by module-level entrypoints."""

    return DistributedSQLDeidentifyUDF()


def deidentify(
    text: str | None,
    profile: str | None = None,
) -> str | None:
    """De-identify one logical SQL scalar value with the process-local worker."""

    return _default_worker().deidentify(text, profile)


def deidentify_batch(
    texts: Sequence[str | None],
    profiles: Sequence[str | None] | str | None = None,
) -> list[str | None]:
    """De-identify a vector window with the process-local worker."""

    return _default_worker().deidentify_batch(texts, profiles)


def _default_loader_factory() -> Any:
    from openmed.core import ModelLoader

    return ModelLoader()


def _canonical_profile(profile: str) -> str:
    normalized = _non_empty_string(profile, "profile").lower().replace("-", "_")
    return canonical_policy_name(_PROFILE_ALIASES.get(normalized, normalized))


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _result_text(result: Any, *, index: int) -> str:
    if isinstance(result, str):
        return result
    value = getattr(result, "deidentified_text", None)
    if isinstance(value, str):
        return value
    raise TypeError(
        "process_batch result for distributed SQL row "
        f"{index} must be a string or expose deidentified_text"
    )


__all__ = [
    "DEFAULT_DISTRIBUTED_SQL_BATCH_SIZE",
    "DEFAULT_DISTRIBUTED_SQL_MODEL",
    "DEFAULT_DISTRIBUTED_SQL_PROFILE",
    "DistributedSQLDeidentifyUDF",
    "DistributedSQLUDFConfig",
    "OPENMED_DEIDENTIFY_DESCRIPTOR",
    "deidentify",
    "deidentify_batch",
]
