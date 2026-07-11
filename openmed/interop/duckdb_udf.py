"""DuckDB UDF registration for local OpenMed privacy and extraction workflows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

Deidentifier = Callable[..., Any]
ClinicalExtractor = Callable[..., Any]

_POLICY_ALIASES: Mapping[str, str] = {
    "safe_harbor": "hipaa_safe_harbor",
    "hipaa": "hipaa_safe_harbor",
}


@dataclass(frozen=True)
class DuckDBUDFConfig:
    """Runtime options forwarded to OpenMed's de-identification engine."""

    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    keep_mapping: bool = False
    use_smart_merging: bool = True
    lang: str = "en"
    normalize_accents: bool | None = None
    use_safety_sweep: bool = True
    consistent: bool = False
    seed: int | None = None
    locale: str | None = None
    default_policy: str = "hipaa_safe_harbor"
    calibration_thresholds_path: str | Path | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_deidentify_kwargs(self, *, policy: str | None = None) -> dict[str, Any]:
        """Return keyword arguments for ``openmed.core.pii.deidentify``."""

        resolved_policy = _canonical_udf_policy(policy or self.default_policy)
        kwargs: dict[str, Any] = {
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "keep_year": self.keep_year,
            "keep_mapping": self.keep_mapping,
            "use_smart_merging": self.use_smart_merging,
            "lang": self.lang,
            "normalize_accents": self.normalize_accents,
            "use_safety_sweep": self.use_safety_sweep,
            "consistent": self.consistent,
            "seed": self.seed,
            "locale": self.locale,
            "policy": resolved_policy,
            "calibration_thresholds_path": self.calibration_thresholds_path,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name

        kwargs.update(dict(self.extra_kwargs))
        return {key: value for key, value in kwargs.items() if value is not None}


class OpenMedDuckDBUDFs:
    """Register and serve OpenMed scalar functions for one DuckDB connection."""

    def __init__(
        self,
        *,
        config: DuckDBUDFConfig | None = None,
        deidentifier: Deidentifier | None = None,
        extractor: ClinicalExtractor | None = None,
    ) -> None:
        self.config = config or DuckDBUDFConfig()
        self._deidentifier = deidentifier
        self._extractor = extractor

    def register(self, con: Any) -> Any:
        """Register OpenMed scalar UDFs on *con* and return the connection."""

        _load_duckdb()
        sqltypes = _load_duckdb_sqltypes()
        _create_scalar_function(
            con,
            "openmed_deidentify",
            self.deidentify,
            parameters=[sqltypes.VARCHAR, sqltypes.VARCHAR],
            return_type=sqltypes.VARCHAR,
        )
        _create_scalar_function(
            con,
            "openmed_pii_count",
            self.pii_count,
            parameters=[sqltypes.VARCHAR],
            return_type=sqltypes.INTEGER,
        )
        _create_scalar_function(
            con,
            "openmed_extract_json",
            self.extract_json,
            parameters=[sqltypes.VARCHAR],
            return_type=sqltypes.VARCHAR,
        )
        return con

    def deidentify(self, text: str | None, policy: str | None) -> str | None:
        """Return de-identified text for one DuckDB scalar cell."""

        if text is None:
            return None
        result = self._run_deidentifier(str(text), policy=policy)
        return _result_text(result)

    def pii_count(self, text: str | None) -> int | None:
        """Return the number of PII entities detected in one scalar cell."""

        if text is None:
            return None
        result = self._run_deidentifier(str(text), policy=self.config.default_policy)
        return len(getattr(result, "pii_entities", ()) or ())

    def extract_json(self, text: str | None) -> str:
        """Return grounded clinical entity rows as JSON for SQL expansion."""

        if text is None:
            return "[]"
        from openmed.interop.clinical_dataframe import extract_text_rows

        rows = extract_text_rows(str(text), extractor=self._extractor)
        return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

    def _run_deidentifier(self, text: str, *, policy: str | None) -> Any:
        kwargs = self.config.to_deidentify_kwargs(policy=policy)
        if self._deidentifier is not None:
            return self._deidentifier(text, **kwargs)

        kwargs["loader"] = _cached_model_loader()
        return _default_deidentifier()(text, **kwargs)


def register_openmed_udfs(
    con: Any,
    *,
    config: DuckDBUDFConfig | None = None,
    deidentifier: Deidentifier | None = None,
    extractor: ClinicalExtractor | None = None,
) -> Any:
    """Register OpenMed DuckDB scalar UDFs on an existing connection.

    The registered functions are:

    - ``openmed_deidentify(text, policy) -> text``
    - ``openmed_pii_count(text) -> integer``
    - ``openmed_extract_json(text) -> JSON text``

    ``duckdb`` is imported only when this function is called. The default
    OpenMed ``ModelLoader`` is cached per process and reused across scalar
    calls so repeated SQL rows do not rebuild the underlying model loader.
    """

    return OpenMedDuckDBUDFs(
        config=config,
        deidentifier=deidentifier,
        extractor=extractor,
    ).register(con)


def clinical_extract_relation(
    con: Any,
    source_sql: str,
    text_column: str,
    *,
    extractor: ClinicalExtractor | None = None,
) -> Any:
    """Return a DuckDB relation of grounded clinical rows for ``source_sql``.

    ``source_sql`` should be a SELECT statement or table expression containing
    ``text_column``. The relation expands ``openmed_extract_json`` with DuckDB's
    native ``json_each`` table function and returns the fixed flat-table schema.
    """

    register_openmed_udfs(con, extractor=extractor)
    return con.sql(clinical_extract_sql(source_sql, text_column))


def clinical_extract_sql(source_sql: str, text_column: str) -> str:
    """Build SQL that expands ``openmed_extract_json`` into flat-table rows."""

    quoted_column = _quote_identifier(text_column)
    return f"""
SELECT
    json_extract_string(openmed_entity.value, '$.entity_label') AS entity_label,
    json_extract_string(openmed_entity.value, '$.normalized_text') AS normalized_text,
    json_extract_string(openmed_entity.value, '$.system') AS system,
    json_extract_string(openmed_entity.value, '$.code') AS code,
    json_extract_string(openmed_entity.value, '$.display') AS display,
    json_extract_string(openmed_entity.value, '$.negation') AS negation,
    json_extract_string(openmed_entity.value, '$.temporality') AS temporality,
    json_extract_string(openmed_entity.value, '$.certainty') AS certainty,
    CAST(NULLIF(json_extract_string(openmed_entity.value, '$.start'), '') AS BIGINT)
        AS start,
    CAST(NULLIF(json_extract_string(openmed_entity.value, '$.end'), '') AS BIGINT)
        AS end,
    json_extract_string(openmed_entity.value, '$.section') AS section
FROM ({source_sql}) AS openmed_source
CROSS JOIN json_each(openmed_extract_json(openmed_source.{quoted_column}))
    AS openmed_entity
ORDER BY 9, 10, 1, 4
""".strip()


def _canonical_udf_policy(policy: str | None) -> str:
    normalized = str(policy or "").strip().lower().replace("-", "_")
    if not normalized:
        return "hipaa_safe_harbor"
    return _POLICY_ALIASES.get(normalized, normalized)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


def _create_scalar_function(
    con: Any,
    name: str,
    function: Deidentifier,
    *,
    parameters: list[Any],
    return_type: Any,
) -> None:
    kwargs = {
        "parameters": parameters,
        "return_type": return_type,
        "null_handling": "special",
    }
    try:
        con.remove_function(name)
    except Exception:
        pass

    try:
        con.create_function(name, function, **kwargs)
    except TypeError:
        kwargs.pop("null_handling")
        con.create_function(name, function, **kwargs)


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _load_duckdb() -> Any:
    try:
        return _import_module("duckdb")
    except ImportError as exc:
        raise ImportError(
            "DuckDB support requires the optional dependency; install "
            "openmed[duckdb] to use openmed.interop.duckdb_udf"
        ) from exc


def _load_duckdb_sqltypes() -> Any:
    try:
        return _import_module("duckdb.sqltypes")
    except ImportError:
        return _import_module("duckdb.typing")


def _default_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


@lru_cache(maxsize=1)
def _cached_model_loader() -> Any:
    from openmed.core import ModelLoader

    return ModelLoader()


__all__ = [
    "ClinicalExtractor",
    "DuckDBUDFConfig",
    "OpenMedDuckDBUDFs",
    "clinical_extract_relation",
    "clinical_extract_sql",
    "register_openmed_udfs",
]
