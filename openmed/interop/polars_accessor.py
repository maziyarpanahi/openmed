"""Polars DataFrame helpers for OpenMed clinical table workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

try:
    import polars as pl
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "Polars support requires the 'polars' extra. "
        "Install with `pip install openmed[polars]`."
    ) from exc

Deidentifier = Callable[..., Any]
RiskReporter = Callable[..., dict[str, Any]]
ClinicalExtractor = Callable[..., Any]


def deidentify_frame(
    frame: Any,
    columns: Sequence[str] | str,
    *,
    method: str = "mask",
    policy: str | None = None,
    deidentifier: Deidentifier | None = None,
    **kwargs: Any,
) -> Any:
    """Return a redacted Polars DataFrame for selected free-text columns.

    Args:
        frame: Polars DataFrame to redact.
        columns: Free-text column names to redact.
        method: De-identification method forwarded to
            :func:`openmed.core.pii.deidentify`.
        policy: Optional policy profile forwarded to de-identification.
        deidentifier: Optional callable used primarily by tests and custom
            embedding contexts.
        **kwargs: Additional keyword arguments forwarded to de-identification.

    Returns:
        A new ``polars.DataFrame`` with selected string cells redacted.
    """

    _ensure_polars_frame(frame)
    selected_columns = _validate_columns(frame, columns)
    redact = deidentifier or _load_deidentifier()
    deidentify_kwargs = _deidentify_kwargs(method, policy, kwargs)

    records = frame.to_dicts()
    for record in records:
        for column in selected_columns:
            record[column] = _redact_value(
                record.get(column),
                redact,
                deidentify_kwargs,
            )

    return pl.DataFrame(records).select(frame.columns)


def risk_report(
    frame: Any,
    qi_columns: Sequence[str] | str | None = None,
    *,
    original: Any | None = None,
    aux: Any | None = None,
    reporter: RiskReporter | None = None,
) -> dict[str, Any]:
    """Return the OpenMed re-identification risk shape for Polars records.

    Args:
        frame: Polars DataFrame to score.
        qi_columns: Optional quasi-identifier columns to include. When omitted,
            all columns are passed to the risk scorer.
        original: Optional original records or Polars DataFrame for leakage checks.
        aux: Optional auxiliary records or Polars DataFrame for linkage checks.
        reporter: Optional risk-report callable used by tests.

    Returns:
        The dictionary shape returned by :func:`openmed.risk.risk_report`.
    """

    _ensure_polars_frame(frame)
    selected_columns = (
        _validate_columns(frame, qi_columns) if qi_columns is not None else None
    )
    risk = reporter or _load_risk_report()
    return risk(
        _records_for_risk(frame, selected_columns),
        original=_records_for_risk(original, selected_columns),
        aux=_records_for_risk(aux, selected_columns),
    )


def extract_frame(
    frame: Any,
    column: str,
    *,
    extractor: ClinicalExtractor | None = None,
    extractor_kwargs: dict[str, Any] | None = None,
    systems: Sequence[str] | None = None,
    top_k: int = 1,
    warn_on_phi: bool = True,
) -> Any:
    """Return grounded clinical entity rows for one Polars text column."""

    _ensure_polars_frame(frame)
    _validate_columns(frame, column)
    from openmed.interop.clinical_dataframe import (
        FLAT_TABLE_COLUMNS,
        extract_records,
    )

    rows = extract_records(
        frame.to_dicts(),
        column,
        extractor=extractor,
        extractor_kwargs=extractor_kwargs,
        systems=systems,
        top_k=top_k,
        warn_on_phi=warn_on_phi,
    )
    if rows:
        return pl.DataFrame(rows).select(list(FLAT_TABLE_COLUMNS))
    return pl.DataFrame(
        schema={column_name: pl.Utf8 for column_name in FLAT_TABLE_COLUMNS}
    )


def ground_frame(
    frame: Any,
    column: str,
    *,
    extractor: ClinicalExtractor | None = None,
    extractor_kwargs: dict[str, Any] | None = None,
    systems: Sequence[str] | None = None,
    top_k: int = 1,
    warn_on_phi: bool = True,
) -> Any:
    """Alias for :func:`extract_frame` emphasizing grounded rows."""

    return extract_frame(
        frame,
        column,
        extractor=extractor,
        extractor_kwargs=extractor_kwargs,
        systems=systems,
        top_k=top_k,
        warn_on_phi=warn_on_phi,
    )


def _register_namespace() -> None:
    api = getattr(pl, "api", None)
    register = getattr(api, "register_dataframe_namespace", None)
    if register is None:
        return

    @register("openmed")
    class OpenMedPolarsDataFrameNamespace:
        """OpenMed helpers attached to ``polars.DataFrame.openmed``."""

        def __init__(self, frame: Any) -> None:
            self._frame = frame

        def deidentify(
            self,
            columns: Sequence[str] | str,
            *,
            method: str = "mask",
            policy: str | None = None,
            deidentifier: Deidentifier | None = None,
            **kwargs: Any,
        ) -> Any:
            """Return a redacted copy of this Polars DataFrame."""

            return deidentify_frame(
                self._frame,
                columns,
                method=method,
                policy=policy,
                deidentifier=deidentifier,
                **kwargs,
            )

        def risk_report(
            self,
            qi_columns: Sequence[str] | str | None = None,
            *,
            original: Any | None = None,
            aux: Any | None = None,
            reporter: RiskReporter | None = None,
        ) -> dict[str, Any]:
            """Return the OpenMed risk report for this Polars DataFrame."""

            return risk_report(
                self._frame,
                qi_columns,
                original=original,
                aux=aux,
                reporter=reporter,
            )

        def extract(
            self,
            column: str,
            *,
            extractor: ClinicalExtractor | None = None,
            extractor_kwargs: dict[str, Any] | None = None,
            systems: Sequence[str] | None = None,
            top_k: int = 1,
            warn_on_phi: bool = True,
        ) -> Any:
            """Return grounded clinical entity rows for one text column."""

            return extract_frame(
                self._frame,
                column,
                extractor=extractor,
                extractor_kwargs=extractor_kwargs,
                systems=systems,
                top_k=top_k,
                warn_on_phi=warn_on_phi,
            )

        def ground(
            self,
            column: str,
            *,
            extractor: ClinicalExtractor | None = None,
            extractor_kwargs: dict[str, Any] | None = None,
            systems: Sequence[str] | None = None,
            top_k: int = 1,
            warn_on_phi: bool = True,
        ) -> Any:
            """Alias for :meth:`extract` emphasizing grounded rows."""

            return self.extract(
                column,
                extractor=extractor,
                extractor_kwargs=extractor_kwargs,
                systems=systems,
                top_k=top_k,
                warn_on_phi=warn_on_phi,
            )


def _load_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _load_risk_report() -> RiskReporter:
    from openmed.risk import risk_report as openmed_risk_report

    return openmed_risk_report


def ensure_registered() -> None:
    """Ensure helpers target the currently imported Polars module."""

    global pl

    import polars as current_pl

    pl = current_pl
    if not _namespace_registered():
        _register_namespace()


def _ensure_polars_frame(frame: Any) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")


def _namespace_registered() -> bool:
    try:
        return hasattr(pl.DataFrame(), "openmed")
    except Exception:
        return False


def _validate_columns(frame: Any, columns: Sequence[str] | str) -> tuple[str, ...]:
    selected = _normalize_columns(columns)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")
    return selected


def _normalize_columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(columns, str):
        normalized = (columns,)
    else:
        normalized = tuple(str(column) for column in columns)

    if not normalized:
        raise ValueError("columns must include at least one column name")
    return normalized


def _deidentify_kwargs(
    method: str,
    policy: str | None,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    kwargs = dict(extra_kwargs)
    kwargs["method"] = method
    if policy is not None:
        kwargs["policy"] = policy
    return kwargs


def _redact_value(
    value: Any,
    deidentifier: Deidentifier,
    deidentify_kwargs: dict[str, Any],
) -> Any:
    if not isinstance(value, str) or value == "":
        return value

    result = deidentifier(value, **deidentify_kwargs)
    if isinstance(result, str):
        return result

    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


def _records_for_risk(
    value: Any | None,
    columns: Sequence[str] | None,
) -> Any | None:
    if value is None:
        return None
    if isinstance(value, pl.DataFrame):
        frame = value.select(list(columns)) if columns is not None else value
        return frame.to_dicts()
    return value


_register_namespace()

__all__ = [
    "ClinicalExtractor",
    "Deidentifier",
    "RiskReporter",
    "deidentify_frame",
    "ensure_registered",
    "extract_frame",
    "ground_frame",
    "risk_report",
]
