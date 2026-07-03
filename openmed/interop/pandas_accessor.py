"""Pandas DataFrame accessor for OpenMed de-identification workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

try:
    import pandas as pd
    from pandas.api.extensions import register_dataframe_accessor
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "Pandas accessor support requires the 'pandas' extra. "
        "Install with `pip install openmed[pandas]`."
    ) from exc

Deidentifier = Callable[..., Any]
RiskReporter = Callable[..., dict[str, Any]]


@register_dataframe_accessor("openmed")
class OpenMedDataFrameAccessor:
    """OpenMed helpers attached to ``pandas.DataFrame.openmed``."""

    def __init__(self, pandas_obj: Any) -> None:
        self._obj = pandas_obj

    def deidentify(
        self,
        columns: Sequence[str] | str,
        *,
        method: str = "mask",
        policy: str | None = None,
        deidentifier: Deidentifier | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a redacted DataFrame copy for selected free-text columns.

        Args:
            columns: Free-text column names to redact.
            method: De-identification method forwarded to
                :func:`openmed.core.pii.deidentify`.
            policy: Optional policy profile forwarded to de-identification.
            deidentifier: Optional callable used primarily by tests and custom
                embedding contexts.
            **kwargs: Additional keyword arguments forwarded to de-identification.

        Returns:
            A new ``pandas.DataFrame`` with selected string cells redacted.
        """

        selected_columns = _validate_columns(self._obj, columns)
        redacted = self._obj.copy(deep=True)
        redact = deidentifier or _load_deidentifier()
        deidentify_kwargs = _deidentify_kwargs(method, policy, kwargs)

        for column in selected_columns:
            redacted[column] = redacted[column].map(
                lambda value: _redact_value(value, redact, deidentify_kwargs)
            )

        return redacted

    def risk_report(
        self,
        qi_columns: Sequence[str] | str | None = None,
        *,
        original: Any | None = None,
        aux: Any | None = None,
        reporter: RiskReporter | None = None,
    ) -> dict[str, Any]:
        """Return the OpenMed re-identification risk shape for table records.

        Args:
            qi_columns: Optional quasi-identifier columns to include. When
                omitted, all columns are passed to the risk scorer.
            original: Optional original records or DataFrame for leakage checks.
            aux: Optional auxiliary records or DataFrame for linkage checks.
            reporter: Optional risk-report callable used by tests.

        Returns:
            The dictionary shape returned by :func:`openmed.risk.risk_report`.
        """

        risk = reporter or _load_risk_report()
        selected_columns = (
            _validate_columns(self._obj, qi_columns) if qi_columns is not None else None
        )
        return risk(
            _records_for_risk(self._obj, selected_columns),
            original=_records_for_risk(original, selected_columns),
            aux=_records_for_risk(aux, selected_columns),
        )


def _load_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _load_risk_report() -> RiskReporter:
    from openmed.risk import risk_report

    return risk_report


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
    if not isinstance(value, str) or value == "" or _is_missing(value):
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


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _records_for_risk(
    value: Any | None,
    columns: Sequence[str] | None,
) -> Any | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        frame = value.loc[:, list(columns)] if columns is not None else value
        return frame.to_dict("records")
    return value


__all__ = [
    "Deidentifier",
    "OpenMedDataFrameAccessor",
    "RiskReporter",
]
