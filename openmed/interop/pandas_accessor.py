"""Pandas DataFrame accessor for OpenMed clinical table workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openmed.risk import (
        AnonymityPolicy,
        AnonymizationResult,
        ReleaseAssessment,
    )

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
ClinicalExtractor = Callable[..., Any]


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

    def assess_release(self, policy: AnonymityPolicy) -> ReleaseAssessment:
        """Return a PHI-safe aggregate release assessment.

        Unlike :meth:`risk_report`, this method accepts an explicit
        patient/privacy-unit policy and returns only the allow-listed aggregate
        schema from :class:`openmed.risk.ReleaseAssessment`.
        """

        from openmed.risk import assess_release

        return assess_release(_records_for_release(self._obj), policy)

    def anonymize_release(
        self,
        policy: AnonymityPolicy,
        *,
        hierarchies: (Mapping[str, Sequence[Mapping[str, Any]]] | None) = None,
    ) -> AnonymizationResult:
        """Generalize and suppress a release under an explicit policy.

        The returned :class:`openmed.risk.AnonymizationResult` keeps transformed
        rows only in ``records``. Its ``to_safe_dict`` and ``to_safe_json``
        methods remain aggregate-only.
        """

        from openmed.risk import anonymize_release

        return anonymize_release(
            _records_for_release(self._obj),
            policy,
            hierarchies=hierarchies,
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
        """Return a long-form clinical entity DataFrame for ``column``.

        The returned columns are ordered exactly like
        :data:`openmed.clinical.exporters.flat_table.FLAT_TABLE_COLUMNS`, so
        pandas/polars/DuckDB rows remain column-consistent with CSV/FHIR export
        paths. ``extractor`` can be injected for model-backed pipelines; the
        default extractor is local and performs no network calls.
        """

        _validate_columns(self._obj, column)
        from openmed.interop.clinical_dataframe import (
            FLAT_TABLE_COLUMNS,
            extract_records,
        )

        rows = extract_records(
            self._obj.to_dict("records"),
            column,
            extractor=extractor,
            extractor_kwargs=extractor_kwargs,
            systems=systems,
            top_k=top_k,
            warn_on_phi=warn_on_phi,
        )
        return pd.DataFrame(rows, columns=list(FLAT_TABLE_COLUMNS))

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
        """Alias for :meth:`extract` emphasizing grounded entities."""

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
    from openmed.risk import risk_report

    return risk_report


def ensure_registered() -> None:
    """Ensure the accessor is registered on the currently imported pandas."""

    global pd, register_dataframe_accessor

    import pandas as current_pd
    from pandas.api.extensions import (
        register_dataframe_accessor as current_register_dataframe_accessor,
    )

    pd = current_pd
    register_dataframe_accessor = current_register_dataframe_accessor
    if "openmed" not in getattr(current_pd.DataFrame, "_accessors", set()):
        current_register_dataframe_accessor("openmed")(OpenMedDataFrameAccessor)


def _validate_columns(frame: Any, columns: Sequence[str] | str) -> tuple[str, ...]:
    selected = _normalize_columns(columns)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns: {', '.join(missing)}")
    return selected


def _normalize_columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    normalized: tuple[str, ...]
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


def _records_for_release(frame: Any) -> list[dict[Any, Any]]:
    columns = list(frame.columns)
    if any(type(field) is not str for field in columns):
        raise TypeError("DataFrame column names must be strings")
    if len(columns) != len(set(columns)):
        raise ValueError("DataFrame column names must be unique")
    records = frame.to_dict("records")
    return [
        {field: _release_scalar(value) for field, value in record.items()}
        for record in records
    ]


def _release_scalar(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        if value.nanosecond:
            raise ValueError(
                "Pandas timestamps with sub-microsecond precision are unsupported"
            )
        return value.to_pydatetime()
    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy":
        if value_type.__name__ == "datetime64":
            timestamp = pd.Timestamp(value)
            if timestamp is pd.NaT:
                return None
            if timestamp.nanosecond:
                raise ValueError(
                    "NumPy timestamps with sub-microsecond precision are unsupported"
                )
            return timestamp.to_pydatetime()
        if value_type.__name__ == "timedelta64":
            raise TypeError("NumPy time durations are unsupported release scalars")
        item = getattr(value, "item", None)
        if callable(item):
            return item()
    return value


__all__ = [
    "ClinicalExtractor",
    "Deidentifier",
    "OpenMedDataFrameAccessor",
    "RiskReporter",
    "ensure_registered",
]
