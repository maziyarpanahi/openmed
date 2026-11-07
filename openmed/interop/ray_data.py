"""Ray Data actor operator for batch column de-identification.

``ray`` is imported only when :func:`deidentify_dataset` is called. Importing
this module, :mod:`openmed.interop`, or :mod:`openmed` never imports Ray.

Actor model-loading lifecycle: Ray constructs one :class:`DeidentifyBatch`
instance per worker actor. The constructor creates one OpenMed ``ModelLoader``;
the first batch loads the requested model into that loader and later batches on
the same actor reuse its cached model and pipeline state. No loaded model is
captured on the driver or serialized between actors.

Raw clinical text must never be logged or interpolated into exceptions here.
Ray can propagate actor failures to driver logs, so errors identify only the
batch contract or column name.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module as _import_module
from typing import Any

_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_COLUMN = "text"


class DeidentifyBatch:
    """De-identify one named column in each Ray Data batch.

    Ray Data treats callable classes as stateful actor transforms. Constructing
    this class once per actor gives that actor one model loader, while repeated
    calls reuse the loader's cached model instead of loading per batch or row.

    Args:
        policy: Policy profile forwarded to OpenMed de-identification.
        column: Name of the text column to transform.
        deidentifier: Optional callable override, primarily for offline tests.
        loader: Optional preconstructed model loader. When omitted, the default
            OpenMed loader is created once per actor unless ``deidentifier`` is
            supplied.
        **deidentify_kwargs: Additional keyword arguments forwarded to the
            de-identification callable for every non-null value.
    """

    def __init__(
        self,
        policy: str = _DEFAULT_POLICY,
        column: str = _DEFAULT_COLUMN,
        *,
        deidentifier: Any = None,
        loader: Any = None,
        **deidentify_kwargs: Any,
    ) -> None:
        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string")

        self.policy = policy
        self.column = column
        self._deidentify_kwargs = dict(deidentify_kwargs)
        self._deidentifier = (
            _default_deidentifier() if deidentifier is None else deidentifier
        )
        if loader is not None:
            self._loader = loader
        elif deidentifier is None:
            self._loader = _default_model_loader()
        else:
            self._loader = None

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Return a batch copy with the configured text column de-identified.

        The input column is never modified in place because Ray may provide a
        read-only zero-copy view into its object store.

        Args:
            batch: Mapping of column names to equal-length batch columns.

        Returns:
            A shallow batch copy whose configured column contains redacted text.

        Raises:
            TypeError: If ``batch`` is not a mapping or the selected column is
                not iterable.
            KeyError: If the selected column is absent.
        """

        if not isinstance(batch, Mapping):
            raise TypeError("Ray Data batch must be a mapping of columns")
        if self.column not in batch:
            raise KeyError(f"Ray Data batch is missing column {self.column!r}")

        source_column = batch[self.column]
        try:
            values = list(source_column)
        except TypeError as exc:
            raise TypeError(
                f"Ray Data column {self.column!r} must be iterable"
            ) from exc

        output = dict(batch)
        output[self.column] = _copy_column_shape(
            source_column,
            [self._deidentify_value(value) for value in values],
        )
        return output

    def _deidentify_value(self, value: Any) -> str | None:
        if value is None:
            return None

        kwargs = dict(self._deidentify_kwargs)
        if self._loader is not None:
            kwargs["loader"] = self._loader
        result = self._deidentifier(value, policy=self.policy, **kwargs)
        return _result_text(result)


def deidentify_dataset(
    ds: Any,
    column: str,
    policy: str = _DEFAULT_POLICY,
    **deidentify_kwargs: Any,
) -> Any:
    """Return ``ds`` with one text column de-identified by Ray worker actors.

    Each actor constructs :class:`DeidentifyBatch` once and therefore owns one
    reusable OpenMed model loader. The driver passes only lightweight
    constructor settings to actors; it never loads or serializes the model.

    Args:
        ds: A Ray Data ``Dataset``.
        column: Name of the text column to de-identify.
        policy: OpenMed policy profile applied to every non-null value.
        **deidentify_kwargs: Additional options forwarded to OpenMed's
            de-identification callable inside each actor.

    Returns:
        The transformed Ray Data ``Dataset`` returned by ``map_batches``.

    Raises:
        ImportError: If the optional Ray dependency is unavailable.
    """

    ray_data = _load_ray_data()
    constructor_kwargs = {
        "column": column,
        "policy": policy,
        **deidentify_kwargs,
    }
    return ds.map_batches(
        DeidentifyBatch,
        batch_format="numpy",
        compute=ray_data.ActorPoolStrategy(),
        fn_constructor_kwargs=constructor_kwargs,
        zero_copy_batch=True,
    )


def _copy_column_shape(source: Any, values: list[str | None]) -> Any:
    if isinstance(source, list):
        return values
    if isinstance(source, tuple):
        return tuple(values)
    if type(source).__module__.partition(".")[0] == "numpy":
        numpy = _import_module("numpy")
        if isinstance(source, numpy.ndarray):
            return numpy.asarray(values)
    return values


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


def _default_deidentifier() -> Any:
    from openmed.core.pii import deidentify

    return deidentify


def _default_model_loader() -> Any:
    from openmed.core import ModelLoader

    return ModelLoader()


def _load_ray_data() -> Any:
    try:
        return _import_module("ray.data")
    except ImportError as exc:
        raise ImportError(
            "Ray Data support requires the optional dependency; install "
            "openmed[ray] to use openmed.interop.ray_data"
        ) from exc


__all__ = ["DeidentifyBatch", "deidentify_dataset"]
