"""Apache Beam transform for worker-local text de-identification.

The Beam SDK is optional and is imported only when this adapter module is
loaded. Importing :mod:`openmed` or :mod:`openmed.interop` therefore never
requires ``apache_beam``. When Beam is installed, :class:`DeidentifyText` is a
regular ``beam.PTransform`` backed by a ``beam.DoFn``.

Each worker constructs one OpenMed ``ModelLoader`` in ``DoFn.setup()`` and
reuses it for every element processed by that ``DoFn`` instance. Raw input is
never included in log messages or adapter-generated exceptions because Beam
worker failures may be forwarded to centralized runner logs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module as _import_module
from typing import Any

try:
    _beam = _import_module("apache_beam")
except ModuleNotFoundError as exc:
    if exc.name != "apache_beam":  # pragma: no cover - broken Beam installation
        raise
    _beam = None

_DoFnBase = _beam.DoFn if _beam is not None else object
_PTransformBase = _beam.PTransform if _beam is not None else object


class _DeidentifyTextDoFn(_DoFnBase):  # type: ignore[misc,valid-type]
    """De-identify string elements or one text field in dictionary records."""

    def __init__(
        self,
        *,
        text_field: str = "text",
        policy: str = "hipaa_safe_harbor",
        deidentifier: Callable[..., Any] | None = None,
        loader_factory: Callable[[], Any] | None = None,
        **deidentify_kwargs: Any,
    ) -> None:
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string")
        if "loader" in deidentify_kwargs:
            raise TypeError("loader is managed by DeidentifyText.setup()")

        self._text_field = text_field
        self._policy = policy
        self._deidentifier = deidentifier
        self._loader_factory = loader_factory
        self._deidentify_kwargs = dict(deidentify_kwargs)
        self._loader: Any = None
        self._setup_complete = False

    def setup(self) -> None:
        """Create the worker-local model loader exactly once."""

        if self._setup_complete:
            return
        factory = self._loader_factory or _new_model_loader
        self._loader = factory()
        self._setup_complete = True

    def process(self, element: str | dict[str, Any]) -> Iterator[Any]:
        """Yield one de-identified element with the same outer shape."""

        if not self._setup_complete:
            raise RuntimeError("DeidentifyText.setup() must run before process()")

        if isinstance(element, str):
            text = element
        elif isinstance(element, dict):
            if self._text_field not in element:
                raise KeyError(f"record is missing text field {self._text_field!r}")
            text = element[self._text_field]
            if not isinstance(text, str):
                raise TypeError(
                    f"record field {self._text_field!r} must contain a string"
                )
        else:
            raise TypeError("DeidentifyText elements must be strings or dictionaries")

        deidentifier = self._deidentifier or _default_deidentifier()
        result = deidentifier(
            text,
            policy=self._policy,
            loader=self._loader,
            **self._deidentify_kwargs,
        )
        redacted = _result_text(result)

        if isinstance(element, str):
            yield redacted
            return

        redacted_record = dict(element)
        redacted_record[self._text_field] = redacted
        yield redacted_record


class DeidentifyText(_PTransformBase):  # type: ignore[misc,valid-type]
    """De-identify text elements in an Apache Beam ``PCollection``.

    Args:
        text_field: Dictionary field containing text. Plain-string elements
            ignore this setting.
        policy: OpenMed de-identification policy name.
        **deidentify_kwargs: Additional keyword arguments forwarded to
            :func:`openmed.deidentify` for every element. ``loader`` is managed
            by the transform and cannot be supplied.
    """

    def __init__(
        self,
        *,
        text_field: str = "text",
        policy: str = "hipaa_safe_harbor",
        **deidentify_kwargs: Any,
    ) -> None:
        super().__init__()
        self._text_field = text_field
        self._policy = policy
        self._deidentify_kwargs = dict(deidentify_kwargs)

    def expand(self, pcoll: Any) -> Any:
        """Apply worker-local de-identification to ``pcoll``."""

        beam = _require_beam()
        return pcoll | beam.ParDo(
            _DeidentifyTextDoFn(
                text_field=self._text_field,
                policy=self._policy,
                **self._deidentify_kwargs,
            )
        )


def _require_beam() -> Any:
    if _beam is None:
        raise ImportError(
            "Apache Beam support requires the optional dependency; install "
            "openmed[beam] to use openmed.interop.beam_transform"
        )
    return _beam


def _new_model_loader() -> Any:
    from openmed import ModelLoader

    return ModelLoader()


def _default_deidentifier() -> Callable[..., Any]:
    from openmed import deidentify

    return deidentify


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


__all__ = ["DeidentifyText"]
