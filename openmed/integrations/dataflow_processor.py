"""Apache Beam/Dataflow record de-identification with bundle-scoped setup.

The transform batches records without deriving keys from their contents. This
keeps PHI and quasi-identifiers out of shuffle keys while allowing one resident
OpenMed batch processor to be reused by each Beam ``DoFn`` instance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import apache_beam as beam
except ImportError as exc:  # pragma: no cover - exercised by packaging users
    raise ImportError(
        "Apache Beam support requires the 'dataflow' extra. "
        "Install with `pip install openmed[dataflow]`."
    ) from exc

from openmed.core.policy import PolicyName, canonical_policy_name

DEFAULT_DATAFLOW_BATCH_SIZE = 32
DEFAULT_DATAFLOW_POLICY = "hipaa_safe_harbor"
CLEANED_TAG = "cleaned"
DEAD_LETTER_TAG = "dead_letter"

ProcessBatchCallable = Callable[..., Any]
ProcessorFactory = Callable[..., Any]


@dataclass(frozen=True)
class DataflowDeadLetter:
    """An element that could not be safely de-identified.

    ``reason`` is a stable, PHI-free category. The original ``element`` is
    retained so callers can route it to a protected dead-letter sink.

    Attributes:
        element: Original input element.
        reason: PHI-free failure category.
    """

    element: Any
    reason: str


@dataclass(frozen=True)
class _PreparedRecord:
    original: Any
    output: dict[str, Any]
    text: str


class _RedactBatchDoFn(beam.DoFn):
    """Redact one Beam-provided batch with a resident OpenMed processor."""

    def __init__(
        self,
        *,
        text_field: str,
        policy: str | None,
        method: str,
        model_name: str,
        batch_size: int,
        processor_factory: ProcessorFactory | None,
        process_batch_fn: ProcessBatchCallable | None,
        deidentify_kwargs: Mapping[str, Any],
    ) -> None:
        self.text_field = text_field
        self.policy = policy
        self.method = method
        self.model_name = model_name
        self.batch_size = batch_size
        self.processor_factory = processor_factory
        self.configured_process_batch = process_batch_fn
        self.deidentify_kwargs = dict(deidentify_kwargs)
        self._process_batch: ProcessBatchCallable | None = None
        self._forward_options = False

    def setup(self) -> None:
        """Create one resident processor for this worker-side ``DoFn``."""

        if self.configured_process_batch is not None:
            self._process_batch = self.configured_process_batch
            self._forward_options = True
            return

        factory = self.processor_factory
        if factory is None:
            from openmed.processing import BatchProcessor

            factory = BatchProcessor

        processor = factory(**self._processor_options())
        process_texts = getattr(processor, "process_texts", None)
        if callable(process_texts):
            self._process_batch = process_texts
        elif callable(processor):
            self._process_batch = processor
        else:
            raise TypeError(
                "processor_factory must return a callable or expose process_texts"
            )

    def process(self, batch: Sequence[Any]):
        """Yield cleaned records and tagged dead letters for one input batch."""

        prepared: list[_PreparedRecord] = []
        for element in batch:
            try:
                prepared_record, failure = self._prepare(element)
            except Exception:
                prepared_record = None
                failure = DataflowDeadLetter(element, "invalid_element")
            if failure is not None:
                yield beam.pvalue.TaggedOutput(DEAD_LETTER_TAG, failure)
            elif prepared_record is None:
                yield dict(element)
            else:
                prepared.append(prepared_record)

        if not prepared:
            return

        try:
            batch_result = self._run_batch([record.text for record in prepared])
            items = _batch_items(batch_result)
        except Exception:
            for record in prepared:
                yield beam.pvalue.TaggedOutput(
                    DEAD_LETTER_TAG,
                    DataflowDeadLetter(record.original, "processing_error"),
                )
            return

        if len(items) != len(prepared):
            for record in prepared:
                yield beam.pvalue.TaggedOutput(
                    DEAD_LETTER_TAG,
                    DataflowDeadLetter(record.original, "invalid_result_count"),
                )
            return

        for record, item in zip(prepared, items):
            try:
                redacted_text, reason = _deidentified_text(item)
            except Exception:
                redacted_text, reason = "", "invalid_result"
            if reason is not None:
                yield beam.pvalue.TaggedOutput(
                    DEAD_LETTER_TAG,
                    DataflowDeadLetter(record.original, reason),
                )
                continue
            record.output[self.text_field] = redacted_text
            yield record.output

    def _prepare(
        self, element: Any
    ) -> tuple[_PreparedRecord | None, DataflowDeadLetter | None]:
        if element is None or not isinstance(element, Mapping):
            return None, DataflowDeadLetter(element, "invalid_element")
        if self.text_field not in element:
            return None, DataflowDeadLetter(element, "missing_text_field")

        value = element[self.text_field]
        if value is None:
            return None, DataflowDeadLetter(element, "null_text")
        if not isinstance(value, str):
            return None, DataflowDeadLetter(element, "invalid_text_type")
        if not value:
            return None, None
        return _PreparedRecord(element, dict(element), value), None

    def _run_batch(self, texts: list[str]) -> Any:
        process_batch = self._process_batch
        if process_batch is None:  # pragma: no cover - Beam lifecycle invariant
            raise RuntimeError("dataflow processor setup was not called")
        if self._forward_options:
            return process_batch(texts, **self._processor_options())
        return process_batch(texts)

    def _processor_options(self) -> dict[str, Any]:
        options = {
            "model_name": self.model_name,
            "operation": "deidentify",
            "batch_size": self.batch_size,
            "continue_on_error": True,
            "method": self.method,
            **self.deidentify_kwargs,
        }
        if self.policy is not None:
            options["policy"] = self.policy
        return options


class DataflowBatchProcessor(beam.PTransform):
    """Batch and de-identify a configured field in Beam record mappings.

    The transform returns a tagged output tuple. Access cleaned records through
    ``outputs.cleaned`` and invalid records through ``outputs.dead_letter``.
    Batches are formed with ``BatchElements`` and never use record contents as
    keys, so PHI and quasi-identifiers are not exposed as partition keys.

    Args:
        text_field: Mapping field containing text to de-identify.
        policy: Optional OpenMed policy profile.
        method: De-identification method forwarded to OpenMed.
        model_name: Model registry key or artifact identifier.
        batch_size: Maximum records passed to one batch inference call.
        processor_factory: Optional factory called once from ``DoFn.setup``.
            It receives the OpenMed batch processor options and must return a
            callable or an object exposing ``process_texts``.
        process_batch_fn: Optional replacement for
            :func:`openmed.processing.process_batch`, primarily for tests.
        **deidentify_kwargs: Additional options forwarded to OpenMed.
    """

    def __init__(
        self,
        text_field: str = "text",
        *,
        policy: str | PolicyName | None = DEFAULT_DATAFLOW_POLICY,
        method: str = "mask",
        model_name: str = "disease_detection_superclinical",
        batch_size: int = DEFAULT_DATAFLOW_BATCH_SIZE,
        processor_factory: ProcessorFactory | None = None,
        process_batch_fn: ProcessBatchCallable | None = None,
        **deidentify_kwargs: Any,
    ) -> None:
        self.text_field = _non_empty_string(text_field, "text_field")
        self.policy = canonical_policy_name(policy) if policy is not None else None
        self.method = _non_empty_string(method, "method")
        self.model_name = _non_empty_string(model_name, "model_name")
        self.batch_size = _positive_int(batch_size, "batch_size")
        if processor_factory is not None and not callable(processor_factory):
            raise TypeError("processor_factory must be callable")
        if process_batch_fn is not None and not callable(process_batch_fn):
            raise TypeError("process_batch_fn must be callable")
        if processor_factory is not None and process_batch_fn is not None:
            raise ValueError("pass only one of processor_factory or process_batch_fn")

        kwargs = dict(deidentify_kwargs)
        for reserved in (
            "batch_size",
            "continue_on_error",
            "ids",
            "method",
            "model_name",
            "operation",
            "policy",
            "texts",
        ):
            if reserved in kwargs:
                raise ValueError(
                    f"deidentify kwargs must not include reserved key {reserved!r}"
                )
        self.processor_factory = processor_factory
        self.process_batch_fn = process_batch_fn
        self.deidentify_kwargs = kwargs

    def expand(self, records):
        """Apply bundle-friendly batching and record de-identification."""

        batches = records | "Batch records without QI keys" >> beam.BatchElements(
            min_batch_size=self.batch_size,
            max_batch_size=self.batch_size,
        )
        return batches | "De-identify record batches" >> beam.ParDo(
            _RedactBatchDoFn(
                text_field=self.text_field,
                policy=self.policy,
                method=self.method,
                model_name=self.model_name,
                batch_size=self.batch_size,
                processor_factory=self.processor_factory,
                process_batch_fn=self.process_batch_fn,
                deidentify_kwargs=self.deidentify_kwargs,
            )
        ).with_outputs(DEAD_LETTER_TAG, main=CLEANED_TAG)


DataflowProcessor = DataflowBatchProcessor


def _batch_items(batch_result: Any) -> list[Any]:
    raw_items = getattr(batch_result, "items", batch_result)
    try:
        return list(raw_items)
    except TypeError as exc:
        raise TypeError("batch processor returned a non-iterable result") from exc


def _deidentified_text(item: Any) -> tuple[str, str | None]:
    if getattr(item, "success", True) is False:
        return "", "processing_error"

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result, None
    if isinstance(result, Mapping):
        text = result.get("deidentified_text")
    else:
        text = getattr(result, "deidentified_text", None)
    if not isinstance(text, str):
        return "", "invalid_result"
    return text, None


def _non_empty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "CLEANED_TAG",
    "DEAD_LETTER_TAG",
    "DEFAULT_DATAFLOW_BATCH_SIZE",
    "DEFAULT_DATAFLOW_POLICY",
    "DataflowBatchProcessor",
    "DataflowDeadLetter",
    "DataflowProcessor",
    "ProcessBatchCallable",
    "ProcessorFactory",
]
