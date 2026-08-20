"""Bounded, schema-preserving redaction for columnar trace batches.

The adapter works on :class:`pyarrow.RecordBatch` values and accepts an
injected text redactor.  PyArrow is optional and imported only when an adapter
function is called.  No model, network client, logger, or report is created by
this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

TextRedactor: TypeAlias = Callable[[str], str]
FieldPath: TypeAlias = tuple[str, ...]
FieldSelection: TypeAlias = str | Sequence[str]

DEFAULT_BATCH_SIZE = 1024
DEFAULT_REDACTION_TEXT = "[REDACTED]"

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_REDACTION_TEXT",
    "ColumnarTraceAdapter",
    "ColumnarTraceAdapterError",
    "ColumnarTraceSchemaAdapter",
    "FieldPath",
    "FieldSelection",
    "TextRedactor",
    "adapt_record_batch",
    "adapt_trace_batches",
    "default_text_redactor",
    "iter_redacted_record_batches",
    "redact_record_batch",
    "redact_trace_batches",
]


class ColumnarTraceAdapterError(ValueError):
    """Raised when a trace batch cannot be adapted safely."""


def default_text_redactor(value: str) -> str:
    """Return the deterministic, safe default replacement for one text value."""

    del value
    return DEFAULT_REDACTION_TEXT


def redact_record_batch(
    batch: Any,
    *,
    text_columns: Sequence[FieldSelection],
    text_redactor: TextRedactor | None = None,
    redactor: TextRedactor | None = None,
) -> Any:
    """Redact configured text leaves in one Arrow record batch.

    ``text_columns`` contains top-level column names or dotted paths through
    nested struct fields.  List, large-list, fixed-size-list, and map values
    can also contain a selected text leaf.  Only the selected arrays are
    rebuilt; the returned batch uses the input schema verbatim, including
    field metadata, nullability, and logical types.

    Args:
        batch: A ``pyarrow.RecordBatch``.
        text_columns: Text column names or nested field paths to redact.
        text_redactor: Deterministic callable receiving one non-empty string
            and returning its replacement.  When omitted, the safe built-in
            replacement ``"[REDACTED]"`` is used.
        redactor: Compatibility alias for ``text_redactor``.

    Returns:
        A new ``pyarrow.RecordBatch`` with the same schema and row count.

    Raises:
        ColumnarTraceAdapterError: If a selected path is missing, does not
            resolve to text values, or the redactor fails.
        ImportError: If PyArrow is not installed.
    """

    pyarrow = _import_pyarrow()
    _require_record_batch(batch, pyarrow)
    paths = _validate_paths(batch.schema, text_columns, pyarrow)
    resolved_redactor = _resolve_redactor(text_redactor, redactor)

    paths_by_column: dict[str, list[FieldPath]] = {}
    for path in paths:
        paths_by_column.setdefault(path[0], []).append(path)

    arrays = list(batch.columns)
    for column_index, column_name in enumerate(batch.schema.names):
        selected_paths = paths_by_column.get(column_name)
        if not selected_paths:
            continue
        rewritten = arrays[column_index]
        for path in selected_paths:
            rewritten = _rewrite_array(
                rewritten,
                path[1:],
                resolved_redactor,
                pyarrow,
            )
        arrays[column_index] = rewritten

    try:
        return pyarrow.RecordBatch.from_arrays(arrays, schema=batch.schema)
    except ColumnarTraceAdapterError:
        raise
    except Exception:
        raise ColumnarTraceAdapterError(
            "The adapted record batch could not preserve its input schema"
        ) from None


def iter_redacted_record_batches(
    source: Any,
    *,
    text_columns: Sequence[FieldSelection],
    text_redactor: TextRedactor | None = None,
    redactor: TextRedactor | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Any]:
    """Yield redacted record batches without materializing the source.

    ``source`` may be one record batch, an Arrow table, a record-batch reader,
    an Arrow dataset, or any iterable yielding record batches.  Incoming
    batches larger than ``batch_size`` are sliced before redaction.  Slicing is
    zero-copy in Arrow, so only one bounded batch is presented at a time.

    The source is not consumed until the returned iterator is advanced.  The
    text redactor is called in stable schema/path order and receives no label
    or unselected-column values.
    """

    _validate_batch_size(batch_size)
    pyarrow = _import_pyarrow()
    paths = _normalize_field_paths(text_columns)
    resolved_redactor = _resolve_redactor(text_redactor, redactor)
    return _iter_batches(
        source,
        paths,
        resolved_redactor,
        batch_size,
        pyarrow,
    )


@dataclass(frozen=True, init=False)
class ColumnarTraceSchemaAdapter:
    """Reusable configuration for bounded trace-batch adaptation.

    The adapter has no model-loading behavior.  Supply a deterministic
    ``text_redactor`` for domain-specific masking, or use the safe default.
    """

    text_columns: tuple[FieldPath, ...] = field(repr=False)
    text_redactor: TextRedactor | None = field(repr=False)
    batch_size: int

    def __init__(
        self,
        text_columns: Sequence[FieldSelection],
        text_redactor: TextRedactor | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        redactor: TextRedactor | None = None,
    ) -> None:
        if text_redactor is not None and redactor is not None:
            raise ValueError("Pass only one text redactor")
        _validate_batch_size(batch_size)
        object.__setattr__(self, "text_columns", _normalize_field_paths(text_columns))
        object.__setattr__(
            self,
            "text_redactor",
            text_redactor if text_redactor is not None else redactor,
        )
        object.__setattr__(self, "batch_size", batch_size)

    def adapt_batch(self, batch: Any) -> Any:
        """Redact one record batch while preserving its exact schema."""

        return redact_record_batch(
            batch,
            text_columns=self.text_columns,
            text_redactor=self.text_redactor,
        )

    def adapt_batches(self, source: Any) -> Iterator[Any]:
        """Return a lazy iterator of bounded redacted record batches."""

        return iter_redacted_record_batches(
            source,
            text_columns=self.text_columns,
            text_redactor=self.text_redactor,
            batch_size=self.batch_size,
        )

    def __call__(self, source: Any) -> Iterator[Any]:
        """Adapt a batch source lazily."""

        return self.adapt_batches(source)


ColumnarTraceAdapter = ColumnarTraceSchemaAdapter
adapt_record_batch = redact_record_batch
adapt_trace_batches = iter_redacted_record_batches
redact_trace_batches = iter_redacted_record_batches


def _iter_batches(
    source: Any,
    paths: Sequence[FieldPath],
    text_redactor: TextRedactor,
    batch_size: int,
    pyarrow: Any,
) -> Iterator[Any]:
    for batch in _source_batches(source, batch_size, pyarrow):
        if batch.num_rows == 0:
            yield redact_record_batch(
                batch,
                text_columns=paths,
                text_redactor=text_redactor,
            )
            continue
        for offset in range(0, batch.num_rows, batch_size):
            bounded = batch.slice(offset, batch_size)
            yield redact_record_batch(
                bounded,
                text_columns=paths,
                text_redactor=text_redactor,
            )


def _source_batches(source: Any, batch_size: int, pyarrow: Any) -> Iterator[Any]:
    if isinstance(source, pyarrow.RecordBatch):
        candidates: Iterable[Any] = (source,)
    elif isinstance(source, pyarrow.Table):
        candidates = source.to_batches(max_chunksize=batch_size)
    else:
        try:
            scanner_factory = getattr(source, "scanner", None)
        except Exception:
            raise ColumnarTraceAdapterError(
                "The columnar source could not be inspected safely"
            ) from None
        if callable(scanner_factory):
            try:
                scanner = scanner_factory(batch_size=batch_size)
            except TypeError:
                try:
                    scanner = scanner_factory()
                except Exception:
                    raise ColumnarTraceAdapterError(
                        "The columnar source scanner could not be created"
                    ) from None
            except Exception:
                raise ColumnarTraceAdapterError(
                    "The columnar source scanner could not be created"
                ) from None
            try:
                candidates = scanner.to_batches()
            except Exception:
                raise ColumnarTraceAdapterError(
                    "The columnar source scanner could not produce batches"
                ) from None
        else:
            try:
                candidates = iter(source)
            except TypeError:
                raise TypeError(
                    "source must be a RecordBatch, Table, dataset, or iterable"
                ) from None
            except Exception:
                raise ColumnarTraceAdapterError(
                    "The columnar source could not be opened safely"
                ) from None

    try:
        iterator = iter(candidates)
    except TypeError:
        raise TypeError(
            "source must be a RecordBatch, Table, dataset, or iterable"
        ) from None
    except Exception:
        raise ColumnarTraceAdapterError(
            "The columnar source could not be opened safely"
        ) from None

    while True:
        try:
            candidate = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ColumnarTraceAdapterError(
                "The columnar source could not yield a record batch"
            ) from None
        if isinstance(candidate, pyarrow.Table):
            try:
                yield from candidate.to_batches(max_chunksize=batch_size)
            except Exception:
                raise ColumnarTraceAdapterError(
                    "A source table could not be split into record batches"
                ) from None
        elif isinstance(candidate, pyarrow.RecordBatch):
            yield candidate
        else:
            raise TypeError("source must yield pyarrow.RecordBatch values")


def _rewrite_array(
    array: Any,
    path: FieldPath,
    text_redactor: TextRedactor,
    pyarrow: Any,
) -> Any:
    if not path:
        return _rewrite_selected_array(array, text_redactor, pyarrow)

    array_type = array.type
    if pyarrow.types.is_struct(array_type):
        field_index = array_type.get_field_index(path[0])
        if field_index < 0:
            raise ColumnarTraceAdapterError(
                f"Selected nested field is missing: {_safe_path_label(path)}"
            )
        children = [array.field(index) for index in range(array_type.num_fields)]
        children[field_index] = _rewrite_array(
            children[field_index],
            path[1:],
            text_redactor,
            pyarrow,
        )
        try:
            return pyarrow.StructArray.from_arrays(
                children,
                mask=array.is_null(),
                type=array_type,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A nested struct could not be rebuilt without schema coercion"
            ) from None

    if pyarrow.types.is_list(array_type) or pyarrow.types.is_large_list(array_type):
        offsets, values, mask = _list_parts(array, pyarrow)
        rewritten_values = _rewrite_array(
            values,
            path,
            text_redactor,
            pyarrow,
        )
        try:
            constructor = (
                pyarrow.LargeListArray
                if pyarrow.types.is_large_list(array_type)
                else pyarrow.ListArray
            )
            return constructor.from_arrays(
                offsets,
                rewritten_values,
                type=array_type,
                mask=mask,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A nested list could not be rebuilt without schema coercion"
            ) from None

    if pyarrow.types.is_fixed_size_list(array_type):
        list_size = array_type.list_size
        start = array.offset * list_size
        values = array.values.slice(start, len(array) * list_size)
        rewritten_values = _rewrite_array(
            values,
            path,
            text_redactor,
            pyarrow,
        )
        try:
            return pyarrow.FixedSizeListArray.from_arrays(
                rewritten_values,
                list_size=list_size,
                mask=array.is_null(),
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A fixed-size list could not be rebuilt without schema coercion"
            ) from None

    if pyarrow.types.is_map(array_type):
        offsets, keys, items, mask = _map_parts(array, pyarrow)
        rewritten_items = _rewrite_array(
            items,
            path,
            text_redactor,
            pyarrow,
        )
        try:
            return pyarrow.MapArray.from_arrays(
                offsets,
                keys,
                rewritten_items,
                type=array_type,
                mask=mask,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A nested map could not be rebuilt without schema coercion"
            ) from None

    raise ColumnarTraceAdapterError(
        "Selected path does not resolve through a supported nested Arrow type"
    )


def _rewrite_selected_array(
    array: Any, text_redactor: TextRedactor, pyarrow: Any
) -> Any:
    array_type = array.type
    if pyarrow.types.is_null(array_type):
        return array

    if pyarrow.types.is_dictionary(array_type):
        dictionary = _rewrite_text_array(array.dictionary, text_redactor, pyarrow)
        if dictionary is array.dictionary:
            return array
        try:
            return pyarrow.DictionaryArray.from_arrays(
                array.indices,
                dictionary,
                ordered=array_type.ordered,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A dictionary text column could not preserve its logical type"
            ) from None

    if pyarrow.types.is_list(array_type) or pyarrow.types.is_large_list(array_type):
        offsets, values, mask = _list_parts(array, pyarrow)
        rewritten_values = _rewrite_selected_array(
            values,
            text_redactor,
            pyarrow,
        )
        try:
            constructor = (
                pyarrow.LargeListArray
                if pyarrow.types.is_large_list(array_type)
                else pyarrow.ListArray
            )
            return constructor.from_arrays(
                offsets,
                rewritten_values,
                type=array_type,
                mask=mask,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A list text column could not preserve its logical type"
            ) from None

    if pyarrow.types.is_fixed_size_list(array_type):
        list_size = array_type.list_size
        start = array.offset * list_size
        values = array.values.slice(start, len(array) * list_size)
        rewritten_values = _rewrite_selected_array(
            values,
            text_redactor,
            pyarrow,
        )
        try:
            return pyarrow.FixedSizeListArray.from_arrays(
                rewritten_values,
                list_size=list_size,
                mask=array.is_null(),
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A fixed-size-list text column could not preserve its logical type"
            ) from None

    if pyarrow.types.is_map(array_type):
        offsets, keys, items, mask = _map_parts(array, pyarrow)
        rewritten_items = _rewrite_selected_array(items, text_redactor, pyarrow)
        try:
            return pyarrow.MapArray.from_arrays(
                offsets,
                keys,
                rewritten_items,
                type=array_type,
                mask=mask,
            )
        except Exception:
            raise ColumnarTraceAdapterError(
                "A map text column could not preserve its logical type"
            ) from None

    return _rewrite_text_array(array, text_redactor, pyarrow)


def _rewrite_text_array(array: Any, text_redactor: TextRedactor, pyarrow: Any) -> Any:
    array_type = array.type
    if not (
        pyarrow.types.is_string(array_type) or pyarrow.types.is_large_string(array_type)
    ):
        raise ColumnarTraceAdapterError(
            "Selected column must contain string or large-string values"
        )

    values = array.to_pylist()
    rewritten: list[str | None] = []
    changed = False
    for value in values:
        if value is None or value == "":
            rewritten.append(value)
            continue
        try:
            replacement = text_redactor(value)
        except Exception:
            raise ColumnarTraceAdapterError(
                "The text redactor failed without exposing source values"
            ) from None
        if not isinstance(replacement, str):
            raise ColumnarTraceAdapterError("The text redactor must return a string")
        try:
            normalized_replacement = str.encode(replacement, "utf-8").decode("utf-8")
        except Exception:
            raise ColumnarTraceAdapterError(
                "The text replacement could not be normalized safely"
            ) from None
        rewritten.append(normalized_replacement)
        changed = changed or normalized_replacement != value

    if not changed:
        return array
    try:
        return pyarrow.array(rewritten, type=array_type)
    except Exception:
        raise ColumnarTraceAdapterError(
            "The text replacement could not preserve the input string type"
        ) from None


def _list_parts(array: Any, pyarrow: Any) -> tuple[Any, Any, Any]:
    offsets = array.offsets
    offset_values = offsets.to_pylist()
    start = int(offset_values[0]) if offset_values else 0
    end = int(offset_values[-1]) if offset_values else start
    normalized_offsets = pyarrow.array(
        [int(value) - start for value in offset_values],
        type=offsets.type,
    )
    values = array.values.slice(start, end - start)
    return normalized_offsets, values, array.is_null()


def _map_parts(array: Any, pyarrow: Any) -> tuple[Any, Any, Any, Any]:
    offsets = array.offsets
    offset_values = offsets.to_pylist()
    start = int(offset_values[0]) if offset_values else 0
    end = int(offset_values[-1]) if offset_values else start
    normalized_offsets = pyarrow.array(
        [int(value) - start for value in offset_values],
        type=offsets.type,
    )
    keys = array.keys.slice(start, end - start)
    items = array.items.slice(start, end - start)
    return normalized_offsets, keys, items, array.is_null()


def _validate_paths(
    schema: Any, selections: Sequence[FieldSelection], pyarrow: Any
) -> tuple[FieldPath, ...]:
    normalized = _normalize_field_paths(selections)
    resolved: list[FieldPath] = []
    for path in normalized:
        joined = ".".join(path)
        safe_path = _safe_path_label(path)
        if schema.get_field_index(joined) >= 0:
            actual_path: FieldPath = (joined,)
        else:
            actual_path = path
        field_index = schema.get_field_index(actual_path[0])
        if field_index < 0:
            raise ColumnarTraceAdapterError(
                f"Selected text column is missing: {safe_path}"
            )
        field_type = schema.field(field_index).type
        for segment in actual_path[1:]:
            while (
                pyarrow.types.is_list(field_type)
                or pyarrow.types.is_large_list(field_type)
                or pyarrow.types.is_fixed_size_list(field_type)
            ):
                field_type = field_type.value_type
            if pyarrow.types.is_map(field_type):
                field_type = field_type.item_type
            if not pyarrow.types.is_struct(field_type):
                raise ColumnarTraceAdapterError(
                    f"Selected path does not resolve through a struct: {safe_path}"
                )
            child_index = field_type.get_field_index(segment)
            if child_index < 0:
                raise ColumnarTraceAdapterError(
                    f"Selected nested field is missing: {safe_path}"
                )
            field_type = field_type.field(child_index).type
        _validate_selected_type(field_type, safe_path, pyarrow)
        if actual_path not in resolved:
            resolved.append(actual_path)
    return tuple(resolved)


def _validate_selected_type(field_type: Any, path: str, pyarrow: Any) -> None:
    if pyarrow.types.is_null(field_type):
        return
    if pyarrow.types.is_string(field_type) or pyarrow.types.is_large_string(field_type):
        return
    if pyarrow.types.is_dictionary(field_type):
        if pyarrow.types.is_string(
            field_type.value_type
        ) or pyarrow.types.is_large_string(field_type.value_type):
            return
    if pyarrow.types.is_list(field_type) or pyarrow.types.is_large_list(field_type):
        _validate_selected_type(field_type.value_type, path, pyarrow)
        return
    if pyarrow.types.is_fixed_size_list(field_type):
        _validate_selected_type(field_type.value_type, path, pyarrow)
        return
    if pyarrow.types.is_map(field_type):
        _validate_selected_type(field_type.item_type, path, pyarrow)
        return
    raise ColumnarTraceAdapterError(
        f"Selected column must resolve to text values: {path}"
    )


def _normalize_field_paths(
    selections: Sequence[FieldSelection],
) -> tuple[FieldPath, ...]:
    if isinstance(selections, str):
        raise TypeError("text_columns must be a sequence of column paths")
    try:
        values = iter(selections)
    except TypeError:
        raise TypeError("text_columns must be a sequence of column paths") from None

    normalized: list[FieldPath] = []
    seen: set[FieldPath] = set()
    while True:
        try:
            selection = next(values)
        except StopIteration:
            break
        except Exception:
            raise ColumnarTraceAdapterError(
                "Text column paths could not be read safely"
            ) from None
        if isinstance(selection, str):
            parts = tuple(selection.split("."))
        else:
            try:
                parts = tuple(selection)
            except TypeError:
                raise TypeError(
                    "Each text column path must be a string sequence"
                ) from None
            except Exception:
                raise ColumnarTraceAdapterError(
                    "A text column path could not be read safely"
                ) from None
        if any(not isinstance(part, str) for part in parts):
            raise TypeError("Each text column path must contain only strings")
        clean = tuple(part.strip() for part in parts)
        if not clean or any(not part for part in clean):
            raise ValueError("Text column paths must not contain empty fields")
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    if not normalized:
        raise ValueError("At least one text column must be selected")
    return tuple(normalized)


def _safe_path_label(path: str | Sequence[str]) -> str:
    """Return a stable path identifier without exposing schema field names."""

    raw_path = path if isinstance(path, str) else ".".join(path)
    digest = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()[:12]
    return f"path_sha256_{digest}"


def _resolve_redactor(
    text_redactor: TextRedactor | None,
    redactor: TextRedactor | None,
) -> TextRedactor:
    if text_redactor is not None and redactor is not None:
        raise ValueError("Pass only one text redactor")
    resolved = text_redactor if text_redactor is not None else redactor
    if resolved is None:
        return default_text_redactor
    if not callable(resolved):
        raise TypeError("text_redactor must be callable")
    return resolved


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _require_record_batch(batch: Any, pyarrow: Any) -> None:
    if not isinstance(batch, pyarrow.RecordBatch):
        raise TypeError("batch must be a pyarrow.RecordBatch")


def _import_pyarrow() -> Any:
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Columnar trace adaptation requires pyarrow. Install openmed[columnar]."
        ) from exc
    return pyarrow
