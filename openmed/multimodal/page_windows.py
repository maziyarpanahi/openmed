"""Deterministic page batching for offline OCR and VLM document inference.

Batches are planned from page counts and per-page pixel counts alone. Nothing
here renders a page, reads document text, or decides which pages are
clinically important, so a plan carries page numbers and integers only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable

PAGE_BATCH_SCHEMA_VERSION: Final[str] = "openmed.multimodal.page_windows.v1"
MAX_DOCUMENT_PAGES: Final[int] = 100_000
MAX_PAGE_PIXELS: Final[int] = 10_000_000_000
MAX_PAGE_BATCH_COUNT: Final[int] = 100_000

_BATCH_FIELDS = (
    "batch_id",
    "batch_index",
    "start_page",
    "end_page",
    "page_count",
    "pixel_total",
    "isolated",
)
_PLAN_FIELDS = (
    "schema_version",
    "page_count",
    "max_pages_per_batch",
    "max_pixels_per_batch",
    "oversize_policy",
    "batch_count",
    "batches",
)


class OversizePolicy(str, Enum):
    """Closed set of rules for a page larger than the per-batch pixel budget.

    Values:
        REJECT: Fail closed; the caller must downscale or split the page.
        ISOLATE: Give the page a batch of its own, flagged ``isolated``.
    """

    REJECT = "reject"
    ISOLATE = "isolate"


class PageBatchError(ValueError):
    """Value-free failure raised for unusable page batching parameters."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PageBatch:
    """One half-open ``[start_page, end_page)`` run of pages."""

    batch_id: str
    batch_index: int
    start_page: int
    end_page: int
    pixel_total: int
    isolated: bool

    @property
    def page_count(self) -> int:
        """Return the number of pages in the batch."""

        return self.end_page - self.start_page

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_count": self.page_count,
            "pixel_total": self.pixel_total,
            "isolated": self.isolated,
        }
        return {field: values[field] for field in _BATCH_FIELDS}


@dataclass(frozen=True, slots=True)
class PageBatchPlan:
    """Reproducible batch plan for one document."""

    page_count: int
    max_pages_per_batch: int
    max_pixels_per_batch: int | None
    oversize_policy: OversizePolicy
    batches: tuple[PageBatch, ...]
    schema_version: str = PAGE_BATCH_SCHEMA_VERSION

    @property
    def batch_count(self) -> int:
        """Return the number of planned batches."""

        return len(self.batches)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "page_count": self.page_count,
            "max_pages_per_batch": self.max_pages_per_batch,
            "max_pixels_per_batch": self.max_pixels_per_batch,
            "oversize_policy": self.oversize_policy.value,
            "batch_count": self.batch_count,
            "batches": [batch.to_dict() for batch in self.batches],
        }
        return {field: values[field] for field in _PLAN_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def plan_page_batches(
    page_count: int,
    *,
    max_pages_per_batch: int,
    page_pixels: Iterable[int] | None = None,
    max_pixels_per_batch: int | None = None,
    oversize_policy: OversizePolicy | str = OversizePolicy.REJECT,
    max_batches: int = MAX_PAGE_BATCH_COUNT,
) -> PageBatchPlan:
    """Plan ordered, gap-free page batches for document inference.

    Pages are zero-indexed and every batch is half-open,
    ``[start_page, end_page)``. Pages are taken in order and a batch is closed
    when it reaches ``max_pages_per_batch`` or when adding the next page would
    exceed ``max_pixels_per_batch``. Batches therefore partition
    ``range(page_count)`` exactly: no page is skipped or duplicated.

    Args:
        page_count: Total pages in the document.
        max_pages_per_batch: Largest number of pages in one batch.
        page_pixels: Per-page pixel counts, required when a pixel budget is
            given and otherwise used only to report ``pixel_total``.
        max_pixels_per_batch: Optional pixel budget for one batch.
        oversize_policy: What to do with a page larger than the pixel budget.
        max_batches: Largest number of batches the plan may contain.

    Returns:
        A :class:`PageBatchPlan` whose batches are ordered and contiguous.

    Raises:
        PageBatchError: If any parameter is out of range, non-integer, boolean,
            inconsistent, or if the plan would exceed ``max_batches``. Under
            :attr:`OversizePolicy.REJECT` an oversize page also fails closed.
    """

    _validate_bounds(page_count, "page_count", MAX_DOCUMENT_PAGES)
    _validate_bounds(max_pages_per_batch, "page_batch_size", MAX_DOCUMENT_PAGES)
    _validate_bounds(max_batches, "page_batch_count", MAX_PAGE_BATCH_COUNT)
    if max_pages_per_batch == 0:
        raise PageBatchError("page_batch_size_invalid")
    if max_batches == 0:
        raise PageBatchError("page_batch_count_invalid")
    policy = _parse_policy(oversize_policy)
    if max_pixels_per_batch is not None:
        _validate_bounds(max_pixels_per_batch, "page_pixel_budget", MAX_PAGE_PIXELS)
        if max_pixels_per_batch == 0:
            raise PageBatchError("page_pixel_budget_invalid")
    pixels = _materialize_pixels(page_pixels, page_count)
    if max_pixels_per_batch is not None and pixels is None:
        raise PageBatchError("page_pixels_required")

    batches: list[PageBatch] = []
    start = 0
    pending = 0
    pending_pixels = 0

    def flush(isolated: bool = False) -> None:
        nonlocal start, pending, pending_pixels
        if pending == 0:
            return
        if len(batches) == max_batches:
            raise PageBatchError("page_batch_count_limit_exceeded")
        batches.append(
            PageBatch(
                batch_id=f"b{len(batches):06d}",
                batch_index=len(batches),
                start_page=start,
                end_page=start + pending,
                pixel_total=pending_pixels,
                isolated=isolated,
            )
        )
        start += pending
        pending = 0
        pending_pixels = 0

    for page in range(page_count):
        page_pixel_count = pixels[page] if pixels is not None else 0
        if max_pixels_per_batch is not None and page_pixel_count > max_pixels_per_batch:
            if policy is OversizePolicy.REJECT:
                raise PageBatchError("page_exceeds_pixel_budget")
            flush()
            pending = 1
            pending_pixels = page_pixel_count
            flush(isolated=True)
            continue
        if pending == max_pages_per_batch or (
            max_pixels_per_batch is not None
            and pending > 0
            and pending_pixels + page_pixel_count > max_pixels_per_batch
        ):
            flush()
        pending += 1
        pending_pixels += page_pixel_count
    flush()

    return PageBatchPlan(
        page_count=page_count,
        max_pages_per_batch=max_pages_per_batch,
        max_pixels_per_batch=max_pixels_per_batch,
        oversize_policy=policy,
        batches=tuple(batches),
    )


def _materialize_pixels(page_pixels: Any, page_count: int) -> tuple[int, ...] | None:
    if page_pixels is None:
        return None
    if isinstance(page_pixels, (str, bytes, bytearray)):
        raise PageBatchError("page_pixels_invalid")
    try:
        values = tuple(page_pixels)
    except TypeError:
        pass
    else:
        if len(values) != page_count:
            raise PageBatchError("page_pixels_length_mismatch")
        for value in values:
            _validate_bounds(value, "page_pixels", MAX_PAGE_PIXELS)
        return values
    raise PageBatchError("page_pixels_invalid")


def _validate_bounds(value: Any, stem: str, maximum: int) -> None:
    if type(value) is not int:
        raise PageBatchError(f"{stem}_not_an_integer")
    if value < 0 or value > maximum:
        raise PageBatchError(f"{stem}_out_of_range")


def _parse_policy(value: Any) -> OversizePolicy:
    if isinstance(value, OversizePolicy):
        return value
    if type(value) is str:
        try:
            return OversizePolicy(value)
        except ValueError:
            pass
    raise PageBatchError("page_oversize_policy_unsupported")


__all__ = [
    "MAX_DOCUMENT_PAGES",
    "MAX_PAGE_BATCH_COUNT",
    "MAX_PAGE_PIXELS",
    "PAGE_BATCH_SCHEMA_VERSION",
    "OversizePolicy",
    "PageBatch",
    "PageBatchError",
    "PageBatchPlan",
    "plan_page_batches",
]
