# Document Page Batches

`plan_page_batches` turns a page count into reproducible batches for offline
OCR or VLM inference. It is an explicit planning helper: it renders nothing,
reads no document content, and makes no judgement about which pages matter
clinically.

```python
from openmed.multimodal.page_windows import OversizePolicy, plan_page_batches

plan = plan_page_batches(
    6,
    max_pages_per_batch=4,
    page_pixels=[5, 5, 20, 1, 1, 1],
    max_pixels_per_batch=25,
    oversize_policy=OversizePolicy.ISOLATE,
)
for batch in plan.batches:
    print(batch.batch_id, batch.start_page, batch.end_page, batch.pixel_total)
```

## Batching rules

Pages are zero-indexed and every batch is half-open,
`[start_page, end_page)`. Pages are taken in order and the current batch is
closed when it reaches `max_pages_per_batch`, or when adding the next page
would exceed `max_pixels_per_batch`. The batches therefore partition
`range(page_count)` exactly: no page is skipped, duplicated, or reordered, and
an empty document plans no batches.

`page_pixels` is optional. Without a pixel budget it only populates each
batch's `pixel_total`; with one it also drives the split, and supplying a
budget without per-page counts fails closed rather than guessing.

A page whose own pixel count exceeds the budget cannot fit any batch.
`OversizePolicy` decides what happens:

| Policy | Effect |
| --- | --- |
| `reject` | Fail closed with `page_exceeds_pixel_budget`; the caller downscales or splits the page. |
| `isolate` | Close the current batch and give the page one of its own, flagged `isolated`. |

Isolation is exact at the edges: a leading, trailing, or consecutive run of
oversize pages still yields contiguous, gap-free coverage.

## Limits and failures

`PageBatchError` is a `ValueError` with a stable `.category`; its string is the
same category. Every parameter must be an integer, which excludes booleans
because `type(value) is int` is checked rather than `isinstance`. Bounds are
`MAX_DOCUMENT_PAGES` (100 000), `MAX_PAGE_PIXELS` (10 000 000 000) and
`MAX_PAGE_BATCH_COUNT` (100 000), and `max_batches` caps one plan.

Categories are `page_count_not_an_integer`, `page_batch_size_not_an_integer`,
`page_batch_count_not_an_integer`, `page_pixel_budget_not_an_integer`,
`page_pixels_not_an_integer`, the matching `_out_of_range` categories,
`page_batch_size_invalid`, `page_batch_count_invalid`,
`page_pixel_budget_invalid`, `page_pixels_invalid`,
`page_pixels_length_mismatch`, `page_pixels_required`,
`page_exceeds_pixel_budget`, `page_batch_count_limit_exceeded` and
`page_oversize_policy_unsupported`. Submitted values are never echoed.

## Privacy and scope

Only page numbers and pixel counts enter the planner, so a plan cannot carry
page images, text, layout, filenames, or clinical content. `to_dict()`
preserves declared field order and `to_json()` sorts keys for byte-identical
payloads.

Rendering pages, choosing clinically important pages, DICOM frame selection,
WSI tiling, and parallel scheduling are out of scope. A plan is an arithmetic
statement about ordering, not a clinical or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/multimodal/test_page_windows.py -q
```

Fixtures are synthetic. Tests pin empty, exact, remainder, variable-pixel,
oversize and batch-limit examples, assert exact partitioning across 30
page-count/batch-size pairs, and cover every failure category.
