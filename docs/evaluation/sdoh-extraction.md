# SDOH extraction benchmark

The registered `sdoh-extraction` suite measures the current `extract_sdoh()`
rules against 33 repository-authored synthetic social-history windows. Gold
contains 26 patient findings across tobacco, alcohol, drug, employment,
living status and food insecurity. Cases include verb forms, negation,
historical status, family experiencers, unanswered screening and an empty
social history. No model, network, restricted corpus or credential is needed.

```python
from openmed.eval.suites.sdoh_extraction import run_sdoh_extraction_benchmark

report = run_sdoh_extraction_benchmark()
print(report.metrics["overall"])
```

The `BenchmarkReport` contains fixed suite/model/device headers, aggregate
counts/rates and SHA-256 digests. It excludes notes, finding values, source
paths, record IDs and source-level offsets. `generated_at` remains `None`.
An injected trusted extractor uses the controlled model name
`injected-sdoh-extractor`.

## Scoring contract

Each input is an already selected social-history window. The default runner
calls `extract_sdoh(text, ())`; it passes no gold spans and applies no extra
experiencer filter, cue change or threshold tuning. Synthetic gold is
patient-focused: family findings count as extra predictions when extracted
as patient findings. These windows test extraction, not section detection.

Gold independently specifies determinant, status and half-open trigger
offsets. Synthetic trigger spans are authored for this regression set, not
SHAC annotations. Finding value, confidence, extent and temporality are
excluded. Unknown categories, missing statuses, invalid offsets and extractor
errors refuse the run with fixed `sdoh_extractor_contract_failed` errors.

Deterministic maximum-cardinality matching pairs each gold and prediction at
most once, within the same window and determinant. Duplicate predictions
remain false positives; duplicate native events remain separate gold findings.
Touching half-open intervals do not overlap.

| Metric | Match rule | Denominator |
| --- | --- | --- |
| `precision` | Same determinant and any trigger overlap | All predictions |
| `recall` | Same determinant and any trigger overlap | All gold findings |
| `status_accuracy` | Same determinant/status and any trigger overlap | All gold findings, including missed triggers |
| `exact_offset_match` | Same determinant and exact trigger offsets; status scored separately | All gold findings |

Status-aware and exact matchings are evaluated independently. In `by_status`,
every match also requires the status label. Overall, determinant and status
slices include gold/prediction counts, overlap/status/exact numerator counts
and false positives/negatives. Zero denominators produce `None`, not successful
scores. Empty case lists are refused.

## Reproduce the synthetic baseline

```python
from openmed.eval.suites.sdoh_extraction import (
    SDOH_REPORT_PATH,
    run_sdoh_extraction_benchmark,
    sdoh_report_digest,
)

report = run_sdoh_extraction_benchmark()
assert report.to_json() + "\n" == SDOH_REPORT_PATH.read_text(encoding="utf-8")
assert report.metadata["report_digest"] == sdoh_report_digest(report)
```

| Count or rate | Result |
| --- | ---: |
| Windows / gold / predictions | 33 / 26 / 29 |
| Overlap / correct status / exact offset matches | 22 / 18 / 18 |
| Missed gold / extra predictions | 4 / 7 |
| Precision / recall | 75.86% / 84.62% |
| Status accuracy / exact offset match, including misses | 69.23% / 69.23% |

The two missing verb-form cases from #3753 remain gold positives. The third
case still has the wrong `unknown` status. Gold is never filtered according
to extractor output. Changes require explicit baseline review; tests compare
the complete committed report.

`fixture_digest` hashes a sorted multiset of per-window text digests and
gold labels/offsets, preserving duplicates without storing text or source IDs.
`prediction_digest` binds projected predictions to those windows.
`report_digest` hashes canonical compact JSON of the aggregate report with
its own field removed. Window order does not affect reproduction. Digests
provide neither anonymity nor authorization; unsalted text hashes may be guessable.

## User-supplied SHAC lane

```python
from openmed.eval.suites.sdoh_extraction import (
    SDOHBenchmarkUnavailable,
    run_shac_sdoh_benchmark,
)

# Uses OPENMED_SHAC_PATH if an authorized user has configured it.
result = run_shac_sdoh_benchmark()
if isinstance(result, SDOHBenchmarkUnavailable):
    print(result.to_dict())  # Controlled unavailable codes; no metrics.
else:
    print(result.metrics["overall"])  # Aggregate results only.
```

The existing `load_shac()` adapter reads only a credentialed local corpus
outside the repository and confines resolved files to its configured root.
The lane fetches no data, writes no corpus cache and bundles no SHAC rows or
report. A path does not establish authorization; the caller must already have
the required DUA access.

Supported native paired BRAT `.txt`/`.ann` files retain explicit status
attributes. Equivalent JSON exports can put `subtype` or an `attributes`
object on argument entities, or supply record-level
`attributes: [{"name": ..., "target": ..., "value": ...}]` entries. Native
event roles are `Status` and `Type`; gold labels are never guessed from text.

| Native event | Required labeled arguments | Gold status mapping |
| --- | --- | --- |
| Alcohol, Drug, Tobacco | `StatusTime` / `StatusTimeVal` | `none`, `current`, `past` retained |
| Employment | `StatusEmploy` / `StatusEmployVal` | `on_disability` to `disabled`; other labels retained, including `homemaker` |
| LivingStatus | `StatusTime` / `StatusTimeVal` and `TypeLiving` / `TypeLivingVal` | Current `alone`, `with_family`, `with_others`, `homeless` to `lives_alone`, `lives_with_family`, `lives_with_others`, `homeless`; past to `former`; future to `future` |

Gold labels `homemaker`, `lives_with_others` and `future` remain in denominators
even when the extractor cannot emit them. Food insecurity has no native SHAC
gold, so SHAC food recall is unobserved (`None`). This view scores triggers
and a selected normalized status; it does not score every event argument or
reproduce the official shared-task score. See the
[native annotation schema](https://github.com/Lybarger/brat_scoring/blob/main/docs/annotation.conf)
and the distinct [official scoring criteria](https://github.com/Lybarger/brat_scoring).

Unavailable outcomes carry no metrics or implied zero: `shac_not_configured`,
`shac_path_unavailable`, `shac_gold_invalid` and `shac_gold_unsupported`.
The entire lane is unavailable if any record lacks required labels, has
ambiguous gold, an unsupported event view, a non-English language or a
discontinuous trigger. No record is silently removed from a scored denominator.

The default extraction runs under the Python offline socket guard. Injected
extractors are trusted application code; this is not a native-code or subprocess
sandbox. Exceptions are replaced with fixed codes outside their original
context. Cases stay protected in memory; general fixture serializers can
contain source values.

This is a synthetic regression and optional local evaluation view, not
clinical validation, deployed patient performance or a qualified SHAC result.
Human review and the [SDOH evidence contract](../clinical/sdoh-evidence.md)
still apply; no autonomous clinical action is added.
