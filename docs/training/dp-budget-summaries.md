# Differential-privacy budget summaries

Governance review needs to know how much validated privacy budget a training
program consumed, grouped by model family and composition method, without
learning which sites participated or what local data they held.
`build_dp_budget_summary()` aggregates already-validated consumption metadata
into a deterministic, bounded summary; `build_dp_budget_summary_from_dicts()`
accepts the same records as JSON-style mappings.

```python
from openmed.training.dp_budget_summary import build_dp_budget_summary_from_dicts

summary = build_dp_budget_summary_from_dicts(
    [
        {
            "round_index": 1,
            "model_family": "clinical-ner",
            "epsilon": 0.5,
            "delta": 0.001,
            "composition": "basic",
            "outcome": "allowed",
        }
    ]
)
print(summary.render_markdown())
```

## Accepted records

Each record describes one already-validated charge for one training round. The
module performs no privacy accounting of its own: records are expected to
describe values already validated by the existing accounting surfaces
(`PrivacyBudgetSpend`, `PrivacyBudgetDecision` and
`PrivacyBudgetLedgerExceeded`).

| Field | Accepted values |
| --- | --- |
| `round_index` | Integer from 1 through 1,000,000, unique per model family |
| `model_family` | Lowercase label matching `[a-z][a-z0-9._-]{0,63}` |
| `epsilon` | Finite number from 0 through 1,000,000 |
| `delta` | Finite number greater than or equal to 0 and less than 1 |
| `composition` | `basic` or `advanced` |
| `outcome` | `allowed`, `denied` or `exhausted` |

Records are grouped into cells keyed by `(model_family, composition)` and the
cells are sorted by those labels, so the input order never changes the output.
A duplicate `(model_family, round_index)` pair, a record with extra fields, a
record missing a field, or more than 10,000 records is rejected.

## Suppression

A cell whose consumption count is below the configured cell floor is reported
as suppressed instead of being counted exactly. The floor defaults to
`MIN_CELL_SIZE` (5) and accepts values from 2 through 1,000.

- A suppressed cell hides `epsilon_total`, `epsilon_max`, `delta_total` and
  `delta_max` entirely.
- An individual outcome count below the floor is replaced by the bucket label
  `<min_cell_size` rather than a number.
- An individual epsilon bucket count below the floor is replaced the same way.

Epsilon values are additionally reported through fixed inclusive buckets:

| Bucket | Range |
| --- | --- |
| `le_0.5` | `epsilon` less than or equal to 0.5 |
| `le_1` | up to 1 |
| `le_3` | up to 3 |
| `le_8` | up to 8 |
| `gt_8` | above 8 |

Totals are computed with `math.fsum` over the sorted values and rounded to six
decimal places, so two orderings of the same records produce the same numbers.
Every cell payload always carries all three outcome keys and all five bucket
keys, whether or not each entry is reported.

## Output

`render_json()` emits sorted-key JSON with two-space indentation and no trailing
newline. `render_markdown()` emits a title, five metadata lines, and one section
per cell with outcome, epsilon-bucket and metric tables. Both renderings are
byte-stable and pinned by golden digests in the focused tests.

No site, client, cohort, path, endpoint, example, tensor, gradient, message,
local-metric, patient-count or other identifying value is accepted. Rejection
messages name the violated rule and never echo the submitted value.

## Out of scope

The summary does not add noise, does not check a composition bound, does not
certify differential privacy, and does not replace a coordinator's ledger
decision. It reports only what already-validated records describe.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_dp_budget_summary.py -q
```
