# Tabular re-identification risk report

`openmed.risk.tabular_risk_report()` produces a deterministic, local-only
summary for a structured export. It is intended for release manifests and
review handoffs where a detailed row-level risk report would disclose more
than the recipient needs.

The report computes exact-match equivalence classes over declared
quasi-identifiers and retains only aggregate evidence:

- source, analyzed, and caller-declared suppressed row counts;
- schema column names, safe scalar kinds, missing counts, and distinct counts;
- class count, minimum `k`, singleton rate, and class-size distribution;
- maximum, mean, and P95 exact-match risk indicators;
- caller-declared generalization coverage;
- suppression rate, configured thresholds, pass/review outcome, and digests.

Source cells are used only during the local computation. Class keys are
fingerprinted in memory and are not included in the report. Row identifiers,
class membership, suppression offsets, raw generalized values, and
generalization-level labels are not retained or serialized. The module makes
no network calls and has no telemetry path.

## Example

```python
from openmed.risk import tabular_risk_report

synthetic_rows = [
    {"age_band": "30-39", "region_band": "north", "outcome": "synthetic-a"},
    {"age_band": "30-39", "region_band": "north", "outcome": "synthetic-b"},
]

report = tabular_risk_report(
    synthetic_rows,
    quasi_identifiers=["age_band", "region_band"],
    generalization={"age_band": "ten-year", "region_band": "district"},
    thresholds={
        "minimum_k": 2,
        "max_singleton_rate": 0.0,
        "max_reidentification_risk": 0.5,
    },
)

json_text = report.to_json()
markdown_text = report.to_markdown()
```

Generalization metadata is a caller declaration. This report does not prove
that a transformation was applied, nor does it infer a safe threshold for a
particular population. Suppression counts likewise describe the export
workflow; the report does not retain the suppressed row offsets.

## Interpretation

The default threshold requires `minimum_k >= 2` and a zero singleton rate. The
default suppression and generalization thresholds are permissive (`100%` and
`0%`) because those policy choices depend on the release context. Supply
explicit thresholds for a release gate. A `review` outcome is a signal for
qualified privacy review, not an automatic clinical or compliance decision.

The exact-match risk indicators are local sample indicators: a class of size
`k` contributes `1/k` risk for each row. They do not estimate population risk,
attacker auxiliary-data risk, or a legal safe harbor. Use a qualified expert
and the relevant release policy before sharing a structured export.
