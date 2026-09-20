# Minimal clinical NLI premise windows

`select_minimal_premise_window()` limits a clinical NLI verifier to the
smallest character window covering every cited source span. It runs locally,
uses only deterministic offset arithmetic, and performs no network calls.

```python
from openmed.clinical.nli_premise_window import (
    EvidenceSpan,
    select_minimal_premise_window,
)

deidentified_note = "[omitted] stable finding; intervening context; follow-up finding"
result = select_minimal_premise_window(
    deidentified_note,
    (EvidenceSpan(10, 24), EvidenceSpan(47, 64)),
    max_characters=64,
    deidentified=True,
)

if result.verification_allowed:
    premise = result.premise
else:
    refusal = result.refusal_reason
```

Offsets are half-open and must address the supplied text. Citation-like objects
with `source_start` and `source_end`, two-integer tuples, and mappings with
`start`/`end` or `source_start`/`source_end` are accepted. Spans are sorted and
deduplicated before the covering window is computed.

## Fail-closed boundary

Verification is refused when:

- the caller does not explicitly mark the input as de-identified;
- no cited source span is supplied; or
- the range from the earliest cited start to the latest cited end is larger
  than `max_characters`.

The ceiling is inclusive. The selector never drops a required span or stitches
distant fragments together to evade it. Invalid spans and configuration raise
fixed-category errors that do not echo submitted text or values.

`PremiseWindowResult.to_dict()` is safe metadata for an audit report: status,
reason code, offsets, character count, and source-span count. It deliberately
omits the selected premise. The premise is also excluded from object `repr`.
Do not log the `premise` property or send it to a remote verifier. The explicit
de-identification marker is a pipeline contract, not a de-identification check.

This utility bounds verifier input; it does not classify claims, certify
privacy, or make clinical decisions.
