# Clinical NLI abstention gate

The clinical NLI gate is a local, deterministic policy boundary for evidence
links. It consumes calibrated entailment, contradiction, and neutral
probabilities. It emits `entailment` or `contradiction` only when the winning
class clears its calibrated threshold and has the configured margin over the
next-best class. Everything else becomes `abstain`.

An abstention is an explicit handoff to human review. It is not a negative
clinical conclusion, and the gate never makes a diagnosis, treatment decision,
or autonomous clinical judgment.

## Example

```python
from openmed.clinical.nli_gate import EvidenceLink, NLIThresholds, evaluate_nli

evidence = EvidenceLink.from_text(
    source_id="synthetic-note-1",
    claim_id="synthetic-claim-1",
    source_text="Synthetic evidence supports the synthetic claim.",
    claim_text="The synthetic claim is supported.",
    start=0,
    end=48,
)

result = evaluate_nli(
    {
        "entailment": 0.96,
        "contradiction": 0.02,
        "neutral": 0.02,
        "calibration_id": "synthetic-calibration-v1",
    },
    evidence,
    thresholds=NLIThresholds(
        entailment=0.90,
        contradiction=0.90,
        margin=0.10,
        calibration_id="synthetic-calibration-v1",
    ),
)

assert result.outcome == "entailment"
assert result.autonomous_decision is False
```

For an ambiguous or neutral score, `result.outcome` is `"abstain"`,
`result.requires_human_review` is `True`, and `result.human_review` contains a
queue-safe reason plus the typed evidence link. The result and audit entry
contain offsets, hashes, score metadata, and provenance only; premise,
hypothesis, and source text are never retained.

`NLIThresholds.calibration_id` identifies the held-out calibration artifact
that selected the operating point. A score set carrying a different
calibration ID is rejected rather than silently compared with the wrong
thresholds. The module performs no model download or mandatory network call.

This is an assistive safety boundary. It does not provide a compliance
certification or a clinical decision guarantee.
