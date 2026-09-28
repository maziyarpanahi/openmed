# Clinical NLI threshold calibration

`openmed.eval.nli_calibration` produces an offline calibration report for an
entailment acceptance gate. It is evaluation evidence, not a compliance
certification or a clinical decision guarantee.

## What is measured

Each synthetic fixture contains a premise, a hypothesis, a gold NLI label, and
an entailment score. For a threshold `t`, scores `>= t` are accepted as
entailment and scores below `t` abstain. The report includes:

- threshold operating points and the abstention count/rate;
- precision, recall, specificity, false-positive rate, false-negative rate,
  accuracy, and F1;
- true-positive, false-positive, true-negative, and false-negative counts; and
- a deterministic recommendation, optionally constrained by precision, recall,
  or false-positive-rate requirements.

Contradiction, neutral, and binary not-entailment labels are retained as gold
label counts. Confusion counts aggregate all non-entailment labels into the
negative class because this gate accepts only entailment.

## Privacy and reproducibility

Premise and hypothesis text are never serialized into JSON or Markdown
reports. The report records a SHA-256 model fingerprint and a SHA-256 fixture
fingerprint; the latter binds normalized text, labels, scores, and fixture
identifiers without disclosing them. Validation errors also omit fixture
values. Keep committed fixtures synthetic and offline.

```python
from openmed.eval import calibrate_nli_thresholds

report = calibrate_nli_thresholds(
    [
        {
            "id": "synthetic-1",
            "premise": "Synthetic premise alpha.",
            "hypothesis": "Synthetic hypothesis alpha.",
            "gold_label": "entailment",
            "score": 0.91,
        },
        {
            "id": "synthetic-2",
            "premise": "Synthetic premise beta.",
            "hypothesis": "Synthetic hypothesis gamma.",
            "gold_label": "neutral",
            "score": 0.42,
        },
    ],
    model_id="local-nli-checkpoint",
    thresholds=(0.4, 0.7, 0.9),
    precision_floor=0.90,
)

print(report.recommended_threshold)
print(report.to_json())
```

The default threshold candidates are the unique fixture scores plus `0.0` and
`1.0`. Recommendations are deterministic. A precision floor selects the
highest-recall feasible point; a recall floor selects the highest-precision
feasible point; an FPR ceiling selects the highest-recall feasible point. If
no point satisfies the requested constraints, the report falls back to maximum
F1 and records `max_f1_no_point_met_constraints`.
