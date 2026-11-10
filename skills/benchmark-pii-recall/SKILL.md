---
name: benchmark-pii-recall
description: "Benchmark an OpenMed PII model with synthetic gold spans and report label-aware exact-span and grapheme recall without emitting identifier surfaces. Use when an agent must compare a model, threshold, backend, or quantized artifact and enforce a recall floor before release."
---

# Benchmark PII recall

Measure PII recall before optimizing F1, size, or latency. A missed direct
identifier is a privacy failure even when aggregate F1 improves.

## Procedure

1. Build synthetic fixtures with exact offsets and canonical PII labels.
2. Include direct identifiers, boundary cases, languages/scripts, and the
   target device or quantization.
3. Run `extract_pii` at the candidate threshold.
4. Normalize prediction labels and score each document separately.
5. Aggregate counts only; do not persist raw text or identifier surfaces.
6. Fail the release when the recall floor or zero-critical-leak requirement is
   not met.

## Runnable synthetic benchmark

Install the model runtime first with `python -m pip install "openmed[hf]"`.

```python
from openmed import extract_pii
from openmed.core.labels import normalize_label
from openmed.eval import compute_character_recall, compute_exact_span_f1

MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
RECALL_FLOOR = 0.99
FIXTURES = [
    {
        "text": (
            "Call the synthetic clinic at 212-555-0198 or email "
            "demo.patient@example.test."
        ),
        "spans": [
            ("PHONE", "212-555-0198"),
            ("EMAIL", "demo.patient@example.test"),
        ],
    },
    {
        "text": (
            "The synthetic callback number is 415-555-0136 and the contact "
            "address is sample.user@example.test."
        ),
        "spans": [
            ("PHONE", "415-555-0136"),
            ("EMAIL", "sample.user@example.test"),
        ],
    },
]

true_positives = false_positives = false_negatives = 0
covered_graphemes = total_graphemes = 0

for fixture in FIXTURES:
    text = fixture["text"]
    gold = []
    for label, surface in fixture["spans"]:
        start = text.index(surface)
        gold.append(
            {"start": start, "end": start + len(surface), "label": label}
        )

    result = extract_pii(
        text,
        model_name=MODEL,
        confidence_threshold=0.5,
        lang="en",
    )
    predicted = [
        {
            "start": entity.start,
            "end": entity.end,
            "label": normalize_label(entity.label),
        }
        for entity in result.entities
        if entity.start is not None and entity.end is not None
    ]

    exact = compute_exact_span_f1(gold, predicted, source_text=text)
    recall = compute_character_recall(gold, predicted, source_text=text)
    true_positives += exact.true_positives
    false_positives += exact.false_positives
    false_negatives += exact.false_negatives
    covered_graphemes += int(recall.numerator)
    total_graphemes += int(recall.denominator)

exact_recall = true_positives / max(true_positives + false_negatives, 1)
grapheme_recall = covered_graphemes / max(total_graphemes, 1)
print(
    {
        "documents": len(FIXTURES),
        "exact_span_recall": exact_recall,
        "grapheme_recall": grapheme_recall,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
)
assert grapheme_recall >= RECALL_FLOOR, "PII recall floor not met"
```

## Release gates

- Require zero misses for critical direct identifiers even if aggregate recall
  passes.
- Report per-label, language, script, section, and device slices.
- Compare quantized and full-precision outputs; reject recall regressions.
- Add hard negatives so over-redaction does not hide behind high recall.
- Store fixture hashes, model identity, threshold, and aggregate counts only.
- Keep DUA-gated corpora outside the repository and load them only from the
  user's approved location.

## Repository example

Read
[the policy and release-evidence walkthrough](../../examples/v16_policy_audit_release_gates.py)
for PHI-free leakage metrics and audit evidence.
