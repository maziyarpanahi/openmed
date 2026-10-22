# Private candidate quantized-deployment recall deltas

A privately improved adapter is merged into a full-precision base before it
ships, and the merged model is then quantized for each deployment format. An
adapter can lose direct-identifier recall once quantized, and an adapter
trained over a quantized base can behave differently when merged into a
full-precision or differently quantized base.
`openmed.eval.private_candidate_quant_gate` produces per-format evidence for
such a candidate. It runs locally, needs no model download or network access,
and receives model inference through injected prediction callables.

The module produces evidence only. Quantization tooling, device benchmarking
and the promotion decision stay separate.

## Recall-delta evidence per format

For each declared deployment format, the gate measures exact-span recall of
every direct-identifier label on a synthetic or sealed holdout, once for the
full-precision merge and once for the quantized deployment. It then calls
`openmed.eval.quant_delta.evaluate_quant_recall_delta`, so the existing
per-format limits apply unchanged: `INT8_RECALL_DELTA_LIMIT` (0.005) for INT8
and `INT4_RECALL_DELTA_LIMIT` (0.010) for INT4. A format fails when any
label's recall loss meets or exceeds its limit.

Direct-identifier labels are those whose canonical policy class is
`DIRECT_IDENTIFIER`. Other gold labels in the holdout are ignored, and a
holdout without any direct-identifier span is rejected.

Formats are a precision token, `fp32`, `bf16`, `fp16`, `int8` or `int4`, with
an optional lowercase runtime prefix such as `onnx-int8`, `coreml-int8` or
`mlx-int4`. Full-precision formats are not quantized under the existing gate
and need no prediction provider. Core ML fp16 span parity remains the job of
`evaluate_coreml_span_parity`.

Each format gets its own result, with its limit, per-label deltas, offending
labels, the gold span count per label (`label_support`) and the holdout item
count. Results are never averaged. The candidate-level `passed` flag is true
only when every format passed, and `failed_formats` names the failing ones, so
a passing format cannot offset a failing one.

## Base precision compatibility

The caller records the precision of the base used during training. Each
deployment format is then classified:

| Training base | Deployment | `precision_compatibility` | Outcome |
| --- | --- | --- | --- |
| `fp32`, `bf16` or `fp16` | any full-precision format | `same_precision` | compatible |
| `fp32`, `bf16` or `fp16` | `int8` or `int4` | `full_precision_reference` | compatible; the recall delta measures the shift |
| `int8` or `int4` | same precision | `same_precision` | compatible |
| `int8` or `int4` | any other precision | `declared` | compatible with a matching declaration |
| `int8` or `int4` | any other precision | `undeclared` | fails with `base_precision_incompatible` |

A `PrecisionCompatibilityDeclaration` names the candidate digest, the exact
training and deployment precisions, and the digest of the evidence supporting
the claim. A declaration for another candidate or another precision pair does
not apply. The applied declaration's evidence digest is bound into the
format result as `compatibility_digest`.

## Example

```python
from openmed.eval.private_candidate_quant_gate import (
    evaluate_private_candidate_quant_evidence,
)

text = "Contact Ana Silva at ana@example.test."
gold = [
    {"label": "PERSON", "start": 8, "end": 17},
    {"label": "EMAIL", "start": 21, "end": 37},
]
holdout = [{"text": f"{text} {index}", "spans": gold} for index in range(100)]


def full_precision_merge(document):
    return gold


def degraded_int8(document):
    # Misses one PERSON span out of 100: a 0.01 loss, over the INT8 limit.
    return gold[1:] if document == holdout[0]["text"] else gold


evidence = evaluate_private_candidate_quant_evidence(
    candidate_digest="sha256:" + "a" * 64,
    base_digest="sha256:" + "b" * 64,
    harness_digest="sha256:" + "c" * 64,
    training_base_precision="bf16",
    deployment_formats=["fp16", "int8", "int4"],
    holdout=holdout,
    reference_predictor=full_precision_merge,
    format_predictors={"int8": degraded_int8, "int4": full_precision_merge},
)
assert evidence.failed_formats == ("int8",)
assert evidence.format_results[1].reason_codes == ("recall_delta_exceeds_limit",)
assert evidence.format_results[2].passed
```

## Reason codes

| Code | Meaning |
| --- | --- |
| `recall_delta_exceeds_limit` | A direct-identifier label lost at least the existing per-format limit. |
| `missing_recall_evidence` | No prediction provider was supplied for a quantized format. |
| `prediction_failed` | The format's provider raised or returned a malformed span. |
| `base_precision_incompatible` | A quantized training base differs from the deployment precision and no matching declaration exists. |

Invalid inputs raise `PrivateCandidateQuantGateError` with a stable `code`,
such as `invalid_digest`, `invalid_precision`, `invalid_deployment_format`,
`duplicate_deployment_format`, `undeclared_format_predictor`,
`invalid_holdout` or `reference_prediction_failed`, and the name of the
offending field. Messages never echo submitted values, and provider exceptions
are suppressed rather than chained.

## Evidence binding and privacy

Every format result binds the candidate, base and harness digests and the
format, and carries an `evidence_digest` over all of its fields, including the
verdict. The candidate-level `evidence_digest` covers the bound digests, the
training base precision, the verdict and each format's evidence digest.

Holdout text is passed only to the prediction callables. Results contain
reason codes, precision tokens, limits, rates, slice sizes and digests; they
never contain text, spans or offsets.
