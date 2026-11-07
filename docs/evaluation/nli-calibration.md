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

## Qualify an existing local artifact

The supported Python library and CLI compose the existing local Torch/ONNX
sequence classifier and calibration reporter. Supply an already available,
materialized local artifact directory, an explicit class-index mapping, and
separate development and evaluation JSON files. Optional runtime dependencies
must already be installed. Loading uses the existing offline-only loader with
remote code disabled. No checkpoint is designated by this command.

```bash
openmed nli-qualify \
  --artifact /local/artifact \
  --label-mapping /local/labels.json \
  --development /local/development.json \
  --evaluation /local/evaluation.json \
  --policy /local/qualification-policy.json \
  --runtime torch --json
```

The mapping file contains, for example,
`{"0":"entailment","1":"contradiction","2":"neutral"}`. Use the meanings
from the caller's checkpoint; do not infer them from its predictions. Each
split has this shape (the example is deliberately synthetic):

```json
{
  "provenance": {"kind": "synthetic", "reference": "synthetic-corpus-v1"},
  "records": [
    {
      "id": "synthetic-dev-001",
      "group_id": "synthetic-source-001",
      "premise": "Synthetic subject received fluids.",
      "hypothesis": "Synthetic subject received fluids.",
      "gold_label": "entailment",
      "slices": ["negation", "temporality"]
    }
  ]
}
```

Provenance kinds are `synthetic`, `caller_supplied`, and `restricted`.
References and source-group identifiers are caller-owned; references are
hashed in the receipt. The caller must group all records from the same source
or patient together and keep those groups disjoint between splits. Duplicate
IDs within a split, shared IDs/groups across splits and identical text pairs
across splits are rejected. This does not authenticate a caller's provenance
claim or detect incorrectly declared source groups. Restricted data requires
independent authorized access; absent data remains unavailable. No restricted
corpus, text or weights are bundled.

The policy file may set `min_per_class`, `precision_floor`, `recall_floor`,
`false_positive_rate_ceiling`, `margin`, and `required_slices`. Defaults are
20 examples of **each** gold class in **each** split and required slice,
precision >= 0.95, recall >= 0.8, FPR <= 0.05, and margin >= 0.05.
These are configurable tooling defaults, not validated clinical requirements.
The predefined slices are `negation`, `temporality`, `experiencer`, `numbers`,
and `medication_status`; all are required by default. An empty required-slice
list explicitly limits qualification to the overall supplied data.

Entailment and contradiction thresholds are selected independently on
**development only**, using the existing entailment reporter with a one-versus-
rest gold mapping for each decisive class. Non-winning, tied and low-margin
scores cannot be accepted. The selected thresholds are then fixed for the
held-out evaluation report and its supported slices. A fallback recommendation
that fails policy constraints never qualifies. All-abstention outcomes fail
qualification even when callers relax recall or precision floors. Evaluation results do not tune
the thresholds. Neutral remains abstention at the deployed gate.

The JSON envelope's `data` contains a versioned receipt, SHA-256 bindings,
provenance kinds and digests, split counts, supported slices, aggregate
calibration reports, thresholds, and controlled insufficiency reasons.
`status` is `unavailable`, `insufficient`, `synthetic_only`, or `qualified`.
Exit status is 0 only for `qualified`; other qualification outcomes exit 1.
Malformed/unreadable input files return a value-free CLI error with exit 1.
Missing splits do not load the model or use the reporter's synthetic defaults.
Insufficiency includes missing class/slice support, split overlap, and unmet
operating constraints. Unknown labels and failed local inference are unavailable.
Synthetic provenance in either split can never produce `qualified=true`.

A successful receipt means only that the supplied artifact passed the
caller-selected policy on the declared supplied data. It is not clinical
approval, proof of provenance, a benchmark claim, or a signed third-party
attestation. Synthetic adapter tests exercise declaration and refusal mechanics
only and cannot establish clinical qualification. Human review is always
required. Keep input datasets in caller-controlled private storage; receipts
and diagnostics contain no raw text, identifiers or local paths.

### Bind the receipt to BriefContext

```python
from openmed.clinical.nli_qualification import (
    NLIQualificationPolicy,
    bind_qualified_nli,
    qualify_local_nli,
)

policy = NLIQualificationPolicy()
options = dict(
    label_mapping=label_mapping,
    development=development_data,
    evaluation=evaluation_data,
    policy=policy,
    runtime="torch",
)
receipt = qualify_local_nli(artifact_directory, **options)
# Raises a value-free uncalibrated/mismatch error unless the receipt qualifies.
verifier = bind_qualified_nli(receipt, artifact_directory, **options)
# Supply to the existing, independently reviewed BriefContext:
# nli_predict=verifier, thresholds=verifier.thresholds
```

The callback returns full three-class probabilities with
`calibration_id=receipt.digest`. Binding verifies weights, tokenizer, config
and every other file in the artifact directory, the label mapping, runtime,
normalized development/evaluation inputs, provenance and policy. It rechecks
before and after each inference and refuses drift. Keep the artifact snapshot
immutable during qualification and serving. Symlinks are rejected; materialize
cached artifacts into a caller-controlled directory first. Keep receipt output
outside that directory. Hashing the complete artifact and both datasets on
callbacks trades throughput for explicit drift detection; the inputs remain
in process and are never persisted by the library.

Use the live library receipt to bind the callback; serialized receipts are
aggregate audit evidence, with no receipt-import or third-party trust API.
Only the qualifier creates live receipt objects. Public construction, audit-shaped
objects and edits to a serialized synthetic receipt cannot authorize calibration.
This is an input boundary inside a trusted Python process, not protection against
arbitrary code execution in that process. Malformed Unicode pairs fail with
controlled errors that retain no private decoder exception.
The existing synthetic-only evidence admission in BriefContext is preserved.
Qualification does not authorize reviewed evidence admission or autonomous
clinical action. This command targets the existing Python local-artifact
runtimes; OpenMedKit's on-device application verifier remains responsible for
its own evidence/NLI packet admission, with no cloud fallback introduced.
