# Federated round status

OpenMed can render a federated training round as a deterministic operator
summary without publishing participant identities or local training data. The
summary accepts aggregate counts and digest references only. It has no fields
for client IDs, site names, patient counts, local losses, gradients, or
per-client metrics. Build summaries with `build_federated_round_status()`;
direct construction is rejected so callers cannot forge the derived quorum or
completion fields.

## Disclosure rules

Participant and completed-participant counts are released only when each count
is at least the configured `minimum_group_size`, which defaults to `5`. Smaller
counts are replaced with a suppression marker in both JSON and Markdown. The
minimum group size must be at least `2`.

Quorum is calculated before suppression, so operators still see `met` or
`not_met` when the exact participant count is hidden. The required quorum is
not included in the rendered summary.

Completion is reported as one of four coarse bands:

| Band | Aggregate meaning |
| --- | --- |
| `not_started` | No participants have completed |
| `under_half` | More than zero but fewer than half have completed |
| `half_or_more` | At least half but fewer than all have completed |
| `complete` | Every participant has completed |

## Usage

```python
from openmed.training import (
    FederatedRoundReasonCode,
    FederatedRoundState,
    build_federated_round_status,
)

status = build_federated_round_status(
    state=FederatedRoundState.HELD,
    participant_count=8,
    completed_participant_count=8,
    required_quorum=5,
    minimum_group_size=5,
    aggregate_digest_refs=("sha256:" + "a" * 64,),
    reason_code=FederatedRoundReasonCode.QUALITY_REVIEW_REQUIRED,
)

json_artifact = status.to_json()
markdown_artifact = status.to_markdown()
```

Digest references must be lowercase `sha256:` values. Duplicate references are
rejected and valid references are sorted before rendering, so caller ordering
cannot change the artifact. Held and aborted rounds require a reason from their
respective stable reason-code sets; other lifecycle states reject reason codes.

| Round state | Allowed reason codes |
| --- | --- |
| `held` | `privacy_review_required`, `quality_review_required`, `policy_review_required` |
| `aborted` | `quorum_not_met`, `privacy_gate_failed`, `quality_gate_failed`, `policy_gate_failed`, `operator_cancelled`, `round_error` |

## Privacy boundary

The builder uses exact aggregate counts only to calculate quorum, completion,
and disclosure. It immediately replaces sub-threshold counts with `null` in the
immutable status object, so later renderers cannot recover the omitted number.

Minimum-group suppression is an output control, not differential privacy or a
privacy accountant. Transport, client attribution, local metrics, and privacy
budget enforcement remain outside this module.

## Label coverage

Sites often annotate different subsets of a round's labels. When a site trains
on its own BIO tags, an entity it never annotates appears as an outside (`O`)
token, and a naive token loss teaches the shared model to suppress that entity.
`openmed.training.federated_label_coverage` defines a coverage contract, a loss
mask, and a coordinator gate for this case. Import it from the submodule.

### Declarations and the round label schema

The coordinator owns a `FederatedRoundLabelSchema`: the exact canonical labels
from `openmed.core.labels` trained in the round, a `sha256:` reference to the
annotation-guideline version, and the accepted text forms (`raw`, `surrogate`).
Each site publishes a `FederatedLabelCoverageDeclaration` listing the labels it
annotated completely under that guideline and its text form. A declaration has
exactly four fields: `schema_version`, `labels`, `guideline_digest`, and
`text_form`. It carries no site name, example text, or counts; any other field
is rejected.

Critical labels are every catalog direct identifier in the schema, plus any
`additional_critical_labels` the coordinator configures. Configuration can add
critical labels but cannot downgrade a catalog direct identifier.

### Loss-mask semantics

`build_label_loss_masks()` turns a site's observed tag sequences into one
`FederatedTokenLossMask` per sequence. For each token the mask lists the
candidate tag classes, meaning the tags the token could truly carry:

| Observed tag | Candidate classes |
| --- | --- |
| `B-X` or `I-X`, `X` declared | Exactly that tag |
| `O` | `O`, plus `B-Y` and `I-Y` for every round label `Y` the site does not declare |

Classes outside the candidate set are definite negatives and stay in the loss.
An `O` token at a site that does not annotate `Y` could be an unannotated `Y`
entity, so it is never trained as a negative for `Y`; it still trains every
declared label as a negative. Training ports apply the mask in either form:

- Softmax cross-entropy: use the marginal loss `-log(sum of p[c] over candidates)`.
  For a single candidate this is ordinary cross-entropy.
- Per-class binary loss: train `positive_targets()` as positives and
  `negative_matrix()` entries as negatives, and skip the other candidate
  classes of ambiguous tokens.

When a site declares every round label, every mask row has one candidate and
training is identical to unmasked training. `run_masked_site_training()` validates
the declaration and every sequence before it calls a `FederatedMaskedTrainingPort`.
One invalid sequence refuses the whole batch, and a port failure is reported only
as `training_port_failed`.

### Coordinator gate and evidence

`check_round_label_coverage()` validates every declaration against the schema
and blocks the round with `critical_label_uncovered` when a critical label has no
annotating site. Operators can abort such a round with the `policy_gate_failed`
round reason code. `build_label_coverage_evidence()` returns the same evidence
without raising. The evidence records the schema and guideline digests, the
minimum group size, the gate decision, and one band per label:

| Band | Annotating sites |
| --- | --- |
| `none` | Zero |
| `suppressed` | One to fewer than the minimum group size |
| `minimum_to_under_double` | Minimum group size to under twice that size |
| `double_to_under_fourfold` | Twice to under four times the minimum |
| `fourfold_or_more` | Four times the minimum or more |

```python
from openmed.core.labels import PERSON, PHONE
from openmed.training.federated_label_coverage import (
    FederatedLabelCoverageDeclaration,
    FederatedRoundLabelSchema,
    FederatedTextForm,
    build_label_loss_masks,
    check_round_label_coverage,
)

# Synthetic guideline digest for this example only.
guideline = "sha256:" + "a" * 64
schema = FederatedRoundLabelSchema(labels=(PERSON, PHONE), guideline_digest=guideline)
full = FederatedLabelCoverageDeclaration(
    labels=(PERSON, PHONE),
    guideline_digest=guideline,
    text_form=FederatedTextForm.SURROGATE,
)
person_only = FederatedLabelCoverageDeclaration(
    labels=(PERSON,),
    guideline_digest=guideline,
    text_form=FederatedTextForm.SURROGATE,
)

evidence = check_round_label_coverage([full, person_only], schema=schema)
(mask,) = build_label_loss_masks(
    [["O", "B-PERSON", "O"]], declaration=person_only, schema=schema
)
assert mask.classes == ("O", "B-PERSON", "I-PERSON", "B-PHONE", "I-PHONE")
assert mask.negative_matrix()[2][3] is False  # an O token is not a PHONE negative
assert mask.negative_matrix()[2][1] is True  # but it is still a PERSON negative
```

Validation failures raise `FederatedLabelCoverageError` with a stable
`reason_code` and a field name only, for example `unknown_label`,
`label_not_in_round_schema`, `guideline_version_mismatch`,
`text_form_not_allowed`, `undeclared_tag_label`, and `critical_label_uncovered`.
Errors never echo labels, digests, or other submitted values.
