# Federated aggregate metrics

OpenMed represents federated utility and safety measurements with a closed,
versioned envelope. The envelope contains one aggregate value and the controls
needed to interpret its release. It has no generic metadata field and no place
for site names, client identifiers, patient counts, local losses, gradients,
examples, endpoints, or per-client collections.

Build envelopes with `build_federated_metric_envelope()`. Direct construction is
rejected so callers cannot supply a participant band that disagrees with the
exact aggregate participant count used at the privacy boundary.

## Supported metric and privacy types

Metric kinds are closed to `count`, `rate`, and `bounded_mean`. Every metric has
finite, increasing clipping bounds, and the aggregate must fall inside them.
Counts require non-negative integer bounds and values. Rates require bounds and
values between zero and one. Bounded means may use any finite interval.

The envelope records one of three privacy mechanisms:

| Mechanism | Meaning |
| --- | --- |
| `threshold_only` | Minimum-group suppression without added noise |
| `laplace` | A Laplace mechanism was applied upstream |
| `gaussian` | A Gaussian mechanism was applied upstream |

`privacy_mechanism_version` is a required stable version such as `v1`. The
envelope records what was applied; computing noise and choosing privacy budgets
remain outside this module.

## Minimum-group rule

`participant_count` is accepted only by the builder and is never retained. If
it is smaller than `minimum_group_size`, which defaults to `5`, the aggregate
value and all uncertainty values are replaced with `null`. The returned band is
`suppressed`.

Released groups use coarse bands relative to the configured minimum:

| Band | Participant count |
| --- | --- |
| `minimum_to_under_double` | At least the minimum and less than twice it |
| `double_to_under_fourfold` | At least twice and less than four times the minimum |
| `fourfold_or_more` | At least four times the minimum |

This preserves useful scale information without publishing an exact participant
count.

## Uncertainty

Released metrics may use `none` or `confidence_interval`. A confidence interval
requires finite lower and upper bounds, must contain the aggregate value, and
must stay inside the clipping interval. Its confidence level must be strictly
between zero and one. Suppressed metrics use `suppressed` and retain no interval
or confidence level.

## Usage

```python
from openmed.training import (
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
    build_federated_metric_envelope,
)

envelope = build_federated_metric_envelope(
    metric_id="safe_completion_rate",
    metric_kind=FederatedMetricKind.RATE,
    aggregate_value=0.82,
    clipping_lower_bound=0.0,
    clipping_upper_bound=1.0,
    privacy_mechanism=FederatedPrivacyMechanism.LAPLACE,
    privacy_mechanism_version="v1",
    participant_count=25,
    minimum_group_size=5,
    uncertainty_method=FederatedUncertaintyMethod.CONFIDENCE_INTERVAL,
    uncertainty_lower_bound=0.75,
    uncertainty_upper_bound=0.88,
    confidence_level=0.95,
)

json_artifact = envelope.to_json()
restored = envelope.from_dict(envelope.to_dict())
```

`to_json()` sorts keys and ends with a newline, so equivalent envelopes produce
byte-identical JSON. `from_dict()` requires exactly the documented fields and
rejects unknown keys, enum values, versions, non-finite numbers, and invalid
cross-field combinations with value-free errors.

## Privacy boundary

The builder accepts only aggregate inputs. It immediately removes a value when
the participant minimum is not met and never stores the exact participant
count. This envelope does not aggregate client values, compute differential
privacy noise, choose privacy budgets, or decide release and promotion gates.

## Pooled evaluation counts

`openmed.training.federated_pooled_eval` produces envelopes for federation-wide
precision and recall without exposing any site's confusion counts or metrics.

1. Every participant agrees on a `PooledEvaluationLayout`: an ordered, closed
   label list and `max_count_per_entry`, the largest count one site may
   contribute to a single entry. Its `digest` binds the labels, the
   direct-identifier partition and the bound.
2. Each site counts strict exact-span matches on its committed local holdout
   with `build_site_count_vector()`, or encodes precomputed `(tp, fp, fn)`
   triples with `encode_site_counts()`. Encoding runs at the site and fails
   closed before anything leaves it when a count is negative, non-integer,
   above the bound, or uses an undeclared label.
3. Sites submit their `SiteCountVector` to an injected secure-sum port. The
   coordinator calls `pool_federated_evaluation()`, which sends the port a
   `SecureSumRequest` (layout digest, vector length, per-entry bound, quorum
   and declared privacy mechanism) and receives only a `SecureSumOutcome`: the
   pooled vector, the contributor count, and whether the port verified every
   contribution against the bound.
4. Pooled precision (`tp / (tp + fp)`) and recall (`tp / (tp + fn)`) are
   released through `build_federated_metric_envelope()` with Wilson score
   intervals and the declared mechanism.

The secure-sum protocol, disclosure budgeting and sealed evaluation are not
part of this module. The port is any object with a
`secure_sum(request) -> SecureSumOutcome` method; tests use an in-memory fake.

### Fail-closed rules

`pool_federated_evaluation()` raises `PooledEvaluationError` with a stable
`code` and a fixed message. Messages never contain labels, counts, site
identities, paths or port exception text.

| Code | Condition |
| --- | --- |
| `configuration_invalid` | Invalid layout, reference, quorum, support, confidence level or mechanism |
| `secure_sum_failed` | The port raised; its exception is not chained |
| `secure_sum_malformed` | The outcome has the wrong type, length or non-negative integer entries |
| `layout_mismatch` | The outcome was computed for another layout |
| `mechanism_mismatch` | The port applied a different mechanism or version |
| `quorum_not_met` | Fewer than `minimum_sites` contributed; checked before the vector is read |
| `contribution_out_of_bounds` | The port did not verify every bound, or a pooled entry exceeds `contributor_count * max_count_per_entry` |

Site-side encoding additionally uses `contribution_malformed`,
`contribution_out_of_bounds` and `undeclared_label`.

### Label slice suppression

Each released rate passes its pooled support (`tp + fp` for precision,
`tp + fn` for recall) to the envelope builder as the participant count, with
`minimum_label_support` as the minimum group size. A slice below the minimum
is suppressed with a `null` value and no interval. For these envelopes the
participant band therefore describes pooled support, while
`site_count_band` on the report describes contributing sites.

### Direct identifiers

Direct-identifier labels come from the canonical label catalog, and callers
may add custom ones with `extra_direct_identifier_labels`; a catalog direct
identifier cannot be reclassified. Metrics are partitioned:

| Partition | Metric ids |
| --- | --- |
| Direct identifiers | `pooled.di.label.<label>.precision`, `pooled.di.label.<label>.recall`, `pooled.di.micro.*` |
| Other labels | `pooled.other.label.<label>.precision`, `pooled.other.label.<label>.recall`, `pooled.other.micro.*` |

Micro aggregates pool counts only within their own partition, and no metric
mixes the two. `report.direct_identifier_recall_gate(minimum_recall=...)`
checks every direct-identifier label recall on its own. A low or suppressed
label fails the gate even when other labels, or the partition micro recall,
are high.

```python
from openmed.training.federated_pooled_eval import (
    PooledEvaluationLayout,
    build_site_count_vector,
    pool_federated_evaluation,
)

layout = PooledEvaluationLayout(
    labels=("EMAIL", "PHONE", "AGE", "DISEASE"),
    max_count_per_entry=5000,
)

# At each site, on its committed local holdout:
vector = build_site_count_vector(layout, site_documents)
secure_sum_client.submit(vector)

# At the coordinator:
report = pool_federated_evaluation(
    layout=layout,
    port=secure_sum_port,
    evaluation_ref="sha256:" + "0" * 64,
    minimum_sites=5,
    minimum_label_support=20,
)
gate = report.direct_identifier_recall_gate(minimum_recall=0.95)
artifact = report.to_json()
```

`site_documents`, `secure_sum_client` and `secure_sum_port` stand for the
deployment's own holdout loader and secure-sum transport.

With `threshold_only`, released rates are exact ratios of pooled counts, so a
released value can still reveal a pooled ratio. Declare `laplace` or
`gaussian` only when the port applies that mechanism and post-processes the
pooled vector into non-negative integers within the bound.
