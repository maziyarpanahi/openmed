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
