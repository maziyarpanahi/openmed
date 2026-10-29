# Private-training JSON schemas

External coordinators and air-gapped clients can validate the structure of
private-training metadata against versioned Draft 2020-12 JSON Schemas.
OpenMed exports one self-contained catalog covering round lifecycle, round
scheduling, round status, dense update metadata, aggregate metric envelopes,
and contribution floor contracts and band-only contribution declarations.

The catalog is import-light and deterministic. Each schema:

- has a unique, version-bound `urn:openmed:training:` identifier;
- uses only fragment-local references and requires no external resolution;
- pins `additionalProperties: false` and exact schema-version constants;
- binds enum values, field sets and individual bounds to the Python sources;
- renders byte-identically, so it can be embedded in offline artifacts.

The schemas describe public JSON projections. They never read model contents,
tensors, gradients, client identifiers, or local metrics. Metric schemas reject
released values in suppressed participant bands, enforce uncertainty presence,
exclude confidence endpoints, and constrain count and rate fields. Status
schemas enforce state-specific reason codes and suppressed participant counts.
Contribution floor schemas require the explicit override field for any floor
below the default of the declared privacy mechanism, and contribution
declarations accept bands only, never exact local counts.

## Catalog

```python
from openmed.training.private_training_schemas import (
    build_private_training_schemas,
    build_schema,
    render_private_training_schemas,
    render_schema,
)

catalog = build_private_training_schemas()
# keys: federated_round_lifecycle, federated_round_schedule,
#       federated_round_status, federated_update_metadata,
#       federated_aggregate_metric, federated_contribution_floor_policy,
#       federated_contribution_declaration
lifecycle = build_schema("federated_round_lifecycle")
stable = render_private_training_schemas()
```

Each builder returns a fresh mapping. Mutating an export cannot affect later
builds. Unknown names raise `PrivateTrainingSchemaError` without echoing input.

## Offline structural validation

Reject non-finite JSON literals and overflowing floats at the parsing boundary.
Enable format checking for calendar-valid schedule timestamps, and use a
registry that rejects external fetches:

```python
import json
import math

from jsonschema import Draft202012Validator
from referencing import Registry
from openmed.training.private_training_schemas import build_schema


def reject_nonfinite(value):
    raise ValueError("non-finite JSON number")


def parse_finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def reject_remote_resolution(uri):
    raise ValueError("external schema resolution is disabled")


raw_json = '{"schema_version":"openmed.training.federated_round.v1","state":"collecting"}'
payload = json.loads(
    raw_json, parse_constant=reject_nonfinite, parse_float=parse_finite_float
)
validator = Draft202012Validator(
    build_schema("federated_round_lifecycle"),
    registry=Registry(retrieve=reject_remote_resolution),
    format_checker=Draft202012Validator.FORMAT_CHECKER,
)
validator.validate(payload)
```

## Semantic validation

Schema validation is the first gate. Draft 2020-12 cannot express arbitrary
comparisons or arithmetic between fields. Python constructors and validators
remain required for schedule ordering and phase durations, count thresholds
and completion bands, sorted digest references, clipping and interval ordering,
shape products, total element counts, unique parameter names, update-policy
compatibility, the contribution batch floor not exceeding the example floor,
and contract and update digest bindings. JSON Schema also treats integral
numbers such as `1.0` as integers, while some Python contracts require actual
`int` values.

For aggregate metrics, pass the structurally validated mapping to
`FederatedMetricEnvelope.from_dict(payload)`. For updates, use
`FederatedUpdateMetadata.from_dict(payload, policy=policy)` with the caller's
local `FederatedUpdatePolicy`. For contributions, parse with
`FederatedContributionDeclaration.from_dict(payload)` and decide with
`check_federated_contribution()` against the coordinator's own
`FederatedContributionFloorPolicy`. For lifecycle, scheduling and round
status, construct them through the existing Python APIs from validated
aggregate inputs. All these checks run locally; no schema registry or network
is needed.
