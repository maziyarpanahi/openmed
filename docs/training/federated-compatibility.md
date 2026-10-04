# Federated capability compatibility

A round manifest describes what a round requires; a capability envelope
describes what one anonymous client can offer. `check_federated_compatibility()`
compares the two before enrollment and returns a deterministic, metadata-only
report with one finding per compared field, so undeclared capabilities fail
closed instead of being assumed satisfied.

```python
from openmed.training.federated_compatibility import (
    FederatedClientCapabilityEnvelope,
    FederatedResourceClass,
    FederatedRoundRequirement,
    FederatedSecureAggregationMode,
    FederatedTrainingBackend,
    check_federated_compatibility,
)
from openmed.training.federated_metrics import FederatedPrivacyMechanism

requirement = FederatedRoundRequirement(
    protocol_version=3,
    training_backend=FederatedTrainingBackend.TORCH,
    model_format="safetensors",
    adapter_format="lora",
    resource_class=FederatedResourceClass.MEDIUM,
    privacy_mechanism=FederatedPrivacyMechanism.GAUSSIAN,
    secure_aggregation=FederatedSecureAggregationMode.SHAMIR,
)
capability = FederatedClientCapabilityEnvelope(
    training_backends=(FederatedTrainingBackend.TORCH,),
    model_formats=("safetensors",),
    adapter_formats=("lora",),
    privacy_mechanisms=(FederatedPrivacyMechanism.GAUSSIAN,),
    secure_aggregation_modes=(FederatedSecureAggregationMode.SHAMIR,),
    minimum_protocol_version=1,
    maximum_protocol_version=5,
    resource_class=FederatedResourceClass.MEDIUM,
    deterministic_kernels=True,
)

report = check_federated_compatibility(requirement, capability)
print(report.verdict, report.ok, report.incompatible_fields, report.review_fields)
```

Findings are returned in the fixed field order
(`protocol_version`, `training_backend`, `model_format`, `adapter_format`,
`quantization_format`, `resource_class`, `deterministic_kernels`,
`privacy_mechanism`, `secure_aggregation`), and `incompatible_fields` and
`review_fields` follow that same order.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `compatible` | Every compared field was declared and supported. |
| `review_required` | No hard mismatch, but at least one optional capability difference needs a human decision. |
| `incompatible` | At least one mandatory field was undeclared, unsupported, or too small. |

`report.ok` is true only for `compatible`. A report is inconsistent if its
verdict does not match its findings, so a hand-built `FederatedCompatibilityReport`
with a mismatched verdict is rejected.

## Reason codes

| Reason code | Verdict | Meaning |
| --- | --- | --- |
| `protocol_version_supported` | compatible | The requirement version is inside the declared client range. |
| `protocol_version_below_minimum` | incompatible | The requirement version is below the client minimum. |
| `protocol_version_above_maximum` | incompatible | The requirement version is above the client maximum. |
| `protocol_version_unknown` | incompatible | The client declared no protocol range. |
| `training_backend_supported` | compatible | The required backend is declared. |
| `training_backend_unsupported` | incompatible | The required backend is not declared. |
| `model_format_supported` | compatible | The required model format is declared. |
| `model_format_unsupported` | incompatible | The required model format is not declared. |
| `adapter_format_supported` | compatible | The required adapter format is declared. |
| `adapter_format_unsupported` | incompatible | The required adapter format is not declared. |
| `quantization_supported` | compatible | The required quantization format is declared. |
| `quantization_unsupported` | incompatible | The required quantization format is not declared. |
| `resource_class_sufficient` | compatible | The declared class is at or above the required class. |
| `resource_class_insufficient` | incompatible | The declared class is below the required class. |
| `deterministic_kernels_supported` | compatible | The declared kernels satisfy the requirement. |
| `deterministic_kernels_missing` | incompatible | The round required deterministic kernels and the client does not declare them. |
| `privacy_mechanism_supported` | compatible | The required mechanism is declared. |
| `privacy_mechanism_unsupported` | incompatible | The required mechanism is not declared. |
| `secure_aggregation_supported` | compatible | The required mode is declared. |
| `secure_aggregation_unsupported` | incompatible | The required mode is not declared. |
| `capability_unknown` | incompatible | A mandatory declaration is missing, so the field cannot be satisfied. |
| `optional_capability_difference` | review_required | A declared capability is narrower or wider than the requirement without a hard mismatch. |

`FEDERATED_COMPATIBILITY_REASON_CODES` exposes the closed set in reason-code
order, and every reason maps to exactly one verdict.

## Mandatory declarations and fail-closed behavior

An empty tuple or a `null` scalar means "not declared", not "not needed", for
every mandatory field. A missing mandatory declaration reports
`capability_unknown` and yields `incompatible`; the validator never assumes a
default capability. When a field is declared but does not contain the required
value, the field reports its unsupported reason instead.

A round with no quantization requirement does not compare quantization as a
mandatory field: a client that declares quantization formats is only flagged with
`optional_capability_difference`, and a client that declares none produces no
finding at all.

Wider declarations are reported separately from mismatches. A client that
declares more than one supported value for a compared field, a resource class
above the requirement, or deterministic kernels for a round that does not require
them keeps its supported finding and additionally produces a non-required
`optional_capability_difference` finding, which makes the verdict
`review_required` rather than `incompatible`. `required=False` is set exactly for
that reason code, and a finding whose `required` flag disagrees with its reason is
rejected.

## Protocol versions

`MIN_PROTOCOL_VERSION` and `MAX_PROTOCOL_VERSION` bound every declared protocol
version on both sides. A client must declare `minimum_protocol_version` and
`maximum_protocol_version` together and in ascending order, and only a declared
range is compared: undeclared bounds are not treated as unbounded, they are
`protocol_version_unknown`.

## Canonicalization and limits

Capability collections are canonicalized on construction: duplicates are removed
and entries are sorted, enum collections by their value and token collections
lexicographically. A single envelope accepts at most
`DEFAULT_MAX_DECLARED_CAPABILITIES` entries per collection (32 by default, 128
hard ceiling). Token values are lowercase identifiers and are never echoed back
in an error message.

## Privacy

The report retains no client identifier, site name, hardware serial, path,
endpoint, patient count, local metric or training example, and it never echoes
client-declared values: findings carry only the compared field, a stable reason
code and the required flag. `FederatedClientCapabilityEnvelope` holds capability
metadata only, so a report can be shared with a coordinator without disclosing
which client produced it.

## Determinism

`to_json()` renders sorted-key indented JSON with a trailing newline, and both the
requirement and the capability envelope round-trip through `to_dict()`/`to_json()`
and `from_dict()`/`from_json()`; enum-valued fields accept either the enum member
or its serialized value. The same input produces the same report bytes, and a
golden digest in the focused tests pins that output.

## Not in scope

Enrollment, client registration, scheduling, aggregation, secure-aggregation
implementation, tensor transport and any judgment about the truthfulness of a
declared capability are out of scope. A `compatible` verdict says nothing about
whether a client will train deterministically or whether a round should run.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_compatibility.py -q
```
