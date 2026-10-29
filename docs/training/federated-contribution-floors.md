# Federated contribution floors

Updates computed from very few local examples or optimization steps are the
easiest to invert back to training text, especially in rounds without a
differential privacy (DP) mechanism. A round contract therefore declares
minimum local contribution floors, and two independent guards enforce them
over metadata only:

- the client-side guard takes the site's exact local counts and refuses to
  seal a declaration below any floor;
- the coordinator-side check rejects declared bands below its own floors.

Declarations carry only power-of-two bands, digest references and the schema
version. Exact local counts never leave the site.

## Round contract

`FederatedContributionFloorPolicy` holds three floors and the round privacy
mechanism from the existing `FederatedPrivacyMechanism` vocabulary.
`threshold_only` means no DP mechanism is declared; `laplace` and `gaussian`
are DP mechanisms. Omitted floors resolve to the defaults for that mechanism:

| Floor | No DP mechanism (`threshold_only`) | DP mechanism (`laplace`, `gaussian`) |
| --- | --- | --- |
| `min_local_examples` | 256 | 64 |
| `min_local_steps` | 8 | 1 |
| `min_effective_batch_size` | 32 | 8 |

Rationale:

- Published gradient-inversion attacks recover inputs most reliably from
  single-step updates over one or a few examples, and reconstruction quality
  drops as each step averages more examples and as more steps are combined.
- Without DP, the floors are the only contribution guard, so they require
  batches of at least 32 examples, at least 8 local steps, and 256 local
  examples (one pass of 8 steps at the batch floor).
- With a declared DP mechanism, clipping and noise bound per-example influence.
  The floors then only keep sampling from degenerating to a handful of
  examples: at least 64 local examples and batches of at least 8 (a sampling
  rate of at most one eighth).

The defaults are exported as `DEFAULT_NON_DP_CONTRIBUTION_FLOORS` and
`DEFAULT_DP_CONTRIBUTION_FLOORS`. Any mechanism that is not a known DP
mechanism uses the stricter values.

Raising a floor needs no extra field. Setting any floor below its default
requires the explicit policy field `allow_floors_below_default=True`, and that
field is rejected when no floor is lowered, so the contract always records
whether a lower floor was deliberate.

| Rule | Contract |
| --- | --- |
| Floor values | Integers that are band lower bounds: 1, 2, 4, ... 65,536 (`MAX_FEDERATED_CONTRIBUTION_FLOOR`) |
| Rejected floors | Booleans, zero, negatives, floats, strings, `null` in JSON, non-powers of two, and values above 65,536 |
| Batch floor | `min_effective_batch_size` must not exceed `min_local_examples` |
| Override field | Strict boolean; required to lower a floor and rejected when nothing is lowered |
| Binding | `contract_digest` is a domain-separated SHA-256 over the canonical contract |

```python
from openmed.training import (
    FederatedContributionFloorPolicy,
    FederatedPrivacyMechanism,
)

no_dp = FederatedContributionFloorPolicy()
assert (no_dp.min_local_examples, no_dp.min_local_steps) == (256, 8)

dp = FederatedContributionFloorPolicy(
    privacy_mechanism=FederatedPrivacyMechanism.GAUSSIAN
)
assert dp.min_effective_batch_size == 8

# Lowering a floor below its default must be explicit.
lowered = FederatedContributionFloorPolicy(
    min_local_steps=4, allow_floors_below_default=True
)
contract_json = lowered.to_json()
assert FederatedContributionFloorPolicy.from_json(contract_json) == lowered
```

The JSON projection lists every floor explicitly; `from_dict()` and
`from_json()` do not fill defaults from missing or `null` fields.

## Client-side guard

`seal_federated_contribution()` runs at the site. It receives the contract, the
update's declared digest reference and the exact local example count, local
step count and effective batch size (examples per optimization step across
gradient accumulation and devices). The batch size must not exceed the local
example count; a batch built by repeating examples adds no contribution.

```python
from openmed.training import (
    FederatedContributionFloorPolicy,
    FederatedContributionRefusedError,
    seal_federated_contribution,
)

contract = FederatedContributionFloorPolicy()
update_digest = "sha256:" + "b" * 64  # Synthetic reference for this example.

declaration = seal_federated_contribution(
    contract,
    update_digest=update_digest,
    local_examples=300,
    local_steps=10,
    effective_batch_size=40,
)
assert declaration.to_dict()["local_example_band"] == "256_to_under_512"

try:
    seal_federated_contribution(
        contract,
        update_digest=update_digest,
        local_examples=300,
        local_steps=2,
        effective_batch_size=40,
    )
except FederatedContributionRefusedError as refused:
    assert [code.value for code in refused.reason_codes] == [
        "local_steps_below_floor"
    ]
```

Below any floor, the guard raises `FederatedContributionRefusedError` and no
declaration exists. Its `reason_codes` list every failed floor in a stable
order. Malformed inputs (booleans, negatives, non-integers, counts above
`2**63 - 1`, a batch larger than the example count, or an invalid digest) raise
`FederatedContributionError` instead. Neither error, nor its message, args or
attributes, contains a count or floor value.

## Coordinator-side check

The coordinator parses the declaration with
`FederatedContributionDeclaration.from_dict()` or `from_json()`, which check
structure only, then calls `check_federated_contribution()` with its own
contract and the expected update digest:

```python
from openmed.training import (
    FederatedContributionDeclaration,
    FederatedContributionFloorPolicy,
    check_federated_contribution,
)

contract = FederatedContributionFloorPolicy()
update_digest = "sha256:" + "b" * 64
payload = {
    "schema_version": "openmed.training.federated_contribution.v1",
    "contract_digest": contract.contract_digest,
    "update_digest": update_digest,
    "local_example_band": "256_to_under_512",
    "local_step_band": "1_to_under_2",
    "effective_batch_size_band": "32_to_under_64",
}

declaration = FederatedContributionDeclaration.from_dict(payload)
decision = check_federated_contribution(
    declaration, contract=contract, update_digest=update_digest
)
assert decision.to_dict() == {
    "accepted": False,
    "reason_codes": ["local_steps_below_floor"],
}
```

The check does not trust the client guard. Because every floor is a band lower
bound, a declared band is at or above a floor exactly when every count in that
band is. The decision lists every failure in this stable order:

| Reason code | Meaning |
| --- | --- |
| `contract_mismatch` | The declaration was sealed against a different contract |
| `update_mismatch` | The declaration is bound to a different update digest |
| `local_examples_below_floor` | The local example band is below the example floor |
| `local_steps_below_floor` | The local step band is below the step floor |
| `effective_batch_size_below_floor` | The effective batch size band is below the batch floor |

Digest comparisons use a constant-time primitive. A rejected contribution can
be excluded from aggregation; round-level abort or hold reporting stays with
[federated round status](federated-round-status.md).

## Bands and privacy boundary

Bands are power-of-two ranges named by their bounds, from `1_to_under_2`
through `32768_to_under_65536`, plus the open top band `65536_or_more`.
`FederatedContributionBand.for_count()` maps a positive count to its band;
each band's `lower_bound` is a valid floor value.

Declarations accept exactly the schema version, contract digest, update digest
and the three bands. Exact count fields, site or client identifiers, paths,
examples, tensors and local metrics are rejected as unknown fields with a fixed
message. Decisions contain only an acceptance flag and reason codes. The module
performs no logging, filesystem access, network calls or training.

The declaration is still a client claim: the coordinator cannot verify the
counts behind a band. The client guard keeps honest sites from submitting
low-contribution updates; the coordinator check keeps a declared band below the
floors from being aggregated. Capability envelopes, local training and DP
mechanisms are separate work. Both payloads are part of the
[private-training JSON schemas](private-training-schemas.md) catalog as
`federated_contribution_floor_policy` and `federated_contribution_declaration`.
See also [federated update metadata](federated-update-metadata.md).
