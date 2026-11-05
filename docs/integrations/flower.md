# Flower strategy bridge

`openmed.integrations.flower_strategy` lets a federation that already runs
[Flower](https://flower.ai/) keep OpenMed's update contracts in the loop. The
bridge does not replace Flower, make it the reference coordinator, or certify
its security. It maps Flower fit results onto the existing
[federated update metadata](../training/federated-update-metadata.md) and
[schema fingerprint](../training/federated-schema-fingerprints.md) checks and
refuses any round that does not satisfy them.

Operators install a compatible Flower release themselves, in their own
coordinator and site environments. The bridge is tested against the
`flwr>=1.42,<2` interfaces:

```bash
pip install openmed "flwr>=1.42,<2"
```

`openmed[flower]` is a marker extra only: it records the opt-in but installs no
framework code, so OpenMed never selects, pins, or upgrades Flower for you.
`import openmed` and `import openmed.integrations.flower_strategy` work without
Flower; the framework is imported only when you call `as_flower_strategy()`,
`as_flower_client()`, or rely on the default tensor serializer.

!!! note "Co-installing with Airflow"
    Flower 1.42 requires `fastapi>=0.138,<0.139`, while Airflow 3.3 requires
    `fastapi<0.137`. Installing both in one environment makes the resolver fall
    back to Airflow 3.2, which has published security advisories fixed in 3.3.
    Keep Flower and Airflow in separate environments.

## Operator responsibility

Transport security and deployment of the external framework remain the
operator's responsibility. OpenMed does not configure TLS, authenticate
SuperLink or SuperNode processes, manage certificates, harden the Flower
deployment, or vet Flower releases. Review the framework's own security
guidance and your federation's agreements before you connect sites. The bridge
validates declared update structure and content binding; it does not prove that
a site trained honestly, clipped correctly, or applied differential privacy.

## Coordinator: gate every round

Wrap the strategy you already use. The trusted policy must come from the
coordinator, never from incoming updates, and it must require clipping.

```python
from flwr.server.strategy import FedAvg

from openmed.integrations.flower_strategy import OpenMedFlowerStrategy
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdatePolicy,
)

# Synthetic digest reference and adapter shapes for this example only.
policy = FederatedUpdatePolicy(
    model_digest="sha256:" + "a" * 64,
    parameters=(
        FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
        FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
    ),
    require_clipped=True,
)

bridge = OpenMedFlowerStrategy(FedAvg(), policy=policy)
strategy = bridge.as_flower_strategy()
```

Pass `strategy` wherever Flower expects a `Strategy`. Before `aggregate_fit`
reaches the delegate, the bridge checks every `(client_proxy, fit_res)` pair:

1. `fit_res.metrics["openmed.update_metadata"]` holds metadata JSON that
   validates against the trusted policy: schema version, model digest, dense
   format, the exact parameter allowlist with shapes and dtypes, the element
   budget, and a `clipped: true` declaration.
2. The metadata's schema fingerprint equals the policy's fingerprint. Pass
   `expected_schema_fingerprint=` to pin a fingerprint recorded elsewhere; the
   constructor refuses a policy that does not match it.
3. `fit_res.parameters` carries one `numpy.ndarray` tensor per allowlisted
   parameter in name order, and a domain-separated SHA-256 over the parameter
   names and transported tensor bytes equals the declared `update_digest`.
4. The example count is a positive integer and no two results in the round
   carry the same update digest.

Any violation refuses the whole round. The delegate is not called, the bridge
returns `(None, {"openmed.round_status": "refused", "openmed.reason_code":
<code>})`, and Flower keeps the previous global parameters. Set
`raise_on_violation=True` to raise `FlowerBridgeError` instead.

Framework-reported failures also refuse the round by default. With
`accept_failures=True` failures are counted, but the delegate always receives
an empty failure list so an unvalidated payload cannot reach aggregation.
`configure_fit`, `configure_evaluate`, `aggregate_evaluate`, `evaluate`, and
`initialize_parameters` are forwarded unchanged; evaluation results carry no
model update and are not gated.

## Site: refuse unsafe updates

Each site wraps an OpenMed-aware trainer whose `fit` returns a
`FlowerClientUpdate` with arrays keyed by parameter name:

```python
from openmed.integrations.flower_strategy import (
    FlowerClientUpdate,
    OpenMedFlowerClient,
)


class LocalTrainer:
    def fit(self, parameters, config):
        arrays = train_and_clip_locally(parameters, config)  # your code
        return FlowerClientUpdate(arrays=arrays, num_examples=128, clipped=True)


client = OpenMedFlowerClient(LocalTrainer(), policy=policy).as_flower_client()
```

The wrapper raises `FlowerBridgeError` and returns nothing when the update does
not declare clipping, omits an allowlisted parameter, includes a parameter
outside the allowlist, or has a shape or dtype that differs from the policy.
Accepted arrays are returned in canonical name order with validated metadata
attached. Trainer metrics are dropped unless their names are listed in
`metric_allowlist`, and only `bool`, `int`, and `float` values are forwarded,
so free-text metrics cannot carry site or patient identifiers.

The digest is computed over the exact bytes Flower transports, using Flower's
`ndarray_to_bytes` by default. Pass `serializer=` only when your deployment
uses a different, deterministic tensor encoding on both sides.

## Transcript and logs

`bridge.transcript` holds content-free `FlowerRoundReceipt` records with the
round number, `aggregated` or `refused`, a fixed reason code, result and
failure counts, the schema fingerprint, and an order-independent digest over
the accepted update digests. The bridge never reads `ClientProxy` identifiers,
and receipts, log messages, and errors contain no client identifiers, metric
values, or tensor values. Errors from the delegate, trainer, or serializer are
replaced by fixed codes without chaining the original exception.

| Reason code | Meaning |
|---|---|
| `metadata_missing` / `metadata_invalid` | Metadata metric absent or not a valid envelope |
| `parameter_set_mismatch` | Declared names, shapes, or dtypes differ from the allowlist |
| `clipping_not_declared` | Update does not declare clipping |
| `model_digest_mismatch` | Update targets a different base model |
| `schema_fingerprint_mismatch` | Schema identity differs from the trusted fingerprint |
| `parameter_count_mismatch` | Tensor count differs from the allowlist |
| `update_digest_mismatch` | Transported tensor bytes do not match the declared digest |
| `invalid_tensor_payload` | Tensors are not bytes or use an unexpected tensor type |
| `duplicate_update` | Two results in one round carry the same update digest |
| `round_failures_present` | Framework reported failures and `accept_failures` is off |
| `delegate_aggregation_failed` | The wrapped strategy raised or returned a malformed value |
| `parameter_not_allowlisted` / `parameter_missing` | Site update outside or short of the allowlist |
| `parameter_shape_mismatch` | Site array shape or dtype differs from the policy |

## Scope

The bridge targets Flower's `Strategy` and `NumPyClient` interfaces and dense
adapter updates. Its tests use fake strategy and client protocols so they run
offline without Flower installed; they do not exercise a live Flower
deployment.
