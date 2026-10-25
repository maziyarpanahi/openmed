# Federated client round runner

`FederatedClientRunner` is the site-side lifecycle for one federated training
round. It receives a round offer, verifies it, decides whether the site
participates, drives local training, seals exactly one update and submits it.
Everything that touches data, models, keys or the network is an injected port:
manifest verification, capability compatibility, data-use policy, training,
clipping, the privacy mechanism, encoding, submission, the clock, cooperative
cancellation and a durable checkpoint store. The runner itself performs no
training, clipping, noise, masking, transport, identity or coordinator work.

The runner lives in `openmed.training.federated_client_runner`. It uses only
the standard library and the existing federated contracts:
`FederatedRoundSchedule`, `FederatedUpdatePolicy`, `FederatedUpdateMetadata`,
`FederatedRoundState` and `FederatedRoundReasonCode`.

## Client states

| State | Meaning | Terminal |
| --- | --- | --- |
| `received` | The offer was recorded and its manifest digest bound | No |
| `verified` | The manifest passed digest and signature checks | No |
| `declined` | The site will not participate in this round | Yes |
| `training` | The training port is running | No |
| `sealed` | One update is bound to the round and persisted | No |
| `submitted` | The sealed update was accepted by the submission port | Yes |
| `aborted` | The round ended without a submission | Yes |
| `expired` | The deadline passed or collection closed first | Yes |

Allowed transitions are published in `FEDERATED_CLIENT_TRANSITIONS`:

| From | To |
| --- | --- |
| `received` | `verified`, `declined`, `aborted`, `expired` |
| `verified` | `training`, `declined`, `aborted`, `expired` |
| `training` | `sealed`, `verified`, `aborted`, `expired` |
| `sealed` | `submitted`, `aborted`, `expired` |

A sealed round never returns to training, so its update digest is final. The
only backward edge is `training` to `verified`, used when a suspended or
crashed run left no update behind.

## Participation decisions

On every run, before the training port is called, the runner checks in order:

1. The manifest bytes still hash to the declared digest and to the digest
   bound at first receipt (`manifest_digest_drift`).
2. The round deadline, cancellation and wall-clock limit.
3. The coordinator round state from the submission port.
4. The manifest verification port (`manifest_unverified`).
5. The capability port (`capability_incompatible`).
6. The data-use policy port (`data_use_denied`).
7. The declared memory class (`memory_class_exceeded`) and step budget
   (`step_budget_exceeded`) against the site limits.
8. The privacy mechanism's site budget (`site_budget_exhausted`).

A port that raises, or returns anything other than `True`, counts as a
refusal. Steps 1 and 4 to 8 end in `declined`.

## Single-submission semantics

- The sealed payload and its `sha256:` digest are written to the checkpoint
  store before any submission attempt. If that write fails, the runner raises
  `checkpoint_store_failed` and submits nothing.
- A retry after a crash or a failed submission resends the stored,
  byte-identical payload. The training port is not called again, so fresh
  noise is never drawn for an already sealed round.
- `resend_update(offer, payload)` refuses any bytes whose digest differs from
  the bound digest with `update_digest_conflict`, and refuses a round with no
  sealed update with `update_not_sealed`.
- No submission happens once the coordinator reports `held` (`round_held`) or
  `aborted` (`round_aborted`), once collection has closed
  (`collection_closed`), or once the update-submission phase has ended
  (`deadline_passed`). Before the update-submission phase opens, the round
  stays `sealed` with `submission_window_not_open`.
- The payload is retained only while the round is `sealed`. Terminal records
  keep the digest and drop the bytes.
- A stored payload that no longer matches its digest is refused with
  `checkpoint_invalid` before any transmission.

## Cancellation and resource limits

`FederatedClientResourceLimits` declares a wall-clock budget per runner call,
a step budget and a memory class. The training port receives a
`FederatedTrainingControl` and must call `control.step()` before every local
step; it may call `control.report_memory_breach()`. The runner discards the
training result once a stop has been requested, even if the trainer catches
`FederatedTrainingStop`, so a partial update is never clipped, encoded, sealed
or submitted.

| Event | Resulting state | Reason |
| --- | --- | --- |
| Cancellation with `FederatedCancellationMode.ABORT` | `aborted` | `cancelled` |
| Cancellation with `FederatedCancellationMode.SUSPEND` | `verified` or `sealed`, resumable | `suspended` |
| Step budget exceeded during training | `aborted` | `step_budget_exceeded` |
| Wall-clock limit exceeded before sealing | `aborted` | `wall_clock_limit_exceeded` |
| Wall-clock limit exceeded after sealing | `sealed`, resumable | `wall_clock_limit_exceeded` |
| Memory breach reported | `aborted` | `memory_limit_exceeded` |
| Round deadline reached | `expired` | `deadline_passed` |
| Training, clipping, privacy or encoding port failure | `aborted` | `training_failed`, `clipping_failed`, `privacy_mechanism_failed`, `encoding_failed` |
| Encoded metadata does not match the payload or policy | `aborted` | `update_metadata_invalid` |
| Submission port failure | `sealed`, resumable | `submission_failed` |

A process that dies while training leaves a `training` checkpoint. The next
run moves it back to `verified`, re-runs every participation check and trains
again; nothing was sealed, so nothing was disclosed.

## Status records

`run_round()` returns a `FederatedClientRoundStatus` with a schema version,
the client state, an optional reason code, the round digest, the manifest
digest, the sealed update digest, a `resumable` flag and a
`round_reason_code`. For `declined`, `aborted` and `expired` rounds,
`round_reason_code` maps the client reason onto the abort reasons accepted by
`build_federated_round_status()`, and update digests use the same `sha256:`
format as aggregate digest references.

Records, reprs and errors never contain the round reference, site identity,
paths, example text, tensor values or local metrics. Errors are
`FederatedClientRunnerError` instances whose message is a fixed reason code;
port exception text is discarded.

## Usage

The example below wires every port to one synthetic in-memory fake.

```python
import hashlib
import json
from datetime import datetime, timezone

from openmed.training.federated_client_runner import (
    FederatedClientResourceLimits,
    FederatedClientRoundOffer,
    FederatedClientRunner,
    FederatedEncodedUpdate,
    FederatedMemoryClass,
)
from openmed.training.federated_round import FederatedRoundState
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

UTC = timezone.utc
parameter = FederatedParameterMetadata(name="lora.a", shape=(2,), dtype="float32")
policy = FederatedUpdatePolicy(
    model_digest="sha256:" + "a" * 64, parameters=(parameter,)
)
manifest = b'{"round": "synthetic-round-1"}'
offer = FederatedClientRoundOffer(
    round_ref="synthetic-round-1",
    manifest=manifest,
    manifest_digest="sha256:" + hashlib.sha256(manifest).hexdigest(),
    schedule=FederatedRoundSchedule(
        enrollment_starts_at=datetime(2026, 1, 1, tzinfo=UTC),
        update_submission_starts_at=datetime(2026, 1, 2, tzinfo=UTC),
        aggregation_starts_at=datetime(2026, 1, 3, tzinfo=UTC),
        evaluation_starts_at=datetime(2026, 1, 4, tzinfo=UTC),
        finishes_at=datetime(2026, 1, 5, tzinfo=UTC),
    ),
    update_policy=policy,
    local_steps=2,
    required_memory_class=FederatedMemoryClass.SMALL,
)


class SyntheticSite:
    def __init__(self):
        self.files = {}
        self.sent = []

    def verify(self, offer):
        return True

    def is_compatible(self, offer):
        return True

    def permits(self, offer):
        return True

    def has_budget(self, offer):
        return True

    def train(self, offer, control):
        for _ in range(offer.local_steps):
            control.step()
        return [0.25, -0.5]

    def clip(self, update, offer):
        return [max(-1.0, min(1.0, value)) for value in update]

    def privatize(self, update, offer):
        return [value + 0.125 for value in update]

    def encode(self, update, offer):
        payload = json.dumps({"lora.a": update}).encode()
        metadata = FederatedUpdateMetadata(
            model_digest=offer.update_policy.model_digest,
            adapter_format="dense",
            parameters=offer.update_policy.parameters,
            total_elements=2,
            update_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            clipped=True,
            policy=offer.update_policy,
        )
        return FederatedEncodedUpdate(payload=payload, metadata=metadata)

    def round_state(self, offer):
        return FederatedRoundState.COLLECTING

    def submit(self, offer, payload, update_digest):
        self.sent.append(payload)

    def now(self):
        return datetime(2026, 1, 2, 6, tzinfo=UTC)

    def load(self, key):
        return self.files.get(key)

    def save(self, key, data):
        self.files[key] = data


site = SyntheticSite()
runner = FederatedClientRunner(
    manifest_verifier=site,
    capability=site,
    data_use_policy=site,
    trainer=site,
    clipper=site,
    privacy_mechanism=site,
    encoder=site,
    submitter=site,
    clock=site,
    store=site,
    limits=FederatedClientResourceLimits(
        max_wall_clock_seconds=3600,
        max_steps=10,
        memory_class=FederatedMemoryClass.MEDIUM,
    ),
)

status = runner.run_round(offer)
assert status.state.value == "submitted"
assert runner.run_round(offer) == status
assert len(site.sent) == 1
status_json = status.to_json()
```

Running the round again returns the stored status without calling any port.

## Boundaries

The checkpoint store must replace values atomically; the runner assumes one
runner instance per store. The structured offer terms (schedule, update
policy, step count and memory class) are trusted only after the manifest
verification port has bound them to the signed manifest. Operator command-line
bindings are out of scope for this module. Minimum-group suppression,
differential privacy accounting and secure aggregation remain with their own
ports and modules.
