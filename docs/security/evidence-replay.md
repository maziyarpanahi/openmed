# Privacy evidence replay

An evidence-bundle integrity hash shows that an artifact was not changed. The
counts-only evidence replay verifier adds a second, narrower check: whether a
recorded set of synthetic policy decisions still produces the same aggregate
result under the current policy and local environment.

Replay is local and deterministic. It does not download models, call a
service, inspect source documents, or make a clinical or compliance decision.
The result is review evidence, not a certification or a guarantee of zero
re-identification risk.

## Build a manifest

Use synthetic category counts, never source text, identifiers, or individual
records. A manifest has schema version `1`, a versioned count-based policy,
safe local environment metadata (or an existing SHA-256 lock digest), synthetic
inputs, and expected aggregate decisions:

```python
from openmed.risk import build_evidence_manifest

manifest = build_evidence_manifest(
    policy={
        "id": "synthetic-privacy-policy",
        "version": "1",
        "rules": {"EMAIL": "mask", "PERSON": "mask"},
        "default_action": "keep",
    },
    environment={"runtime": "local", "offline": True},
    synthetic_inputs=[
        {"category_counts": {"PERSON": 2, "EMAIL": 1}},
        {"category_counts": {"UNKNOWN": 1}},
    ],
)
```

The helper records a policy fingerprint, environment fingerprint, expected
action counts, and a result fingerprint. Each `synthetic_inputs` item may
contain only `category_counts`, with non-negative integer values. Category and
action names are validated identifiers; arbitrary strings and payload-bearing
fields are rejected.

For an environment that is represented by a separately generated lock digest,
pass a value such as `sha256:<64 lowercase hex characters>` instead of a
metadata mapping. The digest is carried through without reading any external
resource.

## Replay and classify drift

Replay the manifest as-is, or provide current policy, environment, and
synthetic inputs explicitly:

```python
from openmed.risk import replay_evidence

report = replay_evidence(
    manifest,
    policy=current_policy,
    environment=current_environment,
    synthetic_inputs=synthetic_inputs,
)

if not report.matched:
    print(report.to_markdown())
```

`replay_evidence` also accepts a local JSON path. `verify_evidence_replay` is
an equivalent spelling for callers that prefer verifier terminology. A report
contains only aggregate action counts, input counts, safe identifiers, and
SHA-256 fingerprints. It does not contain the manifest's category counts,
individual input identifiers, policy rules, or source text.

Mismatches are returned in stable order and are classified as:

- `schema`: the manifest schema version is not supported;
- `environment`: the current environment fingerprint differs;
- `policy`: the current policy fingerprint differs; and
- `result`: action counts, input count, or the stable result fingerprint
  differs.

Malformed manifests and payload-bearing input shapes fail closed with
privacy-safe exceptions. Exception messages do not include input values or
file contents. Callers should retain the source manifest under their existing
access controls and publish only the aggregate replay report.

## Security boundary

A matching report establishes deterministic agreement for the supplied
synthetic counts and count-based policy. It does not prove that production
payloads were handled safely, that a policy is legally sufficient, or that a
model's clinical behavior is correct. Keep replay fixtures synthetic and
offline; do not add credentials, restricted datasets, or raw PHI to manifests,
tests, logs, or reports.
