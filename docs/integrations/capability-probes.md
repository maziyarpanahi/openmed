# Local capability probes

OpenMed integrations can expose a small, deterministic report of which locally
declared optional adapters are usable. The report does not discover providers,
load credentials, import optional packages, or make network requests. Each
adapter receives a caller-supplied zero-argument probe, so the application
controls the local checks that are performed.

## Declare local adapters

```python
from openmed.integrations import CapabilityAdapter, probe_capabilities

report = probe_capabilities(
    [
        CapabilityAdapter(
            name="local-ner",
            provider="local-transformers",
            extra="hf",
            probe=lambda: True,
        ),
        CapabilityAdapter(
            name="local-table",
            provider="local-columnar",
            extra="columnar",
            probe=lambda: False,
        ),
    ]
)

print(report.counts)
print(report.to_json())
```

The probe callable should inspect only local state—for example, an already
installed package, a local model artifact, or an injected adapter's readiness
flag. It should not contact a remote endpoint. A mapping registry is also
accepted for small applications:

```python
report = probe_capabilities(
    {
        "local-ner": lambda: True,
        "local-table": {"available": False, "extra": "columnar"},
    }
)
```

Probes are synchronous, caller-controlled code. OpenMed does not discover or
import optional providers on their behalf, and it cannot make an arbitrary
callable network-safe. Applications must inject trusted local-only probes. One
report accepts at most 10,000 declarations and rejects duplicate normalized
capability names.

## Safe report fields

Reports sort capability entries by their normalized names and expose stable
`total`, `available`, and `unavailable` counts. A false result with an
`extra` declaration is classified as `missing_extra`; an `ImportError` is
classified the same way when an extra is declared. Other probe failures are
classified as `probe_error`. Probe exception messages are discarded.

Provider identifiers and versions are never serialized. Each is represented by
a deterministic `sha256:` provider fingerprint, allowing comparisons across
reports without placing configuration, tokens, or other sensitive values in
logs or artifacts. Provider and version components are Unicode-normalized and
hashed as a structured pair so embedded separators cannot create aliases.
Capability names and extra names should therefore be non-sensitive identifiers
such as package or integration labels. Values shaped like direct identifiers
are fingerprinted before reporting, and raw declaration objects use value-free
representations.

The overall report also has a stable fingerprint and can be serialized with
`report.to_json(indent=None)` for compact JSON. The report is diagnostic only;
it is not a compliance certification, clinical decision, or guarantee that a
provider remains available after the probe completes.
