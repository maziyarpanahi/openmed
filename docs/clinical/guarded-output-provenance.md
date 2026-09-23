# Guarded clinical-output provenance

`openmed.clinical.guarded_provenance` provides a deterministic provenance
boundary for summary and clinical-NLI review packets. A record binds a guarded
output to:

- the SHA-256 fingerprints of its input and generated output;
- safe model identifiers and a model descriptor fingerprint;
- one policy fingerprint;
- opaque evidence references with optional half-open source offsets; and
- an explicit human-review state and its ordered transitions.

The record is review metadata, not a clinical decision, compliance
certification, diagnosis, or treatment recommendation. The builder never calls
the network and never stores generated output, source text, prompts, reviewer
notes, paths, or arbitrary output metadata. Evidence identifiers and reviewer
identifiers are fingerprinted before the immutable records are created.

## Build a record

```python
from openmed.clinical import (
    build_guarded_provenance_manifest,
    check_guarded_provenance,
)

manifest = build_guarded_provenance_manifest(
    output={
        "output_kind": "summary",
        "input_text": "SYNTHETIC_INPUT",
        "summary_text": "SYNTHETIC_GENERATED_CLINICAL_OUTPUT",
        "evidence_ids": ["synthetic-evidence-1"],
    },
    evidence=[
        {
            "id": "synthetic-evidence-1",
            "start": 4,
            "end": 18,
            "text": "SYNTHETIC_EVIDENCE",
        }
    ],
    model={"model_id": "OpenMed/synthetic-summary-model", "revision": "v1"},
    policy={"profile": "synthetic-local-policy", "revision": 1},
)

print(manifest.to_json())
assert check_guarded_provenance(manifest).ok
```

The output text is used only to derive `output_hash`; it is not present in the
manifest or its `repr`. The default review state is `queued`, and an output
must not be treated as ready for clinical use until a human supplies the
contiguous `queued -> in_review -> approved` transitions. Rejected, abstained,
reopened, and needs-revision states remain explicit.

## Detect missing evidence and input drift

At construction, evidence references that are not in the supplied evidence
collection are recorded as opaque missing identifiers. A caller can check the
same manifest against a later local snapshot:

```python
report = check_guarded_provenance(
    manifest,
    current_input="SYNTHETIC_INPUT_AFTER_EDIT",
    current_evidence=[
        {
            "id": "synthetic-evidence-1",
            "start": 4,
            "end": 18,
            "text": "SYNTHETIC_EVIDENCE",
        }
    ],
)

assert report.input_changed
assert "changed_input" in report.reason_codes
assert not report.ok
```

Integrity reports contain only aggregate counts, fixed reason codes, hashes,
and opaque identifiers. Missing or changed evidence makes the report fail
closed and keeps `requires_human_review` true. `release_ready` is true only
when the manifest is intact and every record has an approved human-review
transition.

## Persist and reload locally

`write_guarded_provenance_manifest()` and
`load_guarded_provenance_manifest()` use ordinary local file I/O. The serialized
manifest has a version, per-record hashes, and a manifest hash so accidental
mutation can be detected without retaining protected clinical text.

```python
from openmed.clinical import (
    load_guarded_provenance_manifest,
    write_guarded_provenance_manifest,
)

write_guarded_provenance_manifest("guarded-provenance.json", manifest)
reloaded = load_guarded_provenance_manifest("guarded-provenance.json")
assert reloaded.verify_hash()
```

SHA-256 fingerprints can still be linkable for low-entropy values. Prefer
omitting hashes at upstream boundaries when correlation is not required, and
keep all review and clinical-use decisions with qualified human reviewers.
