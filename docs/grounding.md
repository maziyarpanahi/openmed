# Offline terminology grounding

OpenMed's grounding workbench is local-first and assistive. It suggests
terminology for qualified review; it is not a diagnosis, treatment, billing, or
autonomous clinical-coding system.

## Import a permissive snapshot

RxNorm, LOINC, ICD-10-CM, and HPO are supported as caller-managed snapshots.
The repository contains no terminology payloads. Import a local release once,
pin its release label, and verify its SHA-256 before using it offline:

```bash
openmed grounding import \
  --system icd10cm \
  --input /path/to/permissive/concepts.jsonl \
  --version synthetic-2026 \
  --sha256 sha256:<64 lowercase hex characters> \
  --cache-dir /path/to/openmed-grounding
```

The command writes a deterministic manifest beside the copied artifact. The
runtime validates that manifest and the artifact hash on every load. Public
downloads are also explicit and require a caller-provided checksum:

```bash
openmed grounding download \
  --system hpo \
  --url https://example.invalid/synthetic-hpo.jsonl \
  --sha256 sha256:<64 lowercase hex characters> \
  --version public-release \
  --cache-dir /path/to/openmed-grounding
```

Use `--cache-dir` or `OPENMED_GROUNDING_CACHE_DIR` to point the service and
command line at the same cache. Grounding defaults to `--offline`; the runtime
installs a socket guard for that operation and never attempts a missing
snapshot over the network.

## Python facade and FHIR CodeableConcept

Text and pre-extracted entity records use the same typed `GroundedSpan`
contract. Each result preserves character offsets, the selected canonical
system URI, code, display, confidence, ranked alternatives, section context,
and checksum/version provenance.

```python
from pathlib import Path
import hashlib

from openmed import ground
from openmed.clinical.exporters import to_codeable_concept
from openmed.clinical.grounding import VocabLoader, VocabSource

fixture = Path("openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl")
digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
loader = VocabLoader(
    cache_dir=Path(".openmed-grounding-demo"),
    local_only=True,
    registry={
        "icd10cm": VocabSource(
            system="icd10cm",
            path=fixture,
            sha256=digest,
            version="synthetic-fixture-1",
        )
    },
)

result = ground(
    "type 2 diabetes",
    systems=["icd10cm"],
    loader=loader,
    offline=True,
)
print(result.concepts[0].to_dict())
print(to_codeable_concept(result[0]))
```

The repository's `examples/offline_grounding.py` runs this same path with
synthetic text. Its output is deterministic for the same snapshot bytes.

`openmed.ground()` also accepts pre-extracted entities and can dispatch one
call across multiple local systems. The `lang` and `top_k` arguments are routed
to the local alias indexes and ranked candidates:

```python
from openmed import ground

entity_result = ground(
    [{"text": "type 2 diabetes", "start": 0, "end": 15, "label": "condition"}],
    systems=["icd10cm"],
    lang="en",
    top_k=1,
    snapshot=loader,
)

multi_loader = VocabLoader(
    cache_dir=Path(".openmed-grounding-multi-demo"),
    local_only=True,
    registry={
        system: VocabSource(
            system=system,
            path=fixture,
            sha256=digest,
            version="synthetic-fixture-1",
        )
        for system in ("rxnorm", "loinc", "icd10cm")
    },
)
multi_system_result = ground(
    "metformin 500 mg and type 2 diabetes",
    systems=["rxnorm", "loinc", "icd10cm"],
    lang="en",
    snapshot=multi_loader,
)
for concept in multi_system_result.concepts:
    print(concept.span.start, concept.span.end, concept.system, concept.code)
```

Raw text uses a deterministic vocabulary-backed local entity pass; it does not
download or initialize a model. `GroundingResult` remains sequence-compatible
with the established `GroundedSpan` exporters, while `result.concepts` exposes
typed `GroundedConcept` records with `top_k` alternatives and release
provenance. Grounding suggestions are assistive and require qualified human
review; they are not autonomous clinical coding, diagnosis, treatment, or
billing decisions.

## REST and CLI

`POST /ground` accepts either `text` or an `entities` list. The response is the
same `openmed.grounding.v1` mapping emitted by `openmed ground --json`; the
CLI adds only its standard command envelope. Request bodies, surface forms,
tokens, and credentials are not placed in access logs or grounding audit
provenance. Use the returned result payload for review and use
`GroundedSpan.to_audit_dict()` when a PHI-free audit record is required.

```bash
openmed ground \
  --text "type 2 diabetes" \
  --system icd10cm \
  --cache-dir /path/to/openmed-grounding \
  --json
```

## Restricted terminology

UMLS, SNOMED CT, CPT, and other restricted systems are never bundled or
downloaded. A request without an explicitly configured user-supplied
out-of-process terminology endpoint fails with `GroundingConfigError` before
any network operation. Pass a caller-owned bridge through
`terminology_bridge=` (or the `restricted_endpoint=` alias). License
credentials belong to that endpoint and are never accepted by the REST or CLI
payloads.
