# Reviewed ambient note FHIR export

The passive exporter projects **fixed, caller-reviewed note fixtures** into a
closed FHIR R4 (4.0.1) document subset: a document Bundle containing an ordered
Composition and a Provenance resource. It does not assemble notes, run ASR,
load models, resolve evidence, authenticate clinicians or execute an EHR write.
It has no network, file persistence, credentials, or cloud fallback.

Every document carries a non-diagnostic notice and a requirement for explicit
clinician confirmation before consequential use. Export is not clinical
validation, provider qualification, regulatory conformance or release evidence.

## Python and OpenMedKit

```python
import json
from pathlib import Path
from openmed.clinical.exporters.fhir.ambient import (
    ambient_draft_digest,
    export_ambient_document,
    import_ambient_document,
    validate_ambient_document,
)

# This checked-in fixture contains synthetic reviewed note text only.
fixture = json.loads(
    Path("tests/fixtures/clinical/ambient_fhir.json").read_text()
)
draft = fixture["draft"]
assert ambient_draft_digest(draft) == fixture["digest"]
result = export_ambient_document(
    draft,
    current_evidence_digest=draft["evidence_digest"],
    recorded_at="2026-01-02T03:04:05Z",
    # Omit this argument for preliminary status.
    confirmed_digest=fixture["digest"],
)
assert validate_ambient_document(result.bundle).valid
assert import_ambient_document(result.bundle) == draft
assert result.losses == (
    ("transcript_payload", 1),
    ("audio_payload", 1),
    ("review_history", 2),
)
```

OpenMedKit accepts the same fixture as UTF-8 JSON. The shared golden test checks
identical digests and FHIR dictionaries, including evidence links and UUIDs.

```swift
let digest = try AmbientFHIRDocument.draftDigest(draftJSON: reviewedFixtureJSON)
let result = try AmbientFHIRDocument.export(
    draftJSON: reviewedFixtureJSON,
    currentEvidenceDigest: currentLocalEvidenceDigest,
    recordedAt: "2026-01-02T03:04:05Z",
    confirmedDigest: digest
)
let valid = AmbientFHIRDocument.validate(documentJSON: result.documentJSON)
let restored = try AmbientFHIRDocument.importDocument(
    documentJSON: result.documentJSON
)
```

`result.bundle` and `documentJSON` contain protected clinical note text. Do not
log them. The Python result representation omits its Bundle; the Swift result
description contains a loss-row count only. Validation findings and exceptions
use fixed descriptions or controlled codes and never echo submitted fields.

## Closed v1 fixture

Every key below is required. Unknown keys are rejected at each object boundary.
There are no fields for source transcript text, audio, attachment metadata,
provider responses, paths, credentials or identifiers in plaintext.

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1` |
| `draft_ref` | Canonical lowercase `urn:uuid:` opaque draft reference |
| `evidence_digest` | 64 lowercase hex characters identifying the reviewed evidence snapshot |
| `sections` | Ordered array of 1–32 section objects |
| `loss_counts` | Ordered array of unique, positive controlled category counts |
| `review` | `null`, or `{digest, reviewer}` from a trusted local review workflow |
| `correction_pending` | Boolean from the current local correction ledger |

A section has exactly `code`, `note` and `evidence`. Supported section codes are
`history`, `exam`, `subjective`, `objective`, `assessment` and `plan`. Repeated
codes are allowed and preserve their order. `note` is the clinician-reviewed
note text, with 1–16,384 Unicode scalar values and valid XML 1.0 characters.
All sections together are bounded to 262,144 UTF-8 bytes. Narrative rendering
escapes markup; import restores the exact text rather than executing it.

Each section has 1–128 evidence rows, each with exactly:

- `reference`: canonical lowercase `urn:uuid:` evidence reference;
- `speaker`: canonical lowercase `urn:uuid:` session-local speaker alias;
- `start`, `end`: integer transcript offsets, `0 <= start < end <= 2^31 - 1`.

Offsets are caller-supplied Unicode scalar offsets in the referenced transcript
snapshot. The exporter preserves them without reading or resolving that
transcript. Section-level citations do not assert statement-level entailment.
Opaque references and aliases must be generated independently of patient
identifiers; UUID syntax alone cannot establish their privacy provenance.

The review object contains a lowercase SHA-256 `digest` and an opaque
`reviewer` UUID reference. A receipt is a **caller assertion**, not an exporter
signature or grant of reviewer authority. Authenticate the reviewer, establish
complete evidence support, and collect the reviewed note in the calling system.

## Review and freshness gates

The default is **preliminary**, even for a reviewed draft. Final export requires
an explicit `confirmed_digest` / `confirmedDigest` equal to the complete current
draft commitment. Both statuses refuse:

- absent review (`unreviewed_draft`);
- a receipt for a different note, section order, citation, speaker, offset or
  conversion-loss declaration (`stale_review`);
- an evidence digest different from the required current ledger digest
  (`stale_evidence`);
- pending corrections (`correction_pending`);
- confirmation of any other digest (`confirmation_mismatch`).

Read `current_evidence_digest` and `correction_pending` from the authoritative
local ledger immediately before export. Reusing a historical snapshot as its
own freshness check defeats that caller responsibility. This slice provides
no ledger, note-assembly or correction-propagation contract for #2816/#3682.

The digest excludes review and correction state so a review receipt can be
attached after computing it. It covers the draft reference, evidence digest,
ordered notes, section codes, complete evidence rows and ordered loss counts.
Its preimage begins with `openmed-ambient-v1\n`, then concatenates UTF-8
length-prefixed values (`byte_count:value`) in that order, with section,
evidence and loss-row counts framing the arrays. Integers use decimal ASCII.
This versioned framing is shared by Python and Swift.

The injected `recorded_at` instant must be valid UTC in
`YYYY-MM-DDTHH:MM:SSZ` form. No system clock is consulted.

## Declared FHIR subset and round trip

The Composition carries the opaque draft identifier, status, review assertion,
draft/evidence digests, clinician-confirmation flag, notice and ordered section
narratives. Evidence extensions preserve opaque references, speaker aliases and
offsets. Provenance identifies the reviewer by opaque logical identifier and
lists distinct opaque evidence identifiers. Its target resolves to the
Composition's deterministic Bundle fullUrl. The Bundle has an identifier and
injected timestamp; it contains no transaction, batch or request entries.

This subset intentionally has no Patient/Encounter mapping, attachments,
terminology inference, clinical coding, arbitrary extensions or embedded media.
There is no claim to IPS, US Core or a clinical-document implementation guide.
Python delegates general exchange checks and Bundle assembly to existing FHIR
helpers and adds the closed ambient subset check. Swift enforces the same
subset by reconstruction. These checks are structural and commitment checks,
not complete FHIR specification validation.

Import reconstructs the fixed fixture and rejects any document that differs
from its canonical re-export: altered statuses, digests, section order,
Provenance, attachments, extension payloads, narrative content or write
requests. A round trip preserves evidence references and review metadata; it
**does not authenticate the transported review assertion**. Re-export still
requires the caller's trusted current evidence and correction state. Neither
final status nor a transported confirmation grants permission for a clinical
operation or EHR write.

## Conversion losses and privacy

Loss categories are closed: `audio_payload`, `transcript_payload`,
`model_metadata`, `review_history`, and `unsupported_elements`. The caller
classifies unsupported elements and supplies positive counts (at most
`2^31 - 1`) without supplying their payloads. Duplicate or unknown categories,
boolean/non-integer counts and zero counts are refused. Empty reports are valid
when the caller declares no unsupported elements. No omitted-element count is
inferred or fabricated. Reports are returned as ordered category/count pairs
and preserved in controlled Composition extensions.

Audio and transcript payloads are never accepted or attached, including by
opt-in. Only reviewed note text appears in narrative; all additional mutable
metadata is restricted to codes, counts, offsets, opaque UUID references,
digests and the injected timestamp. PHI retained deliberately in the reviewed
note remains protected clinical output. This exporter does not de-identify
that note or inspect the unseen transcript for leakage.
