# ICD-11 offline grounding

OpenMed grounds diagnosis mentions to the ICD-11 Mortality and Morbidity
Statistics (MMS) linearization from a release-pinned local snapshot. The WHO
ICD-API is contacted only while you explicitly build or refresh that snapshot.
Loading, matching, de-identification, and FHIR export never make a network
request.

## Get WHO ICD-API credentials

Register at the [WHO ICD-API portal](https://icd.who.int/icdapi), create an API
client, and keep its client id and secret outside source control. The builder
uses WHO's OAuth2 client-credentials flow and reads credentials from environment
variables:

```bash
export WHO_ICD_CLIENT_ID="your-client-id"
export WHO_ICD_CLIENT_SECRET="your-client-secret"
```

Do not put credentials in a snapshot, command argument, notebook, or committed
configuration file.

## Build a pinned snapshot

Choose an explicit WHO release and the chapter codes needed by your deployment.
Repeat `--chapter` to include more than one chapter:

```bash
openmed icd11 build-snapshot \
  --release 2026-01 \
  --chapter 01 \
  --chapter 05 \
  --cache-dir /srv/openmed/icd11
```

The builder traverses only those MMS chapters and stores code-bearing entity
URIs, codes, canonical titles, and index/inclusion synonyms. It emits:

- `icd11-mms-<release>-<language>-<chapters>.json`, the canonical snapshot;
- a matching `.manifest.json` with the release, chapter subset, entity count,
  license marker, and SHA-256 integrity digest.

Both files are deterministic for the same API content, release, language, and
chapter subset. No build timestamp or local path is embedded. Set
`OPENMED_ICD11_CACHE_DIR` to change the default cache location.

The network client accepts HTTPS endpoints (plus loopback HTTP for a locally
deployed test service), rejects redirects so OAuth credentials cannot be
forwarded to another origin, caps each JSON response at 8 MiB, and stops a
chapter traversal above 100,000 entities. Generated snapshots are capped at
128 MiB.

Refresh by running the same command with a new explicit release id. Existing
release-pinned files remain separate, so production can validate and switch to
the new snapshot deliberately.

## Ground and export offline

Load the snapshot once and reuse its in-memory exact-match index:

```python
import os

from openmed.interop.icd11_api import (
    ground_to_codeable_concept,
    load_snapshot,
)

snapshot = load_snapshot(
    "/srv/openmed/icd11/icd11-mms-2026-01-en-01-05.json",
    expected_sha256=os.environ["OPENMED_ICD11_SNAPSHOT_SHA256"],
)
concept = ground_to_codeable_concept("type 2 diabetes", snapshot)
```

Matching applies Unicode normalization, case folding, and punctuation/space
normalization to canonical titles and synonyms. It does not use embeddings,
fuzzy matching, foundation-layer codes, postcoordination, or ICD-10 crosswalks.
An unmatched mention returns `None`.

The returned FHIR `CodeableConcept` uses
`http://id.who.int/icd/release/11/mms` as its coding system, the snapshot's
release as `Coding.version`, and a provenance extension containing only the
snapshot SHA-256. `CodeableConcept.text` is the canonical snapshot title; raw
note context and the caller's surface form are not copied into the export.

The loader verifies the sidecar manifest and digest before indexing. Move the
snapshot and its `.manifest.json` file together. A missing or mismatched pair
fails closed. The digest normalizes CRLF checkout line endings to LF so the
canonical JSON remains portable across platforms; every other byte change is
still detected. Snapshot files are limited to 128 MiB,
manifests to 64 KiB, and indexes to 100,000 entities; oversized or structurally
invalid artifacts fail before indexing.

The adjacent manifest detects accidental corruption and single-file changes;
it is not a signature. When a snapshot crosses a trust boundary, obtain the
builder's SHA-256 through a trusted channel and pass it as `expected_sha256` as
shown above. Do not import an untrusted snapshot and its manifest without an
independent digest pin.

## License and attribution

ICD-11 content is provided by the World Health Organization under the
[CC BY-ND 3.0 IGO license](https://icd.who.int/docs/icd-api/license/). Users are
responsible for complying with WHO's current
[ICD-11 license agreement](https://icd.who.int/en/docs/icd11-license.pdf),
including retaining each classification code, title, and URI. Published reports
must identify the work as the Eleventh Revision of the International
Classification of Diseases, credit WHO with the 2019 citation year, link to
`https://icd.who.int/browse11`, and state the CC BY-ND 3.0 IGO license. Use the
current exact citation text from section 1.3 of WHO's license agreement.

OpenMed does not redistribute the full classification. The bundled test
snapshot is small, synthetic, and unsuitable for clinical coding.
