# Multimodal and ambient threat model

This STRIDE-style model covers the v3.2 media and ambient boundaries. It maps
existing, limited controls to enforcing modules and offline tests, and links
unfinished controls to tracked gaps. It adds no enforcement, recording permission,
clinical validation, provider qualification or release approval. The release
scope remains [#3194](https://github.com/maziyarpanahi/openmed/issues/3194).

## Assets, adversaries and trust boundaries

Protect source media, hidden document content, model artifacts and configuration,
microphone consent state, audio buffers, voice biometrics, transcripts, draft
notes, source coordinates and clocks, review decisions, exports and audit evidence.
An input author can supply hostile media or embedded instructions; an artifact
supplier can substitute weights or preprocessing configuration; a participant can
spoof a speaker; an output reader can link supposedly anonymous modalities; an
operator or integration can accidentally bypass consent, review or deletion.

| Boundary | Untrusted input and required separation |
| --- | --- |
| TB1: media to parser/decoder | Headers, archives, pixels, document layers and waveform metadata are untrusted. Bounded preflight does not isolate native code. |
| TB2: artifacts to provider | Weights, tokenizer, preprocessing and prompts need pinned provenance. Source instructions never grant tool authority. |
| TB3: device to session | Microphone permissions do not establish participant consent. Capture, pause, interruption, participant changes and withdrawal need session policy. |
| TB4: session to storage/evidence | Audio, embeddings, transcripts and drafts are sensitive payloads. Logs and audit evidence admit only controlled codes, counts, offsets, opaque references and digests. |
| TB5: modalities to fused evidence | Page coordinates, frame selection, timing, speaker aliases and revisions must remain bound to their source; a digest alone cannot establish identity or truth. |
| TB6: draft to review/export | Generated text is untrusted, non-diagnostic and draft-only. Explicit reviewer confirmation must bind the exact consequential output before any downstream action. |

The baseline contains media helpers and result envelopes, not a complete ambient
pipeline. Python and OpenMedKit deployments share these boundaries; this page
changes neither platform API. Every integration must assess its actual adapters,
device permissions, retention and export paths. Local-first means no mandatory
network after approved model download, no default telemetry, and no PHI cloud
fallback. Apple Foundation Models processing must remain on-device.

## Threat/control matrix

STRIDE categories are Spoofing, Tampering, Repudiation, Information disclosure,
Denial of service and Elevation of privilege. Stable MA identifiers support
review. A **Control** names both a repository module and its enforcing test;
**Gap** names an issue. Limited controls do not imply end-to-end mitigation.
Residual risks are deployment conditions, not blanket risk acceptance.

| ID | STRIDE | Threat and boundary | Control or tracked gap; limits |
| --- | --- | --- | --- |
| MA-01 | Spoofing | A declared media type or digest disguises different bytes at TB1. | Control: [`openmed/multimodal/preflight.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/preflight.py); [`tests/unit/multimodal/test_preflight.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_preflight.py) tests media mismatch and digest mismatch abstention. An accepted manifest does not authenticate its author. |
| MA-02 | Denial of service | Oversized media or missing geometry reaches a decoder at TB1. | Control: [`openmed/multimodal/asset_limits.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/asset_limits.py); [`tests/unit/multimodal/test_asset_limits.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_asset_limits.py) tests ceilings and unevaluable PDF pixels. Gap: [#3832](https://github.com/maziyarpanahi/openmed/issues/3832) binds admission to redaction entrypoints; callers can otherwise bypass the helper. |
| MA-03 | Elevation of privilege / Denial of service | Native PDF, TIFF, DICOM, WSI, audio or video decoding crashes or escapes the parent process at TB1. | Gap: [#3206](https://github.com/maziyarpanahi/openmed/issues/3206), hostile-media isolation and resource tests. Residual R1: OS sandboxing and malware defenses remain operator responsibilities. |
| MA-04 | Denial of service | Nested or expanding document archives exhaust resources at TB1. | Gap: [#3812](https://github.com/maziyarpanahi/openmed/issues/3812), ZIP expansion admission for DOCX, XLSX, PPTX, EPUB and ODT. Header limits alone do not bound decoded archive work. |
| MA-05 | Information disclosure | Source metadata survives an image or document export at TB4/TB6. | Control: [`openmed/multimodal/metadata_scrub.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/metadata_scrub.py); [`tests/unit/multimodal/test_metadata_scrub.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_metadata_scrub.py) tests image EXIF/XMP, PDF properties and DOCX properties. Gap: [#3817](https://github.com/maziyarpanahi/openmed/issues/3817) covers whole-slide label/macro metadata. This is not an exhaustive carrier inventory. |
| MA-06 | Information disclosure | Burned-in identifiers, barcodes or missed image regions survive OCR/redaction at TB1/TB6. | Gap: [#2795](https://github.com/maziyarpanahi/openmed/issues/2795), coverage audit; [#3816](https://github.com/maziyarpanahi/openmed/issues/3816), barcode regions. Residual R2: detector recall and human review cannot prove absence of all identifiers or linkage risk. |
| MA-07 | Information disclosure | A PDF looks covered while selectable text remains underneath at TB6. | Control: [`openmed/multimodal/verify_pdf.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/verify_pdf.py); [`tests/unit/multimodal/test_verify_pdf.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_verify_pdf.py) tests residual text, unchanged regions, empty spans and value-free reports. Verification covers supplied regions; it is not hidden-content discovery. |
| MA-08 | Information disclosure | PDF annotations, forms, attachments/revisions or hidden OOXML parts bypass visible-text redaction at TB1/TB6. | Gap: [#3824](https://github.com/maziyarpanahi/openmed/issues/3824), PDF inventory, and [#3810](https://github.com/maziyarpanahi/openmed/issues/3810), OOXML residual refusal. Pending PRs do not establish baseline enforcement. |
| MA-09 | Elevation of privilege / Tampering | OCR, visual text or spoken instructions redirect a model or tool workflow at TB2/TB6. | Gap: [#2804](https://github.com/maziyarpanahi/openmed/issues/2804), instruction quarantine. A content-free envelope is not a prompt-injection guard; source content must never authorize tool dispatch. |
| MA-10 | Tampering / Elevation of privilege | Substituted VLM/ASR weights, tokenizer or preprocessing alter inference at TB2. | Gap: [#3202](https://github.com/maziyarpanahi/openmed/issues/3202) and [#3203](https://github.com/maziyarpanahi/openmed/issues/3203), qualified offline providers, plus [#3684](https://github.com/maziyarpanahi/openmed/issues/3684), preprocessing vectors. Existing text-model integrity is not evidence for every media adapter. Residual R3: approved artifacts can still be malicious or inaccurate. |
| MA-11 | Information disclosure | A provider places source content or free-form errors in diagnostics at TB4. | Control: [`openmed/multimodal/provider_result.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/provider_result.py); [`tests/unit/multimodal/test_provider_result.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_provider_result.py) rejects unknown fields and sensitive payloads without echo. This protects envelope serialization only, not provider-internal logging or returned text. |
| MA-12 | Tampering | A stale cache result survives a model, policy or preprocessing change at TB4. | Control: [`openmed/multimodal/cache_key.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/multimodal/cache_key.py); [`tests/unit/multimodal/test_cache_key.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/multimodal/test_cache_key.py) tests version/option invalidation and rejects arbitrary metadata. Keys do not enforce retention, authenticate cache writers or encrypt content. |
| MA-13 | Spoofing / Repudiation | Device permission is mistaken for consent; capture resumes after withdrawal, participant change or background interruption at TB3. | Gap: [#3208](https://github.com/maziyarpanahi/openmed/issues/3208), consent/session semantics, and [#3680](https://github.com/maziyarpanahi/openmed/issues/3680), capture binding. Residual R4: consent validity and recording-law obligations remain operator responsibilities. |
| MA-14 | Information disclosure / Repudiation | Buffers, caches, temporary media or crash artifacts retain audio/transcripts/drafts after cancellation, failure or deletion at TB4. | Gap: [#2818](https://github.com/maziyarpanahi/openmed/issues/2818), retention and destruction accounting, with [#3208](https://github.com/maziyarpanahi/openmed/issues/3208), reference invalidation. Residual R1: receipts cannot prove erasure of swap, backups or caller/provider copies. |
| MA-15 | Spoofing / Information disclosure | Persistent voice embeddings enable identification or cross-session speaker linkage at TB3/TB5. | Gap: [#3818](https://github.com/maziyarpanahi/openmed/issues/3818), session-scoped biometric handles, and [#3676](https://github.com/maziyarpanahi/openmed/issues/3676), uncertain diarization. Speaker similarity does not authenticate a participant. |
| MA-16 | Tampering / Information disclosure | Partial, revised, low-confidence or wrong-language transcripts release identifiers or change doses/negation at TB4/TB6. | Gap: [#2815](https://github.com/maziyarpanahi/openmed/issues/2815), unstable-span withholding; [#2814](https://github.com/maziyarpanahi/openmed/issues/2814), revision lineage; [#3819](https://github.com/maziyarpanahi/openmed/issues/3819) and [#3820](https://github.com/maziyarpanahi/openmed/issues/3820), language and critical-token review. No transcript accuracy qualification is claimed. |
| MA-17 | Tampering | Non-speech hallucinations, repetition loops or overlap stitching create unsupported statements at TB5/TB6. | Gap: [#3815](https://github.com/maziyarpanahi/openmed/issues/3815), hallucination checks, and [#3675](https://github.com/maziyarpanahi/openmed/issues/3675), stitching. Measure ASR errors under [#3821](https://github.com/maziyarpanahi/openmed/issues/3821); synthetic tests do not establish clinical accuracy. |
| MA-18 | Spoofing / Tampering / Repudiation | Clock drift, swapped sources, wrong image coordinates or speaker roles produce false cross-modal citations at TB5. | Gap: [#2819](https://github.com/maziyarpanahi/openmed/issues/2819), evidence clocks; [#2796](https://github.com/maziyarpanahi/openmed/issues/2796), fused coordinates; [#2817](https://github.com/maziyarpanahi/openmed/issues/2817), uncertain roles. Digests establish byte equality only, not acquisition authenticity. |
| MA-19 | Tampering / Elevation of privilege | Stale review, corrections or unsupported claims become a finalized note or clinical export at TB6. | Gap: [#3681](https://github.com/maziyarpanahi/openmed/issues/3681), review component; [#3682](https://github.com/maziyarpanahi/openmed/issues/3682), correction propagation; [#3837](https://github.com/maziyarpanahi/openmed/issues/3837), reviewed export; [#3822](https://github.com/maziyarpanahi/openmed/issues/3822), draft evaluation. Explicit reviewer confirmation of the exact output and non-diagnostic notices are required; no autonomous clinical action. |
| MA-20 | Information disclosure | VLM output reconstructs identifiers or raw retained audio remains identifiable despite safe text at TB6. | Gap: [#3813](https://github.com/maziyarpanahi/openmed/issues/3813), generated-text screening, and [#3838](https://github.com/maziyarpanahi/openmed/issues/3838), spoken-identifier muting. Residual R2: text de-identification does not anonymize faces, voices or linked modalities. |

## Unsupported deployments and residual risks

- **R1: host compromise and incomplete erasure.** Untrusted decoders in a shared
  PHI process without effective OS isolation, resource limits and controlled
  temporary storage are unsupported. This model and #3206 do not replace OS
  sandboxing or malware scanning. Host administrators, hostile in-process code,
  device theft, swap, crash dumps and backups remain outside library guarantees.
- **R2: inference and re-identification.** Unreviewed media/transcript exports,
  claimed universal identifier recall, biometric anonymity or clinical accuracy
  are unsupported. Inspect burned-in/hidden content and linked evidence before
  release. Negative fixtures and recall measurements bound evidence, not risk to
  zero. Digests can themselves enable linkage and need access/retention controls.
- **R3: artifact/provider trust.** Unpinned, unqualified or remotely executing
  VLM/ASR artifacts and cloud fallback for PHI are unsupported. Use permissively
  licensed or caller-supplied approved assets; do not bundle restricted weights or
  datasets. Offline execution does not make a model trustworthy. No provider or
  hardware qualification, benchmark result or release readiness is asserted here.
- **R4: capture and review authority.** Invisible/background capture without
  validated consent and interruption handling, indefinite audio/embedding
  retention and cross-session biometric identification are unsupported. The host
  must enforce consent and reviewers' authority. Confirmation is an explicit
  decision bound to current evidence and output, never inferred from a UI visit.
  Automated diagnosis, note finalization and EHR writes are unsupported.

Operators must document disposition of these risks for their deployment; this
page does not accept them on behalf of participants. All gaps remain gaps until
the enforcing code and tests land and the mapping is reviewed.

## Pending work and evidence maintenance

At this page's baseline, separate proposals include hidden PDF inventory
([PR #4043](https://github.com/maziyarpanahi/openmed/pull/4043)), OOXML refusal
([PR #4041](https://github.com/maziyarpanahi/openmed/pull/4041)), voice handles
([PR #4044](https://github.com/maziyarpanahi/openmed/pull/4044)), ASR contract
([PR #3634](https://github.com/maziyarpanahi/openmed/pull/3634)), and reviewed
ambient export ([PR #4050](https://github.com/maziyarpanahi/openmed/pull/4050)).
These links record ownership, not merged controls or accepted evidence.

The reference gate checks every inline repository module/test path and source
link on this page, and requires each threat row to have a module/test pair or an
issue link. Deleting a referenced module or test fails the offline gate. It does
not run or semantically qualify the linked control tests; changes to behavior
require human mapping review and the relevant test suites.

The enforcing docs gates are
[`tests/unit/test_multimodal_threat_model.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/test_multimodal_threat_model.py),
[`tests/integration/test_multimodal_threat_model.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/integration/test_multimodal_threat_model.py)
and [`tests/unit/test_docs_publication.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/test_docs_publication.py).

```sh
.venv/bin/python -m pytest tests/unit/test_multimodal_threat_model.py tests/integration/test_multimodal_threat_model.py tests/unit/test_docs_publication.py -q
make docs-build
```

The page includes no source payload examples, sample PHI, credentials or private
paths. Tests use synthetic reference trees and deletion mutations, not media or
model output. Follow the [disclosure policy](disclosure-policy.md) for security
reports. Related analyses cover the [redactor](threat-model.md),
[multilingual ingestion](threat-model-multilingual-ingestion.md) and
[agent boundary](agent-threat-model.md).
