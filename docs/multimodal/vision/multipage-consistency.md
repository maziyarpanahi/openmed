# Multipage document consistency

`openmed.multimodal.vision.documents.page_consistency` checks whether locally processed pages can be treated as a coherent packet. It compares packet-scoped identity digests, page numbering, template anchors, continuation cues and exact rendered pixel content. Mixed patient identifiers produce ordered candidate groups; conflicts or missing evidence quarantine the affected groups. Splitting never renumbers source pages or rewrites their coordinates and redaction masks.

This is a structural consistency check. It does not detect identifiers, qualify OCR or a renderer, certify patient identity, establish clinical correctness, redact pixels or authorize disclosure. The caller supplies qualified local OCR spans and actual rendered RGB pixels. Unknown observations remain unknown; absence of an observed identifier is not evidence that a page contains no identifier.

## Capture observations without retaining values

Use the existing [fused evidence](../fused-evidence-graphs.md) `EvidenceNode` and `PageGeometry` contracts for the original page and selected OCR spans. Every observation must match its original source digest, page and geometry. OCR offsets are half-open Unicode code-point offsets in the protected text passed to the capture helper. Arbitrary text fields cannot enter reports.

- `identity_from_span(text, evidence, key=..., kind=..., namespace_ref=...)` computes an HMAC from the actual selected identity span. `IdentityKind` distinguishes patient IDs, names and dates of birth. The namespace is an opaque reference to the identifier system, such as a locally configured patient-ID issuer; it is never a patient value.
- `anchor_from_span(...)` captures a selected template label and its original page-space region. Anchor positions are compared after normalization by source-page dimensions.
- `continuation_from_span(...)` captures an incoming or outgoing structural continuation key. Matching cues do not establish clinical continuity and cannot join different identities.
- `numbering_from_span(text, evidence)` reads a selected `1/3` or `Page 1 of 3` span. Other formats require a qualified caller parser producing `PageNumber` with the original evidence; the helper does not search arbitrary clinical text for number-like substrings.
- `capture_page(source, rgb_pixels, size, ...)` hashes actual row-major RGB8 pixels and retains only their digest and dimensions. It does not accept an image path, decode media or keep pixel bytes. The renderer remains responsible for faithfully mapping the original source into those pixels.

Use a random caller-managed key of at least 32 bytes, scoped to one packet and reused for its observations. The library neither generates nor persists it. The report and observation objects contain no key or selected text. A keyed scope fingerprint detects accidental comparisons across different packet keys. Identity, anchor and continuation tokens have distinct purposes and cannot be substituted for one another through the typed API.

Identity normalization is deliberately conservative: Unicode NFC and outer-whitespace removal only. Case and internal whitespace remain significant, including in patient IDs. There is no fuzzy identity match or unsalted hash of low-entropy identity values. Digests and source coordinates still carry linkage information and need appropriate access and retention controls. Python memory is not securely zeroized by this API.

## Grouping and quarantine rules

`assess_page_consistency(pages, policy=ConsistencyPolicy(identity_namespace=...))` returns a deterministic `PacketConsistency` report. A single unambiguous `PATIENT_ID` in the configured namespace anchors grouping. Names or dates alone do not automatically identify a patient. Different anchored identities create candidate boundaries with the explicit `identity_change` reason; uncertain identities create isolated candidates with `identity_uncertain` boundaries. Original order is preserved.

| Observation | Result |
| --- | --- |
| Missing or ambiguous primary identity | Quarantine; never inherit the nearest patient |
| Conflicting secondary fields anywhere in a group | Quarantine the complete group, even if an intervening page omitted that field |
| The same identity appears in separated groups | Quarantine its fragments; never silently reorder or reconnect them |
| Exact duplicate RGB pixels | Flag every occurrence, including copies assigned different identities; never delete a page silently |
| Missing, repeated, reversed or gapped page counters | Quarantine the incomplete or inconsistent group |
| Conflicting totals or a group that does not cover `1..total` | Quarantine; partial packets are not complete documents |
| Missing anchors, changed labels or displaced anchor geometry | Quarantine; layout changes are not assumed harmless |
| Missing, conflicting or cross-identity continuation endpoints | Quarantine the affected candidates |
| Rejected redaction-mask lineage | Preserve the refusal and quarantine the page's group |

Anchor matching compares every page pair, preventing small adjacent shifts from accumulating into a large undetected drift. `anchor_tolerance` is an explicit maximum normalized coordinate displacement and requires task qualification. It does not relax identity checks. Exact duplicate detection binds RGB bytes and dimensions, so container metadata cannot conceal the same raster; visually similar, rescaled or rescanned duplicates require a separate qualified similarity detector.

A same-patient packet containing different forms or independent page-number sequences may be quarantined. The library does not infer that such transitions are safe. A qualified intake workflow can supply separately identified documents for separate assessment.

`whole_packet_consistent` is true only for one complete group without findings. A mixed-patient packet cannot become eligible for whole-packet extraction merely because each separate group is coherent. `require_consistent_group(pages, index, policy=...)` reassesses the original observations before returning one explicitly selected consistent group. It does not accept a caller-edited report as authorization. Human review remains required.

## Executable synthetic example

This constructs two synthetic raster pages and selected OCR observations in memory. In a real workflow, replace the synthetic RGB and text with output from the qualified local renderer and OCR/detector. The fixed demonstration key must not be reused for real packets.

```python
from hashlib import sha256
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.documents.page_consistency import (
    ConsistencyPolicy, IdentityKind, anchor_from_span,
    assess_page_consistency, capture_page, identity_from_span,
    numbering_from_span, require_consistent_group,
)

key = b"synthetic-example-packet-key-2805-only"
namespace = "1" * 32
pages = []
for index in range(2):
    pixels = bytes([index + 1]) * (100 * 100 * 3)
    source = EvidenceNode(
        node_id=f"{index * 10 + 1:032x}", kind=NodeKind.PAGE,
        source_digest=sha256(pixels).hexdigest(),
        geometry=PageGeometry(page=0, width=100, height=100),
    )
    values = ["SYNTHETIC-ID-A", "Synthetic form", f"{index + 1}/2"]
    text = "|".join(values)
    evidence = []
    start = 0
    for slot, value in enumerate(values):
        evidence.append(EvidenceNode(
            node_id=f"{index * 10 + slot + 2:032x}", kind=NodeKind.OCR,
            source_digest=source.source_digest, geometry=source.geometry,
            region=(10, 10, 30, 20), span=(start, start + len(value)),
        ))
        start += len(value) + 1
    identity = identity_from_span(
        text, evidence[0], key=key, kind=IdentityKind.PATIENT_ID,
        namespace_ref=namespace,
    )
    pages.append(capture_page(
        source, pixels, (100, 100), identities=(identity,),
        anchors=(anchor_from_span(text, evidence[1], key=key),),
        numbering=numbering_from_span(text, evidence[2]),
    ))

pages = tuple(pages)
policy = ConsistencyPolicy(identity_namespace=namespace)
report = assess_page_consistency(pages, policy=policy)
assert report.whole_packet_consistent
assert require_consistent_group(pages, 0, policy=policy) == pages
assert "SYNTHETIC-ID-A" not in report.to_json()
```

## PDF coverage and redaction-mask provenance

`pdf_page_sources(source)` uses the existing bounded [PDF geometry reader](../pdf-geometry-preflight.md) on actual original bytes and returns deterministic page evidence in page-tree order. It requires fully readable geometry and uses displayed crop dimensions in points. Renderer, OCR and mask coordinates must use that explicitly declared source frame; original PDF user-space coordinates must not be substituted without the appropriate transform.

`assess_pdf_page_consistency(source, pages, policy=...)` additionally requires exact source-digest, page-count, geometry and page-tree-order coverage. Changed source bytes, omitted pages, duplicated source locations and reordered observations are refused. This verifies source structure and completeness; it does not independently verify that caller-supplied OCR text or RGB pixels were faithfully extracted from that PDF.

An optional existing [mask-lineage result](../redaction-mask-lineage.md) is recomputed from its original source nodes and transform steps before attachment. Its source must match the page; accepted target geometry must match the captured pixel dimensions. Group selection returns the same page observations and preserves source mask nodes, coordinates and transformations. No pixels are redacted or masks silently reprojected by this feature.

## Limits and reports

Packets are limited to 256 pages, each with at most 32 identity observations and 32 anchors. Rendered pages are limited to four million RGB pixels. Selected text spans are bounded to 4,096 code points within a protected text buffer of at most two million code points. Actual PDF geometry reads retain the existing 64 MiB source limit. Resource or contract errors return a fixed `PageConsistencyError` with no source or backend exception context and no partial assessment.

Reports contain a packet digest, original source-node references, packet indexes, controlled boundary reasons and finding codes. The caller resolves those references through its protected source store. Reports are not signatures or self-authenticating evidence: use the assessment functions with original captured observations and current policy before processing a group. No files, network calls, model downloads, telemetry or new dependencies are added.

Run `python -m pytest tests/unit/multimodal/vision/documents/test_page_consistency.py -q` for synthetic packet, provenance, privacy and negative controls.
