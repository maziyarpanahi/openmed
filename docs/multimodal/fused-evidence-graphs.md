# Fused image-text evidence

`openmed.multimodal.evidence` keeps pixel regions, page geometry, OCR spans, metadata fields, normalized facts and fused claims distinct. The graph stores source coordinates and typed derivation links. Clinical text, values, pixels, filenames and metadata field names stay in the caller's protected storage.

This is a Python provenance contract. OpenMedKit parity is tracked separately in #3679. It performs no model inference, network access, file access or logging. A `cited` claim has a structurally valid citation; it is not clinically verified, cleared for disclosure or approved for an autonomous clinical action.

## Construct a source-linked claim

The references and digest below are synthetic. Assign opaque node references independently of patient identifiers; do not encode names, record numbers or clinical values into identifiers. A source digest identifies the exact source asset, not a short clinical value. Treat digests and coordinates as sensitive linkage metadata under the caller's access and retention policy.

```python
from openmed.multimodal.evidence import (
    DerivationEdge,
    EdgeKind,
    EvidenceNode,
    FusedEvidenceGraph,
    NodeKind,
    PageGeometry,
)

pixel_id, ocr_id, fact_id, claim_id = [f"{i:032x}" for i in range(1, 5)]
page = PageGeometry(page=0, width=800, height=1000)
nodes = (
    EvidenceNode(
        node_id=pixel_id,
        kind=NodeKind.PIXELS,
        source_digest="a" * 64,
        geometry=page,
        region=(20, 30, 300, 80),
    ),
    EvidenceNode(
        node_id=ocr_id,
        kind=NodeKind.OCR,
        source_digest="a" * 64,
        geometry=page,
        region=(25, 35, 290, 75),
        span=(0, 12),
    ),
    EvidenceNode(node_id=fact_id, kind=NodeKind.FACT),
    EvidenceNode(node_id=claim_id, kind=NodeKind.CLAIM),
)
graph = FusedEvidenceGraph(
    nodes=nodes,
    edges=(
        DerivationEdge(source=pixel_id, target=ocr_id, kind=EdgeKind.OCR),
        DerivationEdge(source=ocr_id, target=fact_id, kind=EdgeKind.EXTRACTS),
        DerivationEdge(source=fact_id, target=claim_id, kind=EdgeKind.SUPPORTS),
    ),
)
claim = graph.assess_claims()[0]
assert claim.state.value == "cited"
assert claim.source_node_ids == (pixel_id, ocr_id)
report = graph.to_json()
```

## Coordinates and derivation rules

Page indices are zero-based. Regions use half-open `(left, top, right, bottom)` coordinates with a top-left origin, in declared pixels or PDF points. Regions must lie within their declared page. OCR spans use half-open Unicode code-point offsets in the caller's protected OCR text, not UTF-8 byte offsets or normalized-fact offsets. The caller must bind that text to the node reference and verify the span against its actual length; this content-free contract cannot inspect it.

Every source node carries a SHA-256 asset digest. Page, pixel and OCR nodes also carry page geometry. All geometry declarations for the same source asset and page must agree. Metadata nodes carry a bounded field ordinal instead of a field name or value. Facts and claims carry only opaque references and must inherit citations through valid derivations.

| Edge | Allowed source → target | Meaning |
| --- | --- | --- |
| `locates` | Page geometry → pixels | Locate a region in its source page. |
| `ocr` | Pixels → OCR span | Bind recognized text offsets to a contained region on the same source page. |
| `extracts` | OCR span or metadata → normalized fact | Extract a fact from declared textual or metadata evidence. |
| `interprets` | Pixels → normalized fact | Identify a visual inference separately from OCR extraction. |
| `normalizes` | Normalized fact → normalized fact | Retain original source ancestry through normalization. |
| `supports` | Pixels, OCR, metadata or fact → fused claim | Cite evidence for a fused claim. |
| `conflicts` | Any two distinct nodes | Declare an unresolved conflict; this never supplies a citation. |

Edges point from evidence to its derivative. A pixel-to-OCR edge with a different asset, page or non-contained region is a coordinate mismatch. A metadata-to-fact edge labeled `interprets` is an unsupported modality jump. Such edges remain visible in the report, contribute no citation and place downstream claims on hold. Graph construction rejects cycles, duplicate nodes/edges, unknown endpoints, inconsistent page geometry and malformed field shapes with a value-free `FusedEvidenceError`.

## Review behavior and limits

Read `assess_claims()` before consuming a claim. `unsupported` means no valid source citation exists. `review_required` means at least one citation exists, but a conflict, invalid derivation or unsupported ancestor remains. `cited` means the graph found a valid source path and no declared structural problem. A valid parallel source never clears a problematic branch. These states do not replace privacy screening, clinical review or model qualification.

Declare known disagreements with `conflicts` edges. The graph propagates those holds through derivations but cannot discover contradictory clinical values because it does not receive them. It also cannot authenticate an asset, prove OCR accuracy, detect PHI encoded into an opaque reference or perform coordinate transformations. Transformed media needs separately bound provenance; declaring a new coordinate frame does not validate a transform.

Graphs accept at most 1,024 nodes and 4,096 edges, use iterative traversal and sort nodes, edges, findings and source citations for deterministic JSON (`openmed.multimodal.fused-evidence.v1`). No external dependencies, bundled weights or restricted datasets are required.

## Validation

```bash
.venv/bin/python -m pytest tests/unit/multimodal/evidence/test_fused_graph.py -q
```

Synthetic controls cover fused ancestry, source coordinates, span integrity, conflicting branches, unsupported modality jumps, mixed valid/invalid evidence, ordering, cycles, missing endpoints, deep lineage and content-free failures.
