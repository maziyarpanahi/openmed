"""Synthetic native-intake vectors shared by Python, Swift and Kotlin tests.

Each region is authored as a truth box on the original page. The OCR-image box
is produced here by an independent forward transform (crop, then rotate with
``PageTransform``, then scale), so the shared reverse mapping must recover the
authored truth. Expected records come from the Python reference mapper.
"""

import hashlib
import json
import math
from pathlib import Path

from openmed.multimodal.evidence.fused_graph import SpatialUnit
from openmed.multimodal.native_intake import (
    BOX_TOLERANCE,
    PAGE_SEPARATOR,
    SCHEMA_VERSION,
    IntakeLimits,
    IntakeOutcome,
    IntakePageGeometry,
    IntakePageInput,
    IntakeReason,
    PageStatus,
    Recognition,
    RecognizedRegion,
    map_document,
    refuse_document,
)
from openmed.multimodal.page_rotation import PageTransform

PATH = Path(__file__).resolve().parents[1] / "parity" / "document_intake_v1.json"


def digest(label):
    return hashlib.sha256(f"synthetic-{label}".encode()).hexdigest()


def forward(geometry, size, truth, *, exact=True):
    """Map an original-page truth box into OCR-image pixels."""
    width, height = geometry["width"], geometry["height"]
    left, top, right, bottom = geometry["crop"] or (0, 0, width, height)
    local = (truth[0] - left, truth[1] - top, truth[2] - left, truth[3] - top)
    transform = PageTransform((right - left, bottom - top), geometry["rotation"])
    upright = transform.bbox(local)
    scale_x = size[0] / transform.target_size.width
    scale_y = size[1] / transform.target_size.height
    box = (
        upright[0] * scale_x,
        upright[1] * scale_y,
        upright[2] * scale_x,
        upright[3] * scale_y,
    )
    if exact:
        assert all(float(value).is_integer() for value in box)
        return [int(value) for value in box]
    # A real engine reports whole pixels; round outward so the box still
    # covers the authored region.
    return [
        math.floor(box[0]),
        math.floor(box[1]),
        math.ceil(box[2]),
        math.ceil(box[3]),
    ]


def page(width, height, ocr, text, words, *, unit="pixel", rotation=0, crop=None):
    """Author a rendered page whose words are ``(word, truth_box)`` pairs."""
    geometry = {
        "width": width,
        "height": height,
        "unit": unit,
        "rotation": rotation,
        "crop": crop,
    }
    return {"geometry": geometry, "ocr": [*ocr], "text": text, "words": words}


def held(reason):
    return {"unavailable": reason}


def spans(text, words):
    """Locate each authored word in the page text, in order, as scalar spans."""
    cursor, result = 0, []
    for word in words:
        start = text.index(word, cursor)
        cursor = start + len(word)
        result.append([start, cursor])
    return result


def build_page(spec, *, exact):
    if "unavailable" in spec:
        return {"unavailable": spec["unavailable"], "geometry": None, "ocr": None}
    geometry, (width, height) = spec["geometry"], spec["ocr"]
    regions = []
    words = [word for word, _ in spec["words"]]
    for (word, truth), span in zip(spec["words"], spans(spec["text"], words)):
        override = None
        if isinstance(truth, dict):
            override, truth = truth.get("box"), truth.get("truth")
        if override is None:
            box = forward(geometry, (width, height), truth, exact=exact)
        else:
            box = override
        regions.append({"span": span, "box": box, "truth": truth})
    for span, box in spec.get("raw_regions", []):
        regions.append({"span": span, "box": box, "truth": None})
    return {
        "unavailable": None,
        "geometry": geometry,
        "ocr": {
            "width": width,
            "height": height,
            "text": spec["text"],
            "regions": regions,
        },
    }


def page_input(row):
    if row["unavailable"] is not None:
        return IntakePageInput(unavailable=IntakeReason(row["unavailable"]))
    geometry = row["geometry"]
    ocr = row["ocr"]
    return IntakePageInput(
        geometry=IntakePageGeometry(
            width=geometry["width"],
            height=geometry["height"],
            unit=SpatialUnit(geometry["unit"]),
            rotation=geometry["rotation"],
            crop=None if geometry["crop"] is None else tuple(geometry["crop"]),
        ),
        recognition=Recognition(
            width=ocr["width"],
            height=ocr["height"],
            text=ocr["text"],
            regions=tuple(
                RecognizedRegion(span=tuple(region["span"]), box=tuple(region["box"]))
                for region in ocr["regions"]
            ),
        ),
    )


def run(case):
    """Run the Python reference mapper on one shared case."""
    if case["refusal"] is not None:
        return refuse_document(IntakeReason(case["refusal"]))
    limits = IntakeLimits(**case["limits"]) if case["limits"] else None
    return map_document(
        case["source_digest"],
        [page_input(row) for row in case["pages"]],
        limits=limits,
    )


def expected(case, probes):
    report = run(case)
    pages = []
    for record in report.pages:
        row = {
            "page": record.page,
            "status": record.status.value,
            "reason": None if record.reason is None else record.reason.value,
            "text_span": None if record.text_span is None else list(record.text_span),
            "node_ref": record.node_ref,
            "evidence_digest": None,
            "display_width_f64": None,
            "display_height_f64": None,
            "crop_node_ref": None,
            "crop_evidence_digest": None,
            "crop_display_box_f64": None,
        }
        if record.evidence is not None:
            payload = record.evidence.payload()
            row["evidence_digest"] = record.evidence.evidence_digest
            row["display_width_f64"] = payload["display_width_f64"]
            row["display_height_f64"] = payload["display_height_f64"]
        if record.crop_evidence is not None:
            payload = record.crop_evidence.payload()
            row["crop_node_ref"] = payload["node_ref"]
            row["crop_evidence_digest"] = record.crop_evidence.evidence_digest
            row["crop_display_box_f64"] = payload["display_box_f64"]
        pages.append(row)
    regions = []
    for region in report.regions:
        payload = region.evidence.payload()
        regions.append(
            {
                "page": region.page,
                "span": list(region.span),
                "node_ref": region.node_ref,
                "box_f64": payload["box_f64"],
                "display_box_f64": payload["display_box_f64"],
                "evidence_digest": region.evidence.evidence_digest,
            }
        )
    return {
        "outcome": report.outcome.value,
        "reasons": [reason.value for reason in report.reasons],
        "text": report.text,
        "pages": pages,
        "regions": regions,
        "probes": [
            {
                "span": list(span),
                "node_refs": [r.node_ref for r in report.regions_for_span(*span)],
            }
            for span in probes
        ],
    }


PDF = {"unit": "point"}
LETTER = (612, 792)


def specs():
    """Authored synthetic documents; every value is synthetic."""
    first = "Synthetic Jane Roe\nVisit 2026-01-02"
    mixed = "Patient 患者 रवि 😀 José"
    return [
        {
            "id": "multipage_pdf",
            "covers": ["multipage", "rotated"],
            "pages": [
                page(
                    *LETTER,
                    (1224, 1584),
                    first,
                    [
                        ("Synthetic", (72, 72, 162, 90)),
                        ("Jane", (168, 72, 204, 90)),
                        ("Roe", (210, 72, 237, 90)),
                        ("Visit", (72, 100, 117, 118)),
                        ("2026-01-02", (123, 100, 213, 118)),
                    ],
                    **PDF,
                ),
                page(
                    *LETTER,
                    (1224, 1584),
                    "Plan follow-up",
                    [
                        ("Plan", (72, 700, 108, 718)),
                        ("follow-up", (114, 700, 195, 718)),
                    ],
                    rotation=180,
                    **PDF,
                ),
            ],
            "probes": [(25, 35), (30, 41), (35, 37)],
        },
        {
            "id": "rotated_pdf_90",
            "covers": ["rotated"],
            "pages": [
                page(
                    *LETTER,
                    (1584, 1224),
                    "Rotated note\nSynthetic",
                    [
                        ("Rotated", (72, 72, 135, 90)),
                        ("note", (141, 72, 177, 90)),
                        ("Synthetic", (72, 100, 153, 118)),
                    ],
                    rotation=90,
                    **PDF,
                ),
            ],
            "probes": [(0, 7), (8, 22)],
        },
        {
            "id": "rotated_image_270",
            "covers": ["rotated"],
            "pages": [
                page(
                    400,
                    300,
                    (300, 400),
                    "Upright text",
                    [("Upright", (40, 30, 110, 50)), ("text", (120, 30, 160, 50))],
                    rotation=270,
                ),
            ],
            "probes": [(8, 12)],
        },
        {
            "id": "cropped_pdf",
            "covers": ["cropped"],
            "pages": [
                page(
                    *LETTER,
                    (1080, 1296),
                    "Cropped region",
                    [
                        ("Cropped", (90, 108, 153, 126)),
                        ("region", (159, 108, 213, 126)),
                    ],
                    crop=[36, 72, 576, 720],
                    **PDF,
                ),
            ],
            "probes": [(0, 14)],
        },
        {
            "id": "cropped_rotated_image",
            "covers": ["cropped", "rotated"],
            "pages": [
                page(
                    1000,
                    800,
                    (400, 600),
                    "Crop and turn",
                    [
                        ("Crop", (150, 100, 230, 130)),
                        ("and", (240, 100, 290, 130)),
                        ("turn", (300, 100, 370, 130)),
                    ],
                    rotation=90,
                    crop=[100, 50, 700, 450],
                ),
            ],
            "probes": [(5, 8), (9, 13)],
        },
        {
            "id": "mixed_script_image",
            "covers": ["mixed-script"],
            "pages": [
                page(
                    800,
                    200,
                    (800, 200),
                    mixed,
                    [
                        ("Patient", (10, 20, 90, 40)),
                        ("患者", (100, 20, 140, 40)),
                        ("रवि", (150, 20, 190, 40)),
                        # The engine overshoots the top edge by one pixel; the
                        # overshoot is within tolerance and clamped to the edge.
                        ("😀", {"box": [200, -1, 220, 40], "truth": (200, 0, 220, 40)}),
                        ("José", (230, 20, 290, 40)),
                    ],
                ),
            ],
            "probes": [(15, 16), (8, 14), (17, 22)],
        },
        {
            "id": "fractional_scale_pdf",
            "covers": ["rotated"],
            "exact": False,
            "tolerance": 72 / 150,
            "pages": [
                page(
                    *LETTER,
                    (1650, 1275),
                    "Scaled render",
                    [
                        ("Scaled", (72, 72, 126, 90)),
                        ("render", (132, 72, 186, 90)),
                    ],
                    rotation=90,
                    **PDF,
                ),
            ],
            "probes": [(7, 13)],
        },
        {
            "id": "missing_and_failed_pages",
            "covers": ["multipage", "partial"],
            "pages": [
                page(100, 100, (100, 100), "First", [("First", (10, 10, 50, 20))]),
                held("page_missing"),
                held("recognition_failed"),
                page(100, 100, (100, 100), "Last", [("Last", (10, 10, 40, 20))]),
            ],
            "probes": [(0, 5), (7, 11)],
        },
        {
            "id": "permission_lost",
            "covers": ["partial"],
            "pages": [
                page(100, 100, (100, 100), "Before", [("Before", (10, 10, 60, 20))]),
                held("permission_lost"),
                page(100, 100, (100, 100), "After", [("After", (10, 10, 50, 20))]),
            ],
            "probes": [(0, 6)],
        },
        {
            "id": "ambiguous_transforms",
            "covers": ["partial"],
            "pages": [
                page(100, 100, (100, 100), "Tilted", [], rotation=45),
                page(100, 100, (100, 100), "Kept", [("Kept", (10, 10, 40, 20))]),
                page(100, 100, (100, 100), "Outside", [], crop=[0, 0, 140, 100]),
            ],
            "probes": [(0, 4)],
        },
        {
            "id": "unrotated_render",
            "covers": ["refused"],
            "pages": [
                page(*LETTER, (1224, 1584), "Swapped", [], rotation=90, **PDF),
            ],
            "probes": [],
        },
        {
            "id": "region_out_of_bounds",
            "covers": ["partial"],
            "pages": [
                {
                    **page(100, 100, (100, 100), "Far edge", []),
                    "raw_regions": [([0, 3], [90, 10, 140, 30])],
                },
                page(100, 100, (100, 100), "Inside", [("Inside", (10, 10, 60, 20))]),
            ],
            "probes": [(0, 6)],
        },
        {
            "id": "invalid_recognition",
            "covers": ["refused"],
            "pages": [
                {
                    **page(100, 100, (100, 100), "Short", []),
                    "raw_regions": [([3, 99], [10, 10, 40, 20])],
                },
            ],
            "probes": [],
        },
        {
            "id": "page_limit",
            "covers": ["multipage", "partial"],
            "limits": {"max_pages": 2},
            "pages": [
                page(100, 100, (100, 100), "One", [("One", (10, 10, 40, 20))]),
                page(100, 100, (100, 100), "Two", [("Two", (10, 10, 40, 20))]),
                page(100, 100, (100, 100), "Three", [("Three", (10, 10, 50, 20))]),
            ],
            "probes": [(5, 8)],
        },
        {
            "id": "resource_limit",
            "covers": ["partial"],
            "limits": {"max_regions_per_page": 1},
            "pages": [
                page(
                    100,
                    100,
                    (100, 100),
                    "Too many",
                    [("Too", (10, 10, 30, 20)), ("many", (35, 10, 60, 20))],
                ),
                page(100, 100, (100, 100), "Single", [("Single", (10, 10, 60, 20))]),
            ],
            "probes": [(0, 6)],
        },
        {
            "id": "permission_denied",
            "covers": ["refused"],
            "refusal": "permission_denied",
        },
        {"id": "document_locked", "covers": ["refused"], "refusal": "document_locked"},
        {"id": "empty_document", "covers": ["refused"], "pages": []},
    ]


def fixture():
    cases = []
    for spec in specs():
        exact = spec.get("exact", True)
        case = {
            "id": spec["id"],
            "covers": spec["covers"],
            "source_digest": digest(spec["id"]),
            "refusal": spec.get("refusal"),
            "limits": spec.get("limits"),
            "tolerance": spec.get("tolerance", 0),
            "pages": [build_page(row, exact=exact) for row in spec.get("pages", [])],
        }
        case["expected"] = expected(case, spec.get("probes", []))
        cases.append(case)
    defaults = IntakeLimits()
    return {
        "version": 1,
        "schema": SCHEMA_VERSION,
        "synthetic_only": True,
        "page_separator": PAGE_SEPARATOR,
        "box_tolerance": BOX_TOLERANCE,
        "default_limits": {
            "max_pages": defaults.max_pages,
            "max_regions_per_page": defaults.max_regions_per_page,
            "max_page_scalars": defaults.max_page_scalars,
        },
        "outcomes": [outcome.value for outcome in IntakeOutcome],
        "reasons": [reason.value for reason in IntakeReason],
        "page_statuses": [status.value for status in PageStatus],
        "cases": cases,
    }


def render():
    """Return the committed fixture text."""
    return json.dumps(fixture(), ensure_ascii=False, indent=2) + "\n"


if __name__ == "__main__":
    PATH.write_text(render(), encoding="utf-8")
