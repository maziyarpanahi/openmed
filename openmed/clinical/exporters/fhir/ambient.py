"""Passive, closed-subset FHIR R4 export of caller-reviewed ambient fixtures.

This adapter does not assemble notes or confer reviewer authority. Input and
output are protected clinical documents; diagnostics contain controlled codes
only. No source media, transport, credentials or model are accepted.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openmed.interop.fhir.validation import validation_result

from .bundle import to_bundle
from .validate import ValidationFinding, ValidationResult

__all__ = [
    "AmbientDocumentError",
    "AmbientDocumentExport",
    "ambient_draft_digest",
    "export_ambient_document",
    "import_ambient_document",
    "validate_ambient_document",
]

BASE = "https://openmed.dev/fhir/StructureDefinition/ambient-"
NOTICE = (
    "Non-diagnostic ambient note. Explicit clinician confirmation is required "
    "before consequential use. This export performs no EHR write."
)
_SECTION_CODES = frozenset(
    {"history", "exam", "subjective", "objective", "assessment", "plan"}
)
_LOSS_CATEGORIES = frozenset(
    {
        "audio_payload",
        "transcript_payload",
        "model_metadata",
        "review_history",
        "unsupported_elements",
    }
)
_DIV_START = '<div xmlns="http://www.w3.org/1999/xhtml"><p>'
_DIV_END = "</p></div>"


class AmbientDocumentError(ValueError):
    """A controlled, value-free ambient export refusal."""


@dataclass(frozen=True)
class AmbientDocumentExport:
    """Protected document and counts-only conversion loss report.

    Attributes:
        bundle: FHIR R4 document Bundle. Never log this clinical payload.
        losses: Ordered pairs of controlled category and omitted-element count.
        draft_digest: SHA-256 commitment to the complete fixed draft snapshot.
    """

    bundle: dict[str, Any] = field(repr=False)
    losses: tuple[tuple[str, int], ...]
    draft_digest: str


def _require(ok: bool, code: str = "invalid_draft") -> None:
    if not ok:
        raise AmbientDocumentError(code)


def _keys(value: Any, names: set[str]) -> None:
    _require(isinstance(value, dict) and set(value) == names)


def _opaque(value: Any) -> None:
    _require(
        isinstance(value, str) and bool(re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", value))
    )
    try:
        valid = str(uuid.UUID(value[9:])) == value[9:]
    except ValueError:
        valid = False
    _require(valid)


def _digest(value: Any) -> None:
    _require(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)))


def _count(value: Any, maximum: int = 2**31 - 1) -> None:
    _require(type(value) is int and 0 <= value <= maximum)


def _snapshot(draft: Mapping[str, Any]) -> dict[str, Any]:
    # Deep copy prevents a caller's later edits from mutating the reviewed view.
    _keys(
        draft,
        {
            "schema_version",
            "draft_ref",
            "evidence_digest",
            "sections",
            "loss_counts",
            "review",
            "correction_pending",
        },
    )
    result = copy.deepcopy(dict(draft))
    _require(type(result["schema_version"]) is int and result["schema_version"] == 1)
    _opaque(result["draft_ref"])
    _digest(result["evidence_digest"])
    _require(type(result["correction_pending"]) is bool)
    sections = result["sections"]
    _require(isinstance(sections, list) and 1 <= len(sections) <= 32)
    total = 0
    for section in sections:
        _keys(section, {"code", "note", "evidence"})
        _require(isinstance(section["code"], str) and section["code"] in _SECTION_CODES)
        note = section["note"]
        _require(isinstance(note, str) and 1 <= len(note) <= 16384)
        # XML 1.0 narrative cannot carry controls, surrogates or noncharacters.
        _require(
            all(
                c in "\t\n\r"
                or 0x20 <= ord(c) <= 0xD7FF
                or 0xE000 <= ord(c) <= 0xFFFD
                or 0x10000 <= ord(c) <= 0x10FFFF
                for c in note
            )
        )
        total += len(note.encode("utf-8"))
        evidence = section["evidence"]
        _require(isinstance(evidence, list) and 1 <= len(evidence) <= 128)
        for item in evidence:
            _keys(item, {"reference", "speaker", "start", "end"})
            _opaque(item["reference"])
            _opaque(item["speaker"])
            _count(item["start"])
            _count(item["end"])
            _require(item["start"] < item["end"])
    _require(total <= 262144)
    losses = result["loss_counts"]
    _require(isinstance(losses, list) and len(losses) <= len(_LOSS_CATEGORIES))
    seen = set()
    for item in losses:
        _keys(item, {"category", "count"})
        category = item["category"]
        _require(
            isinstance(category, str)
            and category in _LOSS_CATEGORIES
            and category not in seen
        )
        _count(item["count"])
        _require(item["count"] > 0)
        seen.add(category)
    review = result["review"]
    if review is not None:
        _keys(review, {"digest", "reviewer"})
        _digest(review["digest"])
        _opaque(review["reviewer"])
    return result


def _commitment(draft: dict[str, Any]) -> str:
    values = [draft["draft_ref"], draft["evidence_digest"], str(len(draft["sections"]))]
    for section in draft["sections"]:
        values.extend((section["code"], section["note"], str(len(section["evidence"]))))
        for item in section["evidence"]:
            values.extend(
                (
                    item["reference"],
                    item["speaker"],
                    str(item["start"]),
                    str(item["end"]),
                )
            )
    values.append(str(len(draft["loss_counts"])))
    for item in draft["loss_counts"]:
        values.extend((item["category"], str(item["count"])))
    framed = b"openmed-ambient-v1\n"
    for value in values:
        encoded = value.encode("utf-8")
        framed += str(len(encoded)).encode("ascii") + b":" + encoded
    return hashlib.sha256(framed).hexdigest()


def ambient_draft_digest(draft: Mapping[str, Any]) -> str:
    """Commit section order, notes, citations, speakers and declared losses.

    Args:
        draft: Closed v1 fixture described in the ambient FHIR documentation.
            Review and correction state are excluded from the content digest.

    Returns:
        A lowercase SHA-256 digest for a trusted local reviewer to approve.

    Raises:
        AmbientDocumentError: If the fixed snapshot violates the contract.
    """
    return _commitment(_snapshot(draft))


def _extension(name: str, kind: str, value: Any) -> dict[str, Any]:
    return {"url": BASE + name, kind: value}


def _evidence(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": BASE + "evidence",
        "extension": [
            _extension("reference", "valueUri", item["reference"]),
            _extension("speaker", "valueUri", item["speaker"]),
            _extension("start", "valueUnsignedInt", item["start"]),
            _extension("end", "valueUnsignedInt", item["end"]),
        ],
    }


def _instant(value: Any) -> None:
    _require(
        isinstance(value, str)
        and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value)),
        "invalid_recorded_time",
    )
    try:
        valid = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ") is not None
    except ValueError:
        valid = False
    _require(valid, "invalid_recorded_time")


def export_ambient_document(
    draft: Mapping[str, Any],
    *,
    current_evidence_digest: str,
    recorded_at: str,
    confirmed_digest: str | None = None,
) -> AmbientDocumentExport:
    """Export a reviewed fixed draft without any network or EHR operation.

    Args:
        draft: Protected closed v1 snapshot, with trusted local review receipt.
        current_evidence_digest: Digest from the caller's current evidence ledger.
        recorded_at: Injected UTC instant, exactly ``YYYY-MM-DDTHH:MM:SSZ``.
        confirmed_digest: Explicit clinician confirmation of this exact digest.
            Omit for preliminary export; supply for final export.

    Returns:
        Protected Composition/Provenance document and counts-only loss report.

    Raises:
        AmbientDocumentError: For invalid, unreviewed, stale, correction-pending
            or incorrectly confirmed drafts. Errors contain controlled codes.
    """
    snapshot = _snapshot(draft)
    _digest(current_evidence_digest)
    _instant(recorded_at)
    digest = _commitment(snapshot)
    _require(not snapshot["correction_pending"], "correction_pending")
    _require(snapshot["evidence_digest"] == current_evidence_digest, "stale_evidence")
    review = snapshot["review"]
    _require(review is not None, "unreviewed_draft")
    _require(review["digest"] == digest, "stale_review")
    _require(
        confirmed_digest is None or confirmed_digest == digest, "confirmation_mismatch"
    )
    return _render(snapshot, digest, recorded_at, confirmed_digest is not None)


def _render(
    draft: dict[str, Any], digest: str, recorded: str, final: bool
) -> AmbientDocumentExport:
    reviewer = {
        "identifier": {
            "system": BASE + "reviewer",
            "value": draft["review"]["reviewer"],
        }
    }
    losses = tuple((item["category"], item["count"]) for item in draft["loss_counts"])
    composition = {
        "resourceType": "Composition",
        "id": "ambient-note",
        "identifier": {"system": BASE + "draft", "value": draft["draft_ref"]},
        "status": "final" if final else "preliminary",
        "type": {
            "coding": [
                {
                    "system": "https://openmed.dev/fhir/CodeSystem/document-type",
                    "code": "ambient-note",
                }
            ]
        },
        "date": recorded,
        "author": [reviewer],
        "title": "Reviewed ambient note",
        "text": {"status": "additional", "div": _DIV_START + NOTICE + _DIV_END},
        "extension": [
            _extension("schema", "valueUnsignedInt", 1),
            _extension("draft-digest", "valueString", digest),
            _extension("evidence-digest", "valueString", draft["evidence_digest"]),
            _extension("review-status", "valueCode", "reviewed"),
            _extension("clinician-confirmed", "valueBoolean", final),
        ]
        + [
            {
                "url": BASE + "conversion-loss",
                "extension": [
                    _extension("category", "valueCode", category),
                    _extension("count", "valueUnsignedInt", count),
                ],
            }
            for category, count in losses
        ],
        "section": [
            {
                "code": {
                    "coding": [
                        {
                            "system": "https://openmed.dev/fhir/CodeSystem/ambient-section",
                            "code": section["code"],
                        }
                    ]
                },
                "text": {
                    "status": "additional",
                    "div": _DIV_START + html.escape(section["note"]) + _DIV_END,
                },
                "extension": [_evidence(item) for item in section["evidence"]],
            }
            for section in draft["sections"]
        ],
    }
    sources = sorted(
        {
            item["reference"]
            for section in draft["sections"]
            for item in section["evidence"]
        }
    )
    provenance = {
        "resourceType": "Provenance",
        "id": "ambient-provenance",
        "target": [{"reference": "Composition/ambient-note"}],
        "recorded": recorded,
        "activity": {
            "coding": [
                {
                    "system": "https://openmed.dev/fhir/CodeSystem/ambient-activity",
                    "code": "reviewed-export",
                }
            ]
        },
        "agent": [{"who": reviewer}],
        "entity": [
            {
                "role": "source",
                "what": {
                    "identifier": {
                        "system": BASE + "transcript-evidence",
                        "value": reference,
                    }
                },
            }
            for reference in sources
        ],
    }
    bundle = to_bundle([composition, provenance], doc_id=digest, bundle_type="document")
    bundle["identifier"] = {"system": BASE + "draft-digest", "value": digest}
    bundle["timestamp"] = recorded
    return AmbientDocumentExport(bundle, losses, digest)


def _read_document(bundle: Mapping[str, Any]) -> dict[str, Any]:
    composition = bundle["entry"][0]["resource"]
    extensions = composition["extension"]
    draft: dict[str, Any] = {
        "schema_version": extensions[0]["valueUnsignedInt"],
        "draft_ref": composition["identifier"]["value"],
        "evidence_digest": extensions[2]["valueString"],
        "review": {
            "digest": extensions[1]["valueString"],
            "reviewer": composition["author"][0]["identifier"]["value"],
        },
        "correction_pending": False,
        "loss_counts": [
            {
                "category": row["extension"][0]["valueCode"],
                "count": row["extension"][1]["valueUnsignedInt"],
            }
            for row in extensions[5:]
        ],
        "sections": [],
    }
    for section in composition["section"]:
        narrative = section["text"]["div"]
        _require(
            isinstance(narrative, str)
            and narrative.startswith(_DIV_START)
            and narrative.endswith(_DIV_END)
        )
        draft["sections"].append(
            {
                "code": section["code"]["coding"][0]["code"],
                "note": html.unescape(narrative[len(_DIV_START) : -len(_DIV_END)]),
                "evidence": [
                    {
                        "reference": row["extension"][0]["valueUri"],
                        "speaker": row["extension"][1]["valueUri"],
                        "start": row["extension"][2]["valueUnsignedInt"],
                        "end": row["extension"][3]["valueUnsignedInt"],
                    }
                    for row in section["extension"]
                ],
            }
        )
    status = composition["status"]
    _require(status in {"preliminary", "final"})
    expected = export_ambient_document(
        draft,
        current_evidence_digest=draft["evidence_digest"],
        recorded_at=composition["date"],
        confirmed_digest=draft["review"]["digest"] if status == "final" else None,
    )
    # Closed subset: attachments, arbitrary extensions, write requests and
    # altered narratives are all rejected, rather than silently discarded.
    _require(
        json.dumps(bundle, sort_keys=True, ensure_ascii=False)
        == json.dumps(expected.bundle, sort_keys=True, ensure_ascii=False)
    )
    _require(validation_result(bundle).valid)
    return draft


def validate_ambient_document(bundle: Mapping[str, Any]) -> ValidationResult:
    """Validate the exact v1 subset offline with value-free findings.

    This is a structural/commitment check, not a signature or authority check,
    clinical validation, complete FHIR validation or implementation-guide claim.
    """
    try:
        _read_document(bundle)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return ValidationResult(
            errors=(
                ValidationFinding(
                    "error", "Bundle", "Invalid ambient document subset.", "invalid"
                ),
            )
        )
    return ValidationResult()


def import_ambient_document(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip notes, section order and opaque citations from the subset.

    Returned review metadata is an untrusted transported assertion. Callers
    must authenticate reviewer authority and check their current correction
    ledger before exporting again. This function grants no clinical permission.
    """
    if not validate_ambient_document(bundle).valid:
        raise AmbientDocumentError("invalid_document")
    return _read_document(bundle)
