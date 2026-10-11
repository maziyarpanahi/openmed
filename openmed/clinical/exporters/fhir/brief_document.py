"""Passive, preliminary R4 documents for the existing guarded brief contract.

This is an intentionally closed export subset, not a general FHIR importer or
an attestation API. The source brief always requires human review. Importing a
projection cannot recreate evidence approval or a ClinicalBrief.
"""

from __future__ import annotations

import copy
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from openmed.clinical.brief import ClinicalBrief, _digest
from openmed.clinical.guarded_provenance import check_guarded_provenance
from openmed.clinical.review_packet_privacy import scan_review_packet_privacy
from openmed.clinical.summary_envelope import SUMMARY_SAFETY_DISCLAIMER
from openmed.core.iso_temporal import parse_iso_datetime
from openmed.core.offline import network_blocked_if_offline
from openmed.interop.fhir.reference_integrity import check_bundle_reference_integrity

from .bundle import to_bundle
from .references import deterministic_fullurl

BRIEF_DOCUMENT_SUBSET = "openmed.clinical.brief.fhir-r4.v1"
BRIEF_DOCUMENT_EXTENSION = (
    "https://openmed.ai/fhir/StructureDefinition/clinical-brief-document"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOSSES = (
    "profile_sections_unavailable",
    "source_payloads_omitted",
    "evaluation_metrics_omitted",
    "review_packet_details_omitted",
    "model_details_omitted",
)


class BriefDocumentError(ValueError):
    """Controlled, value-free export/import failure code."""


class _Blocked(Exception):
    def __init__(self, code):
        self.code = code


@dataclass(frozen=True)
class BriefFHIRDocument:
    """Protected document with a separate value-free conversion-loss report.

    Use ``to_response`` only inside an authorized local application. It contains
    narrative. ``to_dict`` and the repr contain only codes, counts and digests.
    """

    _json: str = field(repr=False)
    brief_digest: str
    citation_count: int

    def to_response(self) -> dict[str, Any]:
        """Return a defensive copy of the Bundle and its DocumentReference."""
        return json.loads(self._json)

    def to_dict(self) -> dict[str, Any]:
        """Return value-free review status and explicit conversion losses."""
        return {
            "subset": BRIEF_DOCUMENT_SUBSET,
            "brief_digest": self.brief_digest,
            "citation_count": self.citation_count,
            "status": "preliminary",
            "review_status": "queued",
            "requires_human_review": True,
            "conversion_loss": list(_LOSSES),
        }


def export_brief_document(
    brief: ClinicalBrief,
    *,
    recorded_at: datetime,
    privacy_detector: Callable[[str], Any],
    original_identifiers: tuple[str, ...] = (),
    status: str = "preliminary",
) -> BriefFHIRDocument:
    """Export a guarded brief without attesting, writing or contacting an EHR.

    Args:
        brief: Successful existing ClinicalBrief; refused results are rejected.
        recorded_at: Explicit timezone-aware export time, never a source date.
        privacy_detector: Configured local detector; every finding blocks output.
        original_identifiers: Original tokens to exclude from all string surfaces.
        status: Only ``preliminary`` is supported, including for reviewed evidence.

    Returns:
        An offline R4 document subset and count/code-only conversion-loss report.

    Raises:
        BriefDocumentError: A controlled code without input or callback errors.
    """
    return _guard(
        lambda: _export(brief, recorded_at, status),
        privacy_detector,
        original_identifiers,
    )


def import_brief_document(
    document: Mapping[str, Any],
    *,
    privacy_detector: Callable[[str], Any],
    original_identifiers: tuple[str, ...] = (),
) -> BriefFHIRDocument:
    """Validate and round-trip only this closed preliminary document subset.

    Args:
        document: Protected response from ``export_brief_document``.
        privacy_detector: Configured local detector, required even on import.
        original_identifiers: Original tokens prohibited in every string surface.

    Returns:
        A defensive document projection, never a newly approved ClinicalBrief.

    Raises:
        BriefDocumentError: A value-free code for invalid or unsafe documents.
    """
    return _guard(lambda: _import(document), privacy_detector, original_identifiers)


def _fail(code: str) -> None:
    raise _Blocked(code)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_digest(value: Any) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _guard(build, detector, identifiers):
    # Raise outside handlers: callback exceptions may themselves contain PHI.
    code = "invalid_document"
    try:
        with network_blocked_if_offline(local_only=True):
            if (
                not callable(detector)
                or type(identifiers) not in (tuple, list)
                or len(identifiers) > 1024
                or any(type(item) is not str for item in identifiers)
                or sum(len(item.encode()) for item in identifiers) > 16384
            ):
                _fail("privacy_configuration")
            payload, metadata = build()
            rendered = _canonical(payload)
            if len(rendered.encode()) > 262144:
                _fail("document_limit")
            # Scan both decoded leaves (XHTML entity escaping must not hide an
            # identifier) and the complete assembled document/attachment metadata.
            leaves = "\n".join(html.unescape(item) for item in _strings(payload))
            folded = leaves.casefold()
            for identifier in identifiers:
                original = identifier.strip().casefold()
                if original and original in folded:
                    _fail("privacy")
                tokens = re.findall(r"\w+", identifier.casefold())
                if any(len(token) >= 3 and token in folded for token in tokens):
                    _fail("privacy")
            for text in (leaves, rendered):
                if scan_review_packet_privacy(text, detector).findings:
                    _fail("privacy")
            return BriefFHIRDocument(
                rendered, metadata["brief_digest"], len(metadata["citations"])
            )
    except _Blocked as error:
        code = error.code
    except Exception:
        pass
    raise BriefDocumentError(code)


def _strings(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def _export(brief, date, status):
    if type(status) is not str or status != "preliminary":
        _fail("unsupported_finalization")
    if type(brief) is not ClinicalBrief or brief.refusal_reason is not None:
        _fail("refused_brief")
    if type(date) is not datetime or date.tzinfo is None or date.utcoffset() is None:
        _fail("invalid_export_time")
    date = date.astimezone(timezone.utc).isoformat(timespec="seconds")
    audit = brief.to_dict()
    summary = brief.summary
    if (
        type(summary) is not str
        or not summary
        or len(summary.encode()) > 16384
        or audit["schema_version"] != 1
        or audit["status"] != "needs_review"
        or audit["refusal_reason"] is not None
        or audit["summary_digest"] != _digest(summary)
        or audit["summary_characters"] != len(summary)
        or audit["envelope"]["requires_human_review"] is not True
        or audit["envelope"]["human_review_mode"] is not True
        or audit["envelope"]["is_diagnostic"] is not False
        or audit["envelope"]["status"] != "ready"
        or audit["review_packet"]["review_status"] != "review_required"
    ):
        _fail("invalid_brief")
    provenance = audit["provenance"]
    if (
        not provenance
        or provenance.get("review_status") != "queued"
        or not check_guarded_provenance(provenance, current_output=summary).ok
        or not _is_digest(audit["envelope"]["provenance"]["content_hash"])
        or audit["envelope"]["provenance"].get("verified") is not True
    ):
        _fail("missing_provenance")
    metadata = {
        "subset": BRIEF_DOCUMENT_SUBSET,
        "brief_digest": brief.digest,
        "summary_digest": audit["summary_digest"],
        "source_digest": audit["envelope"]["provenance"]["content_hash"],
        "provenance_digest": provenance["record_hash"],
        "policy_digest": provenance["policy_fingerprint"],
        "review_status": "queued",
        "requires_human_review": True,
        "citations": [],
        "conversion_loss": list(_LOSSES),
    }
    for citation in audit["citations"]:
        matches = [
            evidence
            for evidence in provenance["evidence"]
            if evidence["source_offsets"]
            == {"start": citation["source_start"], "end": citation["source_end"]}
        ]
        if len(matches) != 1:
            _fail("missing_provenance")
        metadata["citations"].append(
            {
                **citation,
                "evidence_id": matches[0]["evidence_id"],
                "evidence_hash": matches[0]["evidence_hash"],
            }
        )
    if audit["verdicts"] != [
        {"claim_index": i, "label": "entailment"}
        for i in range(len(metadata["citations"]))
    ]:
        _fail("invalid_brief")
    _validate_metadata(metadata, summary)
    # Apply the same closed-subset/reference validator on export and import.
    return _import(_build(metadata, summary, date))


def _validate_metadata(metadata, summary):
    if set(metadata) != {
        "subset",
        "brief_digest",
        "summary_digest",
        "source_digest",
        "provenance_digest",
        "policy_digest",
        "review_status",
        "requires_human_review",
        "citations",
        "conversion_loss",
    }:
        _fail("unsupported_subset")
    if (
        metadata["subset"] != BRIEF_DOCUMENT_SUBSET
        or metadata["review_status"] != "queued"
        or metadata["requires_human_review"] is not True
        or metadata["conversion_loss"] != list(_LOSSES)
        or any(
            not _is_digest(metadata[key])
            for key in (
                "brief_digest",
                "summary_digest",
                "source_digest",
                "provenance_digest",
                "policy_digest",
            )
        )
        or metadata["summary_digest"] != _digest(summary)
        or not 1 <= len(metadata["citations"]) <= 64
    ):
        _fail("invalid_provenance")
    end = 0
    for i, citation in enumerate(metadata["citations"]):
        if set(citation) != {
            "claim_index",
            "source_start",
            "source_end",
            "output_start",
            "output_end",
            "evidence_id",
            "evidence_hash",
        }:
            _fail("unsupported_subset")
        if (
            any(
                type(citation[key]) is not int
                for key in (
                    "claim_index",
                    "source_start",
                    "source_end",
                    "output_start",
                    "output_end",
                )
            )
            or citation["claim_index"] != i
            or not 0 <= citation["source_start"] < citation["source_end"] <= 16384
            or not end
            <= citation["output_start"]
            < citation["output_end"]
            <= len(summary)
            or summary[end : citation["output_start"]].strip()
            or not _is_digest(citation["evidence_id"])
            or not _is_digest(citation["evidence_hash"])
        ):
            _fail("invalid_citation")
        end = citation["output_end"]
    if summary[end:].strip():
        _fail("invalid_citation")


def _narrative(text):
    return {
        "status": "generated",
        "div": '<div xmlns="http://www.w3.org/1999/xhtml">'
        + html.escape(text)
        + "</div>",
    }


def _extension(metadata):
    return [{"url": BRIEF_DOCUMENT_EXTENSION, "valueString": _canonical(metadata)}]


def _build(metadata, summary, date):
    seed = _digest({"metadata": metadata, "recorded_at": date})
    device = {"resourceType": "Device", "id": "brief-exporter", "status": "active"}
    composition = {
        "resourceType": "Composition",
        "id": "brief",
        "status": "preliminary",
        "type": {"text": "Clinical brief"},
        "date": date,
        "author": [{"reference": "Device/brief-exporter"}],
        "title": "Clinical brief for human review",
        "text": _narrative(summary),
        "extension": _extension(metadata),
        "section": [],
    }
    resources = [composition, device]
    for i, citation in enumerate(metadata["citations"]):
        resource_id = f"evidence-{i}"
        resources.append(
            {
                "resourceType": "DocumentReference",
                "id": resource_id,
                "status": "current",
                "description": "Source evidence commitment",
                "identifier": [
                    {"system": "urn:openmed:evidence", "value": citation["evidence_id"]}
                ],
                "content": [
                    {
                        "attachment": {
                            "contentType": "text/plain",
                            "title": "Source evidence (payload omitted)",
                            "url": "urn:sha256:" + citation["evidence_hash"][7:],
                        }
                    }
                ],
            }
        )
        composition["section"].append(
            {
                "title": f"Claim {i + 1}",
                "text": _narrative(
                    summary[citation["output_start"] : citation["output_end"]]
                ),
                "entry": [{"reference": f"DocumentReference/{resource_id}"}],
            }
        )
    composition["section"].append(
        {
            "title": "Limitations",
            "text": _narrative(SUMMARY_SAFETY_DISCLAIMER),
        }
    )
    bundle = to_bundle(resources, doc_id=seed, bundle_type="document")
    bundle["id"] = seed[7:]
    bundle["identifier"] = {"system": "urn:openmed:brief-document", "value": seed}
    bundle["timestamp"] = date
    document_reference = {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "preliminary",
        "type": {"text": "Clinical brief"},
        "date": date,
        "extension": _extension(metadata),
        "content": [
            {
                "attachment": {
                    "contentType": "application/fhir+json",
                    "url": deterministic_fullurl(seed, -1),
                    "title": "Preliminary clinical brief",
                }
            }
        ],
    }
    return {
        "bundle": bundle,
        "document_reference": document_reference,
        "document_url": deterministic_fullurl(seed, -1),
    }


def _import(document):
    # Bound parsing before walking arbitrary nested FHIR content.
    if len(_canonical(document).encode()) > 262144:
        _fail("document_limit")
    composition = document["bundle"]["entry"][0]["resource"]
    metadata = json.loads(composition["extension"][0]["valueString"])
    narrative = composition["text"]["div"]
    prefix = '<div xmlns="http://www.w3.org/1999/xhtml">'
    if not narrative.startswith(prefix) or not narrative.endswith("</div>"):
        _fail("unsupported_subset")
    summary = html.unescape(narrative[len(prefix) : -6])
    if not summary or len(summary.encode()) > 16384:
        _fail("document_limit")
    _validate_metadata(metadata, summary)
    date = document["bundle"]["timestamp"]
    parsed = parse_iso_datetime(date)
    if (
        parsed.tzinfo is None
        or parsed.astimezone(timezone.utc).isoformat(timespec="seconds") != date
    ):
        _fail("invalid_export_time")
    expected = _build(metadata, summary, date)
    if _canonical(document) != _canonical(expected):
        _fail("unsupported_subset")
    if not check_bundle_reference_integrity(document["bundle"]).valid:
        _fail("invalid_reference")
    return copy.deepcopy(expected), metadata
