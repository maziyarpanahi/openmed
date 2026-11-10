"""Round-trip evaluation suite for grounding provenance reproducibility.

Builds a small, algorithmically generated synthetic gold set of grounded spans
with a known vocabulary version and retrieval/rerank method per code, then
proves that :func:`openmed.clinical.grounding.GroundingProvenance` captures
enough to re-derive which terminology version and method produced each emitted
code — and that the chain never carries the raw note text.

All vocabulary content is synthetic; no real patient data or licensed
terminology is used. The suite is self-contained, offline, and deterministic,
and is intentionally not registered in ``DEFAULT_SUITES`` (it is a provenance
reproducibility gate, not a model benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass

from openmed.clinical.exporters.codeable_concept import GroundedSpan
from openmed.clinical.grounding import (
    Candidate,
    GroundingProvenance,
    scan_provenance_for_raw_text,
)
from openmed.core.audit import stable_hash

__all__ = [
    "GROUNDING_PROVENANCE",
    "GroundingProvenanceCase",
    "build_synthetic_gold_set",
    "grounding_provenance_metadata",
    "run_grounding_provenance_roundtrip",
]

#: Suite identifier (deliberately absent from ``DEFAULT_SUITES``).
GROUNDING_PROVENANCE = "grounding_provenance"

# (system, code, display, mention surface, method, vocabulary version)
# Mention surfaces are opaque synthetic tokens deliberately disjoint from the
# vocabulary displays and codes, so the no-PHI scan can prove the raw surface is
# never stored (only its offsets and content hash are kept).
_SYNTHETIC_ROWS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("LOINC", "L0001", "Synthetic serum marker", "mzx01", "sparse", "loinc-syn-2024a"),
    (
        "RXNORM",
        "R0002",
        "Synthetic analgesic tablet",
        "mzx02",
        "dense",
        "rxnorm-syn-2024b",
    ),
    (
        "ICD10CM",
        "C0003",
        "Synthetic chronic condition",
        "mzx03",
        "rerank",
        "icd10cm-syn-2024c",
    ),
    (
        "HPO",
        "H0004",
        "Synthetic phenotype trait",
        "mzx04",
        "composite",
        "hpo-syn-2024d",
    ),
    (
        "SNOMED",
        "S0005",
        "Synthetic clinical finding",
        "mzx05",
        "post-coordinated",
        "snomed-syn-2024e",
    ),
)


@dataclass(frozen=True)
class GroundingProvenanceCase:
    """One synthetic grounded span with the decision it should reproduce.

    ``span`` is the grounded input (surface, offsets, ranked candidates),
    ``method`` is the retrieval/rerank stage that chose the top candidate, and
    ``expected_*`` are the values a reviewer must be able to re-derive from the
    provenance record for the emitted code.
    """

    span: GroundedSpan
    method: str
    expected_system: str
    expected_code: str
    expected_vocab_version: str


def build_synthetic_gold_set() -> list[GroundingProvenanceCase]:
    """Return the deterministic synthetic gold set of grounding cases.

    Offsets are laid out contiguously in a single synthetic note so each span
    carries realistic, non-overlapping character positions. Each span gets a
    strong exact candidate plus a weaker fuzzy alternative to exercise the
    chosen-versus-alternatives split.
    """
    cases: list[GroundingProvenanceCase] = []
    cursor = 0
    for system, code, display, mention, method, version in _SYNTHETIC_ROWS:
        start = cursor
        end = start + len(mention)
        cursor = end + 1  # single-space separator between mentions
        candidates = (
            Candidate(
                system=system,
                code=code,
                display=display,
                score=1.0,
                source=method,
                matched_alias=mention.replace(" ", ""),
                match_kind="exact",
                vocab_version=version,
            ),
            Candidate(
                system=system,
                code=f"{code}-alt",
                display=f"{display} (alternative)",
                score=0.42,
                source=method,
                matched_alias=mention.replace(" ", ""),
                match_kind="fuzzy",
                vocab_version=version,
            ),
        )
        cases.append(
            GroundingProvenanceCase(
                span=GroundedSpan(
                    text=mention,
                    start=start,
                    end=end,
                    candidates=candidates,
                ),
                method=method,
                expected_system=system,
                expected_code=code,
                expected_vocab_version=version,
            )
        )
    return cases


def run_grounding_provenance_roundtrip(
    cases: list[GroundingProvenanceCase] | None = None,
    *,
    encoder_index_key: str = "synthetic-index",
) -> dict[str, object]:
    """Run the provenance round-trip over ``cases`` (default: the gold set).

    For every emitted code, the provenance record's ``(system, code,
    vocab_version, method)`` is compared against the case's expectations, proving
    the record is sufficient to re-derive the terminology version and method that
    produced the code. The report also confirms no raw span surface leaks into
    the chain and carries a ``repro_hash`` so identical runs are verifiably
    stable.
    """
    gold = build_synthetic_gold_set() if cases is None else cases

    records: list[GroundingProvenance] = []
    mismatches: list[dict[str, str]] = []
    for case in gold:
        record = GroundingProvenance.from_candidates(
            start=case.span.start,
            end=case.span.end,
            span_text=case.span.text,
            candidates=case.span.candidates,
            method=case.method,
            encoder_index_key=encoder_index_key,
        )
        records.append(record)
        rederived = (
            record.system,
            record.code,
            record.vocab_version,
            record.method,
        )
        expected = (
            case.expected_system,
            case.expected_code,
            case.expected_vocab_version,
            case.method,
        )
        if rederived != expected:
            mismatches.append(
                {"expected": repr(expected), "rederived": repr(rederived)}
            )

    forbidden = [case.span.text for case in gold]
    leaked = scan_provenance_for_raw_text(records, forbidden)
    emitted = [record for record in records if not record.abstained]

    return {
        "suite": GROUNDING_PROVENANCE,
        "case_count": len(gold),
        "emitted_code_count": len(emitted),
        "reproduced": len(emitted) - len(mismatches),
        "mismatches": mismatches,
        "leaked_raw_text": list(leaked),
        "phi_safe": not leaked,
        "passed": not mismatches and not leaked,
        "repro_hash": stable_hash([record.to_dict() for record in records]),
    }


def grounding_provenance_metadata() -> dict[str, object]:
    """Return static metadata describing the grounding provenance suite."""
    return {
        "suite": GROUNDING_PROVENANCE,
        "data_provenance": "synthetic",
        "deterministic": True,
        "registered_in_default_suites": False,
    }
