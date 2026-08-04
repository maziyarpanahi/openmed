"""Focused tests for composite splitting, re-linking, and safe provenance."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

from openmed.clinical.grounding import (
    Candidate,
    VocabLoader,
    VocabSource,
    decompose_and_relink,
    ground,
)
from openmed.clinical.normalization import (
    KNOWN_ATOMIC_COMPOSITES,
    normalize_composite,
)
from openmed.eval.suites.composite_normalization import (
    build_composite_normalization_gold,
    evaluate_composite_normalization,
)


def _candidate(surface: str, code: str) -> Candidate:
    return Candidate(
        system="ICD10CM",
        code=code,
        display=f"synthetic {code}",
        score=1.0,
        source="sparse",
        matched_alias=surface,
        match_kind="exact",
        vocab_version="sha256:synthetic-composite-v1",
    )


def _strict_linker(mapping: dict[str, str]):
    def link(surface: str) -> tuple[Candidate, ...]:
        code = mapping.get(surface)
        return (_candidate(surface, code),) if code is not None else ()

    return link


def _loader(tmp_path: Path, rows: list[dict[str, object]]) -> VocabLoader:
    path = tmp_path / "composite-vocab.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=path,
                sha256=checksum,
            )
        },
    )


def test_normalize_composite_gold_has_byte_accurate_offsets() -> None:
    cases = build_composite_normalization_gold()

    assert len(cases) >= 30
    for case in cases:
        result = normalize_composite(case.mention, is_linkable=lambda _: True)
        assert tuple(child.text for child in result.children) == case.children
        encoded = case.mention.encode("utf-8")
        for child in result.children:
            assert case.mention[child.start : child.end] == child.text
            assert (
                encoded[child.byte_start : child.byte_end].decode("utf-8") == child.text
            )


def test_byte_and_character_bases_are_preserved_independently() -> None:
    mention = "  nausée and vomiting  "
    result = normalize_composite(
        mention,
        start=100,
        byte_start=200,
        is_linkable=lambda _: True,
    )

    first, second = result.children
    first_local = mention.index("nausée")
    second_local = mention.index("vomiting")
    assert (first.start, first.end) == (100 + first_local, 100 + first_local + 6)
    assert first.byte_start == 200 + len(mention[:first_local].encode("utf-8"))
    assert first.byte_end == first.byte_start + len("nausée".encode("utf-8"))
    assert (second.start, second.end) == (
        100 + second_local,
        100 + second_local + len("vomiting"),
    )


def test_known_atomic_multiword_concepts_have_zero_false_splits() -> None:
    results = [normalize_composite(term) for term in KNOWN_ATOMIC_COMPOSITES]

    assert results
    assert all(result.strategy == "atomic" for result in results)
    assert all(not result.was_split for result in results)
    assert all(result.children == (result.parent,) for result in results)


def test_linkability_check_withholds_split_for_uncodable_child() -> None:
    result = normalize_composite(
        "asthma with synthetic uncodable qualifier",
        is_linkable=lambda surface: surface == "asthma",
    )

    assert not result.was_split
    assert result.needs_postcoordination
    assert result.blocked_reason == "unlinked_child"
    assert tuple(child.text for child in result.proposed_children) == (
        "asthma",
        "synthetic uncodable qualifier",
    )


def test_relink_emits_multiple_child_codes_when_whole_span_is_not_exact() -> None:
    result = decompose_and_relink(
        "nausea and vomiting",
        linker=_strict_linker({"nausea": "SYN-NAUSEA", "vomiting": "SYN-VOMITING"}),
        start=14,
        byte_start=14,
    )

    assert result.decision == "multiple"
    assert [span.text for span in result.spans] == ["nausea", "vomiting"]
    assert [span.codes for span in result.spans] == [
        {"icd10cm": "SYN-NAUSEA"},
        {"icd10cm": "SYN-VOMITING"},
    ]
    assert [(span.start, span.end) for span in result.spans] == [(14, 20), (25, 33)]
    assert result.postcoordination is None


def test_exact_whole_span_wins_as_precoordinated_concept() -> None:
    surface = "type 2 diabetes with diabetic nephropathy"
    result = decompose_and_relink(
        surface,
        linker=_strict_linker(
            {
                surface: "SYN-DM-NEPHRO",
                "type 2 diabetes": "SYN-DM2",
                "diabetic nephropathy": "SYN-NEPHRO",
            }
        ),
    )

    assert result.decision == "precoordinated"
    assert len(result.spans) == 1
    assert result.spans[0].text == surface
    assert result.spans[0].codes == {"icd10cm": "SYN-DM-NEPHRO"}
    decomposition = result.spans[0].provenance["composite_decomposition"]
    assert decomposition["decision"] == "precoordinated"
    assert len(decomposition["children"]) == 2


def test_uncodable_composite_routes_to_postcoordination_without_raw_phi() -> None:
    surface = "asthma with Patient-Zeta-493 qualifier"
    result = decompose_and_relink(
        surface,
        linker=_strict_linker({"asthma": "SYN-ASTHMA"}),
        start=7,
        byte_start=7,
    )

    assert result.decision == "postcoordination"
    assert len(result.spans) == 1
    assert result.spans[0].text == surface
    assert result.spans[0].candidates == ()
    assert result.postcoordination is not None
    assert [span.codes for span in result.postcoordination.linked_children] == [
        {"icd10cm": "SYN-ASTHMA"},
        {},
    ]
    serialized = json.dumps(result.provenance.to_dict(), sort_keys=True)
    assert "Patient-Zeta-493" not in serialized
    assert "asthma" not in serialized
    assert result.provenance.parent_text_hash
    assert all(child.text_hash for child in result.provenance.children)


def test_ground_facade_composite_stage_is_opt_in_and_reuses_ranking(
    tmp_path: Path,
) -> None:
    loader = _loader(
        tmp_path,
        [
            {
                "code": "SYN-NAUSEA",
                "preferred_term": "nausea",
                "synonyms": [],
            },
            {
                "code": "SYN-VOMITING",
                "preferred_term": "vomiting",
                "synonyms": [],
            },
        ],
    )
    raw_span = {
        "text": "nausea and vomiting",
        "start": 5,
        "end": 24,
        "label": "condition",
    }

    default = ground([raw_span], systems=["icd10cm"], loader=loader)
    normalized = ground(
        [raw_span],
        systems=["icd10cm"],
        loader=loader,
        normalize_composites=True,
    )

    assert [span.text for span in default] == ["nausea and vomiting"]
    assert [span.text for span in normalized] == ["nausea", "vomiting"]
    assert [span.codes for span in normalized] == [
        {"icd10cm": "SYN-NAUSEA"},
        {"icd10cm": "SYN-VOMITING"},
    ]
    assert [(span.start, span.end) for span in normalized] == [(5, 11), (16, 24)]
    assert all(span.canonical_label == "CONDITION" for span in normalized)
    assert all("composite_decomposition" in span.provenance for span in normalized)


def test_synthetic_eval_reports_all_acceptance_metrics_offline(monkeypatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket, "socket", fail_socket)
    report = evaluate_composite_normalization()

    assert report["case_count"] >= 30
    assert report["top1_accuracy"] >= 0.80
    assert report["over_split_rate"] == 0.0
    assert report["under_split_rate"] == 0.0
    assert report["offset_accuracy"] == 1.0
    assert report["atomic_false_splits"] == 0
    assert report["metadata"]["offline"] is True
