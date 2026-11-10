from __future__ import annotations

from dataclasses import dataclass

import pytest

from openmed.interop import get_adapter, quickumls, scispacy_linker


@dataclass
class StubExtensions:
    kb_ents: list[tuple[str, float]]


@dataclass
class StubSpan:
    text: str
    start_char: int
    end_char: int
    _: StubExtensions


@dataclass
class StubDoc:
    ents: list[StubSpan]


def test_registry_loads_umls_adapters_lazily():
    assert get_adapter("scispacy-linker") is scispacy_linker
    assert get_adapter("quickumls") is quickumls


def test_scispacy_cui_and_score_propagate_to_canonical_span_codes():
    linked = StubDoc(
        ents=[
            StubSpan(
                text="diabetes mellitus",
                start_char=12,
                end_char=29,
                _=StubExtensions(kb_ents=[("C0011849", 0.97), ("C0011860", 0.82)]),
            )
        ]
    )

    assert scispacy_linker.to_canonical(linked) == [
        {
            "text": "diabetes mellitus",
            "start": 12,
            "end": 29,
            "codes": [
                {"system": "UMLS", "code": "C0011849", "score": 0.97},
                {"system": "UMLS", "code": "C0011860", "score": 0.82},
            ],
        }
    ]


def test_scispacy_mapping_stub_is_supported_without_importing_dependency():
    linked = {
        "entities": [
            {
                "text": "metformin",
                "start": 0,
                "end": 9,
                "kb_ents": [{"concept_id": "C0025598", "score": 0.91}],
            }
        ]
    }

    assert scispacy_linker.to_canonical(linked)[0]["codes"] == [
        {"system": "UMLS", "code": "C0025598", "score": 0.91}
    ]


def test_scispacy_missing_linker_resource_is_actionable():
    span_without_linker = {"text": "diabetes", "start": 0, "end": 8}

    with pytest.raises(
        scispacy_linker.ScispaCyLinkerResourceError,
        match=r"user-supplied licensed UMLS resources",
    ):
        scispacy_linker.to_canonical([span_without_linker])


def test_scispacy_runtime_import_is_guarded(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(scispacy_linker, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[scispacy\]"):
        scispacy_linker.link_to_canonical("diabetes", nlp=lambda text: text)


def test_quickumls_maps_nested_candidates_to_same_code_fields():
    matches = [
        [
            {
                "start": 4,
                "end": 12,
                "ngram": "diabetes",
                "term": "Diabetes mellitus",
                "cui": "C0011849",
                "similarity": 0.96,
            },
            {
                "start": 4,
                "end": 12,
                "ngram": "diabetes",
                "term": "Diabetes",
                "cui": "C0011860",
                "similarity": 0.84,
            },
        ]
    ]

    assert quickumls.to_canonical(matches) == [
        {
            "text": "diabetes",
            "start": 4,
            "end": 12,
            "codes": [
                {"system": "UMLS", "code": "C0011849", "score": 0.96},
                {"system": "UMLS", "code": "C0011860", "score": 0.84},
            ],
        }
    ]


def test_quickumls_configured_matcher_needs_no_resource_import():
    class StubMatcher:
        def match(self, text: str, *, best_match: bool, ignore_syntax: bool):
            assert text == "metformin"
            assert best_match is True
            assert ignore_syntax is False
            return [
                {
                    "start": 0,
                    "end": 9,
                    "term": "metformin",
                    "cui": "C0025598",
                    "similarity": 0.93,
                }
            ]

    result = quickumls.match_to_canonical("metformin", matcher=StubMatcher())

    assert result[0]["codes"] == [{"system": "UMLS", "code": "C0025598", "score": 0.93}]


def test_quickumls_missing_user_resource_is_actionable():
    with pytest.raises(
        quickumls.QuickUMLSResourceError,
        match=r"user-supplied QuickUMLS data directory",
    ):
        quickumls.match_to_canonical("diabetes")
