from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter, medspacy_context


@dataclass
class StubSpan:
    text: str
    start_char: int
    end_char: int
    label_: str
    _: SimpleNamespace


@dataclass
class StubDoc:
    text: str
    ents: tuple[StubSpan, ...]


def _span(text: str, surface: str, *, label: str = "PROBLEM", **flags: bool):
    start = text.index(surface)
    return StubSpan(
        text=surface,
        start_char=start,
        end_char=start + len(surface),
        label_=label,
        _=SimpleNamespace(**flags),
    )


def test_registry_loads_medspacy_context_adapter_without_optional_imports():
    assert get_adapter("medspacy_context") is medspacy_context
    assert get_adapter("medspaCy") is medspacy_context


def test_fixture_doc_propagates_context_flags_to_canonical_metadata():
    text = "No pneumonia. Family history of asthma."
    doc = StubDoc(
        text=text,
        ents=(
            _span(text, "pneumonia", is_negated=True),
            _span(text, "asthma", is_historical=True, is_family=True),
        ),
    )

    entities = medspacy_context.to_canonical(doc)

    assert [(entity.start, entity.end) for entity in entities] == [
        (text.index("pneumonia"), text.index("pneumonia") + len("pneumonia")),
        (text.index("asthma"), text.index("asthma") + len("asthma")),
    ]
    assert entities[0].metadata["clinical_context"] == {
        "negation": "negated",
        "uncertainty": "certain",
        "temporality": "recent",
        "experiencer": "patient",
    }
    assert entities[1].metadata["clinical_context"] == {
        "negation": "affirmed",
        "uncertainty": "certain",
        "temporality": "historical",
        "experiencer": "family",
    }
    assert entities[1].metadata["medspacy_context"] == {
        "is_negated": False,
        "is_uncertain": False,
        "is_historical": True,
        "is_family": True,
    }


def test_exact_offsets_attach_context_without_mutating_openmed_span():
    text = "Family history of asthma."
    medspacy_span = _span(text, "asthma", is_historical=True, is_family=True)
    original = PIIEntity(
        text="asthma",
        label="PROBLEM",
        entity_type="PROBLEM",
        confidence=0.8,
        start=medspacy_span.start_char,
        end=medspacy_span.end_char,
        metadata={"source": "openmed"},
    )

    enriched = medspacy_context.to_canonical(
        StubDoc(text=text, ents=(medspacy_span,)),
        openmed_spans=[original],
    )

    assert original.metadata == {"source": "openmed"}
    assert enriched[0] is not original
    assert enriched[0].metadata["source"] == "medspacy"
    assert enriched[0].metadata["clinical_context"]["temporality"] == "historical"
    assert enriched[0].metadata["clinical_context"]["experiencer"] == "family"


def test_mapping_doc_stub_and_unmatched_offsets_are_safe():
    text = "History of asthma."
    start = text.index("asthma")
    doc = {
        "text": text,
        "ents": [
            {
                "text": "asthma",
                "start_char": start,
                "end_char": start + len("asthma"),
                "label_": "PROBLEM",
                "_": {"is_historical": True},
            }
        ],
    }
    unmatched = PIIEntity(
        text="history",
        label="PROBLEM",
        entity_type="PROBLEM",
        confidence=0.5,
        start=0,
        end=7,
    )

    output = medspacy_context.to_canonical(doc, spans=[unmatched])

    assert output[0] is not unmatched
    assert output[0].metadata is None


def test_runtime_helper_reports_missing_optional_extra(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(medspacy_context, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[medspacy\]"):
        medspacy_context.process_to_canonical("synthetic", nlp=lambda value: value)


def test_core_import_does_not_import_medspacy_or_spacy():
    code = """
import sys
import openmed
from openmed.interop import get_adapter

assert get_adapter("medspacy_context") is not None
blocked = [
    name for name in sys.modules
    if name == "medspacy" or name.startswith("medspacy.")
    or name == "spacy" or name.startswith("spacy.")
]
assert blocked == [], blocked
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("offset", [True, 1.5, "1"])
def test_offsets_are_not_silently_truncated(offset):
    doc = {"text": "abc", "ents": [{"start_char": offset, "end_char": 2}]}
    with pytest.raises(ValueError, match="offsets must be integers"):
        medspacy_context.to_canonical(doc)


def test_explicit_text_must_match_processed_doc():
    doc = StubDoc("alpha", (_span("alpha", "alpha"),))
    with pytest.raises(ValueError, match="must match"):
        medspacy_context.to_canonical(doc, text="bravo")


def test_same_offsets_on_different_existing_text_do_not_attach_context():
    doc = StubDoc("alpha", (_span("alpha", "alpha", is_negated=True),))
    span = {"start": 0, "end": 5, "text": "bravo"}
    with pytest.raises(ValueError, match="must match"):
        medspacy_context.to_canonical(doc, spans=[span])


def test_pipeline_failure_traceback_does_not_echo_source(monkeypatch):
    import traceback

    monkeypatch.setattr(medspacy_context, "_require_runtime_dependencies", lambda: None)

    def pipeline(text):
        raise RuntimeError("synthetic-private-5550199")

    with pytest.raises(ValueError) as caught:
        medspacy_context.process_to_canonical("synthetic", nlp=pipeline)
    assert "synthetic-private-5550199" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_oversized_default_confidence_is_rejected_cleanly():
    with pytest.raises(ValueError):
        medspacy_context.MedspacyContextAdapterConfig(default_confidence=10**400)
