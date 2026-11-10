from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters


@dataclass
class EntityLike:
    label: str
    start: int
    end: int
    confidence: float
    canonical_label: str | None = None


def test_registry_lists_spacy_adapter_without_importing_spacy():
    assert "spacy" in available_adapters()
    assert adapter_spec("spacy").extra == "spacy"

    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "spacy" or name.startswith("spacy."):
        raise AssertionError(f"unexpected spaCy import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import openmed
from openmed.interop import available_adapters

assert openmed is not None
assert "spacy" in available_adapters()
"""

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_blank_spacy_pipeline_projects_openmed_spans(monkeypatch):
    spacy = pytest.importorskip("spacy", exc_type=ImportError)
    from openmed.interop import spacy as spacy_adapter

    def fake_extract_pii(text: str, **kwargs):
        assert text == "Patient Jane Roe called 555-0100."
        assert kwargs["confidence_threshold"] == 0.6
        assert kwargs["lang"] == "en"
        return SimpleNamespace(
            entities=[
                EntityLike("PERSON", 8, 16, 0.98, canonical_label=None),
                EntityLike("PHONE", 24, 32, 0.93),
            ]
        )

    monkeypatch.setattr(spacy_adapter, "_load_extract_pii", lambda: fake_extract_pii)

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "openmed_deid",
        config={
            "confidence_threshold": 0.6,
            "lang": "en",
        },
    )

    doc = nlp("Patient Jane Roe called 555-0100.")

    spans = doc.spans["openmed_pii"]
    assert [span.text for span in spans] == ["Jane Roe", "555-0100"]
    assert [span.label_ for span in spans] == ["PERSON", "PHONE"]
    assert [(span.start_char, span.end_char) for span in spans] == [(8, 16), (24, 32)]
    assert [
        (span.label, span.start, span.end, span.score) for span in doc._.openmed_pii
    ] == [
        ("PERSON", 8, 16, 0.98),
        ("PHONE", 24, 32, 0.93),
    ]


def test_spacy_component_resolves_overlaps_when_merging_doc_ents(monkeypatch):
    spacy = pytest.importorskip("spacy", exc_type=ImportError)
    from openmed.interop import spacy as spacy_adapter

    def fake_extract_pii(text: str, **kwargs):
        del text, kwargs
        return SimpleNamespace(
            entities=[
                EntityLike("PERSON", 0, 4, 0.87),
                EntityLike("PERSON", 0, 8, 0.99),
            ]
        )

    monkeypatch.setattr(spacy_adapter, "_load_extract_pii", lambda: fake_extract_pii)

    nlp = spacy.blank("en")
    nlp.add_pipe("openmed_deid", config={"merge_ents": True})

    doc = nlp("Jane Roe called.")

    assert [span.text for span in doc.spans["openmed_pii"]] == ["Jane", "Jane Roe"]
    assert [ent.text for ent in doc.ents] == ["Jane Roe"]
    assert [ent.label_ for ent in doc.ents] == ["PERSON"]
