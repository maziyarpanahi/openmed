"""Check the shared clinical-entity pack vectors against the Python reference.

The Kotlin and Swift ``ClinicalEntityPack`` tests replay these synthetic
vectors through a fake backend. This test keeps their expected output equal to
what ``normalize_label`` and the generated task pack manifest imply, and keeps
every offset a Unicode scalar offset into the case text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.labels import normalize_label
from openmed.core.task_packs import resolve

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "tests" / "fixtures" / "parity" / "clinical_entity_pack_v1.json"


def _vectors() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def test_vectors_declare_the_clinical_entity_pack_and_scalar_offsets() -> None:
    vectors = _vectors()

    assert vectors["schema"] == "openmed.task_pack_vectors"
    assert vectors["schema_version"].startswith("1.")
    assert vectors["pack_id"] == "clinical_entity"
    assert vectors["offset_unit"] == "unicode_scalar"
    assert len(vectors["cases"]) >= 3
    assert any(not case["text"].isascii() for case in vectors["cases"])
    assert any(not case["expected"] for case in vectors["cases"])


@pytest.mark.parametrize("case", _vectors()["cases"], ids=lambda case: case["id"])
def test_expected_spans_follow_normalized_pack_labels(case: dict) -> None:
    vectors = _vectors()
    canonical = set(resolve(vectors["pack_id"], "android").canonical_labels)
    threshold = vectors["confidence_threshold"]
    text = case["text"]

    implied = [
        {
            "category": normalize_label(span["label"]),
            "start": span["start"],
            "end": span["end"],
            "text": text[span["start"] : span["end"]],
        }
        for span in case["model_output"]
        if normalize_label(span["label"]) in canonical
        and span["confidence"] >= threshold
    ]

    assert case["expected"] == implied
    for span in case["expected"]:
        assert 0 <= span["start"] < span["end"] <= len(text)
        assert text[span["start"] : span["end"]] == span["text"]
