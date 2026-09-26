"""Tests for the deterministic synthetic clinical fixture generator."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from openmed.eval.clinical_fixtures import (
    ASSERTION_VALUES,
    DEFAULT_PROFILES,
    SYNTHETIC_CODE_SYSTEM,
    ClinicalFixture,
    generate_fixture,
    generate_fixtures,
)


def test_generation_is_deterministic_and_seeded() -> None:
    first = generate_fixtures(seed=23)
    second = generate_fixtures(seed=23)
    different = generate_fixtures(seed=24)

    assert first == second
    assert [fixture.profile for fixture in first] == list(DEFAULT_PROFILES)
    assert [fixture.text for fixture in first] != [
        fixture.text for fixture in different
    ]
    assert [fixture.text_hash for fixture in first] == [
        fixture.text_hash for fixture in second
    ]


def test_generation_does_not_modify_global_random_state() -> None:
    random.seed(101)
    expected_next = random.random()

    random.seed(101)
    generate_fixtures(seed=101)

    assert random.random() == expected_next


def test_profiles_cover_offsets_sections_assertions_and_codes() -> None:
    fixtures = (*generate_fixtures(seed=41), generate_fixture("pathology", seed=41))
    assertions = set()

    for fixture in fixtures:
        fixture.validate()
        assert fixture.sections
        assert fixture.expected_structured_fields
        assert fixture.metadata["synthetic"] is True
        assert fixture.metadata["phi"] is False
        assert any(item.code is not None for item in fixture.expected_fields)

        for section in fixture.sections:
            assert fixture.text[section.start : section.end]

        for span in fixture.gold_spans:
            assert fixture.span_text(span) == span.text
            assert fixture.text[span.start : span.end] == span.text
            assert span.section in {section.name for section in fixture.sections}
            assertions.add(span.assertion)
            if span.code is not None:
                assert span.code.system
                assert span.code.code

        known_span_ids = {span.span_id for span in fixture.gold_spans}
        assert all(
            item.span_id is None or item.span_id in known_span_ids
            for item in fixture.expected_fields
        )

    assert {"absent", "historical", "uncertain"} <= assertions
    assert assertions <= set(ASSERTION_VALUES)
    assert any(
        span.code.system == SYNTHETIC_CODE_SYSTEM
        for fixture in fixtures
        for span in fixture.gold_spans
        if span.code
    )


def test_privacy_safe_serialization_omits_document_and_span_text() -> None:
    fixtures = (*generate_fixtures(seed=7), generate_fixture("pathology", seed=7))

    for fixture in fixtures:
        payload = fixture.to_dict()
        serialized = fixture.to_json()
        encoded = json.dumps(payload, sort_keys=True)

        assert "text" not in payload
        assert '"text":' not in serialized
        assert fixture.text not in encoded
        assert all(span.text not in encoded for span in fixture.gold_spans)
        assert all("value" not in item for item in payload["expected_fields"])
        assert payload["text_sha256"].startswith("sha256:")
        assert payload["synthetic"] is True
        assert payload["phi"] is False
        assert payload["metadata"]["synthetic"] is True
        assert payload["metadata"]["phi"] is False
        assert fixture.text not in repr(fixture)
        assert all(span.text not in repr(span) for span in fixture.gold_spans)

    tainted_metadata = replace(
        fixtures[0],
        metadata={"source": "synthetic-private-note", "text": "raw-value"},
    )
    tainted_report = json.dumps(tainted_metadata.to_dict(), sort_keys=True)
    assert "synthetic-private-note" not in tainted_report
    assert "raw-value" not in tainted_report


def test_text_inclusive_serialization_round_trips_locally() -> None:
    fixture = generate_fixture("radiology", seed=12)

    payload = fixture.to_dict(include_text=True)
    restored = ClinicalFixture.from_mapping(payload)

    assert restored.text == fixture.text
    assert restored.gold_spans == fixture.gold_spans
    assert restored.sections == fixture.sections
    assert restored.expected_fields == fixture.expected_fields
    assert restored.to_dict(include_text=True) == payload


def test_profile_aliases_and_order_independent_seed_derivation() -> None:
    assert generate_fixture("radiology", seed=5) == generate_fixture(
        "radiology_report", seed=5
    )

    ordered = generate_fixtures(("lab", "radiology"), seed=8)
    reversed_order = generate_fixtures(("radiology", "lab"), seed=8)

    assert ordered[0] == reversed_order[1]
    assert ordered[1] == reversed_order[0]

    with pytest.raises(ValueError, match="unknown clinical fixture profile"):
        generate_fixture("not-a-profile", seed=1)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        generate_fixtures(("lab", "lab_report"), seed=1)


def test_validation_errors_do_not_include_source_text() -> None:
    fixture = generate_fixture("lab_report", seed=19)
    broken_span = replace(
        fixture.gold_spans[0],
        end=len(fixture.text) + 1,
        text="",
    )

    with pytest.raises(ValueError) as raised:
        replace(fixture, gold_spans=(broken_span, *fixture.gold_spans[1:]))

    assert fixture.text not in str(raised.value)
    assert all(span.text not in str(raised.value) for span in fixture.gold_spans)
