"""Golden and safety tests for span-native clinical coreference."""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical import (
    COREFERENCE_FEATURES,
    COREFERENCE_RESOLUTION_ADVISORY,
    CoreferenceChain,
    resolve_coreference,
)
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/clinical/coreference_gold.json"


def _load_cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _offset(text: str, surface: str, occurrence: int = 0) -> tuple[int, int]:
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.index(surface, cursor)
        cursor = start + len(surface)
    return start, start + len(surface)


def _spans_for(case: Mapping[str, Any]) -> list[OpenMedSpan]:
    text = str(case["text"])
    spans = []
    for mention in case["mentions"]:
        surface = str(mention["surface"])
        start, end = _offset(text, surface, int(mention.get("occurrence", 0)))
        spans.append(
            OpenMedSpan(
                doc_id=str(case["id"]),
                start=start,
                end=end,
                text_hash=hmac_text_hash(surface, "synthetic-fixture-secret"),
                entity_type=str(mention.get("entity_type", mention["canonical_label"])),
                canonical_label=str(mention["canonical_label"]),
                section=mention.get("section"),
                metadata=mention.get("metadata", {}),
            )
        )
    return spans


def _predicted_partition(
    case: Mapping[str, Any],
    spans: list[OpenMedSpan],
) -> dict[int, str]:
    _chains, index = resolve_coreference(spans, str(case["text"]))
    return {
        position: index[(span.doc_id, (span.start, span.end))]
        for position, span in enumerate(spans)
    }


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
def test_golden_accept_and_reject_cases(case: Mapping[str, Any]) -> None:
    spans = _spans_for(case)

    predicted = _predicted_partition(case, spans)
    mentions = case["mentions"]

    for left in range(len(spans)):
        for right in range(len(spans)):
            expected_same = (
                mentions[left]["gold_chain"] == mentions[right]["gold_chain"]
            )
            assert (predicted[left] == predicted[right]) is expected_same


def test_repeated_mentions_keep_original_offsets_and_representative() -> None:
    case = _load_cases()[0]
    spans = _spans_for(case)

    chains, index = resolve_coreference(spans, case["text"])

    assert len(chains) == 1
    chain = chains[0]
    assert isinstance(chain, CoreferenceChain)
    assert chain.members == tuple(spans)
    assert chain.member_spans == tuple(spans)
    assert chain.representative == spans[0]
    assert chain.representative_mention == spans[0]
    assert 0.70 <= chain.confidence <= 1.0
    assert chain.advisory == COREFERENCE_RESOLUTION_ADVISORY
    assert set(index) == {(span.doc_id, (span.start, span.end)) for span in spans}


def test_resolution_is_input_order_invariant() -> None:
    case = _load_cases()[1]
    spans = _spans_for(case)

    chains, index = resolve_coreference(spans, case["text"])
    reversed_chains, reversed_index = resolve_coreference(reversed(spans), case["text"])

    assert reversed_chains == chains
    assert reversed_index == index


def test_experiencer_boundary_never_merges_patient_and_family() -> None:
    case = next(item for item in _load_cases() if item["id"] == "experiencer_boundary")
    spans = _spans_for(case)

    chains, index = resolve_coreference(spans, case["text"])

    family_key = (spans[0].doc_id, (spans[0].start, spans[0].end))
    patient_key = (spans[1].doc_id, (spans[1].start, spans[1].end))
    pronoun_key = (spans[2].doc_id, (spans[2].start, spans[2].end))
    assert index[family_key] != index[patient_key]
    assert index[patient_key] == index[pronoun_key]
    assert sorted(len(chain.members) for chain in chains) == [1, 2]


def test_resolver_is_offline_and_emits_no_raw_text_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    case = _load_cases()[0]
    spans = _spans_for(case)

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    resolve_coreference(spans, case["text"])

    captured = caplog.text.casefold()
    assert "left lung lesion" not in captured
    assert "the lesion" not in captured


def test_feature_contract_names_all_required_deterministic_signals() -> None:
    assert COREFERENCE_FEATURES == (
        "section",
        "sentence_distance",
        "head_noun",
        "entity_type",
    )


def test_invalid_offsets_duplicates_and_threshold_are_rejected() -> None:
    case = _load_cases()[0]
    span = _spans_for(case)[0]

    with pytest.raises(ValueError, match="within text"):
        resolve_coreference([span], "short")
    with pytest.raises(ValueError, match="unique document offsets"):
        resolve_coreference([span, span], case["text"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_coreference([span], case["text"], threshold=1.1)


def test_mixed_document_ids_are_rejected() -> None:
    case = _load_cases()[0]
    first, second = _spans_for(case)[:2]
    other_document = OpenMedSpan.from_dict(
        {**second.to_dict(), "doc_id": "another-document"}
    )

    with pytest.raises(ValueError, match="one document"):
        resolve_coreference([first, other_document], case["text"])


def test_overlapping_spans_do_not_form_antecedent_links() -> None:
    text = "rash."
    spans = [
        OpenMedSpan(
            doc_id="overlap",
            start=0,
            end=4,
            text_hash=hmac_text_hash("rash", "synthetic-fixture-secret"),
            entity_type="condition",
            canonical_label="CONDITION",
        ),
        OpenMedSpan(
            doc_id="overlap",
            start=0,
            end=5,
            text_hash=hmac_text_hash("rash.", "synthetic-fixture-secret"),
            entity_type="condition",
            canonical_label="CONDITION",
        ),
    ]

    chains, index = resolve_coreference(spans, text)

    assert len(chains) == 2
    assert index[("overlap", (0, 4))] != index[("overlap", (0, 5))]
