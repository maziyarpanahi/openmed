from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from openmed.core.africa_context import (
    DATA_FILE,
    load_africa_context_data,
    profile_defaults_for,
)
from openmed.core.labels import (
    ACCOUNT_NUMBER,
    ETHNICITY,
    ORGANIZATION,
    SENSITIVE_ATTRIBUTE,
)
from openmed.core.pii import deidentify
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.policy import PolicyProfile, list_policies, load_policy
from openmed.processing.outputs import PredictionResult

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "africa_context_safety_sweep.json"
_FIXTURE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_AFRICAN_PROFILES = {
    "africa_malabo_baseline",
    "eg_pdpl",
    "ke_dpa",
    "ma_law_09_08",
    "ng_ndpa",
    "za_popia",
}
_PROTECTED_CONTEXT_LABELS = (ETHNICITY, ACCOUNT_NUMBER, ORGANIZATION)


def _profile_for_contract(profile_name: str) -> PolicyProfile:
    if profile_name in list_policies():
        return load_policy(profile_name)

    base = load_policy("strict_no_leak")
    defaults = profile_defaults_for(profile_name)
    actions = dict(base.actions)
    actions.update(defaults["actions"])
    policy_label_actions = dict(base.policy_label_actions)
    policy_label_actions.update(defaults["policy_label_actions"])
    return base.derive(
        name=profile_name,
        actions=actions,
        policy_label_actions=policy_label_actions,
    )


def _empty_extract(
    source_text: str, *args: object, **kwargs: object
) -> PredictionResult:
    return PredictionResult(
        text=source_text,
        entities=[],
        model_name="synthetic-africa-context-fixture",
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_data_file_declares_every_initial_african_profile() -> None:
    payload = load_africa_context_data()

    assert payload["schema_version"] == 1
    assert set(payload["african_profile_defaults"]) == _AFRICAN_PROFILES
    assert DATA_FILE.name == "africa_context_terms.json"


@pytest.mark.parametrize("profile_name", sorted(_AFRICAN_PROFILES))
def test_african_profile_context_defaults_are_non_keep(profile_name: str) -> None:
    profile = _profile_for_contract(profile_name)

    assert profile.policy_label_actions[SENSITIVE_ATTRIBUTE] != "keep"
    for label in _PROTECTED_CONTEXT_LABELS:
        assert profile.action_for(label) != "keep"


@pytest.mark.parametrize("profile_name", sorted(_AFRICAN_PROFILES))
@pytest.mark.parametrize(
    "record",
    _FIXTURE["records"],
    ids=[record["id"] for record in _FIXTURE["records"]],
)
def test_african_context_leakage_gate_redacts_every_planted_surface(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    record: Mapping[str, Any],
) -> None:
    from openmed.core import pii

    monkeypatch.setattr(pii, "extract_pii", _empty_extract)
    result = deidentify(
        record["text"],
        policy=_profile_for_contract(profile_name),
        use_smart_merging=False,
        use_safety_sweep=True,
    )

    detected_labels = {entity.canonical_label for entity in result.pii_entities}
    for planted in record["planted"]:
        assert planted["surface"] not in result.deidentified_text
        assert planted["label"] in detected_labels


def test_context_terms_do_not_trigger_without_identifying_shape_or_context() -> None:
    from openmed.core.safety_sweep import safety_sweep

    text = "A Zulu translation was requested after the routine clinic visit."
    entities = safety_sweep(text, [])

    assert not {entity.label for entity in entities} & set(_PROTECTED_CONTEXT_LABELS)


@pytest.mark.parametrize(
    "text",
    [
        "A Zulu translation was requested after the routine clinic visit.",
        "The raceway trace showed a Yoruba translation request.",
    ],
)
def test_ethnicity_terms_require_word_bounded_context_in_every_pattern_path(
    text: str,
) -> None:
    from openmed.core.safety_sweep import safety_sweep

    assert ETHNICITY not in {entity.label for entity in safety_sweep(text, [])}
    assert ETHNICITY not in {unit[2] for unit in find_semantic_units(text)}


def test_generic_lowercase_facility_reference_is_not_a_named_organization() -> None:
    from openmed.core.safety_sweep import safety_sweep

    text = "The patient returned to the district hospital for a routine review."

    assert ORGANIZATION not in {entity.label for entity in safety_sweep(text, [])}


def test_country_descriptor_is_not_a_named_healthcare_facility() -> None:
    from openmed.core.safety_sweep import safety_sweep

    text = "Ghana clinic registration recorded a synthetic identifier."

    assert ORGANIZATION not in {entity.label for entity in safety_sweep(text, [])}


def test_engine_module_contains_no_african_term_arrays() -> None:
    merger_source = (
        Path(__file__).parents[3] / "openmed" / "core" / "pii_entity_merger.py"
    ).read_text(encoding="utf-8")

    for term in ("M-Pesa", "MTN MoMo", "Kikuyu", "Yoruba", "Zulu"):
        assert term not in merger_source
