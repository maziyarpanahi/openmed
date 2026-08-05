from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii import deidentify
from openmed.core.policy import CompiledPolicy, compile_policy, load_policy
from openmed.processing.outputs import EntityPrediction, PredictionResult

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "africa_statute_coverage.json"
_DOC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "compliance"
    / "africa-malabo-baseline.md"
)
_MATRIX = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_INITIAL_AFRICAN_PROFILES = {
    "africa_malabo_baseline",
    "eg_pdpl",
    "ke_dpa",
    "ma_law_09_08",
    "ng_ndpa",
    "za_popia",
}
_MALABO_ARTICLE_14_CATEGORIES = (
    "racial, ethnic and regional origin",
    "parental filiation",
    "political opinions",
    "religious or philosophical beliefs",
    "trade union membership",
    "sex life",
    "genetic information",
    "data on the state of health of the data subject",
)


def _profile_matrix() -> Mapping[str, Mapping[str, Any]]:
    profiles = _MATRIX["profiles"]
    assert isinstance(profiles, dict)
    return profiles


def _coverage_cases() -> list[pytest.ParameterSet]:
    cases: list[pytest.ParameterSet] = []
    for profile_name, profile_entry in _profile_matrix().items():
        classes = profile_entry["classes"]
        assert isinstance(classes, list)
        for statute_class in classes:
            assert isinstance(statute_class, dict)
            class_name = statute_class["statute_class"]
            cases.append(
                pytest.param(
                    profile_name,
                    statute_class,
                    id=f"{profile_name}:{class_name}",
                )
            )
    return cases


def _leakage_cases() -> list[pytest.ParameterSet]:
    cases = _MATRIX["leakage_cases"]
    assert isinstance(cases, list)
    return [
        pytest.param(case, id=str(case["id"]))
        for case in cases
        if isinstance(case, dict)
    ]


@lru_cache(maxsize=None)
def _compiled_policy(profile_name: str) -> CompiledPolicy:
    return compile_policy(profile_name)


def test_coverage_matrix_declares_every_initial_african_profile() -> None:
    profiles = _profile_matrix()

    assert _MATRIX["schema_version"] == 1
    assert _INITIAL_AFRICAN_PROFILES <= set(profiles)
    for profile_name, profile_entry in profiles.items():
        assert profile_entry["statute"], profile_name
        assert profile_entry["source_url"], profile_name
        assert profile_entry["classes"], profile_name


@pytest.mark.parametrize(
    ("profile_name", "statute_class"),
    _coverage_cases(),
)
def test_declared_statute_class_has_compiled_non_keep_coverage(
    profile_name: str,
    statute_class: Mapping[str, Any],
) -> None:
    class_name = statute_class["statute_class"]
    citation = statute_class["citation"]
    labels = statute_class["canonical_labels"]
    assert isinstance(class_name, str) and class_name
    assert isinstance(citation, str) and citation
    assert isinstance(labels, list) and labels

    unknown_labels = set(labels) - CANONICAL_LABELS
    assert not unknown_labels, (
        f"{profile_name}:{class_name} declares non-canonical labels "
        f"{sorted(unknown_labels)}"
    )

    profile = load_policy(profile_name)
    compiled = _compiled_policy(profile_name)
    for label in labels:
        assert profile.action_for(label) != "keep", (
            f"{profile_name}:{class_name} dropped non-keep coverage for {label}"
        )
        plan_entry = compiled.plan.entry_for(label)
        assert plan_entry is not None, (
            f"{profile_name}:{class_name} has no compiled coverage for {label}"
        )
        assert plan_entry.action != "keep", (
            f"{profile_name}:{class_name} compiled a keep action for {label}"
        )


def test_malabo_categories_are_verbatim_in_matrix_and_documentation() -> None:
    article = _MATRIX["malabo_article_14_1"]
    assert article["citation"] == "Article 14(1)"
    assert tuple(article["categories_verbatim"]) == _MALABO_ARTICLE_14_CATEGORIES

    baseline = _profile_matrix()["africa_malabo_baseline"]
    baseline_categories = tuple(
        statute_class["statute_class"] for statute_class in baseline["classes"]
    )
    assert baseline_categories == _MALABO_ARTICLE_14_CATEGORIES
    assert all(
        statute_class["citation"] == "Article 14(1)"
        for statute_class in baseline["classes"]
    )

    documentation = _DOC_PATH.read_text(encoding="utf-8")
    for category in _MALABO_ARTICLE_14_CATEGORIES:
        assert f"| `{category}` |" in documentation


@pytest.mark.parametrize("case", _leakage_cases())
def test_malabo_baseline_has_zero_pan_african_direct_identifier_leakage(
    monkeypatch: pytest.MonkeyPatch,
    case: Mapping[str, Any],
) -> None:
    from openmed.core import pii

    text = case["text"]
    assert isinstance(text, str)
    entities: list[EntityPrediction] = []
    surfaces: list[str] = []
    for planted in case["entities"]:
        surface = planted["surface"]
        label = planted["label"]
        assert isinstance(surface, str)
        assert isinstance(label, str)
        start = text.index(surface)
        surfaces.append(surface)
        entities.append(
            EntityPrediction(
                text=surface,
                label=label,
                start=start,
                end=start + len(surface),
                confidence=0.99,
            )
        )

    def fake_extract(
        source_text: str, *args: object, **kwargs: object
    ) -> PredictionResult:
        assert source_text == text
        return PredictionResult(
            text=source_text,
            entities=entities,
            model_name="synthetic-pan-african-fixture",
            timestamp="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract)
    result = deidentify(
        text,
        policy="africa_malabo_baseline",
        use_smart_merging=False,
        use_safety_sweep=False,
    )
    audit = deidentify(
        text,
        policy="africa_malabo_baseline",
        use_smart_merging=False,
        use_safety_sweep=False,
        audit=True,
    )

    assert result.mapping is None
    assert all(surface not in result.deidentified_text for surface in surfaces)
    assert result.metadata["policy"]["name"] == "africa_malabo_baseline"
    assert result.metadata["safety_sweep"]["source"] == "safety_sweep"
    assert audit.policy == "africa_malabo_baseline"
    assert audit.safety_sweep["enabled"] is True
    assert audit.residual_risk["risk_report"]["leakage_rate"] == 0.0
    assert all(span.action != "keep" for span in audit.spans)
