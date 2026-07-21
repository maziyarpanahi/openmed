import json
from importlib import resources

import pytest

from openmed.core.arbitration import MODE_BALANCED, MODE_HIGH_RECALL_UNION, arbitrate
from openmed.core.cascade import R2_BASE, CascadeRouter
from openmed.core.pipeline import Pipeline
from openmed.core.schemas.span import ACTION_VALUES, OpenMedSpan, hmac_text_hash
from openmed.core.script_detect import INDIC_SCRIPTS
from openmed.core.thresholds import (
    DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING,
    fit_thresholds,
    load_thresholds,
    lookup_threshold,
    membership_defense_for_profile,
    profile_recall_floor,
    profile_script_leakage_ceiling,
    profile_script_recall_floors,
    recall_floor_guard,
    update_thresholds,
    validate_threshold_matrix,
)


def _span(
    *,
    start: int = 0,
    end: int = 8,
    label: str = "ID_NUM",
    score: float = 0.5,
    detector: str = "model:tiny",
) -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="doc-1",
        start=start,
        end=end,
        text_hash=hmac_text_hash(f"{start}:{end}:{label}", "test-secret"),
        entity_type=label,
        canonical_label=label,
        score=score,
        detector=detector,
    )


def test_lookup_threshold_exact_and_fallback_order():
    exact = lookup_threshold("ID_NUM", "en", "strict_no_leak")
    wildcard = lookup_threshold("ID_NUM", "de", "strict_no_leak")
    profile_default = lookup_threshold("ORGANIZATION", "en", "strict_no_leak")

    assert {
        "keep_floor",
        "escalate_below",
        "action",
    }.issubset(exact)
    assert exact["source"] == "exact"
    assert exact["language"] == "en"
    assert wildcard["source"] == "wildcard_language"
    assert wildcard["language"] == "*"
    assert profile_default["source"] == "profile_default"
    assert profile_default["canonical_label"] == "ORGANIZATION"


def test_thresholds_json_has_schema_version_and_valid_actions():
    payload = json.loads(
        resources.files("openmed.core")
        .joinpath("thresholds.json")
        .read_text(encoding="utf-8")
    )

    assert isinstance(payload["schema_version"], int)
    validate_threshold_matrix(payload)
    for profile in payload["profiles"].values():
        assert profile["default"]["action"] in ACTION_VALUES
        for label_entries in profile["labels"].values():
            for entry in label_entries.values():
                assert entry["action"] in ACTION_VALUES


def test_profiles_define_configurable_strict_cjk_and_indic_floors():
    matrix = load_thresholds()
    floors = profile_script_recall_floors("balanced", matrix=matrix)

    assert INDIC_SCRIPTS <= floors.keys()
    assert floors["Han"] >= 0.99
    assert floors["Devanagari"] >= 0.99
    assert floors["Telugu"] >= 0.99
    assert (
        profile_recall_floor("balanced", matrix=matrix, script="Han") == (floors["Han"])
    )
    assert profile_script_leakage_ceiling("balanced", matrix=matrix) <= 0.01

    matrix["profiles"]["balanced"]["script_recall_floors"]["Han"] = 0.997
    matrix["profiles"]["balanced"]["script_leakage_ceiling"] = 0.003
    validate_threshold_matrix(matrix)
    assert profile_recall_floor("balanced", matrix=matrix, script="Han") == 0.997
    assert profile_script_leakage_ceiling("balanced", matrix=matrix) == 0.003


def test_recall_floor_guard_blocks_drop_below_profile_floor():
    result = recall_floor_guard(
        {"ID_NUM": 0.996},
        {"ID_NUM": 0.992},
        policy_profile="strict_no_leak",
        protected_labels=("ID_NUM",),
    )

    assert result.block is True
    assert result.violations[0]["canonical_label"] == "ID_NUM"
    assert result.recall_floor == 0.995


def test_fit_and_update_thresholds_return_valid_versioned_matrix():
    matrix = load_thresholds()
    fitted = fit_thresholds(
        [
            {
                "canonical_label": "EMAIL",
                "language": "en",
                "score": 0.42,
                "target": True,
            }
        ],
        policy_profile="balanced",
        base_matrix=matrix,
    )
    updated = update_thresholds(
        fitted,
        {
            ("EMAIL", "en", "balanced"): {
                "keep_floor": 0.4,
                "escalate_below": 0.45,
                "action": "mask",
            }
        },
        bump_schema_version=True,
    )

    validate_threshold_matrix(updated)
    assert updated["schema_version"] == matrix["schema_version"] + 1
    assert (
        lookup_threshold("EMAIL", "en", "balanced", matrix=updated)["keep_floor"] == 0.4
    )


def test_membership_defense_resolves_from_policy_profile():
    matrix = load_thresholds()
    matrix["profiles"]["balanced"]["membership_defense"] = {
        "enabled": True,
        "clip_min": 0.4,
        "clip_max": 0.6,
        "temperature": 2.0,
        "smoothing": 0.5,
    }

    policy = membership_defense_for_profile("balanced", matrix=matrix)

    assert policy.enabled is True
    assert policy.recall_floor == matrix["profiles"]["balanced"]["recall_floor"]
    assert policy.advantage_ceiling == DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING
    assert 0.5 <= policy.apply_score(0.99) <= 0.6


def test_threshold_matrix_rejects_malformed_membership_defense():
    matrix = load_thresholds()
    matrix["profiles"]["balanced"]["membership_defense"] = {
        "enabled": True,
        "clip_min": 0.8,
        "clip_max": 0.2,
    }

    with pytest.raises(ValueError, match="clip_min"):
        validate_threshold_matrix(matrix)


def test_load_thresholds_rejects_malformed_json(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in threshold file") as exc_info:
        load_thresholds(path)

    assert str(path) in str(exc_info.value)


def test_arbitration_reads_matrix_keep_floor_by_mode():
    weak_id = _span(score=0.2)

    assert arbitrate([weak_id], mode=MODE_BALANCED) == ()
    assert arbitrate([weak_id], mode=MODE_HIGH_RECALL_UNION) == (weak_id,)


def test_cascade_reads_matrix_escalate_below():
    base_calls = []
    weak_id = _span(score=0.57)

    def base_detector(text, **kwargs):
        base_calls.append(kwargs["reason"])
        return []

    router = CascadeRouter(
        tiny_detector=lambda text, **kwargs: [weak_id],
        base_detector=base_detector,
    )

    result = router.run("MRN 123")

    assert result.reached(R2_BASE)
    assert base_calls == ["low_confidence"]


def test_pipeline_default_policy_action_comes_from_threshold_matrix():
    span = _span(label="EMAIL", score=0.9)
    router = CascadeRouter(tiny_detector=lambda text, **kwargs: [span])

    result = Pipeline(
        cascade_router=router,
        use_safety_sweep=False,
    ).run("email a@b.co")

    assert result.stage("policy_actions").spans[0].action == "mask"
    assert "threshold_action" in result.stage("policy_actions").spans[0].metadata
