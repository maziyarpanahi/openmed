"""Offline tests for the GLiNER2.5 fixed-option decision backend."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.ner.families import gliner2_decision as decision_module
from openmed.ner.families.gliner2 import GLiNER2Handle
from openmed.ner.families.gliner2_decision import GLiNER2DecisionBackend
from openmed.structured.decision import (
    DecisionBackend,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    decide,
)

COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


class FakeClassifier:
    """Records ``classify_text`` calls and answers with fixed label scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def classify_text(self, text: str, tasks: Any, **kwargs: Any) -> Any:
        self.calls.append({"tasks": tasks, "kwargs": kwargs})
        return {
            task: [
                {"label": label, "confidence": self.scores[label]}
                for label in config["labels"]
                if label in self.scores
            ]
            for task, config in tasks.items()
        }


def _backend(
    scores: dict[str, float],
    *,
    model_id: str = "fastino/GLiNER2.5-multi-Decide",
    **kwargs: Any,
) -> tuple[GLiNER2DecisionBackend, FakeClassifier]:
    model = FakeClassifier(scores)
    handle = GLiNER2Handle(
        model_id=model_id,
        model=model,
        revision=COMMIT,
        license_id="Apache-2.0",
        architecture="boundary",
    )
    return GLiNER2DecisionBackend(handle, **kwargs), model


def _request(
    mode: DecisionMode, options: tuple[str, ...], text: str
) -> DecisionRequest:
    return DecisionRequest(mode=mode, input_text=text, options=options)


def test_backend_identity_records_model_revision_and_license() -> None:
    backend, _ = _backend({})

    assert isinstance(backend, DecisionBackend)
    identity = backend.identity.to_dict()
    assert identity == {
        "backend_id": "openmed.gliner2.classify_text.v1",
        "kind": "encoder",
        "runtime": "gliner2",
        "model_id": "fastino/GLiNER2.5-multi-Decide",
        "revision": COMMIT,
        "license_id": "Apache-2.0",
        "local": True,
        "deterministic": False,
    }


@pytest.mark.parametrize("license_id", [None, "GPL-3.0-only"])
def test_backend_requires_a_permissive_license(license_id) -> None:
    handle = GLiNER2Handle(
        model_id="someone/unreviewed-gliner2",
        model=FakeClassifier({}),
        revision=COMMIT,
        license_id=license_id,
    )

    with pytest.raises(DecisionError, match="license"):
        GLiNER2DecisionBackend(handle)


def test_fixed_choice_maps_to_a_single_label_softmax_task() -> None:
    backend, model = _backend(
        {"urgent": 0.86, "routine": 0.1, "deferred": 0.04}, task="priority"
    )

    result = decide(
        _request(
            DecisionMode.FIXED_CHOICE,
            ("urgent", "routine", "deferred"),
            "Synthetic referral notes same-day chest pain.",
        ),
        backend=backend,
    )

    assert model.calls[0]["tasks"] == {
        "priority": {
            "labels": ["urgent", "routine", "deferred"],
            "multi_label": True,
            "cls_threshold": 0.0,
            "class_act": "softmax",
        }
    }
    assert result.state is DecisionState.SUCCESS
    assert result.choice == "urgent"
    assert [item["score"] for item in result.option_scores] == [0.86, 0.1, 0.04]
    assert result.confidence == pytest.approx(0.86)
    assert result.margin == pytest.approx(0.76)
    assert result.backend["revision"] == COMMIT
    assert result.autonomous_action is False
    assert result.review["required"] is True


def test_ambiguous_scores_abstain_instead_of_choosing() -> None:
    backend, _ = _backend({"urgent": 0.48, "routine": 0.52})

    result = decide(
        _request(
            DecisionMode.FIXED_CHOICE,
            ("urgent", "routine"),
            "Synthetic note is unclear.",
        ),
        backend=backend,
    )

    assert result.state is DecisionState.ABSTAINED
    assert result.choice is None
    assert [item["score"] for item in result.option_scores] == [0.48, 0.52]


@pytest.mark.parametrize(
    ("options", "expected_scores", "choice"),
    [
        (("no", "yes"), [0.07, 0.93], "yes"),
        (("Yes", "No"), [0.93, 0.07], "Yes"),
        (("TRUE", "false"), [0.93, 0.07], "TRUE"),
    ],
)
def test_boolean_choice_maps_to_yes_no_labels(options, expected_scores, choice) -> None:
    backend, model = _backend({"yes": 0.93, "no": 0.07}, task="needs_follow_up")

    result = decide(
        _request(DecisionMode.BOOLEAN_CHOICE, options, "Synthetic follow-up note."),
        backend=backend,
    )

    task = model.calls[0]["tasks"]["needs_follow_up"]
    assert task["labels"] == ["yes", "no"]
    assert task["class_act"] == "softmax"
    assert [item["score"] for item in result.option_scores] == expected_scores
    assert [item["option"] for item in result.option_scores] == list(options)
    assert result.choice == choice


def test_boolean_choice_with_other_wording_uses_the_caller_labels() -> None:
    backend, model = _backend({"present": 0.2, "absent": 0.8})

    result = decide(
        _request(
            DecisionMode.BOOLEAN_CHOICE,
            ("present", "absent"),
            "Synthetic finding was ruled out.",
        ),
        backend=backend,
    )

    assert model.calls[0]["tasks"]["decision"]["labels"] == ["present", "absent"]
    assert result.choice == "absent"


def test_multi_label_returns_every_score_and_uses_the_profile_threshold() -> None:
    backend, model = _backend({"diabetes": 0.91, "renal": 0.74, "oncology": 0.02})

    result = decide(
        _request(
            DecisionMode.MULTI_LABEL,
            ("diabetes", "renal", "oncology"),
            "Synthetic chart mentions diabetes and kidney disease.",
        ),
        backend=backend,
    )

    task = model.calls[0]["tasks"]["decision"]
    assert task["class_act"] == "sigmoid"
    assert task["cls_threshold"] == 0.0
    assert [item["score"] for item in result.option_scores] == [0.91, 0.74, 0.02]
    assert result.state is DecisionState.SUCCESS
    assert result.choices == ("diabetes", "renal")


def test_scalar_score_is_the_affirmative_probability() -> None:
    backend, model = _backend({"yes": 0.12, "no": 0.88})

    result = decide(
        _request(DecisionMode.SCALAR_SCORE, (), "Synthetic note shows no concern."),
        backend=backend,
    )

    assert model.calls[0]["tasks"]["decision"]["labels"] == ["yes", "no"]
    assert result.state is DecisionState.SUCCESS
    assert result.scalar_score == pytest.approx(0.12)
    assert result.confidence == pytest.approx(0.88)


def test_ordered_preference_ranks_with_softmax_scores() -> None:
    backend, _ = _backend({"low": 0.05, "medium": 0.15, "high": 0.8})

    result = decide(
        _request(
            DecisionMode.ORDERED_PREFERENCE,
            ("low", "medium", "high"),
            "Synthetic triage severity.",
        ),
        backend=backend,
    )

    assert result.ranking == ("high", "medium", "low")


def test_missing_option_scores_become_a_conflict_not_a_success() -> None:
    backend, _ = _backend({"urgent": 0.99})

    result = decide(
        _request(
            DecisionMode.FIXED_CHOICE,
            ("urgent", "routine"),
            "Synthetic referral.",
        ),
        backend=backend,
    )

    assert result.state is DecisionState.CONFLICT
    assert result.code == "backend_conflict"
    assert result.choice is None


def test_non_english_input_is_scored_unchanged() -> None:
    backend, model = _backend({"sí": 0.9, "no": 0.1})
    text = "Paciente sintético refiere dolor torácico desde ayer."

    result = decide(
        _request(DecisionMode.FIXED_CHOICE, ("sí", "no"), text),
        backend=backend,
    )

    assert result.choice == "sí"
    assert model.calls[0]["tasks"]["decision"]["labels"] == ["sí", "no"]


def test_from_pretrained_loads_through_the_family_loader(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_loader(model_id: str, **kwargs: Any) -> GLiNER2Handle:
        seen["model_id"] = model_id
        seen.update(kwargs)
        return GLiNER2Handle(
            model_id=model_id,
            model=FakeClassifier({}),
            revision=COMMIT,
            license_id="Apache-2.0",
        )

    monkeypatch.setattr(decision_module, "load_gliner2_handle", fake_loader)

    backend = GLiNER2DecisionBackend.from_pretrained(
        "fastino/GLiNER2.5-Decide",
        revision=COMMIT,
        local_files_only=True,
        task="eligible",
    )

    assert seen["model_id"] == "fastino/GLiNER2.5-Decide"
    assert seen["revision"] == COMMIT
    assert seen["local_files_only"] is True
    assert backend.task == "eligible"
    assert backend.identity.model_id == "fastino/GLiNER2.5-Decide"


def test_decision_results_do_not_carry_input_text() -> None:
    marker = "Synthetic Orlaith Vantablack"
    backend, _ = _backend({"urgent": 0.9, "routine": 0.1})

    result = decide(
        _request(DecisionMode.FIXED_CHOICE, ("urgent", "routine"), f"{marker} note"),
        backend=backend,
    )

    assert marker not in repr(result.to_dict())
