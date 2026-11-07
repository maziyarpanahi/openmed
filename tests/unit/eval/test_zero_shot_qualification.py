"""Gate tests for zero-shot export qualification with oracle runners.

No model runtime is needed: the reference and candidates are rule-based
logits runners over feeds built by the profile contract with a pure-Python
character tokenizer, so these tests run everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmed.eval import export_qualification as eq
from openmed.eval import zero_shot_qualification as zq
from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    VariantCandidate,
)
from openmed.mlx.models import normalize_model_config
from openmed.ner.families import zero_shot_export as zs
from tests.fixtures.zero_shot_export_stub import (
    GLICLASS_LABELS,
    GLICLASS_MARKERS,
    GLINER_LABELS,
    GLINER_MARKERS,
    SENTINELS,
    CharTokenizer,
    gliclass_config,
    gliclass_fixture_rows,
    gliner_config,
    gliner_fixture_rows,
    rule_class_logits,
    rule_span_logits,
    write_jsonl,
)

SOURCE = {"kind": "content_sha256", "revision": "sha256:" + "0" * 64}


def _setup(tmp_path: Path, profile: zs.ZeroShotProfile, **config_overrides: Any):
    if profile is zs.GLINER_SPAN_PROFILE:
        config = gliner_config(**config_overrides)
        markers, rows = GLINER_MARKERS, gliner_fixture_rows()
    else:
        config = gliclass_config(**config_overrides)
        markers, rows = GLICLASS_MARKERS, gliclass_fixture_rows()
    normalized = normalize_model_config(config)
    spec = zs.validate_zero_shot_config(profile, normalized)
    tokenizer = CharTokenizer(markers)
    prompt = zs.resolve_prompt_spec(profile, normalized, spec, tokenizer)
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "model.onnx").write_bytes(b"graph")
    (bundle / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")
    return SimpleNamespace(
        profile=profile,
        spec=spec,
        tokenizer=tokenizer,
        prompt=prompt,
        bounds=zs.default_bounds(spec),
        bundle=bundle,
        fixtures=write_jsonl(tmp_path / "suite.jsonl", rows),
    )


def _candidate(runner: Any, setup: SimpleNamespace, **kwargs: Any) -> VariantCandidate:
    options = {
        "name": "onnx-fp32",
        "runner": runner,
        "precision": "float32",
        "provider": "CPUExecutionProvider",
        "artifact_paths": (setup.bundle / "model.onnx",),
    }
    options.update(kwargs)
    return VariantCandidate(**options)


def _qualify(setup: SimpleNamespace, *candidates: VariantCandidate, **kwargs: Any):
    reference = (
        rule_span_logits
        if setup.profile is zs.GLINER_SPAN_PROFILE
        else rule_class_logits
    )
    options = {
        "fixtures_path": setup.fixtures,
        "environment": {"python": "3.11", "packages": {}},
    }
    options.update(kwargs)
    return zq.qualify_zero_shot_export(
        profile=setup.profile,
        spec=setup.spec,
        prompt=setup.prompt,
        bounds=setup.bounds,
        tokenizer=setup.tokenizer,
        reference_runner=reference,
        candidates=list(candidates),
        source=SOURCE,
        artifact_root=setup.bundle,
        artifact_paths=[setup.bundle / "tokenizer.json"],
        **options,
    )


def _no_private_values(payload: Any, *extra: str) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    for value in SENTINELS:
        assert value not in text
    for value in extra:
        assert json.dumps(value) not in text


# ---------------------------------------------------------------------------
# GLiNER span gates
# ---------------------------------------------------------------------------


def test_identical_span_candidate_qualifies_with_value_free_evidence(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)
    report = _qualify(setup, _candidate(rule_span_logits, setup))
    assert report.passed, report.reasons
    payload = report.to_dict()
    assert payload["task"] == "zero-shot-ner"
    assert payload["offset_contract"]["unit"] == "unicode_code_point"
    assert payload["offset_contract"]["source"] == "word_splitter"
    assert payload["profile"]["label_orders"] == ["suite", "reversed"]
    assert payload["label_map"]["kind"] == "caller_supplied_labels"
    variant = payload["variants"][0]
    assert variant["claim"] == "qualified"
    assert variant["direct_identifier_recall"]["reference"] == {
        "ID_NUM": 1.0,
        "PERSON": 1.0,
    }
    assert variant["span_parity"]["identical"] is True
    assert variant["span_parity"]["offsets_valid"] is True
    assert variant["span_parity"]["fixtures_compared"] == 12  # 6 x 2 orders
    assert variant["logit_parity"]["label_orders"] == ["reversed", "suite"]
    assert variant["critical_leakage"]["new_leaks"] == 0
    assert payload["suite"]["candidate_label_count"] == 2
    _no_private_values(payload, *GLINER_LABELS)

    path = eq.write_qualification_report(
        report, setup.bundle / eq.EXPORT_QUALIFICATION_FILENAME
    )
    verified = eq.verify_qualification_report(path, artifact_root=setup.bundle)
    assert verified["qualified_variants"] == ["onnx-fp32"]
    (setup.bundle / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ExportQualificationError) as error:
        eq.verify_qualification_report(path, artifact_root=setup.bundle)
    assert error.value.reason is ExportReason.ARTIFACT_DIGEST_MISMATCH


def test_dropped_person_scores_fail_recall_leakage_and_span_gates(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)

    def dropped(feed: Any) -> Any:
        logits = rule_span_logits(feed)
        ids = [int(value) for value in feed["input_ids"][0]]
        markers = [index for index, token in enumerate(ids) if token == 5]
        for column, index in enumerate(markers):
            if chr(ids[index + 1] - 100) == "p":
                logits[..., column] = np.minimum(logits[..., column], -8.0)
        return logits

    report = _qualify(setup, _candidate(dropped, setup))
    assert not report.passed
    reasons = set(report.variants[0].reasons)
    assert {
        ExportReason.LOGIT_PARITY_FAILED.value,
        ExportReason.SPAN_PARITY_FAILED.value,
        ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED.value,
        ExportReason.CRITICAL_LEAKAGE_DETECTED.value,
    } <= reasons
    leakage = report.variants[0].critical_leakage
    assert leakage["new_leaks_by_label"] == {"PERSON": 12}
    with pytest.raises(ExportQualificationError):
        report.raise_for_failure()
    _no_private_values(report.to_dict())


def test_label_positions_frozen_at_export_fail_the_reversed_order_gate(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)

    def frozen(feed: Any) -> Any:
        # Pretends the labels always arrive in suite order.
        ids = feed["input_ids"].copy()
        markers = [index for index, token in enumerate(ids[0]) if token == 5]
        keys = [100 + ord("i"), 100 + ord("p")]
        for index, key in zip(markers, keys):
            ids[0, index + 1] = key
        return rule_span_logits({**feed, "input_ids": ids})

    report = _qualify(setup, _candidate(frozen, setup))
    assert not report.passed
    assert ExportReason.SPAN_PARITY_FAILED.value in report.variants[0].reasons
    span_parity = report.variants[0].span_parity
    assert span_parity["fixtures_mismatched"] == 6  # every reversed request


def test_broken_runners_fail_closed_without_echoing_inputs(tmp_path: Path) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)

    def wrong_shape(feed: Any) -> Any:
        return rule_span_logits(feed)[:, :-1]

    def raises(feed: Any) -> Any:
        raise RuntimeError(f"bad input {feed['input_ids'].tolist()}")

    report = _qualify(
        setup,
        _candidate(wrong_shape, setup, name="onnx-shape"),
        _candidate(raises, setup, name="onnx-raises"),
    )
    reasons = {item.name: item.reasons for item in report.variants}
    assert reasons == {
        "onnx-shape": (ExportReason.IO_CONTRACT_MISMATCH.value,),
        "onnx-raises": (ExportReason.INFERENCE_FAILED.value,),
    }
    assert "bad input" not in json.dumps(report.to_dict())


def test_undeclared_runtime_and_provider_are_refused_before_inference(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)
    calls: list[str] = []

    def counting(feed: Any) -> Any:
        calls.append("run")
        return rule_span_logits(feed)

    with pytest.raises(ExportQualificationError) as error:
        _qualify(setup, _candidate(counting, setup), runtime="coreml")
    assert error.value.reason is ExportReason.UNSUPPORTED_RUNTIME
    for kwargs, reason in (
        ({"provider": "CoreMLExecutionProvider"}, ExportReason.UNSUPPORTED_PROVIDER),
        ({"precision": "float16"}, ExportReason.UNSUPPORTED_PRECISION),
        ({"quantization": "int8"}, ExportReason.UNSUPPORTED_QUANTIZATION),
    ):
        with pytest.raises(ExportQualificationError) as error:
            _qualify(setup, _candidate(counting, setup, **kwargs))
        assert error.value.reason is reason
    assert calls == []


def test_suite_must_fit_the_declared_bounds(tmp_path: Path) -> None:
    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)
    setup.bounds = zs.default_bounds(setup.spec, max_sequence_length=40)
    with pytest.raises(zs.ZeroShotRequestError) as error:
        _qualify(setup, _candidate(rule_span_logits, setup))
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH


def test_default_phi_suite_qualifies_through_label_prompts(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        zs.GLINER_SPAN_PROFILE,
        max_position_embeddings=512,
        max_len=384,
    )
    report = _qualify(setup, _candidate(rule_span_logits, setup), fixtures_path=None)
    assert report.passed, report.reasons
    suite = report.to_dict()["suite"]
    assert suite["default"] is True
    assert suite["candidate_label_count"] == len(suite["labels"]) == 9
    assert "candidate_labels" not in suite


def test_span_suite_needs_a_prompt_for_every_gold_label(tmp_path: Path) -> None:
    rows = gliner_fixture_rows()
    path = write_jsonl(tmp_path / "suite.jsonl", rows)
    with pytest.raises(ExportQualificationError) as error:
        zq.load_zero_shot_fixtures(
            zs.GLINER_SPAN_PROFILE, path, label_prompts={"PERSON": "person"}
        )
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    fixtures, identity = zq.load_zero_shot_fixtures(zs.GLINER_SPAN_PROFILE, path)
    assert fixtures[0].labels == ("identifier", "person")
    assert fixtures[0].canonical == {"identifier": "ID_NUM", "person": "PERSON"}
    assert identity["synthetic"] is True and identity["default"] is False


def test_onnx_token_classifier_loader_refuses_zero_shot_bundles(
    tmp_path: Path,
) -> None:
    from openmed.onnx import inference

    setup = _setup(tmp_path, zs.GLINER_SPAN_PROFILE)
    report = _qualify(setup, _candidate(rule_span_logits, setup))
    eq.write_qualification_report(
        report, setup.bundle / eq.EXPORT_QUALIFICATION_FILENAME
    )
    with pytest.raises(ExportQualificationError) as error:
        inference._check_qualified_artifact(
            setup.bundle,
            setup.bundle / "model.onnx",
            providers=["CPUExecutionProvider"],
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_TASK


# ---------------------------------------------------------------------------
# GLiClass gates
# ---------------------------------------------------------------------------


def test_identical_classifier_candidate_qualifies(tmp_path: Path) -> None:
    setup = _setup(tmp_path, zs.GLICLASS_PROFILE)
    report = _qualify(setup, _candidate(rule_class_logits, setup))
    assert report.passed, report.reasons
    variant = report.to_dict()["variants"][0]
    parity = variant["classification_parity"]
    assert parity["ranking_identical"] and parity["decisions_identical"]
    assert parity["requests_compared"] == 8  # 4 fixtures x 2 label orders
    assert parity["label_orders"] == ["reversed", "suite"]
    assert variant["span_parity"] == {"applicable": False}
    assert variant["direct_identifier_recall"] == {"applicable": False}
    assert report.to_dict()["offset_contract"]["source"] == "not_applicable"
    _no_private_values(report.to_dict(), *GLICLASS_LABELS)


def test_classifier_ranking_and_decision_gates(tmp_path: Path) -> None:
    setup = _setup(tmp_path, zs.GLICLASS_PROFILE)

    def reversed_ranking(feed: Any) -> Any:
        return -rule_class_logits(feed)

    def shifted(feed: Any) -> Any:
        return rule_class_logits(feed) + 1.0

    report = _qualify(
        setup,
        _candidate(reversed_ranking, setup, name="onnx-reversed"),
        _candidate(shifted, setup, name="onnx-shifted"),
    )
    reasons = {item.name: set(item.reasons) for item in report.variants}
    assert ExportReason.LABEL_ORDER_PARITY_FAILED.value in reasons["onnx-reversed"]
    assert ExportReason.DECISION_PARITY_FAILED.value in reasons["onnx-shifted"]
    assert ExportReason.LABEL_ORDER_PARITY_FAILED.value not in reasons["onnx-shifted"]
    assert ExportReason.LOGIT_PARITY_FAILED.value in reasons["onnx-shifted"]


def test_default_classification_suite_is_synthetic_and_value_free(
    tmp_path: Path,
) -> None:
    fixtures, identity = zq.load_zero_shot_fixtures(zs.GLICLASS_PROFILE)
    assert identity["default"] is True and identity["synthetic"] is True
    assert identity["fixture_count"] == len(fixtures) == 8
    assert all(fixture.gold_labels for fixture in fixtures)
    assert "cardiology" not in json.dumps(identity)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "no labels"}\n', encoding="utf-8")
    with pytest.raises(ExportQualificationError) as error:
        zq.load_zero_shot_fixtures(zs.GLICLASS_PROFILE, bad)
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
