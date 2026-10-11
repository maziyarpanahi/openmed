"""Synthetic adapter controls; these results are never clinical quality evidence."""

import copy
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from openmed.clinical.nli_gate import EvidenceLink, evaluate_nli
from openmed.clinical.nli_qualification import (
    NLIQualificationError,
    NLIQualificationPolicy,
    bind_qualified_nli,
    qualify_local_nli,
)
from openmed.eval.nli_error_slices import CLINICAL_NLI_PHENOMENA

MAPPING = {"0": "entailment", "1": "contradiction", "2": "neutral"}
POLICY = NLIQualificationPolicy(min_per_class=1)


class FakeLoader:
    """Deterministic injected runtime, wholly independent of real checkpoints."""

    def __init__(self, *, wrong=False, tie=False, failure=False):
        self.calls = 0
        self.wrong = wrong
        self.tie = tie
        self.failure = failure

    def load_local_sequence_classifier(self, reference, *, revision, runtime):
        self.calls += 1
        assert revision is None
        if self.failure:
            raise RuntimeError("private-path SECRET 555-0101")

        def tokenizer(premise, hypothesis, **kwargs):
            assert kwargs["truncation"] is False
            label = premise.split()[0]
            index = (
                list(MAPPING.values()).index(label) if label in MAPPING.values() else 0
            )
            return {"input_ids": [[(index + int(self.wrong)) % 3]]}

        def model(**encoded):
            index = encoded["input_ids"][0][0]
            scores = [0.0 if self.tie or i != index else 8.0 for i in range(3)]
            return SimpleNamespace(logits=[SimpleNamespace(tolist=lambda: scores)])

        return {"model": model, "tokenizer": tokenizer}


def dataset(prefix, *, kind="synthetic"):
    return {
        "provenance": {"kind": kind, "reference": "private-source SECRET 555-0101"},
        "records": [
            {
                "id": f"{prefix}-{label}",
                "group_id": f"group-{prefix}-{label}",
                "premise": f"{label} synthetic {prefix} 555-0101",
                "hypothesis": f"Synthetic {prefix}",
                "gold_label": label,
                "slices": list(CLINICAL_NLI_PHENOMENA),
            }
            for label in MAPPING.values()
        ],
    }


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("qualification attempted network access")

    monkeypatch.setattr(socket, "create_connection", no_network)
    artifact = tmp_path / "private-artifact-SECRET"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"synthetic weights")
    (artifact / "tokenizer.json").write_text('{"synthetic": true}')
    (artifact / "config.json").write_text('{"num_labels": 3}')
    return dict(
        model_path=artifact,
        label_mapping=copy.deepcopy(MAPPING),
        development=dataset("dev"),
        evaluation=dataset("eval"),
        policy=POLICY,
    )


def declared_inputs(inputs):
    """Exercise caller provenance declarations; no real clinical data is used."""
    result = copy.deepcopy(inputs)
    for name in ("development", "evaluation"):
        result[name]["provenance"]["kind"] = "caller_supplied"
    return result


def test_synthetic_reports_are_deterministic_safe_and_never_qualified(inputs):
    receipt = qualify_local_nli(**inputs, loader=FakeLoader())
    payload = receipt.to_dict()
    assert payload["status"] == "synthetic_only"
    assert payload["qualified"] is False
    assert payload["reasons"] == ["synthetic_not_clinical_evidence"]
    assert payload["supported_slices"] == list(CLINICAL_NLI_PHENOMENA)
    assert payload["split_counts"] == {"development": 3, "evaluation": 3}
    assert (
        receipt.to_json() == qualify_local_nli(**inputs, loader=FakeLoader()).to_json()
    )
    for marker in (
        "SECRET",
        "555-0101",
        str(inputs["model_path"]),
        "group-dev",
        "Synthetic eval",
    ):
        assert marker not in receipt.to_json()
        assert marker not in repr(receipt)
    payload["qualified"] = True
    assert receipt.to_dict()["qualified"] is False
    with pytest.raises(NLIQualificationError, match="uncalibrated"):
        bind_qualified_nli(receipt, **inputs, loader=FakeLoader())


def test_declared_caller_data_passes_policy_and_binds_gate_scores(inputs):
    options = declared_inputs(inputs)
    receipt = qualify_local_nli(**options, loader=FakeLoader())
    assert receipt.to_dict()["status"] == "qualified"
    backend = bind_qualified_nli(receipt, **options, loader=FakeLoader())
    scores = backend("entailment synthetic", "synthetic claim")
    assert (
        scores["calibration_id"] == receipt.digest == backend.thresholds.calibration_id
    )
    assert set(scores) == {
        "entailment",
        "contradiction",
        "neutral",
        "calibrated",
        "calibration_id",
    }
    verdict = evaluate_nli(
        scores, EvidenceLink("source", "claim"), thresholds=backend.thresholds
    )
    assert verdict.outcome == "entailment"


@pytest.mark.parametrize("missing", ["development", "evaluation"])
def test_missing_restricted_data_is_unavailable_without_loading(inputs, missing):
    options = declared_inputs(inputs)
    options["development"]["provenance"]["kind"] = "restricted"
    options[missing] = None
    loader = FakeLoader()
    payload = qualify_local_nli(**options, loader=loader).to_dict()
    assert payload["status"] == "unavailable"
    assert payload["reasons"] == [f"{missing}_unavailable"]
    assert payload["reports"] == {}
    assert loader.calls == 0


@pytest.mark.parametrize("mutation", ["class", "slice", "group", "pair", "empty"])
def test_insufficiency_and_leaking_splits_refuse_before_loading(inputs, mutation):
    options = declared_inputs(inputs)
    rows = options["evaluation"]["records"]
    if mutation == "class":
        rows.pop()
    elif mutation == "slice":
        rows[0]["slices"].remove("negation")
    elif mutation == "group":
        rows[0]["group_id"] = options["development"]["records"][0]["group_id"]
    elif mutation == "pair":
        rows[0]["premise"] = options["development"]["records"][0]["premise"]
        rows[0]["hypothesis"] = options["development"]["records"][0]["hypothesis"]
    else:
        rows.clear()
    loader = FakeLoader()
    payload = qualify_local_nli(**options, loader=loader).to_dict()
    assert payload["status"] == "insufficient"
    assert payload["reasons"]
    assert loader.calls == 0


@pytest.mark.parametrize("target", ["mapping", "gold"])
def test_unknown_labels_fail_closed_without_echoing_values(inputs, target):
    if target == "mapping":
        inputs["label_mapping"]["0"] = "private UNKNOWN"
    else:
        inputs["evaluation"]["records"][0]["gold_label"] = "private UNKNOWN"
    payload = qualify_local_nli(**inputs, loader=FakeLoader()).to_dict()
    assert payload["status"] == "unavailable"
    assert "UNKNOWN" not in json.dumps(payload)
    assert payload["reasons"] == ["unknown_label"]


@pytest.mark.parametrize(
    "drift",
    [
        "weights",
        "tokenizer",
        "config",
        "mapping",
        "development",
        "evaluation",
        "provenance",
        "policy",
        "runtime",
    ],
)
def test_every_artifact_or_calibration_input_drift_invalidates_receipt(inputs, drift):
    options = declared_inputs(inputs)
    receipt = qualify_local_nli(**options, loader=FakeLoader())
    backend = bind_qualified_nli(receipt, **options, loader=FakeLoader())
    if drift in ("weights", "tokenizer", "config"):
        filename = {
            "weights": "model.safetensors",
            "tokenizer": "tokenizer.json",
            "config": "config.json",
        }[drift]
        (options["model_path"] / filename).write_bytes(b"changed")
    elif drift == "mapping":
        options["label_mapping"].update({"0": "neutral", "2": "entailment"})
    elif drift in ("development", "evaluation"):
        options[drift]["records"][0]["hypothesis"] = (
            "changed synthetic calibration input"
        )
    elif drift == "provenance":
        options["evaluation"]["provenance"]["reference"] = "changed opaque source"
    elif drift == "policy":
        options["policy"] = replace(POLICY, min_per_class=2)
    else:
        options["runtime"] = "onnx"
    with pytest.raises(NLIQualificationError, match="receipt_mismatch"):
        bind_qualified_nli(receipt, **options, loader=FakeLoader())
    if drift not in ("policy", "runtime"):
        with pytest.raises(NLIQualificationError, match="receipt_mismatch"):
            backend("entailment", "synthetic")


def test_evaluation_failure_cannot_reselect_development_threshold(inputs):
    options = declared_inputs(inputs)
    options["evaluation"]["records"][0]["premise"] = "neutral synthetic eval failure"
    payload = qualify_local_nli(**options, loader=FakeLoader()).to_dict()
    assert payload["qualified"] is False
    assert "evaluation_constraints_entailment" in payload["reasons"]
    for label in ("entailment", "contradiction"):
        reports = payload["reports"][label]
        assert (
            reports["evaluation"]["recommended_threshold"]
            == reports["development"]["recommended_threshold"]
        )
        assert len(reports["evaluation"]["threshold_points"]) == 1


@pytest.mark.parametrize("mode", ["tie", "wrong", "failure"])
def test_bad_adapter_cannot_qualify_or_disclose_exceptions(inputs, mode):
    payload = qualify_local_nli(
        **declared_inputs(inputs), loader=FakeLoader(**{mode: True})
    ).to_dict()
    assert payload["qualified"] is False
    assert payload["reasons"]
    assert "SECRET" not in json.dumps(payload)
    if mode == "failure":
        assert payload["reasons"] == ["local_inference_unavailable"]


def test_symlinked_artifact_is_unavailable(inputs, tmp_path):
    (inputs["model_path"] / "outside").symlink_to(tmp_path / "missing")
    assert qualify_local_nli(**inputs).to_dict()["reasons"] == ["artifact_unavailable"]


def test_relaxed_constraints_do_not_accept_all_abstention(inputs):
    options = declared_inputs(inputs)
    options["policy"] = replace(
        POLICY, precision_floor=0, recall_floor=0, false_positive_rate_ceiling=1
    )
    payload = qualify_local_nli(**options, loader=FakeLoader(tie=True)).to_dict()
    assert payload["qualified"] is False
    assert "development_constraints_entailment" in payload["reasons"]


def test_repeated_identifier_cannot_enter_heldout(inputs):
    inputs["evaluation"]["records"][0]["id"] = inputs["development"]["records"][0]["id"]
    assert "split_overlap" in qualify_local_nli(**inputs).to_dict()["reasons"]


def test_synthetic_audit_payload_cannot_construct_a_live_qualified_receipt(inputs):
    from openmed.clinical.nli_qualification import NLIQualificationReceipt

    payload = qualify_local_nli(**inputs, loader=FakeLoader()).to_dict()
    payload.update(qualified=True, status="qualified", reasons=[])
    payload.pop("receipt_digest")
    with pytest.raises(NLIQualificationError, match="invalid_receipt"):
        fake = NLIQualificationReceipt(json.dumps(payload))
        bind_qualified_nli(fake, **inputs, loader=FakeLoader())


def test_audit_shaped_object_is_not_live_calibration_authority(inputs):
    payload = qualify_local_nli(**inputs, loader=FakeLoader()).to_dict()
    payload.update(qualified=True, status="qualified", reasons=[])
    with pytest.raises(NLIQualificationError, match="invalid_receipt"):
        bind_qualified_nli(
            SimpleNamespace(to_dict=lambda: payload, digest="c" * 64),
            **inputs,
            loader=FakeLoader(),
        )


def test_predict_controls_unicode_hash_failure_without_retaining_pair(inputs):
    from openmed.clinical.nli_backends import EncoderNLIBackend, LocalNLIError
    from openmed.clinical.nli_gate import NLIThresholds

    backend = EncoderNLIBackend(
        inputs["model_path"],
        label_mapping=MAPPING,
        thresholds=NLIThresholds(),
        loader=FakeLoader(),
    )
    with pytest.raises(LocalNLIError) as caught:
        backend.predict("entailment SYNTHETIC_PRIVATE_IDENTIFIER\ud800", "Synthetic")
    assert caught.value.__context__ is None
    assert "SYNTHETIC_PRIVATE_IDENTIFIER" not in str(caught.value)


def test_score_provider_exception_is_value_free_without_private_context(inputs):
    from openmed.clinical.nli_backends import EncoderNLIBackend, LocalNLIError
    from openmed.clinical.nli_gate import NLIThresholds

    backend = EncoderNLIBackend(
        inputs["model_path"],
        label_mapping=MAPPING,
        thresholds=NLIThresholds(),
        loader=FakeLoader(failure=True),
    )
    with pytest.raises(LocalNLIError) as caught:
        backend.predict_scores("entailment synthetic", "synthetic")
    assert caught.value.__context__ is None
    assert "SECRET" not in str(caught.value)


def test_binding_label_failure_retains_no_private_decoder_context(inputs):
    options = declared_inputs(inputs)
    receipt = qualify_local_nli(**options, loader=FakeLoader())
    options["label_mapping"]["0"] = "SYNTHETIC_PRIVATE_IDENTIFIER"
    with pytest.raises(NLIQualificationError, match="unknown_label") as caught:
        bind_qualified_nli(receipt, **options, loader=FakeLoader())
    assert caught.value.__context__ is None
    assert "SYNTHETIC_PRIVATE_IDENTIFIER" not in str(caught.value)
