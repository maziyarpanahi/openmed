from __future__ import annotations

import json

import pytest

from openmed.eval.datasets import (
    CLINICAL_MODEL_FAMILIES,
    DUA_GATED_CORPORA,
    PUBLIC_DATASETS,
    DUACredentialRequired,
    assert_no_gated_content_committed,
    clinical_family_dataset_bindings,
    license_for,
    load_dua_corpus,
    load_public_dataset,
    map_public_label,
    validate_clinical_family_dataset_evidence,
)


def test_public_adapter_loads_common_schema_and_maps_labels(tmp_path):
    text = "Patient Jordan Smith called 555-111-2222."
    payload = {
        "records": [
            {
                "id": "shield-1",
                "text": text,
                "split": "sample",
                "spans": [
                    {"start": 8, "end": 20, "label": "patient", "text": "Jordan Smith"},
                    {"start": 28, "end": 40, "label": "phone", "text": "555-111-2222"},
                ],
                "metadata": {"source": "unit-test"},
            }
        ]
    }
    path = tmp_path / "shield.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_public_dataset("shield", path)

    assert result.skipped is False
    assert len(result.records) == 1
    record = result.records[0]
    assert record.license == license_for("shield")
    assert [span.label for span in record.spans] == ["PERSON", "PHONE"]
    fixture = record.to_benchmark_fixture()
    assert fixture.fixture_id == "shield-1"
    assert fixture.gold_spans[0].label == "PERSON"
    assert fixture.metadata["license"]["redistribution"] == "reference-only"


def test_public_adapters_skip_cleanly_when_source_absent(tmp_path):
    for dataset in PUBLIC_DATASETS:
        result = load_public_dataset(dataset, tmp_path / f"{dataset}.json")
        assert result.skipped is True
        assert result.records == ()
        assert result.license is not None


def test_public_label_maps_are_canonical():
    assert map_public_label("shield", "id") == "ID_NUM"
    assert map_public_label("drugprot", "CHEMICAL") == "OTHER"
    assert map_public_label("ncbi_disease", "Disease") == "OTHER"
    assert map_public_label("bc5cdr", "chemical") == "OTHER"


def test_clinical_family_dataset_policy_is_public_first_and_eval_only_for_dua():
    assert CLINICAL_MODEL_FAMILIES == (
        "doctype",
        "section",
        "relex_med",
        "relex_ade",
        "link",
    )

    for family in ("relex_med", "relex_ade"):
        bindings = {
            binding.dataset: binding
            for binding in clinical_family_dataset_bindings(family)
        }
        assert bindings["drugprot"].uses == ("train", "eval")
        assert bindings["drugprot"].access == "public"
        for dataset in ("n2c2", "made"):
            assert bindings[dataset].uses == ("eval",)
            assert bindings[dataset].access == "dua-eval-only"
            assert bindings[dataset].required is False

    link_bindings = {
        binding.dataset: binding for binding in clinical_family_dataset_bindings("link")
    }
    assert link_bindings["redistributable_vocabulary"].uses == ("train",)
    assert link_bindings["medmentions"].uses == ("eval",)
    assert link_bindings["umls"].user_key_gated is True
    assert link_bindings["snomed"].user_key_gated is True


def test_clinical_family_dataset_evidence_requires_public_training_and_metrics():
    digest = "sha256:" + "1" * 64
    relation = validate_clinical_family_dataset_evidence(
        "relex_med",
        {
            "drugprot": {
                "uses": ["train", "eval"],
                "manifest_hash": digest,
                "metrics": {"micro_f1": 0.91},
                "corpus_bundled": False,
                "restricted_vocabulary_bundled": False,
            }
        },
    )
    assert relation["drugprot"]["access"] == "public"

    linking = validate_clinical_family_dataset_evidence(
        "link",
        {
            "redistributable_vocabulary": {
                "uses": ["train"],
                "manifest_hash": digest,
                "metrics": {},
                "corpus_bundled": False,
                "restricted_vocabulary_bundled": False,
                "redistributable": True,
                "vocab": "HPO",
            },
            "medmentions": {
                "uses": ["eval"],
                "manifest_hash": digest,
                "metrics": {"top1_accuracy": 0.71},
                "corpus_bundled": False,
                "restricted_vocabulary_bundled": False,
            },
        },
    )
    assert linking["medmentions"]["metrics"]["top1_accuracy"] == 0.71


def test_clinical_family_dataset_evidence_rejects_restricted_training_or_bundles():
    digest = "sha256:" + "2" * 64
    with pytest.raises(ValueError, match="uses"):
        validate_clinical_family_dataset_evidence(
            "relex_ade",
            {
                "drugprot": {
                    "uses": ["train", "eval"],
                    "manifest_hash": digest,
                    "metrics": {},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": False,
                },
                "n2c2": {
                    "uses": ["train"],
                    "manifest_hash": digest,
                    "metrics": {},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": False,
                },
            },
        )

    with pytest.raises(ValueError, match="restricted vocabulary"):
        validate_clinical_family_dataset_evidence(
            "link",
            {
                "redistributable_vocabulary": {
                    "uses": ["train"],
                    "manifest_hash": digest,
                    "metrics": {},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": True,
                    "redistributable": True,
                    "vocab": "HPO",
                },
                "medmentions": {
                    "uses": ["eval"],
                    "manifest_hash": digest,
                    "metrics": {"top1_accuracy": 0.71},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": False,
                },
            },
        )

    with pytest.raises(ValueError, match="redistributable vocabulary"):
        validate_clinical_family_dataset_evidence(
            "link",
            {
                "redistributable_vocabulary": {
                    "uses": ["train"],
                    "manifest_hash": digest,
                    "metrics": {},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": False,
                    "redistributable": True,
                    "vocab": "UMLS",
                },
                "medmentions": {
                    "uses": ["eval"],
                    "manifest_hash": digest,
                    "metrics": {"top1_accuracy": 0.71},
                    "corpus_bundled": False,
                    "restricted_vocabulary_bundled": False,
                },
            },
        )


def test_dua_stubs_refuse_without_credentialed_path():
    for corpus in DUA_GATED_CORPORA:
        with pytest.raises(DUACredentialRequired):
            load_dua_corpus(corpus)


def test_dua_stub_accepts_existing_credentialed_path_as_eval_only(tmp_path):
    result = load_dua_corpus("i2b2", tmp_path)

    assert result.skipped is True
    assert result.reason.startswith("eval-only gated corpus stub")


@pytest.mark.parametrize("marker", ["UMLS", "snomed", "cPt", "n2C2", "made"])
def test_guard_rejects_gated_payload_markers(tmp_path, marker):
    data_file = tmp_path / "payload.jsonl"
    data_file.write_text(json.dumps({"label": marker}) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="gated dataset content"):
        assert_no_gated_content_committed(tmp_path)


def test_guard_ignores_python_source_and_clean_payloads(tmp_path):
    (tmp_path / "adapter.py").write_text('MARKER = "UMLS"\n', encoding="utf-8")
    (tmp_path / "payload.jsonl").write_text(
        '{"label": "PERSON", "text": "The synthetic patient made progress."}\n',
        encoding="utf-8",
    )

    assert_no_gated_content_committed(tmp_path)
