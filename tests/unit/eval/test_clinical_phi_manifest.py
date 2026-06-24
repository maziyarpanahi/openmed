from __future__ import annotations

import re

import pytest

from openmed.core.labels import (
    AGE,
    DATE,
    DATE_OF_BIRTH,
    ID_NUM,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    URL,
)
from openmed.eval.datasets import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PHI_MANIFEST_REF,
    CLINICAL_PRIVACY_MODEL_ID,
    DUACredentialRequired,
    clinical_phi_manifest_hash,
    load_clinical_phi_manifest,
    resolve_clinical_phi_source,
)
from openmed.eval.datasets.public import PUBLIC_LABEL_MAPS
from openmed.eval.suites.shield import SHIELD
from openmed.training import load_preset


def test_clinical_phi_manifest_names_flagship_and_sources() -> None:
    manifest = load_clinical_phi_manifest()

    assert manifest.manifest_id == CLINICAL_PHI_MANIFEST_ID
    assert manifest.model_id == CLINICAL_PRIVACY_MODEL_ID
    assert manifest.tier == "tier0"
    assert manifest.recipe_mode == "C"
    assert manifest.benchmark_suite == SHIELD

    source_ids = {source.source_id for source in manifest.sources}
    assert source_ids == {
        "shield_public_sample",
        "synthetic_golden_deid",
        "i2b2_eval_only",
        "n2c2_eval_only",
    }


def test_shield_reference_is_public_by_reference_and_label_mapped() -> None:
    source = load_clinical_phi_manifest().source("shield_public_sample")

    assert source.dataset == SHIELD
    assert source.role == "public_comparison"
    assert source.access == "public_reference"
    assert source.redistribution == "reference-only"
    assert source.label_map == PUBLIC_LABEL_MAPS[SHIELD]
    assert set(source.labels) == {
        AGE,
        DATE,
        ID_NUM,
        LOCATION,
        ORGANIZATION,
        PERSON,
        PHONE,
        URL,
    }
    assert "huggingface.co/datasets" in source.source_url


def test_gate_requirements_cover_required_clinical_phi_families() -> None:
    manifest = load_clinical_phi_manifest()

    g1a = manifest.gate("G1a")
    g2 = manifest.gate("G2")
    g3 = manifest.gate("G3")

    assert g1a.metric == "recall"
    assert g1a.threshold == pytest.approx(0.990)
    assert {PERSON, DATE, AGE, LOCATION, ID_NUM, PHONE, URL} <= set(g1a.labels)

    assert g2.metric == "name_address_date_recall"
    assert g2.threshold == pytest.approx(0.980)
    assert {PERSON, DATE, DATE_OF_BIRTH, LOCATION} <= set(g2.labels)

    assert g3.metric == "critical_leakage_count"
    assert g3.comparator == "=="
    assert g3.threshold == pytest.approx(0.0)
    assert ID_NUM in g3.labels


def test_dua_sources_refuse_without_credentials_and_accept_local_path(tmp_path) -> None:
    manifest = load_clinical_phi_manifest()
    for source_id in ("i2b2_eval_only", "n2c2_eval_only"):
        source = manifest.source(source_id)
        assert source.eval_only is True
        assert source.requires_credentials is True
        assert "not redistributed" in source.redistribution

        with pytest.raises(DUACredentialRequired):
            resolve_clinical_phi_source(source_id)

        result = resolve_clinical_phi_source(
            source_id,
            credentialed_paths={source_id: tmp_path},
        )
        assert result.skipped is True
        assert result.reason.startswith("eval-only gated corpus stub")


def test_synthetic_source_resolves_to_committed_golden_fixtures() -> None:
    fixtures = resolve_clinical_phi_source("synthetic_golden_deid")

    assert fixtures
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert any(
        ID_NUM in {span.label for span in fixture.gold_spans} for fixture in fixtures
    )


def test_manifest_hash_is_stable_and_recipe_points_to_manifest() -> None:
    manifest_hash = clinical_phi_manifest_hash()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash)
    assert clinical_phi_manifest_hash() == manifest_hash

    recipe = load_preset("C")
    assert recipe.dapt.corpus_ref == CLINICAL_PHI_MANIFEST_REF
