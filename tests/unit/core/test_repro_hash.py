"""Tests for training reproducibility provenance hashes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from openmed.core.repro_hash import (
    SPAN_SET_HASH_SCHEMA_VERSION,
    ReproducibilityVerificationError,
    build_training_provenance,
    canonicalize_span_records,
    compute_environment_lock_digest,
    compute_reproducibility_hash,
    compute_span_set_hash,
    load_training_provenance,
    verify_reproducibility,
    write_training_provenance,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


def _provenance() -> dict[str, object]:
    return build_training_provenance(
        rng_seeds={"python": 13, "numpy": 21, "torch": 34},
        data_manifest_hash=HASH_A,
        recipe_config_hash=HASH_B,
        env_lock_digest=HASH_C,
        base_model="OpenMed/base-model",
        base_model_revision="7b4f2ca",
        git_sha="abc123",
        repo_id="OpenMed/test-model",
        checkpoint_id="checkpoint-001",
    )


def test_reproducibility_hash_folds_training_inputs() -> None:
    base = compute_reproducibility_hash(
        recipe={"config_hash": HASH_B},
        data_manifest={"hash": HASH_A},
        base_model={"id": "OpenMed/base-model", "revision": "7b4f2ca"},
        git_sha="abc123",
        rng_seeds={"python": 13, "numpy": 21},
        recipe_config_hash=HASH_B,
        env_lock_digest=HASH_C,
    )
    changed_seed = compute_reproducibility_hash(
        recipe={"config_hash": HASH_B},
        data_manifest={"hash": HASH_A},
        base_model={"id": "OpenMed/base-model", "revision": "7b4f2ca"},
        git_sha="abc123",
        rng_seeds={"python": 13, "numpy": 22},
        recipe_config_hash=HASH_B,
        env_lock_digest=HASH_C,
    )
    changed_lock = compute_reproducibility_hash(
        recipe={"config_hash": HASH_B},
        data_manifest={"hash": HASH_A},
        base_model={"id": "OpenMed/base-model", "revision": "7b4f2ca"},
        git_sha="abc123",
        rng_seeds={"python": 13, "numpy": 21},
        recipe_config_hash=HASH_B,
        env_lock_digest="sha256:" + "d" * 64,
    )

    assert base != changed_seed
    assert base != changed_lock
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", base)


def test_verify_reproducibility_rederives_recorded_hash_and_fails_drift() -> None:
    provenance = _provenance()

    assert verify_reproducibility(provenance) == provenance["reproducibility_hash"]

    drifted = dict(provenance)
    drifted["rng_seeds"] = {"python": 13, "numpy": 21, "torch": 35}
    with pytest.raises(
        ReproducibilityVerificationError,
        match="does not match pinned training inputs",
    ):
        verify_reproducibility(drifted)


@pytest.mark.parametrize(
    "missing", ["rng_seeds", "data_manifest_hash", "env_lock_digest"]
)
def test_verify_reproducibility_fails_closed_when_required_fields_are_missing(
    missing: str,
) -> None:
    provenance = _provenance()
    provenance.pop(missing)

    with pytest.raises(ReproducibilityVerificationError, match=missing):
        verify_reproducibility(provenance)


def test_write_training_provenance_persists_verified_json(tmp_path: Path) -> None:
    provenance = _provenance()

    path = write_training_provenance(tmp_path / "checkpoint", provenance)

    assert path.name == "training_provenance.json"
    assert load_training_provenance(path) == provenance
    assert json.loads(path.read_text(encoding="utf-8")) == provenance


def test_environment_lock_digest_matches_uv_lock_sha256(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "demo"\nversion = "1.0.0"\n')
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()

    assert compute_environment_lock_digest(lock) == f"sha256:{expected}"


def test_canonicalize_span_records_is_order_independent_and_phi_free() -> None:
    forward = [
        {"start": 30, "end": 43, "label": "EMAIL", "confidence": 0.8},
        {"start": 7, "end": 16, "label": "NAME", "confidence": 0.9},
    ]
    reverse = list(reversed(forward))

    assert canonicalize_span_records(forward) == canonicalize_span_records(reverse)
    # No raw text field leaks into the canonical record.
    for record in canonicalize_span_records(forward):
        assert set(record) == {
            "start",
            "end",
            "label",
            "confidence",
            "action",
            "replacement_digest",
        }


def test_compute_span_set_hash_is_stable_and_order_independent() -> None:
    spans = [
        {"start": 30, "end": 43, "label": "EMAIL", "confidence": 0.8},
        {"start": 7, "end": 16, "label": "NAME", "confidence": 0.9},
    ]
    digest = compute_span_set_hash(spans, text_length=64, method="mask")

    assert digest.startswith("sha256:")
    assert digest == compute_span_set_hash(
        list(reversed(spans)), text_length=64, method="mask"
    )


def test_compute_span_set_hash_reflects_applied_replacement() -> None:
    base = [{"start": 0, "end": 5, "label": "NAME", "surrogate": "Alpha"}]
    changed = [{"start": 0, "end": 5, "label": "NAME", "surrogate": "Bravo"}]

    assert compute_span_set_hash(base) != compute_span_set_hash(changed)


def test_compute_span_set_hash_binds_method_and_text_length() -> None:
    spans = [{"start": 0, "end": 5, "label": "NAME", "confidence": 0.5}]

    assert compute_span_set_hash(spans, method="mask") != compute_span_set_hash(
        spans, method="replace"
    )
    assert compute_span_set_hash(spans, text_length=10) != compute_span_set_hash(
        spans, text_length=11
    )


def test_span_set_hash_schema_version_is_pinned() -> None:
    assert SPAN_SET_HASH_SCHEMA_VERSION == "openmed.span_set.v1"
