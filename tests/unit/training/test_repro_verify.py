"""Unit tests for the reproducibility-hash recompute-and-verify harness (OM-118)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from openmed.cli.main import main
from openmed.core.repro_hash import compute_reproducibility_hash
from openmed.training.repro_verify import (
    ReproVerificationResult,
    verify_reproducibility_inputs,
)

SAMPLE_RECIPE = {"learning_rate": 0.0001, "batch_size": 32, "mode": "A"}
SAMPLE_DATA_MANIFEST = {"corpus": "dapt_sample.jsonl", "rows": 1000}
SAMPLE_BASE_MODEL = "OpenMed/base-medical-encoder"
SAMPLE_GIT_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"


@pytest.fixture
def reference_provenance() -> dict[str, Any]:
    return {
        "recipe": SAMPLE_RECIPE,
        "data_manifest": SAMPLE_DATA_MANIFEST,
        "base_model": SAMPLE_BASE_MODEL,
        "git_sha": SAMPLE_GIT_SHA,
    }


@pytest.fixture
def claimed_hash(reference_provenance: dict[str, Any]) -> str:
    return compute_reproducibility_hash(
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
    )


def test_match_when_inputs_identical(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    result = verify_reproducibility_inputs(
        claimed_hash=claimed_hash,
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        reference_provenance=reference_provenance,
    )
    assert result.status == "MATCH"
    assert result.matched is True
    assert result.recomputed_hash == claimed_hash
    assert result.claimed_hash == claimed_hash
    assert result.diverging_inputs == ()


@pytest.mark.parametrize(
    "key",
    [
        "recipe",
        "data_manifest",
        "base_model",
        "git_sha",
        "rng_seeds",
        "recipe_config_hash",
        "env_lock_digest",
    ],
)
def test_single_key_divergence(
    reference_provenance: dict[str, Any], claimed_hash: str, key: str
) -> None:
    full_reference = dict(reference_provenance)
    full_reference.update(
        {
            "rng_seeds": {"python": 13, "numpy": 21},
            "recipe_config_hash": "sha256:" + "b" * 64,
            "env_lock_digest": "sha256:" + "c" * 64,
        }
    )
    full_claimed = compute_reproducibility_hash(
        recipe=full_reference["recipe"],
        data_manifest=full_reference["data_manifest"],
        base_model=full_reference["base_model"],
        git_sha=full_reference["git_sha"],
        rng_seeds=full_reference["rng_seeds"],
        recipe_config_hash=full_reference["recipe_config_hash"],
        env_lock_digest=full_reference["env_lock_digest"],
    )

    candidate = dict(full_reference)
    if key == "recipe":
        candidate["recipe"] = {"learning_rate": 0.0005, "batch_size": 64}
    elif key == "data_manifest":
        candidate["data_manifest"] = {"corpus": "other_corpus.jsonl"}
    elif key == "base_model":
        candidate["base_model"] = "FacebookAI/roberta-base"
    elif key == "git_sha":
        candidate["git_sha"] = "ffffffffffffffffffffffffffffffffffffffff"
    elif key == "rng_seeds":
        candidate["rng_seeds"] = {"python": 99, "numpy": 21}
    elif key == "recipe_config_hash":
        candidate["recipe_config_hash"] = "sha256:" + "9" * 64
    elif key == "env_lock_digest":
        candidate["env_lock_digest"] = "sha256:" + "8" * 64

    result = verify_reproducibility_inputs(
        claimed_hash=full_claimed,
        recipe=candidate["recipe"],
        data_manifest=candidate["data_manifest"],
        base_model=candidate["base_model"],
        git_sha=candidate["git_sha"],
        rng_seeds=candidate["rng_seeds"],
        recipe_config_hash=candidate["recipe_config_hash"],
        env_lock_digest=candidate["env_lock_digest"],
        reference_provenance=full_reference,
    )
    assert result.status == "MISMATCH"
    assert result.matched is False
    assert result.diverging_inputs == (key,)
    assert result.recomputed_hash != full_claimed


def test_multi_factor_mismatch(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    candidate = dict(reference_provenance)
    candidate["base_model"] = "FacebookAI/roberta-large"
    candidate["git_sha"] = "0000000000000000000000000000000000000000"

    result = verify_reproducibility_inputs(
        claimed_hash=claimed_hash,
        recipe=candidate["recipe"],
        data_manifest=candidate["data_manifest"],
        base_model=candidate["base_model"],
        git_sha=candidate["git_sha"],
        reference_provenance=reference_provenance,
    )
    assert result.status == "MISMATCH"
    assert result.matched is False
    assert set(result.diverging_inputs) == {"base_model", "git_sha"}


def test_unverifiable_when_claimed_hash_missing(
    reference_provenance: dict[str, Any],
) -> None:
    result = verify_reproducibility_inputs(
        claimed_hash=None,
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
    )
    assert result.status == "UNVERIFIABLE"
    assert result.matched is False
    assert result.recomputed_hash is None


def test_missing_optional_field_tolerated(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    # Reference and candidate both have no rng_seeds/env_lock_digest -> match
    result = verify_reproducibility_inputs(
        claimed_hash=claimed_hash,
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        reference_provenance=reference_provenance,
        rng_seeds=None,
        env_lock_digest=None,
    )
    assert result.status == "MATCH"
    assert result.matched is True
    assert result.diverging_inputs == ()


def test_claimed_hash_from_manifest_row(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    manifest_row = {
        "repo_id": "OpenMed/test-model",
        "reproducibility_hash": claimed_hash,
        "base_model": reference_provenance["base_model"],
        "recipe": reference_provenance["recipe"],
        "data_manifest": reference_provenance["data_manifest"],
        "git_sha": reference_provenance["git_sha"],
    }
    result = verify_reproducibility_inputs(
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        manifest_row=manifest_row,
    )
    assert result.status == "MATCH"
    assert result.claimed_hash == claimed_hash


def test_claimed_hash_from_model_card_markdown(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    card_text = f"""
# Model Card

| Field | Value |
|---|---|
| Repository | `OpenMed/test-model` |
| Reproducibility hash | `{claimed_hash}` |
"""
    result = verify_reproducibility_inputs(
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        model_card_text=card_text,
    )
    assert result.status == "MATCH"
    assert result.claimed_hash == claimed_hash


def test_cli_repro_verify_match(
    tmp_path: Path, reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    manifest_file = tmp_path / "models.jsonl"
    manifest_row = {
        "repo_id": "OpenMed/test-model",
        "reproducibility_hash": claimed_hash,
        "base_model": reference_provenance["base_model"],
        "recipe": reference_provenance["recipe"],
        "data_manifest": reference_provenance["data_manifest"],
        "git_sha": reference_provenance["git_sha"],
    }
    manifest_file.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "repro",
                "verify",
                "--repo",
                "OpenMed/test-model",
                "--manifest",
                str(manifest_file),
            ]
        )
    assert code == 0
    assert stdout.getvalue().strip() == "MATCH"


def test_cli_repro_verify_mismatch_names_diverged_input(
    tmp_path: Path, reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    manifest_file = tmp_path / "models.jsonl"
    manifest_row = {
        "repo_id": "OpenMed/test-model",
        "reproducibility_hash": claimed_hash,
        "base_model": reference_provenance["base_model"],
        "recipe": reference_provenance["recipe"],
        "data_manifest": reference_provenance["data_manifest"],
        "git_sha": reference_provenance["git_sha"],
    }
    manifest_file.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "repro",
                "verify",
                "--repo",
                "OpenMed/test-model",
                "--manifest",
                str(manifest_file),
                "--git-sha",
                "ffffffffffffffffffffffffffffffffffffffff",
            ]
        )
    assert code == 1
    assert "MISMATCH" in stdout.getvalue()
    assert "git_sha" in stdout.getvalue()


def test_cli_repro_verify_json_output(
    tmp_path: Path, reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    manifest_file = tmp_path / "models.jsonl"
    manifest_row = {
        "repo_id": "OpenMed/test-model",
        "reproducibility_hash": claimed_hash,
        "base_model": reference_provenance["base_model"],
        "recipe": reference_provenance["recipe"],
        "data_manifest": reference_provenance["data_manifest"],
        "git_sha": reference_provenance["git_sha"],
    }
    manifest_file.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "repro",
                "verify",
                "--repo",
                "OpenMed/test-model",
                "--manifest",
                str(manifest_file),
                "--json",
            ]
        )
    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "MATCH"
    assert payload["matched"] is True
    assert payload["claimed_hash"] == claimed_hash
    assert payload["recomputed_hash"] == claimed_hash


def test_cli_repro_verify_unverifiable_for_unknown_repo(tmp_path: Path) -> None:
    manifest_file = tmp_path / "models.jsonl"
    manifest_file.write_text("", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "repro",
                "verify",
                "--repo",
                "OpenMed/non-existent-model",
                "--manifest",
                str(manifest_file),
            ]
        )
    assert code == 1
    assert "UNVERIFIABLE" in stdout.getvalue()


def test_dictionary_key_ordering_invariance(
    reference_provenance: dict[str, Any], claimed_hash: str
) -> None:
    # Construct dictionary with reordered keys
    reordered_recipe = {
        "mode": "A",
        "batch_size": 32,
        "learning_rate": 0.0001,
    }
    reordered_manifest = {
        "rows": 1000,
        "corpus": "dapt_sample.jsonl",
    }
    result = verify_reproducibility_inputs(
        claimed_hash=claimed_hash,
        recipe=reordered_recipe,
        data_manifest=reordered_manifest,
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        reference_provenance=reference_provenance,
    )
    assert result.status == "MATCH"
    assert result.matched is True
    assert result.diverging_inputs == ()


def test_malformed_hash_and_invalid_inputs_fail_closed(
    reference_provenance: dict[str, Any],
) -> None:
    # Non-hex claimed hash -> UNVERIFIABLE
    res_malformed = verify_reproducibility_inputs(
        claimed_hash="not_a_valid_sha256_hash",
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
    )
    assert res_malformed.status == "UNVERIFIABLE"
    assert res_malformed.matched is False

    # Invalid seed types causing computation failure -> MISMATCH
    res_invalid = verify_reproducibility_inputs(
        claimed_hash="sha256:" + "a" * 64,
        recipe=reference_provenance["recipe"],
        data_manifest=reference_provenance["data_manifest"],
        base_model=reference_provenance["base_model"],
        git_sha=reference_provenance["git_sha"],
        rng_seeds={"python": True},  # Boolean is rejected as invalid seed
    )
    assert res_invalid.status == "MISMATCH"
    assert res_invalid.matched is False
    assert "inputs_invalid" in res_invalid.diverging_inputs


def test_cli_corrupted_file_handling(tmp_path: Path) -> None:
    corrupt_card = tmp_path / "corrupt_card.md"
    corrupt_card.write_bytes(b"\x80\x81\x82\xff\xfe")  # Invalid UTF-8 bytes

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "repro",
                "verify",
                "--repo",
                "OpenMed/test-model",
                "--card",
                str(corrupt_card),
            ]
        )
    assert code == 1
    assert "UNVERIFIABLE" in stdout.getvalue()
