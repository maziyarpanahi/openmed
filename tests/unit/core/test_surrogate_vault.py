"""Tests for cross-document surrogate vaults."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from openmed.core.pii import deidentify, reidentify
from openmed.core.surrogate_vault import (
    HMAC_SCHEME,
    SCHEMA_VERSION,
    SurrogateVault,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _prediction_result(text: str, surfaces: Iterable[str]) -> PredictionResult:
    entities = []
    for surface in surfaces:
        start = text.index(surface)
        entities.append(
            EntityPrediction(
                text=surface,
                label="NAME",
                start=start,
                end=start + len(surface),
                confidence=0.99,
            )
        )
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="test-pii",
        timestamp="now",
    )


def test_shared_vault_reuses_surrogates_across_deidentify_calls(monkeypatch):
    """A shared vault stabilizes replacements across separate documents."""

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        if "Alice Zephyr" in text:
            return _prediction_result(text, ["Alice Zephyr"])
        return _prediction_result(text, ["Bruno Quill"])

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")

    first = deidentify(
        "Patient Alice Zephyr was admitted.",
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )
    second = deidentify(
        "Alice Zephyr returned for follow-up.",
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )
    third = deidentify(
        "Patient Bruno Quill was admitted.",
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    alice_surrogate = first.pii_entities[0].surrogate
    assert alice_surrogate == second.pii_entities[0].surrogate
    assert alice_surrogate != third.pii_entities[0].surrogate
    assert "Alice Zephyr" not in first.deidentified_text
    assert "Bruno Quill" not in third.deidentified_text


def test_json_vault_persists_only_hmac_hashes_and_surrogates(monkeypatch, tmp_path):
    """Persisted vault files do not include raw source identifiers."""

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        surfaces = [
            surface for surface in ("Alice Zephyr", "Bruno Quill") if surface in text
        ]
        return _prediction_result(text, surfaces)

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    path = tmp_path / "surrogate-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")

    deidentify(
        "Alice Zephyr and Bruno Quill enrolled.",
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    raw_json = path.read_text(encoding="utf-8")
    assert "Alice Zephyr" not in raw_json
    assert "Bruno Quill" not in raw_json

    payload = json.loads(raw_json)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["hmac_scheme"] == HMAC_SCHEME
    assert len(payload["entries"]) == 2
    assert [entry["canonical_label"] for entry in payload["entries"]] == [
        "PERSON",
        "PERSON",
    ]
    assert all(
        set(entry) == {"canonical_label", "lang", "text_hash", "surrogate"}
        for entry in payload["entries"]
    )
    assert all(
        entry["text_hash"].startswith(f"{HMAC_SCHEME}:") for entry in payload["entries"]
    )


def test_json_vault_load_rejects_malformed_json(tmp_path):
    path = tmp_path / "surrogate-vault.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt surrogate vault") as exc_info:
        SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")

    assert str(path) in str(exc_info.value)


def test_json_vault_reload_continues_stable_surrogates(monkeypatch, tmp_path):
    """Save/load preserves the mapping deterministically."""

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        return _prediction_result(text, ["Alice Zephyr"])

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    path = tmp_path / "surrogate-vault.json"
    first_vault = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")
    first = deidentify(
        "Alice Zephyr enrolled.",
        method="replace",
        surrogate_vault=first_vault,
        use_safety_sweep=False,
    )
    persisted = path.read_text(encoding="utf-8")

    reloaded = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")
    second = deidentify(
        "Alice Zephyr enrolled.",
        method="replace",
        surrogate_vault=reloaded,
        use_safety_sweep=False,
    )
    reloaded.save()

    assert second.pii_entities[0].surrogate == first.pii_entities[0].surrogate
    assert path.read_text(encoding="utf-8") == persisted


def test_vault_backed_replace_keeps_reidentify_roundtrip(monkeypatch):
    """The vault stabilizes surrogates without replacing reversible mappings."""

    text = "Alice Zephyr met Bruno Quill."

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        return _prediction_result(text, ["Alice Zephyr", "Bruno Quill"])

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")

    result = deidentify(
        text,
        method="replace",
        keep_mapping=True,
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    assert result.mapping is not None
    assert reidentify(result.deidentified_text, result.mapping) == text
    assert all(
        entry.key.text_hash.startswith(f"{HMAC_SCHEME}:") for entry in vault.entries()
    )
