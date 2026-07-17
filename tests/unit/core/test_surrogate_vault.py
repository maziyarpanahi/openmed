"""Tests for cross-document surrogate vaults."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from openmed.core import surrogate_vault as vault_module
from openmed.core.pii import deidentify, reidentify
from openmed.core.schemas.span import hmac_text_hash
from openmed.core.surrogate_vault import (
    ENCRYPTION_SCHEME,
    HMAC_SCHEME,
    SCHEMA_VERSION,
    SurrogateSource,
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


def test_shared_vault_reuses_one_surrogate_across_indic_scripts(monkeypatch):
    """ISO linkage keeps cross-script mentions consistent within a document."""

    text = "राम met ராம"

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        return _prediction_result(text, ["राम", "ராம"])

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")

    result = deidentify(
        text,
        method="replace",
        lang="hi",
        keep_mapping=True,
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    assert len(result.pii_entities) == 2
    assert result.pii_entities[0].surrogate == result.pii_entities[1].surrogate
    assert len(vault.entries()) == 1
    assert result.mapping is not None
    assert reidentify(result.deidentified_text, result.mapping) == text


def test_vault_reads_pre_transliteration_person_keys():
    """Existing raw-surface entries remain readable after key normalization."""

    source = SurrogateSource("Alice Zephyr", "NAME")
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")
    legacy_key = vault._legacy_key_for_epoch(source, vault._epoch_manager.current_key)
    vault.store.set(legacy_key, "Casey Example", key_id=vault.current_key_id)

    assert legacy_key != vault.key_for(source.source_text, label=source.label)
    assert vault.get(source.source_text, label=source.label) == "Casey Example"


def test_json_vault_persists_only_hmac_hashes_and_encrypted_surrogates(
    monkeypatch,
    tmp_path,
):
    """Persisted vault files do not include raw sources or plaintext surrogates."""

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        surfaces = [
            surface for surface in ("Alice Zephyr", "Bruno Quill") if surface in text
        ]
        return _prediction_result(text, surfaces)

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    path = tmp_path / "surrogate-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")

    result = deidentify(
        "Alice Zephyr and Bruno Quill enrolled.",
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    raw_json = path.read_text(encoding="utf-8")
    assert "Alice Zephyr" not in raw_json
    assert "Bruno Quill" not in raw_json
    assert all(
        entity.surrogate not in raw_json
        for entity in result.pii_entities
        if entity.surrogate is not None
    )

    payload = json.loads(raw_json)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["hmac_scheme"] == HMAC_SCHEME
    assert payload["encryption_scheme"] == ENCRYPTION_SCHEME
    assert payload["current_epoch"]["key_id"] == vault.current_key_id
    assert len(payload["entries"]) == 2
    assert [entry["canonical_label"] for entry in payload["entries"]] == [
        "PERSON",
        "PERSON",
    ]
    assert all(
        set(entry)
        == {
            "canonical_label",
            "lang",
            "text_hash",
            "key_id",
            "surrogate_ciphertext",
            "surrogate_nonce",
            "surrogate_tag",
        }
        for entry in payload["entries"]
    )
    assert all(
        entry["text_hash"].startswith(f"{HMAC_SCHEME}:") for entry in payload["entries"]
    )
    assert all(entry["key_id"] == vault.current_key_id for entry in payload["entries"])

    with pytest.raises(ValueError, match="key_id|authentication"):
        SurrogateVault.from_file(path, hmac_secret="wrong-secret")


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
    assert all(entry.key_id == vault.current_key_id for entry in vault.entries())


def test_rotation_preserves_cross_document_surrogates():
    sources = (
        SurrogateSource("Alice Zephyr", "NAME"),
        SurrogateSource("Bruno Quill", "NAME"),
    )
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")
    expected = {
        "Alice Zephyr": "Casey Example",
        "Bruno Quill": "Riley Sample",
    }
    for source in sources:
        vault.get_or_create(
            source.source_text,
            label=source.label,
            lang=source.lang,
            create_surrogate=lambda attempt, source=source: expected[
                source.source_text
            ],
        )

    before = {
        source.source_text: vault.get(
            source.source_text,
            label=source.label,
            lang=source.lang,
        )
        for source in sources
    }
    old_hashes = {entry.key.text_hash for entry in vault.entries()}

    result = vault.rotate(sources, target_sequence=2)

    assert result.migrated_entries == 2
    assert result.consistency is not None
    assert result.consistency.passed
    assert vault.current_epoch_sequence == 2
    for source in sources:
        assert (
            vault.get(source.source_text, label=source.label, lang=source.lang)
            == before[source.source_text]
        )
    assert {entry.key.text_hash for entry in vault.entries()}.isdisjoint(old_hashes)
    assert all(entry.key_id == vault.current_key_id for entry in vault.entries())


def test_current_epoch_key_cannot_link_prior_epoch_hashes():
    source = SurrogateSource("Alice Zephyr", "NAME")
    vault = SurrogateVault.in_memory("unit-test-hmac-secret")
    vault.get_or_create(
        source.source_text,
        label=source.label,
        create_surrogate=lambda attempt: "Casey Example",
    )
    prior_key = vault._epoch_manager.current_key
    prior_payload = vault.to_payload()
    prior_text_hash = vault.entries()[0].key.text_hash

    vault.rotate((source,), target_sequence=2)

    current_key = vault._epoch_manager.current_key
    assert current_key.key_id != prior_key.key_id
    assert current_key.linkage_key != prior_key.linkage_key
    assert (
        hmac_text_hash(source.source_text, current_key.linkage_key) != prior_text_hash
    )
    with pytest.raises(ValueError, match="key_id|authentication"):
        vault_module._decrypt_entry_payload(prior_payload["entries"][0], current_key)


def test_revoke_current_epoch_reencrypts_forward_and_retires_old_key(tmp_path):
    source = SurrogateSource("Alice Zephyr", "NAME")
    path = tmp_path / "surrogate-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")
    vault.get_or_create(
        source.source_text,
        label=source.label,
        create_surrogate=lambda attempt: "Casey Example",
    )
    retired_key = vault._epoch_manager.current_key
    old_payload = json.loads(path.read_text(encoding="utf-8"))
    assert (
        vault_module._decrypt_entry_payload(
            old_payload["entries"][0], retired_key
        ).surrogate
        == "Casey Example"
    )

    result = vault.revoke_current_epoch((source,), target_sequence=2)

    assert retired_key.key_id in result.revoked_key_ids
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert retired_key.key_id in payload["revoked_key_ids"]
    assert retired_key.key_id not in {entry["key_id"] for entry in payload["entries"]}
    with pytest.raises(ValueError, match="key_id|authentication"):
        vault_module._decrypt_entry_payload(payload["entries"][0], retired_key)
    assert (
        vault.get(source.source_text, label=source.label, lang=source.lang)
        == "Casey Example"
    )


def test_rotation_is_atomic_and_idempotent_after_interrupted_save(
    monkeypatch,
    tmp_path,
):
    sources = (
        SurrogateSource("Alice Zephyr", "NAME"),
        SurrogateSource("Bruno Quill", "NAME"),
    )
    path = tmp_path / "surrogate-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")
    for source in sources:
        vault.get_or_create(
            source.source_text,
            label=source.label,
            create_surrogate=lambda attempt, source=source: f"{source.label}-{attempt}",
        )
    original_payload = path.read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(vault_module.os, "replace", fail_replace)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            vault.rotate(sources, target_sequence=2)

    assert path.read_text(encoding="utf-8") == original_payload
    assert vault.current_epoch_sequence == 1
    assert len(vault.entries()) == 2

    reloaded = SurrogateVault.from_file(path, hmac_secret="unit-test-hmac-secret")
    result = reloaded.rotate(sources, target_sequence=2)
    assert result.consistency is not None
    assert result.consistency.passed
    assert len(reloaded.entries()) == 2
    resumed_payload = path.read_text(encoding="utf-8")

    idempotent = reloaded.rotate(sources, target_sequence=2)
    assert idempotent.migrated_entries == 0
    assert idempotent.consistency is not None
    assert idempotent.consistency.passed
    assert path.read_text(encoding="utf-8") == resumed_payload
    assert len(reloaded.entries()) == 2
