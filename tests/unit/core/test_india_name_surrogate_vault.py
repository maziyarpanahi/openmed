"""India multi-script name consistency and leakage regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core.pii import deidentify
from openmed.core.script_detect import canonical_indian_name, detect_script
from openmed.core.surrogate_vault import HMAC_SCHEME, SurrogateVault
from openmed.processing.outputs import EntityPrediction, PredictionResult

_FIXTURE_PATH = Path("tests/fixtures/pii/india_transliterated_names.json")
_SECRET = "india-name-unit-test-secret"


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _prediction(
    text: str,
    surface: str,
    *,
    metadata: dict[str, str] | None = None,
) -> PredictionResult:
    start = text.index(surface)
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=surface,
                label="NAME",
                start=start,
                end=start + len(surface),
                confidence=0.99,
                metadata=metadata,
            )
        ],
        model_name="synthetic-india-name-fixture",
        timestamp="now",
    )


def _fixture_documents() -> list[dict[str, str]]:
    documents = _fixture()["documents"]
    assert isinstance(documents, list)
    return [dict(document) for document in documents]


def test_multiscript_name_aliases_round_trip_through_one_vault_entry(tmp_path):
    fixture = _fixture()
    documents = _fixture_documents()
    expected_canonical = str(fixture["canonical_name"])
    path = tmp_path / "india-name-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret=_SECRET)

    keys = {
        vault.key_for(
            document["surface"],
            label="PERSON",
            lang=document["language"],
        )
        for document in documents
    }
    assert {canonical_indian_name(document["surface"]) for document in documents} == {
        expected_canonical
    }
    assert len(keys) == 1

    rendered = {}
    for document in documents:
        rendered[document["id"]] = vault.get_or_create(
            document["surface"],
            label="PERSON",
            lang=document["language"],
            create_surrogate=lambda attempt: "Aarav Mehta",
        )

    assert len(vault.entries()) == 1
    assert vault.entries()[0].surrogate == canonical_indian_name("Aarav Mehta")
    assert detect_script(rendered["india-name-devanagari"]) == "Devanagari"
    assert detect_script(rendered["india-name-tamil"]) == "Tamil"
    assert rendered["india-name-latin-standard"] == rendered["india-name-latin-variant"]
    assert detect_script(rendered["india-name-latin-standard"]) == "Latin"

    reloaded = SurrogateVault.from_file(path, hmac_secret=_SECRET)
    for document in documents:
        assert (
            reloaded.get(
                document["surface"],
                label="PERSON",
                lang=document["language"],
            )
            == rendered[document["id"]]
        )


def test_similar_romanized_people_do_not_collapse_to_one_identity():
    fixture = _fixture()
    first = "Sunil Sharma"
    second = str(fixture["negative_surface"])
    vault = SurrogateVault.in_memory(_SECRET)

    assert canonical_indian_name(first) != canonical_indian_name(second)
    assert vault.key_for(first, label="PERSON", lang="hi") != vault.key_for(
        second,
        label="PERSON",
        lang="hi",
    )
    first_surrogate = vault.get_or_create(
        first,
        label="PERSON",
        lang="hi",
        create_surrogate=lambda attempt: "Aarav Mehta",
    )
    second_surrogate = vault.get_or_create(
        second,
        label="PERSON",
        lang="hi",
        create_surrogate=lambda attempt: "Kabir Rao",
    )

    assert first_surrogate != second_surrogate
    assert len(vault.entries()) == 2


def test_canonical_name_is_absent_from_vault_payload_and_audit(
    monkeypatch,
    tmp_path,
):
    fixture = _fixture()
    documents = _fixture_documents()
    canonical_name = str(fixture["canonical_name"])
    surfaces = {document["surface"] for document in documents}

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        surface = next(surface for surface in surfaces if surface in text)
        return _prediction(
            text,
            surface,
            metadata={"canonical_transliteration_key": canonical_name},
        )

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    monkeypatch.setattr(
        "openmed.core.anonymizer.Anonymizer.surrogate",
        lambda self, original, label, **kwargs: "Aarav Mehta",
    )
    path = tmp_path / "india-name-vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret=_SECRET)

    audit_payloads = []
    for document in documents:
        report = deidentify(
            document["text"],
            method="replace",
            lang="hi",
            surrogate_vault=vault,
            use_safety_sweep=False,
            audit=True,
        )
        audit_payloads.append(report.to_json())

    raw_vault = path.read_text(encoding="utf-8")
    assert canonical_name not in raw_vault
    assert all(surface not in raw_vault for surface in surfaces)
    assert all(canonical_name not in payload for payload in audit_payloads)
    assert all(
        surface not in payload for surface in surfaces for payload in audit_payloads
    )
    persisted = json.loads(raw_vault)
    assert len(persisted["entries"]) == 1
    assert persisted["entries"][0]["text_hash"].startswith(f"{HMAC_SCHEME}:")


def test_multiscript_fixture_deidentification_has_no_name_leakage(monkeypatch):
    documents = _fixture_documents()
    surfaces = {document["surface"] for document in documents}

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        surface = next(surface for surface in surfaces if surface in text)
        return _prediction(text, surface)

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    monkeypatch.setattr(
        "openmed.core.anonymizer.Anonymizer.surrogate",
        lambda self, original, label, **kwargs: "Aarav Mehta",
    )
    vault = SurrogateVault.in_memory(_SECRET)

    for document in documents:
        result = deidentify(
            document["text"],
            method="replace",
            lang="hi",
            surrogate_vault=vault,
            use_safety_sweep=False,
        )
        assert all(surface not in result.deidentified_text for surface in surfaces)
        assert (
            detect_script(result.pii_entities[0].surrogate or "") == document["script"]
        )


def test_non_india_name_keys_keep_exact_source_and_language_behavior():
    vault = SurrogateVault.in_memory(_SECRET)
    source = "Alice Zephyr"

    key = vault.key_for(source, label="PERSON", lang="en")

    assert key.lang == "en"
    assert key.text_hash == vault.text_hash(source)
