from openmed.core.model_card import render_model_card
from openmed.eval.datasets.licenses import (
    PERMISSIVE_ENCODER_LICENSES,
    encoder_license_for,
)
from openmed.ner.families.indic import INDIC_ENCODER_SPECS


def _row():
    return {
        "repo_id": "Example/indic-pii-adapter",
        "family": "PII",
        "task": "token-classification",
        "languages": ["hi", "te"],
        "canonical_labels": ["PERSON", "PHONE"],
        "license": "Apache-2.0",
        "formats": ["safetensors"],
        "encoder_provenance": {
            "family": "MuRIL",
            "source": "google/muril-base-cased",
            "license": "Apache-2.0",
            "provenance": "user-supplied",
            "weights": "user-supplied; not bundled",
            "supports_transliterated_text": True,
        },
    }


def test_permissive_encoder_license_registry_matches_loader_metadata():
    assert set(PERMISSIVE_ENCODER_LICENSES) == {"muril", "indicbert"}
    assert encoder_license_for("MuRIL").license_id == "Apache-2.0"
    assert encoder_license_for("indic-bert").license_id == "MIT"
    for family, metadata in INDIC_ENCODER_SPECS.items():
        registered = encoder_license_for(family)
        assert registered.license_id == metadata.license_id
        assert registered.source_url == metadata.source_url
        assert registered.redistribution == "user-supplied-reference-only"


def test_model_card_surfaces_encoder_license_and_provenance():
    card = render_model_card(_row())

    assert "## Encoder Provenance" in card
    assert "| Encoder family | MuRIL |" in card
    assert "| Source | `google/muril-base-cased` |" in card
    assert "| License | Apache-2.0 |" in card
    assert "| Provenance | user-supplied |" in card
    assert "| Weights | user-supplied; not bundled |" in card
    assert "| Transliterated text | Yes |" in card
