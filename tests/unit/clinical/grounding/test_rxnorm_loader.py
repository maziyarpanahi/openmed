"""Focused tests for the local RxNorm vocabulary loader."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.grounding import (
    RXNORM_SYSTEM_URI,
    RxNormLoader,
    VocabularyLoaderRegistry,
    validate_vocabulary_loader,
)


def _conso_row(
    rxcui: str,
    tty: str,
    term: str,
    *,
    preferred: bool = True,
) -> str:
    fields = [""] * 18
    fields[0] = rxcui
    fields[1] = "ENG"
    fields[6] = "Y" if preferred else "N"
    fields[11] = "RXNORM"
    fields[12] = tty
    fields[13] = rxcui
    fields[14] = term
    return "|".join(fields) + "|"


def _relation_row(source: str, target: str, relation: str) -> str:
    fields = [""] * 15
    fields[0] = source
    fields[3] = "RO"
    fields[4] = target
    fields[7] = relation
    return "|".join(fields) + "|"


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "rxnorm"
    release.mkdir()
    (release / "RXNCONSO.RRF").write_text(
        "\n".join(
            [
                _conso_row("100", "IN", "acetaminophen"),
                _conso_row("200", "SBD", "Tylenol 500 MG Oral Tablet"),
                _conso_row("300", "BN", "Tylenol"),
                _conso_row("400", "DF", "Oral Tablet"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (release / "RXNREL.RRF").write_text(
        "\n".join(
            [
                _relation_row("200", "100", "has_ingredient"),
                _relation_row("300", "200", "has_tradename"),
                _relation_row("200", "400", "has_dose_form"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return release


def test_brand_and_product_resolve_to_ingredient_with_tty_and_dose_form(tmp_path):
    loader = RxNormLoader(_release(tmp_path))

    match = loader.resolve_one("Tylenol")
    product = loader.resolve_one("Tylenol 500 MG Oral Tablet", tty="SBD")

    assert match is not None
    assert match.code == "100"
    assert match.metadata["matched_rxcui"] == "300"
    assert match.metadata["tty"] == "BN"
    assert match.metadata["normalized_ingredient"] == "acetaminophen"
    assert product is not None
    assert product.code == "100"
    assert product.metadata["tty"] == "SBD"
    assert product.metadata["dose_form"] == "Oral Tablet"


def test_tty_filter_uses_requested_type_and_deterministic_fallback(tmp_path):
    loader = RxNormLoader(_release(tmp_path))

    requested = loader.resolve_one("Tylenol 500 MG Oral Tablet", tty="SBD")
    fallback = loader.resolve_one("acetaminophen", tty="SBD")
    repeated = loader.resolve_one("acetaminophen", tty="SBD")

    assert requested is not None
    assert requested.metadata["tty"] == "SBD"
    assert fallback is not None
    assert fallback.metadata["tty"] == "IN"
    assert fallback == repeated
    assert loader.available_ttys == ("SBD", "BN", "IN", "DF")


def test_loader_conforms_to_registry_and_returns_matcher_terms(tmp_path):
    loader = RxNormLoader(_release(tmp_path))

    assert validate_vocabulary_loader(loader) is loader
    registry = VocabularyLoaderRegistry()
    registry.register(loader)
    matcher = registry.matcher(RXNORM_SYSTEM_URI)

    matches = matcher.lookup("Tylenol")
    assert matches
    assert matches[0].code == "100"
    assert matches[0].metadata["tty"] == "BN"


def test_jsonl_projection_supports_inline_ingredient_relationship(tmp_path):
    projection = tmp_path / "rxnorm.jsonl"
    projection.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "rxcui": "100",
                        "tty": "IN",
                        "term": "acetaminophen",
                    }
                ),
                json.dumps(
                    {
                        "rxcui": "200",
                        "tty": "SBD",
                        "term": "Synthetic Brand",
                        "relationships": [
                            {
                                "source": "200",
                                "target": "100",
                                "relation": "has_ingredient",
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    match = RxNormLoader(projection).resolve_one("Synthetic Brand")

    assert match is not None
    assert match.code == "100"
    assert match.metadata["tty"] == "SBD"
