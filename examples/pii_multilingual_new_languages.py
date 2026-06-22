#!/usr/bin/env python3
"""Smoke example for recently added multilingual PII support."""

from __future__ import annotations

import os
from collections import defaultdict

from openmed import (
    extract_pii,
    find_semantic_units,
    get_default_pii_model,
    get_pii_models_by_language,
)
from openmed.core.pii_i18n import get_patterns_for_language

RUN_LIVE_EXTRACTION = os.getenv("OPENMED_RUN_LIVE_EXAMPLES") == "1"


SAMPLES = {
    "nl": {
        "label": "Dutch",
        "text": (
            "Pati\u00ebnt: Eva de Vries, geboortedatum: 15 januari 1984, "
            "BSN: 123456782, telefoon: +31 6 12345678, adres: Keizersgracht 123, 1012 AB Amsterdam"
        ),
    },
    "hi": {
        "label": "Hindi",
        "text": (
            "\u0930\u094b\u0917\u0940: \u0905\u0928\u0940\u0924\u093e \u0936\u0930\u094d\u092e\u093e, "
            "\u091c\u0928\u094d\u092e\u0924\u093f\u0925\u093f: 15 \u091c\u0928\u0935\u0930\u0940 1984, "
            "\u092b\u094b\u0928: +91 9876543210, \u092a\u0924\u093e: 12 \u0917\u0932\u0940 \u0938\u0902\u0916\u094d\u092f\u093e 5, "
            "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940 110001"
        ),
    },
    "te": {
        "label": "Telugu",
        "text": (
            "\u0c30\u0c4b\u0c17\u0c3f: \u0c38\u0c3f\u0c24\u0c3e \u0c30\u0c46\u0c21\u0c4d\u0c21\u0c3f, "
            "\u0c1c\u0c28\u0c4d\u0c2e \u0c24\u0c47\u0c26\u0c40: 15 \u0c1c\u0c28\u0c35\u0c30\u0c3f 1984, "
            "\u0c2b\u0c4b\u0c28\u0c4d: +91 9876543210, \u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e: 12 \u0c35\u0c40\u0c27\u0c3f 5, "
            "\u0c39\u0c48\u0c26\u0c30\u0c3e\u0c2c\u0c3e\u0c26\u0c4d 500001"
        ),
    },
    "pt": {
        "label": "Portuguese",
        "text": (
            "Paciente: Pedro Almeida, data de nascimento: 15 de mar\u00e7o de 1985, "
            "CPF: 123.456.789-09, email: pedro@hospital.pt, telefone: +351 912 345 678, "
            "endere\u00e7o: Rua das Flores 25, 1200-195 Lisboa"
        ),
    },
    "ar": {
        "label": "Arabic",
        "text": (
            "\u0627\u0644\u0645\u0631\u064a\u0636\u0629 \u0644\u064a\u0644\u0649 \u062d\u0633\u0646\u060c "
            "\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f 15/03/1985\u060c "
            "\u0627\u0644\u0647\u0627\u062a\u0641 +20 10 1234 5678\u060c "
            "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a 29801011234567"
        ),
    },
    "ja": {
        "label": "Japanese",
        "text": (
            "\u60a3\u8005 \u4f50\u85e4 \u82b1\u5b50\u3001"
            "\u751f\u5e74\u6708\u65e5 1985\u5e743\u670815\u65e5\u3001"
            "\u96fb\u8a71 +81 90 1234 5678\u3001"
            "\u30de\u30a4\u30ca\u30f3\u30d0\u30fc 1234 5678 9012"
        ),
    },
    "tr": {
        "label": "Turkish",
        "text": (
            "Hasta Ay\u015fe Y\u0131lmaz, do\u011fum tarihi 15.03.1985, "
            "telefon +90 532 123 45 67, TCKN 10000000146, adres Atat\u00fcrk Caddesi 12"
        ),
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_patterns(lang: str, text: str) -> None:
    patterns = get_patterns_for_language(lang)
    matches = find_semantic_units(text, patterns)
    grouped = defaultdict(list)
    for match in matches:
        start, end, entity_type, score = match[:4]
        grouped[entity_type].append((text[start:end], score))

    print(f"Regex/semantic-unit matches for {lang}:")
    for entity_type in sorted(grouped):
        preview = ", ".join(
            f"{value!r} ({score:.2f})" for value, score in grouped[entity_type][:3]
        )
        print(f"  - {entity_type}: {preview}")


def run_live_extraction(lang: str, text: str) -> None:
    model_id = get_default_pii_model(lang)
    print(f"Model: {model_id}")
    try:
        result = extract_pii(
            text,
            lang=lang,
            model_name=model_id,
            confidence_threshold=0.4,
            use_smart_merging=True,
        )
    except Exception as exc:
        print(f"Live extraction unavailable: {exc}")
        return

    if not result.entities:
        print("Live extraction returned 0 entities.")
        return

    print("Live extraction entities:")
    for entity in result.entities:
        print(f"  - {entity.label:<18} {entity.text!r} ({entity.confidence:.3f})")


def main() -> None:
    print_header("New Multilingual PII Smoke Demo")
    for lang, payload in SAMPLES.items():
        label = payload["label"]
        text = payload["text"]
        models = get_pii_models_by_language(lang)
        print_header(f"{label} ({lang})")
        print(f"Registered models: {len(models)}")
        print(f"Default model:    {get_default_pii_model(lang)}")
        print(f"Sample text:      {text}")
        summarize_patterns(lang, text)
        if RUN_LIVE_EXTRACTION:
            run_live_extraction(lang, text)
        else:
            print(
                "Live extraction skipped (set OPENMED_RUN_LIVE_EXAMPLES=1 to run HF models)."
            )


if __name__ == "__main__":
    main()
