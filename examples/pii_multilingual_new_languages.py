#!/usr/bin/env python3
"""Smoke example for recently added multilingual PII support."""

from __future__ import annotations

from collections import defaultdict
import os

from openmed import extract_pii, find_semantic_units, get_default_pii_model, get_pii_models_by_language
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
        preview = ", ".join(f"{value!r} ({score:.2f})" for value, score in grouped[entity_type][:3])
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
            print("Live extraction skipped (set OPENMED_RUN_LIVE_EXAMPLES=1 to run HF models).")


if __name__ == "__main__":
    main()
