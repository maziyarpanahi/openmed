#!/usr/bin/env python3
"""De-identify fabricated Hindi and code-mixed Hinglish clinical notes.

Every name and identifier in this example is synthetic test data. None of the
values identify a real person, clinician, organization, or account.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openmed import deidentify
from openmed.core.pii_i18n import validate_aadhaar

HINDI_MODEL_ID = "OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1"
DEFAULT_OUTPUT_DIR = Path("hindi_hinglish_redacted")

SYNTHETIC_HINDI_NAME = "अनन्या मेहता"
SYNTHETIC_HINGLISH_NAME = "Ananya Mehta"
SYNTHETIC_CLINICIAN_NAME = "डॉ. रोहन कपूर"
SYNTHETIC_AADHAAR = "2467 7832 5484"
SYNTHETIC_ABHA = "91-0000-0000-0000"
SYNTHETIC_PHONE = "+91 98765 43210"
SYNTHETIC_EMAIL = "ananya.mehta@example.test"
SYNTHETIC_MEDICAL_RECORD_NUMBER = "HI-DEMO-708"

# यह नोट और इसमें दिए गए सभी नाम व पहचान संख्याएँ काल्पनिक परीक्षण डेटा हैं।
# This note and every name and identifier in it are fabricated test data.
SYNTHETIC_HINDI_NOTE = (
    "【पूरी तरह काल्पनिक उदाहरण】रोगी अनन्या मेहता, मेडिकल रिकॉर्ड HI-DEMO-708, "
    "आधार 2467 7832 5484 और आभा 91-0000-0000-0000। फोन +91 98765 43210, "
    "ईमेल ananya.mehta@example.test। डॉ. रोहन कपूर ने हल्के बुखार के लिए आराम की सलाह दी।"
)

# यह Hinglish नोट और इसके सभी पहचानकर्ता भी काल्पनिक परीक्षण डेटा हैं।
# This Hinglish note and all of its identifiers are also fabricated test data.
SYNTHETIC_HINGLISH_NOTE = (
    "【Synthetic Hinglish example】Patient Ananya Mehta ka follow-up aaj hai. "
    "Medical record HI-DEMO-708, Aadhaar 2467 7832 5484 aur ABHA "
    "91-0000-0000-0000 file mein hain. Phone +91 98765 43210 aur email "
    "ananya.mehta@example.test hai. Patient ko do din se halka bukhar hai."
)

NOTES = {
    "hindi": SYNTHETIC_HINDI_NOTE,
    "hinglish": SYNTHETIC_HINGLISH_NOTE,
}

SYNTHETIC_IDENTIFIERS_BY_NOTE = {
    "hindi": (
        SYNTHETIC_HINDI_NAME,
        SYNTHETIC_CLINICIAN_NAME,
        SYNTHETIC_AADHAAR,
        SYNTHETIC_ABHA,
        SYNTHETIC_PHONE,
        SYNTHETIC_EMAIL,
        SYNTHETIC_MEDICAL_RECORD_NUMBER,
    ),
    "hinglish": (
        SYNTHETIC_HINGLISH_NAME,
        SYNTHETIC_AADHAAR,
        SYNTHETIC_ABHA,
        SYNTHETIC_PHONE,
        SYNTHETIC_EMAIL,
        SYNTHETIC_MEDICAL_RECORD_NUMBER,
    ),
}

# India policy ABDM recognizer Aadhaar और ABHA पहचानता है। Local terms केवल
# काल्पनिक names, MRN और spaced phone display form को deterministic बनाते हैं।
# The India policy enables the ABDM recognizer for Aadhaar and ABHA. Local terms
# make only the fabricated names, medical-record number, and spaced phone
# display form deterministic; the safety sweep independently covers the email.
INDIA_CUSTOM_RECOGNIZER = {
    "case_sensitive": False,
    "deny": {
        "terms": [
            {"term": SYNTHETIC_HINDI_NAME, "label": "PERSON"},
            {"term": SYNTHETIC_HINGLISH_NAME, "label": "PERSON"},
            {"term": SYNTHETIC_CLINICIAN_NAME, "label": "PERSON"},
            {"term": SYNTHETIC_PHONE, "label": "PHONE"},
            {"term": SYNTHETIC_MEDICAL_RECORD_NUMBER, "label": "ID_NUM"},
        ]
    },
}

Deidentifier = Callable[..., Any]


def assert_synthetic_identifiers_removed(
    note_name: str,
    deidentified_text: str,
) -> None:
    """Fail closed if any fabricated identifier survives in one output."""

    leaked = [
        identifier
        for identifier in SYNTHETIC_IDENTIFIERS_BY_NOTE[note_name]
        if identifier in deidentified_text
    ]
    if leaked:
        raise AssertionError(
            f"Synthetic identifiers survived in {note_name}: {leaked!r}"
        )


def structured_entities(result: Any) -> list[dict[str, Any]]:
    """Return JSON-serializable rows for the entities in a result."""

    return [
        {
            "label": entity.canonical_label or entity.entity_type or entity.label,
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
            "confidence": round(float(entity.confidence), 3),
            "replacement": entity.redacted_text,
            "sources": list(entity.sources),
        }
        for entity in result.pii_entities
    ]


def run_hindi_hinglish_deidentification(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    deidentifier: Deidentifier = deidentify,
    loader: Any = None,
) -> dict[str, Any]:
    """De-identify both notes, verify them, and save their redacted text.

    Args:
        output_dir: Directory for the two redacted UTF-8 text files.
        deidentifier: Injectable ``deidentify``-compatible callable for tests.
        loader: Optional cached or test model loader forwarded to OpenMed.

    Returns:
        Results keyed by ``"hindi"`` and ``"hinglish"``.
    """

    # पहले काल्पनिक आधार की जाँच करें। / Validate the fabricated Aadhaar first.
    if not validate_aadhaar(SYNTHETIC_AADHAAR):
        raise AssertionError("The fabricated Aadhaar-format value is invalid")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for note_name, note in NOTES.items():
        kwargs: dict[str, Any] = {
            "method": "replace",
            "model_name": HINDI_MODEL_ID,
            "lang": "hi",
            "locale": "en_IN",
            "policy": "india_dpdp_act",
            "use_safety_sweep": True,
            "custom_recognizer": INDIA_CUSTOM_RECOGNIZER,
            "consistent": True,
            "seed": 708,
        }
        if loader is not None:
            kwargs["loader"] = loader

        # पहचान हटाएँ और फ़ाइल लिखने से पहले रिसाव रोकें। / De-identify, then fail closed.
        result = deidentifier(note, **kwargs)
        assert_synthetic_identifiers_removed(note_name, result.deidentified_text)
        # UTF-8 में संशोधित पाठ सहेजें। / Save the redacted text as UTF-8.
        (destination / f"{note_name}_note_redacted.txt").write_text(
            result.deidentified_text + "\n",
            encoding="utf-8",
        )
        results[note_name] = result

    return results


def main() -> None:
    """Run both examples and print their structured output."""

    results = run_hindi_hinglish_deidentification()
    for note_name, result in results.items():
        print(f"=== {note_name.title()} de-identified text ===")
        print(result.deidentified_text)
        print("\n=== संरचित इकाइयाँ / Structured entities ===")
        print(json.dumps(structured_entities(result), ensure_ascii=False, indent=2))
        print()

    print(f"सहेजा गया / Saved under: {DEFAULT_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
