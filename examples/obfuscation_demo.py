"""Demo: language-agnostic PHI/PII obfuscation.

Shows the four behaviors the new ``Anonymizer`` engine unlocks for
``deidentify(method="replace", ...)``:

1. **Random surrogates** — every run produces different fakes.
2. **Within-document consistency** — repeated mentions of the same name
   resolve to one fake identity (``consistent=True``).
3. **Cross-run reproducibility** — pin the surrogate stream for
   regression testing or stable test fixtures (``seed=42``).
4. **Locale-aware generation** — surrogates look right for the language
   (German names for ``lang="de"``, French phone formats for ``lang="fr"``,
   Brazilian CPFs for ``locale="pt_BR"``, ...).

Run:
    python examples/obfuscation_demo.py
"""

from __future__ import annotations

from unittest.mock import patch

from openmed import deidentify
from openmed.core.anonymizer import Anonymizer
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _mock_extraction(text: str, entities: list[EntityPrediction]):
    """Helper so the demo runs without downloading a real model."""
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="mock",
        timestamp="2026-04-27",
    )


def heading(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_anonymizer_direct() -> None:
    """Use the Anonymizer class directly without going through extract_pii."""
    heading("1. Anonymizer used directly")

    anon = Anonymizer(lang="en", consistent=True, seed=42)
    print(f"  PERSON: {anon.surrogate('John Doe', 'name')}")
    print(f"  EMAIL : {anon.surrogate('john@hospital.org', 'email')}")
    print(f"  PHONE : {anon.surrogate('(415) 555-1234', 'phone_number')}")
    print(f"  DATE  : {anon.surrogate('01/15/1970', 'date_of_birth')}")

    print("\n  Same input twice (consistent=True) -> same output:")
    print(f"    {anon.surrogate('Maria Silva', 'name')}")
    print(f"    {anon.surrogate('Maria Silva', 'name')}")


def demo_random_vs_consistent() -> None:
    heading("2. Random vs consistent surrogates")

    text = "Dr. Smith met John Doe. John Doe later called Dr. Smith."
    entities = [
        EntityPrediction(text="Dr. Smith", label="name", start=0, end=9, confidence=0.9),
        EntityPrediction(text="John Doe", label="name", start=14, end=22, confidence=0.95),
        EntityPrediction(text="John Doe", label="name", start=24, end=32, confidence=0.95),
        EntityPrediction(text="Dr. Smith", label="name", start=46, end=55, confidence=0.9),
    ]

    with patch("openmed.core.pii.extract_pii",
               return_value=_mock_extraction(text, entities)):
        rand = deidentify(text, method="replace", lang="en")
        cons = deidentify(text, method="replace", lang="en", consistent=True, seed=99)

    print("  Random  :", rand.deidentified_text)
    print("  Consist.:", cons.deidentified_text)
    print("\n  Note how consistent=True makes both 'John Doe' mentions and both")
    print("  'Dr. Smith' mentions resolve to the same fake names.")


def demo_locales() -> None:
    heading("3. Locale-aware surrogates across languages")

    samples = [
        ("en", "Patient Jane Smith born 03/15/1985"),
        ("fr", "Patient Marie Dupont née le 15/03/1985"),
        ("de", "Patient Anna Müller geboren am 15.03.1985"),
        ("it", "Paziente Marco Rossi nato il 15/03/1985"),
        ("es", "Paciente Carlos García nacido el 15/03/1985"),
        ("pt", "Paciente Pedro Almeida nascido em 15/03/1985"),
        ("nl", "Patiënt Daan Jansen geboren op 15-03-1985"),
    ]

    for lang, text in samples:
        # Mock detection of a name + a date for each locale demo
        words = text.split()
        # Find the name token (positions 1-2) and the date (last token)
        name_start = text.index(words[1])
        name_end = name_start + len(" ".join(words[1:3]))
        date_token = next(t for t in words if any(c.isdigit() for c in t))
        date_start = text.rindex(date_token)
        date_end = date_start + len(date_token)

        entities = [
            EntityPrediction(text=text[name_start:name_end], label="name",
                             start=name_start, end=name_end, confidence=0.95),
            EntityPrediction(text=text[date_start:date_end], label="date_of_birth",
                             start=date_start, end=date_end, confidence=0.95),
        ]
        with patch("openmed.core.pii.extract_pii",
                   return_value=_mock_extraction(text, entities)):
            r = deidentify(text, method="replace", lang=lang, consistent=True, seed=7)
        print(f"  [{lang}] {r.deidentified_text}")


def demo_brazilian_portuguese_cpf() -> None:
    heading("4. pt_BR locale override (CPF generation)")

    text = "Paciente João Silva, CPF: 123.456.789-09"
    entities = [
        EntityPrediction(text="João Silva", label="name", start=9, end=19, confidence=0.95),
        EntityPrediction(text="123.456.789-09", label="ID_NUM", start=27, end=41, confidence=0.95),
    ]
    with patch("openmed.core.pii.extract_pii",
               return_value=_mock_extraction(text, entities)):
        r = deidentify(text, method="replace", lang="pt", locale="pt_BR",
                       consistent=True, seed=42)

    print(f"  Original: {text}")
    print(f"  Surrogate: {r.deidentified_text}")
    surrogate_cpf = next(e.redacted_text for e in r.pii_entities if e.entity_type == "ID_NUM")
    from openmed.core.pii_i18n import validate_portuguese_cpf
    valid = validate_portuguese_cpf(surrogate_cpf)
    print(f"  Surrogate CPF {surrogate_cpf!r} passes checksum: {valid}")


def demo_format_preservation() -> None:
    heading("5. Format-preserving phone numbers")

    text = "Call +1 (415) 555-1234 or +33 6 12 34 56 78"
    entities = [
        EntityPrediction(text="+1 (415) 555-1234", label="phone_number", start=5, end=22, confidence=0.95),
        EntityPrediction(text="+33 6 12 34 56 78", label="phone_number", start=26, end=43, confidence=0.95),
    ]
    with patch("openmed.core.pii.extract_pii",
               return_value=_mock_extraction(text, entities)):
        r = deidentify(text, method="replace", lang="en", consistent=True, seed=42)

    print(f"  Original  : {text}")
    print(f"  Surrogate : {r.deidentified_text}")
    print("  -> digit groups, separators, and country-code position preserved")


def main() -> None:
    print("OpenMed obfuscation demo")
    print("Faker-backed, locale-aware, optionally deterministic surrogates.")
    demo_anonymizer_direct()
    demo_random_vs_consistent()
    demo_locales()
    demo_brazilian_portuguese_cpf()
    demo_format_preservation()
    print()


if __name__ == "__main__":
    main()
