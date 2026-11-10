"""Text-only synthetic ASR qualification inputs; no recorded voices or patients.

The optional device profile renders these with owner-installed macOS voices.
Voice assets and generated audio are never distributed with OpenMed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSpeechCase:
    """One fixed synthetic utterance and its local rendering language."""

    reference: str
    language: str
    voice: str
    contains_identifier: bool = False


SPEECH_CASES = (
    SyntheticSpeechCase(
        "The blue folder is on the table near the window.", "en", "Samantha"
    ),
    SyntheticSpeechCase(
        "Please close the door and turn off the light.", "en", "Samantha"
    ),
    SyntheticSpeechCase(
        "Le dossier bleu est sur la table près de la fenêtre.", "fr", "Thomas"
    ),
    SyntheticSpeechCase(
        "Veuillez fermer la porte et éteindre la lumière.", "fr", "Thomas"
    ),
    SyntheticSpeechCase(
        "La carpeta azul está sobre la mesa cerca de la ventana.", "es", "Mónica"
    ),
    SyntheticSpeechCase("Por favor cierre la puerta y apague la luz.", "es", "Mónica"),
    SyntheticSpeechCase(
        "My name is Alice Carter and my reference number is eight four two seven.",
        "en",
        "Samantha",
        True,
    ),
    SyntheticSpeechCase(
        "Je m'appelle Alice Martin et mon numéro de référence est huit quatre deux sept.",
        "fr",
        "Thomas",
        True,
    ),
    SyntheticSpeechCase(
        "Me llamo Alicia García y mi número de referencia es ocho cuatro dos siete.",
        "es",
        "Mónica",
        True,
    ),
)
