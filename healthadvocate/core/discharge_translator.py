"""Discharge translator — PII-safe NER + structured LLM plain language conversion."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def translate_discharge(engine: HealthEngine, text: str, lang: str = "en", profile_id: str | None = None) -> dict:
    if not text or not text.strip():
        return {"explanation": "No discharge text provided.", "action_items": [], "red_flags": [], "validation": None}

    diseases = engine.extract_diseases(text, confidence=0.5)
    drugs = engine.extract_drugs(text, confidence=0.5)
    anatomy = engine.extract_anatomy(text, confidence=0.5)

    # Deidentify before LLM
    safe_text, pii_map = engine.deidentify_for_llm(text, method="mask")

    all_entities = list(diseases.entities) + list(drugs.entities) + list(anatomy.entities)
    entity_desc = format_entities_with_confidence(all_entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient received these discharge instructions:\n\n{safe_text[:2000]}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "Translate these discharge instructions into plain language. "
        "Be very clear about warning signs that require immediate medical attention."
    )

    system = (
        "You are a patient health advocate translating discharge instructions into plain language. "
        "Medical jargon must become everyday English. Be very clear about warning signs "
        "that require immediate medical attention."
    )

    llm_output = chat_structured(prompt, module_type="discharge_translation", system=system)
    validation = cross_validate(all_entities, llm_output)

    return {
        "medications_detected": [{"name": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in drugs.entities],
        "conditions_detected": [{"name": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in diseases.entities],
        "anatomy_detected": [{"name": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in anatomy.entities],
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "medication_instructions": llm_output.get("medication_instructions", []),
        "warning_signs": llm_output.get("warning_signs", []),
        "follow_up_steps": llm_output.get("follow_up_steps", []),
        "structured_output": llm_output,
        "validation": {
            "confirmed": validation.confirmed,
            "ner_only": validation.ner_only,
            "llm_only": validation.llm_only,
            "reliability": validation.reliability,
            "urgency_disagreement": validation.urgency_disagreement,
        },
        "pii_scrubbed": len(pii_map) > 0,
    }
