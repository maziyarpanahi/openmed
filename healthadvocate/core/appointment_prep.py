"""Appointment preparation — PII-safe NER + structured LLM talking points."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def prepare_appointment(engine: HealthEngine, symptoms: str, concern: str = "", profile_id: str | None = None) -> dict:
    combined = f"{symptoms} {concern}".strip()
    if not combined:
        return {"explanation": "Please provide your symptoms or concerns.", "action_items": [], "red_flags": [], "validation": None}

    diseases = engine.extract_diseases(combined, confidence=0.5)
    drugs = engine.extract_drugs(combined, confidence=0.5)
    anatomy = engine.extract_anatomy(combined, confidence=0.5)

    # Deidentify before LLM
    safe_text, pii_map = engine.deidentify_for_llm(combined, method="mask")

    all_entities = list(diseases.entities) + list(drugs.entities) + list(anatomy.entities)
    entity_desc = format_entities_with_confidence(all_entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient is preparing for a doctor's appointment.\n"
        f"Symptoms/concern: {safe_text}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "Prepare them with key talking points, questions to ask, and an advocacy script."
    )

    system = (
        "You are a patient health advocate helping someone prepare for a medical appointment. "
        "Give them concrete talking points, questions, and a script. "
        "Empower them to advocate for themselves."
    )

    llm_output = chat_structured(prompt, module_type="appointment_prep", system=system)
    validation = cross_validate(all_entities, llm_output)

    return {
        "conditions_found": [{"text": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in diseases.entities],
        "medications_found": [{"text": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in drugs.entities],
        "anatomy_found": [{"text": e.text, "label": e.label, "confidence": round(e.confidence, 2)} for e in anatomy.entities],
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "talking_points": llm_output.get("talking_points", []),
        "questions_to_ask": llm_output.get("questions_to_ask", []),
        "advocacy_script": llm_output.get("advocacy_script", ""),
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
