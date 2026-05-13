"""Second opinion brief — PII-safe NER + structured LLM specialist preparation."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def create_brief(engine: HealthEngine, records: str, lang: str = "en", profile_id: str | None = None) -> dict:
    if not records or not records.strip():
        return {"explanation": "No medical records provided.", "action_items": [], "red_flags": [], "validation": None}

    diseases = engine.extract_diseases(records, confidence=0.5)
    drugs = engine.extract_drugs(records, confidence=0.5)
    anatomy = engine.extract_anatomy(records, confidence=0.5)

    # BUG FIX: deidentify BEFORE sending to LLM (was computed but not used)
    deidentified_text, pii_map = engine.deidentify_for_llm(records, method="mask")

    all_entities = list(diseases.entities) + list(drugs.entities) + list(anatomy.entities)
    entity_desc = format_entities_with_confidence(all_entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient is seeking a second opinion. Here are their de-identified records:\n\n"
        f"{deidentified_text[:2000]}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "Help them prepare for the second opinion with a clear brief, key questions, "
        "and what specific information to bring."
    )

    system = (
        "You are a patient health advocate helping prepare for a second medical opinion. "
        "Help the patient organize their medical history, formulate key questions, "
        "and make the most of their specialist visit."
    )

    llm_output = chat_structured(prompt, module_type="second_opinion_brief", system=system)
    validation = cross_validate(all_entities, llm_output)

    return {
        "deidentified_records": deidentified_text,
        "conditions": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in diseases.entities],
        "medications": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in drugs.entities],
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "key_questions": llm_output.get("key_questions", []),
        "records_to_bring": llm_output.get("records_to_bring", []),
        "treatment_concerns": llm_output.get("treatment_concerns", []),
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
