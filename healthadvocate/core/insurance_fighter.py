"""Insurance denial fighter — PII-safe NER + structured LLM appeal generation."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def fight_denial(engine: HealthEngine, denial_text: str, patient_info: str = "", profile_id: str | None = None) -> dict:
    if not denial_text or not denial_text.strip():
        return {"explanation": "No denial text provided.", "action_items": [], "red_flags": [], "validation": None}

    # Deidentify before sending to LLM
    safe_denial, denial_pii = engine.deidentify_for_llm(denial_text, method="mask")
    safe_patient = ""
    if patient_info and patient_info.strip():
        safe_patient, _ = engine.deidentify_for_llm(patient_info, method="mask")

    diseases = engine.extract_diseases(denial_text, confidence=0.5)
    drugs = engine.extract_drugs(denial_text, confidence=0.5)
    if patient_info and patient_info.strip():
        pi_diseases = engine.extract_diseases(patient_info, confidence=0.5)
        pi_drugs = engine.extract_drugs(patient_info, confidence=0.5)
        diseases.entities.extend(pi_diseases.entities)
        drugs.entities.extend(pi_drugs.entities)

    entity_desc = format_entities_with_confidence(list(diseases.entities) + list(drugs.entities))

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    patient_context = f"\n\nPatient context: {safe_patient.strip()[:400]}" if safe_patient else ""

    prompt = (
        f"An insurance company sent this denial letter:\n\n{safe_denial}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{patient_context}{family_block}\n\n"
        "As a patient health advocate, fight this denial. "
        "Explain what it means, why it may be wrong, and write a draft appeal letter."
    )

    system = (
        "You are a patient health advocate specializing in insurance appeals. "
        "You help patients fight unfair denials. Be knowledgeable about insurance law, "
        "appeal processes, and patients' rights. Include a draft appeal letter."
    )

    llm_output = chat_structured(prompt, module_type="appeal_strategy", system=system, max_tokens=2000)
    all_entities = list(diseases.entities) + list(drugs.entities)
    validation = cross_validate(all_entities, llm_output)

    return {
        "denial_text": denial_text[:200],
        "entities_found": {
            "conditions": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in diseases.entities],
            "medications": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in drugs.entities],
        },
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "denial_reason": llm_output.get("denial_reason", ""),
        "appeal_arguments": llm_output.get("appeal_arguments", []),
        "draft_appeal": llm_output.get("draft_appeal", ""),
        "structured_output": llm_output,
        "validation": {
            "confirmed": validation.confirmed,
            "ner_only": validation.ner_only,
            "llm_only": validation.llm_only,
            "reliability": validation.reliability,
            "urgency_disagreement": validation.urgency_disagreement,
        },
        "pii_scrubbed": len(denial_pii) > 0,
    }
