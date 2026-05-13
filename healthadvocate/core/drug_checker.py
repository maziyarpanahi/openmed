"""Drug checker — NER verification + structured LLM drug information."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def check_drug(engine: HealthEngine, drug_name: str, profile_id: str | None = None) -> dict:
    if not drug_name or not drug_name.strip():
        return {"drug": "", "explanation": "No drug name provided.", "action_items": [], "red_flags": [], "validation": None}

    ner_result = engine.extract_drugs(drug_name, confidence=0.3)
    ner_verified = any(e.text.lower() == drug_name.strip().lower() for e in ner_result.entities)

    entity_desc = format_entities_with_confidence(ner_result.entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    ner_note = "NER verified this is a recognized drug." if ner_verified else "NER could not verify this drug name — provide information cautiously."

    prompt = (
        f"A patient wants to know about the drug: {drug_name.strip()}\n\n"
        f"NER Analysis: {entity_desc}\n{ner_note}\n\n"
        f"{family_block}\n\n"
        "Provide practical information as a health advocate. "
        "If the patient's current medications are listed, check for drug-drug interactions."
    )

    system = (
        "You are a patient health advocate helping someone understand a medication. "
        "Be accurate, concise, and practical. Use plain language, not medical jargon."
    )

    llm_output = chat_structured(prompt, module_type="drug_info", system=system)
    validation = cross_validate(ner_result.entities, llm_output)

    return {
        "drug": drug_name.strip(),
        "ner_verified": ner_verified,
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "drug_class": llm_output.get("drug_class", ""),
        "generic_available": llm_output.get("generic_available"),
        "common_side_effects": llm_output.get("common_side_effects", []),
        "warnings": llm_output.get("warnings", []),
        "doctor_questions": llm_output.get("doctor_questions", []),
        "structured_output": llm_output,
        "validation": {
            "confirmed": validation.confirmed,
            "ner_only": validation.ner_only,
            "llm_only": validation.llm_only,
            "reliability": validation.reliability,
            "urgency_disagreement": validation.urgency_disagreement,
        },
    }
