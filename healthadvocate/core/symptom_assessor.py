"""Symptom assessment — NER extraction + structured LLM analysis."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def assess_symptoms(engine: HealthEngine, symptoms: str, profile_id: str | None = None) -> dict:
    if not symptoms or not symptoms.strip():
        return {
            "conditions": [], "urgency": "low",
            "explanation": "No symptoms provided.", "action_items": [],
            "red_flags": [], "validation": None, "structured_output": None,
        }

    result = engine.extract_diseases(symptoms)
    conditions = [{"name": e.text, "confidence": round(e.confidence, 2), "label": e.label} for e in result.entities]

    # Deidentify before sending to LLM
    safe_symptoms, pii_map = engine.deidentify_for_llm(symptoms, method="mask")

    entity_desc = format_entities_with_confidence(result.entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient reports these symptoms: {safe_symptoms}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "As a health advocate, provide a structured assessment. "
        "Assess urgency realistically — serious symptoms should be high. "
        "Mild and common symptoms should be low."
    )

    system = (
        "You are a patient health advocate helping assess symptoms. "
        "Be concise, practical, empathetic. "
        "Always recommend seeing a healthcare provider for anything beyond minor issues."
    )

    llm_output = chat_structured(prompt, module_type="symptom_assessment", system=system)
    validation = cross_validate(result.entities, llm_output)

    urgency = llm_output.get("urgency", "medium")
    if validation.urgency_disagreement:
        urgency = "high"

    return {
        "conditions": conditions,
        "urgency": urgency,
        "explanation": llm_output.get("summary", ""),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "possible_conditions": llm_output.get("possible_conditions", []),
        "recommended_specialist": llm_output.get("recommended_specialist"),
        "structured_output": llm_output,
        "validation": {
            "confirmed": validation.confirmed,
            "ner_only": validation.ner_only,
            "llm_only": validation.llm_only,
            "reliability": validation.reliability,
            "urgency_disagreement": validation.urgency_disagreement,
        },
        "model_used": result.model_used,
        "processing_time": result.processing_time,
        "pii_scrubbed": len(pii_map) > 0,
    }
