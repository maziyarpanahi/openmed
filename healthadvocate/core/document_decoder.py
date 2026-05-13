"""Document decoder — PII-safe NER + structured LLM explanation."""

from __future__ import annotations

from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker


def decode_document(engine: HealthEngine, text: str, lang: str = "en", profile_id: str | None = None) -> dict:
    if not text or not text.strip():
        return {"entities": [], "explanation": "No document text provided.", "action_items": [], "red_flags": [], "validation": None}

    diseases = engine.extract_diseases(text)
    drugs = engine.extract_drugs(text)
    anatomy = engine.extract_anatomy(text)
    pii = engine.extract_pii(text, lang=lang)

    # Deidentify before sending to LLM
    safe_text, pii_map = engine.deidentify_for_llm(text, method="mask")

    entities = []
    for e in diseases.entities:
        entities.append({"text": e.text, "category": "disease", "confidence": round(e.confidence, 2)})
    for e in drugs.entities:
        entities.append({"text": e.text, "category": "drug", "confidence": round(e.confidence, 2)})
    for e in anatomy.entities:
        entities.append({"text": e.text, "category": "anatomy", "confidence": round(e.confidence, 2)})

    all_ner_entities = list(diseases.entities) + list(drugs.entities) + list(anatomy.entities)
    entity_desc = format_entities_with_confidence(all_ner_entities)

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient shared this medical document:\n\n{safe_text[:2000]}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "Explain this document to the patient in plain language. "
        "Explain medical terms, highlight concerns, and suggest follow-up questions."
    )

    system = (
        "You are a patient health advocate explaining medical documents. "
        "Translate medical jargon into plain language. Help the patient understand "
        "what the document means for their health and what actions to take."
    )

    llm_output = chat_structured(prompt, module_type="document_explanation", system=system)
    validation = cross_validate(all_ner_entities, llm_output)

    return {
        "entities": entities,
        "pii_found": [{"text": e.text, "category": "pii"} for e in pii.entities],
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "medical_terms_explained": llm_output.get("medical_terms_explained", []),
        "follow_up_needed": llm_output.get("follow_up_needed", False),
        "structured_output": llm_output,
        "validation": {
            "confirmed": validation.confirmed,
            "ner_only": validation.ner_only,
            "llm_only": validation.llm_only,
            "reliability": validation.reliability,
            "urgency_disagreement": validation.urgency_disagreement,
        },
        "entity_counts": {
            "diseases": len(diseases.entities),
            "drugs": len(drugs.entities),
            "anatomy": len(anatomy.entities),
            "pii": len(pii.entities),
        },
        "pii_scrubbed": len(pii_map) > 0,
    }
