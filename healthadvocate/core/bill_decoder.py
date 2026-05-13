"""Bill decoder — price extraction + PII-safe structured LLM analysis."""

from __future__ import annotations

import re
from .engine import HealthEngine, format_entities_with_confidence
from .llm_client import chat_structured
from .cross_validation import cross_validate
from . import family_tracker

_PRICE_PATTERN = re.compile(r'\$\s?([\d,]+(?:\.\d{2})?)')


def decode_bill(engine: HealthEngine, bill_text: str, profile_id: str | None = None) -> dict:
    if not bill_text or not bill_text.strip():
        return {"line_items": [], "explanation": "No bill text provided.", "action_items": [], "red_flags": [], "validation": None}

    safe_text, pii_map = engine.deidentify_for_llm(bill_text, method="mask")

    prices = [(float(m.group(1).replace(',', '')), m.start()) for m in _PRICE_PATTERN.finditer(safe_text)]
    diseases = engine.extract_diseases(bill_text, confidence=0.5)
    drugs = engine.extract_drugs(bill_text, confidence=0.5)

    entity_desc = format_entities_with_confidence(list(diseases.entities) + list(drugs.entities))

    family_block = ""
    if profile_id:
        profile = family_tracker.get_profile(profile_id)
        family_block = "\n\n" + family_tracker.format_family_context(profile)

    prompt = (
        f"A patient received this medical bill:\n\n{safe_text[:2000]}\n\n"
        f"Charges found: {len(prices)} line items totaling ${sum(a for a, _ in prices):,.2f}\n\n"
        f"NER Analysis:\n{entity_desc}\n\n"
        f"{family_block}\n\n"
        "Analyze this bill. Flag suspicious charges, explain what each charge is for, "
        "and tell them their rights."
    )

    system = (
        "You are a patient health advocate helping someone understand a medical bill. "
        "Medical billing is often confusing and sometimes contains errors. "
        "Be thorough in identifying potential overcharges or incorrect items."
    )

    llm_output = chat_structured(prompt, module_type="bill_analysis", system=system)
    all_entities = list(diseases.entities) + list(drugs.entities)
    validation = cross_validate(all_entities, llm_output)

    line_items = []
    for i, (amt, pos) in enumerate(prices):
        start = max(0, pos - 60)
        end = min(len(safe_text), pos + 30)
        context = safe_text[start:end].replace('\n', ' ').strip()
        line_items.append({"amount": amt, "description": context})

    total = round(sum(amt for amt, _ in prices), 2) if prices else None

    return {
        "line_items": line_items,
        "total": f"${total:,.2f}" if total else None,
        "entities_found": {
            "diseases": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in diseases.entities],
            "drugs": [{"text": e.text, "confidence": round(e.confidence, 2)} for e in drugs.entities],
        },
        "explanation": llm_output.get("summary", ""),
        "urgency": llm_output.get("urgency", "medium"),
        "action_items": llm_output.get("action_items", []),
        "red_flags": llm_output.get("red_flags", []),
        "suspicious_charges": llm_output.get("suspicious_charges", []),
        "billing_rights": llm_output.get("billing_rights", []),
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
