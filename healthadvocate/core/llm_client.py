"""Medical LLM client — calls LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://172.26.0.1:1234/v1")
_MODEL_NAME = os.environ.get("MEDICAL_LLM_MODEL", "meditron3-8b")

SYSTEM_PROMPT = (
    "You are a patient health advocate. Help patients understand medical information, "
    "navigate the healthcare system, and fight for their rights. "
    "Be concise, practical, and empathetic. Use plain language. "
    "If symptoms sound urgent, say so clearly. "
    "Never diagnose — always recommend seeing a healthcare provider."
)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

_MODULE_SCHEMAS = {
    "symptom_assessment": """{
  "summary": "string - plain language assessment of symptoms",
  "urgency": "high or medium or low",
  "action_items": ["string - specific actions the patient should take"],
  "red_flags": ["string - warning signs requiring immediate attention"],
  "possible_conditions": [{"name": "string", "likelihood": "likely or possible or unlikely"}],
  "recommended_specialist": "string or null"
}""",
    "drug_info": """{
  "summary": "string - what this drug does in plain language",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string - dangerous interactions or warnings"],
  "drug_class": "string",
  "generic_available": "boolean or null",
  "common_side_effects": ["string"],
  "warnings": ["string"],
  "doctor_questions": ["string - questions to ask their doctor"]
}""",
    "bill_analysis": """{
  "summary": "string - what this bill is for in plain language",
  "urgency": "high or medium or low",
  "action_items": ["string - steps to dispute or understand charges"],
  "red_flags": ["string - suspicious or incorrect charges"],
  "suspicious_charges": ["string"],
  "billing_rights": ["string - patient rights regarding this bill"],
  "estimated_overcharge": "string or null"
}""",
    "appeal_strategy": """{
  "summary": "string - what the denial means in plain language",
  "urgency": "high or medium or low",
  "action_items": ["string - specific steps to fight the denial"],
  "red_flags": ["string - unfair practices or errors by the insurer"],
  "denial_reason": "string",
  "appeal_arguments": ["string - specific arguments for appeal"],
  "draft_appeal": "string - a ready-to-send appeal letter"
}""",
    "document_explanation": """{
  "summary": "string - what this document says in plain language",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string"],
  "medical_terms_explained": [{"term": "string", "explanation": "string"}],
  "follow_up_needed": "boolean"
}""",
    "appointment_prep": """{
  "summary": "string - how to prepare for this appointment",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string"],
  "talking_points": ["string - key points to raise with the doctor"],
  "questions_to_ask": ["string"],
  "advocacy_script": "string - what to say to advocate for yourself"
}""",
    "discharge_translation": """{
  "summary": "string - what happened and what to do in plain language",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string - warning signs that require immediate medical attention"],
  "medication_instructions": ["string - each medication with dose and schedule"],
  "warning_signs": ["string - when to go back to the ER"],
  "follow_up_steps": ["string"]
}""",
    "second_opinion_brief": """{
  "summary": "string - situation summary for the new doctor",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string - concerns about current treatment"],
  "key_questions": ["string - most important questions for the specialist"],
  "records_to_bring": ["string"],
  "treatment_concerns": ["string"]
}""",
    "bulletin_analysis": """{
  "summary": "string - what this health bulletin claims",
  "urgency": "high or medium or low",
  "action_items": ["string"],
  "red_flags": ["string"],
  "credibility": "high or medium or low",
  "scientific_context": "string - what the science actually says",
  "recommended_action": "string"
}""",
}


def chat(user_message: str, system: str = SYSTEM_PROMPT, max_tokens: int = 400, temperature: float = 0.6) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=_LM_STUDIO_URL, api_key="lm-studio")
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    if "</think" in text:
        text = text.split("</think")[-1].strip()
    return text.strip()


def chat_structured(
    user_message: str,
    module_type: str = "general",
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 1200,
    temperature: float = 0.1,
) -> dict:
    from openai import OpenAI

    schema = _MODULE_SCHEMAS.get(module_type, "")
    json_system = (
        "CRITICAL: You MUST respond with ONLY a single JSON object. No other text.\n"
        "Example of correct response format:\n"
        '{"summary": "Brief explanation", "urgency": "low", "action_items": ["item1"], "red_flags": []}\n\n'
        "Do NOT write labels like 'Summary:' or 'Urgency:' outside the JSON.\n"
        "Do NOT use markdown or code fences.\n"
        "The ENTIRE response must be parseable JSON.\n\n"
    )
    if schema:
        json_system += f"JSON schema:\n{schema}\n\n"
    json_system += (
        "If you cannot determine a value, use null. Never add fields not in the schema.\n\n"
        + system
    )

    client = OpenAI(base_url=_LM_STUDIO_URL, api_key="lm-studio")
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {"role": "system", "content": json_system},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    if "</think" in text:
        text = text.split("</think")[-1].strip()
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try sanitizing control characters inside JSON strings
    if text.startswith("{"):
        sanitized = _sanitize_json(text)
        try:
            result = json.loads(sanitized)
            if isinstance(result, dict):
                logger.debug("Parsed JSON after sanitizing control characters")
                return result
        except json.JSONDecodeError:
            pass

    # Try repairing truncated JSON (model hit max_tokens)
    if text.startswith("{"):
        repaired = _repair_truncated_json(text)
        if repaired:
            return repaired

    # Try parsing labeled text format (Summary: ...\nUrgency: ...\nAction Items:\n- ...)
    labeled = _parse_labeled_text(text)
    if labeled:
        return labeled

    logger.warning("chat_structured: failed to parse LLM response, returning fallback")
    return {
        "summary": text[:500],
        "urgency": "medium",
        "action_items": [],
        "red_flags": [],
        "_raw_text": True,
    }


def _sanitize_json(text: str) -> str:
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def _repair_truncated_json(text: str) -> dict | None:
    """Try to parse truncated JSON by closing open strings/braces."""
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    in_string = False
    for i, ch in enumerate(text):
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string

    repaired = text
    if in_string:
        repaired += '"'
    if open_brackets > 0:
        repaired += "]" * open_brackets
    if open_braces > 0:
        repaired += "}" * open_braces

    logger.debug("JSON repair attempt: braces=%d brackets=%d in_string=%s", open_braces, open_brackets, in_string)
    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Labeled-text parser — handles Meditron3-8B's preferred output format
# ---------------------------------------------------------------------------

_LABEL_SECTION_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(Summary|Urgency|Action\s+Items?|Red\s+Flags?|Warning\s+Signs?"
    r"|Medication\s+Instructions|Follow[\s-]?up\s+Steps?"
    r"|Key\s+Questions|Records\s+to\s+Bring|Treatment\s+Concerns"
    r"|Talking\s+Points|Questions?\s+to\s+Ask|Advocacy\s+Script"
    r"|Appeal\s+Arguments|Draft\s+Appeal|Denial\s+Reason"
    r"|Drug\s+Class|Generic\s+Available|Common\s+Side\s+Effects"
    r"|Warnings|Doctor\s+Questions|Suspicious\s+Charges"
    r"|Billing\s+Rights|Estimated\s+Overcharge|Medical\s+Terms\s+Explained"
    r"|Follow[\s-]?up\s+Needed|Possible\s+Conditions|Recommended\s+Specialist"
    r"|Credibility|Scientific\s+Context|Recommended\s+Action"
    r")\s*[:：]\s*",
    re.IGNORECASE,
)

_FIELD_MAP = {
    "summary": "summary", "urgency": "urgency",
    "action items": "action_items", "action item": "action_items",
    "red flags": "red_flags", "red flag": "red_flags",
    "warning signs": "warning_signs", "warning sign": "warning_signs",
    "medication instructions": "medication_instructions",
    "follow-up steps": "follow_up_steps", "follow up steps": "follow_up_steps",
    "followup steps": "follow_up_steps",
    "key questions": "key_questions",
    "records to bring": "records_to_bring",
    "treatment concerns": "treatment_concerns",
    "talking points": "talking_points",
    "questions to ask": "questions_to_ask",
    "question to ask": "questions_to_ask",
    "advocacy script": "advocacy_script",
    "appeal arguments": "appeal_arguments",
    "draft appeal": "draft_appeal",
    "denial reason": "denial_reason",
    "drug class": "drug_class",
    "generic available": "generic_available",
    "common side effects": "common_side_effects",
    "warnings": "warnings",
    "doctor questions": "doctor_questions",
    "suspicious charges": "suspicious_charges",
    "billing rights": "billing_rights",
    "estimated overcharge": "estimated_overcharge",
    "medical terms explained": "medical_terms_explained",
    "follow-up needed": "follow_up_needed", "follow up needed": "follow_up_needed",
    "followup needed": "follow_up_needed",
    "possible conditions": "possible_conditions",
    "recommended specialist": "recommended_specialist",
    "credibility": "credibility",
    "scientific context": "scientific_context",
    "recommended action": "recommended_action",
}

_LIST_FIELDS = {
    "action_items", "red_flags", "warning_signs", "medication_instructions",
    "follow_up_steps", "key_questions", "records_to_bring", "treatment_concerns",
    "talking_points", "questions_to_ask", "appeal_arguments", "common_side_effects",
    "warnings", "doctor_questions", "suspicious_charges", "billing_rights",
    "possible_conditions",
}


def _parse_labeled_text(text: str) -> dict | None:
    """Parse labeled-section format into a structured dict."""
    matches = list(_LABEL_SECTION_RE.finditer(text))
    if len(matches) < 2:
        return None

    result: dict = {}
    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        field_name = _FIELD_MAP.get(label)
        if not field_name:
            continue

        # Content runs from end of this match to start of next (or end of text)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if field_name in _LIST_FIELDS:
            # Parse bullet points or numbered items
            items = re.split(r"\n\s*(?:[-•*]|\d+[.\):])\s*", content)
            items = [item.strip() for item in items if item.strip()]
            result[field_name] = items
        elif field_name == "urgency":
            low = content.lower()
            if "high" in low:
                result[field_name] = "high"
            elif "low" in low:
                result[field_name] = "low"
            else:
                result[field_name] = "medium"
        elif field_name in ("generic_available", "follow_up_needed"):
            low = content.lower()
            result[field_name] = "yes" in low or "true" in low
        else:
            result[field_name] = content

    return result if result else None
