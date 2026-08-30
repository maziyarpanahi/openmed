"""FHIR R4 Structured Data Capture (SDC) form serialization.

The serializers implement the supported, offline subset needed by clinical
document intake: Questionnaire items, QuestionnaireResponse answers, local
confidence/provenance extensions, and structural validation.  They never
perform terminology-server calls or fetch an implementation guide.  Values are
privacy-transformed before they enter a FHIR resource.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from openmed.structured.forms import (
    FormExtractionResult,
    _sanitize_value,
)

SDC_VERSION = "4.0.0"
SDC_QUESTIONNAIRE_PROFILE = (
    "http://hl7.org/fhir/uv/sdc/StructureDefinition/sdc-questionnaire"
)
SDC_QUESTIONNAIRE_RESPONSE_PROFILE = (
    "http://hl7.org/fhir/uv/sdc/StructureDefinition/sdc-questionnaireresponse"
)
OPENMED_CONFIDENCE_EXTENSION = (
    "https://openmed.dev/fhir/StructureDefinition/extraction-confidence"
)
OPENMED_PROVENANCE_EXTENSION = (
    "https://openmed.dev/fhir/StructureDefinition/source-provenance"
)
OPENMED_REVIEW_EXTENSION = (
    "https://openmed.dev/fhir/StructureDefinition/review-required"
)
OPENMED_WARNING_EXTENSION = (
    "https://openmed.dev/fhir/StructureDefinition/review-warning"
)
OPENMED_SDC_VERSION_EXTENSION = (
    "https://openmed.dev/fhir/StructureDefinition/sdc-version"
)
_ALLOWED_TYPES = frozenset(
    {
        "boolean",
        "choice",
        "date",
        "dateTime",
        "decimal",
        "display",
        "group",
        "integer",
        "open-choice",
        "quantity",
        "string",
        "text",
        "time",
        "url",
    }
)


class SDCValidationError(ValueError):
    """Raised when a generated resource is outside the supported SDC subset."""


@dataclass(frozen=True)
class SDCValidationReport:
    """Structural SDC validation results."""

    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether no supported-subset violations were found."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe validation report."""
        return {"valid": self.valid, "issues": list(self.issues)}

    def __bool__(self) -> bool:
        return self.valid


def _value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _field_items(fields: Any) -> tuple[Any, ...]:
    if isinstance(fields, FormExtractionResult):
        return fields.fields
    if isinstance(fields, Mapping):
        nested = _value(fields, "fields", "items", default=None)
        if nested is not None:
            if isinstance(nested, Mapping):
                return tuple(nested.values())
            return tuple(nested)
        if fields.get("resourceType") == "Questionnaire":
            return tuple(fields.get("item", ()))
        return tuple(
            {"linkId": key, "text": key, "value": value}
            for key, value in fields.items()
        )
    if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
        return tuple(fields)
    if fields is None:
        return ()
    return (fields,)


def _question_specs(fields: Any) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    by_link_id: dict[str, dict[str, Any]] = {}
    for item in _field_items(fields):
        link_id = str(_value(item, "linkId", "link_id", "id", default="field"))
        text = str(_value(item, "text", "label", "key", default=link_id))
        data_type = str(
            _value(item, "type", "data_type", "value_type", default="string")
        )
        if data_type not in _ALLOWED_TYPES:
            data_type = "string"
        spec: dict[str, Any] = {
            "linkId": link_id,
            "text": text,
            "type": data_type,
        }
        for source, target in (
            ("required", "required"),
            ("repeats", "repeats"),
            ("readOnly", "readOnly"),
        ):
            value = _value(item, source, target, default=None)
            if value is not None:
                spec[target] = bool(value)
        options = _value(
            item, "answerOption", "answer_options", "options", default=None
        )
        if options is not None:
            spec["answerOption"] = [_answer_option(option) for option in options]
        previous = by_link_id.get(link_id)
        if previous is None:
            by_link_id[link_id] = spec
            specs.append(spec)
        elif bool(spec.get("repeats")):
            previous["repeats"] = True
    return tuple(specs)


def _answer_option(option: Any) -> dict[str, Any]:
    if isinstance(option, Mapping):
        if "value[x]" in option:
            return dict(option)
        if any(key.startswith("value") for key in option):
            return dict(option)
        value = option.get("value", option.get("code", option.get("display", "")))
    else:
        value = option
    if isinstance(value, bool):
        return {"valueBoolean": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"valueInteger": value}
    if isinstance(value, float):
        return {"valueDecimal": value}
    return {"valueString": str(value)}


def _safe_field_value(
    item: Any,
    *,
    pii_detector: Callable[..., Any] | None,
    transformer: Callable[..., Any] | None,
) -> tuple[Any, str]:
    raw = _value(item, "redacted_value", default=None)
    link_id = str(_value(item, "linkId", "link_id", "id", default="field"))
    label = str(_value(item, "text", "label", "key", default=link_id))
    if raw is None:
        raw = _value(item, "value", "answer", "valueString", default=None)
        if isinstance(raw, Mapping):
            raw = _value(raw, "value", "valueString", "display", default="")
        raw_text = "" if raw is None else str(raw)
        raw = _sanitize_value(
            raw_text,
            key=link_id or label,
            field=item,
            pii_detector=pii_detector,
            transformer=transformer,
        )
    return raw, label


def _safe_provenance(item: Any, value: Any) -> dict[str, Any]:
    provenance = _value(item, "provenance", default=None)
    if provenance is not None:
        if isinstance(provenance, Mapping):
            result = dict(provenance)
        elif hasattr(provenance, "to_dict"):
            result = dict(provenance.to_dict())
        else:
            result = {}
    else:
        result = {}
    page = _value(item, "page", default=result.get("page"))
    bbox = _value(item, "bbox", default=result.get("bbox"))
    start = _value(item, "start", default=result.get("start"))
    end = _value(item, "end", default=result.get("end"))
    if page is not None:
        result["page"] = int(page)
    if bbox is not None:
        try:
            result["bbox"] = tuple(float(value_) for value_ in bbox)
        except (TypeError, ValueError):
            result.pop("bbox", None)
    if start is not None and end is not None:
        result["start"] = int(start)
        result["end"] = int(end)
    result.pop("text", None)
    result.pop("value", None)
    result["value_sha256"] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return result


def _safe_numeric(value: Any, *, integer: bool) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if integer:
        if not number.is_integer():
            return None
        return int(number)
    return number


def _answer_value(data_type: str, value: Any) -> dict[str, Any]:
    if data_type == "boolean":
        if isinstance(value, bool):
            return {"valueBoolean": value}
        lowered = str(value).strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return {"valueBoolean": True}
        if lowered in {"false", "no", "n", "0"}:
            return {"valueBoolean": False}
    if data_type == "integer":
        number = _safe_numeric(value, integer=True)
        if number is not None:
            return {"valueInteger": number}
    if data_type == "decimal":
        number = _safe_numeric(value, integer=False)
        if number is not None:
            return {"valueDecimal": number}
    if data_type == "date":
        return {"valueDate": str(value)}
    if data_type == "dateTime":
        return {"valueDateTime": str(value)}
    if data_type == "time":
        return {"valueTime": str(value)}
    if data_type == "url":
        return {"valueUrl": str(value)}
    return {"valueString": str(value)}


def _meta(profile: str) -> dict[str, Any]:
    return {
        "profile": [profile],
        "extension": [
            {"url": OPENMED_SDC_VERSION_EXTENSION, "valueString": SDC_VERSION}
        ],
    }


def to_questionnaire(
    fields: Any,
    *,
    questionnaire_id: str = "openmed-clinical-form",
    title: str = "Clinical form",
    url: str | None = None,
    version: str = SDC_VERSION,
    status: str = "active",
    subject_type: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Serialize field definitions into a FHIR R4 SDC Questionnaire."""
    if not questionnaire_id or not isinstance(questionnaire_id, str):
        raise ValueError("questionnaire_id must be a non-empty string")
    if status not in {"draft", "active", "retired", "unknown"}:
        raise ValueError("status must be a FHIR Questionnaire status")
    specs = _question_specs(fields)
    resource: dict[str, Any] = {
        "resourceType": "Questionnaire",
        "id": questionnaire_id,
        "meta": _meta(SDC_QUESTIONNAIRE_PROFILE),
        "url": url or f"https://openmed.dev/fhir/Questionnaire/{questionnaire_id}",
        "version": version,
        "status": status,
        "title": title,
        "item": [
            {key: value for key, value in spec.items() if value is not None}
            for spec in specs
        ],
    }
    if subject_type:
        resource["subjectType"] = list(subject_type)
    issues = validate_questionnaire(resource)
    if issues:
        raise SDCValidationError("generated Questionnaire failed supported validation")
    return resource


def _response_id(items: Sequence[Any]) -> str:
    stable = []
    for item in items:
        stable.append(
            "|".join(
                str(_value(item, name, default=""))
                for name in ("linkId", "link_id", "start", "end", "page")
            )
        )
    digest = hashlib.sha256("\n".join(stable).encode("utf-8")).hexdigest()[:24]
    return f"openmed-response-{digest}"


def _questionnaire_reference(questionnaire: Any) -> str | None:
    if questionnaire is None:
        return None
    if isinstance(questionnaire, str):
        return questionnaire
    if isinstance(questionnaire, Mapping):
        if questionnaire.get("resourceType") != "Questionnaire":
            return None
        if questionnaire.get("id"):
            return f"Questionnaire/{questionnaire['id']}"
        if questionnaire.get("url"):
            return str(questionnaire["url"])
    return None


def to_questionnaire_response(
    fields: Any,
    *,
    questionnaire: Any = None,
    response_id: str | None = None,
    status: str | None = None,
    subject: Mapping[str, Any] | None = None,
    authored: str | None = None,
    pii_detector: Callable[..., Any] | None = None,
    transformer: Callable[..., Any] | None = None,
    privacy_transform: Callable[..., Any] | None = None,
    transform: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Serialize extracted fields into a privacy-safe FHIR QuestionnaireResponse.

    Review-required fields remain present with a SDC-compatible extension and
    force ``in-progress`` status unless the caller explicitly supplies a
    different valid status.  No original value is copied into the resource.
    """
    configured_transforms = [
        candidate
        for candidate in (transformer, privacy_transform, transform)
        if candidate is not None
    ]
    if len(configured_transforms) > 1:
        raise ValueError("provide only one privacy transformer")
    transformer = configured_transforms[0] if configured_transforms else None
    items = _field_items(fields)
    review_required = bool(
        _value(fields, "review_required", default=False)
        or _value(fields, "warnings", default=())
        or any(
            bool(_value(item, "review_required", "needs_review", default=False))
            for item in items
        )
    )
    if status is None:
        status = "in-progress" if review_required else "completed"
    if status not in {
        "in-progress",
        "completed",
        "amended",
        "entered-in-error",
        "stopped",
    }:
        raise ValueError("status must be a FHIR QuestionnaireResponse status")
    response: dict[str, Any] = {
        "resourceType": "QuestionnaireResponse",
        "id": response_id or _response_id(items),
        "meta": _meta(SDC_QUESTIONNAIRE_RESPONSE_PROFILE),
        "status": status,
        "item": [],
    }
    questionnaire_reference = _questionnaire_reference(questionnaire)
    if questionnaire_reference is not None:
        response["questionnaire"] = questionnaire_reference
    if subject is not None:
        response["subject"] = deepcopy(dict(subject))
    if authored is not None:
        response["authored"] = authored
    warnings = tuple(_value(fields, "warnings", default=()) or ())
    if warnings:
        response["extension"] = [
            {"url": OPENMED_WARNING_EXTENSION, "valueString": str(warning)}
            for warning in warnings
        ]
    for item in items:
        link_id = str(_value(item, "linkId", "link_id", "id", default="field"))
        label = str(_value(item, "text", "label", "key", default=link_id))
        data_type = str(
            _value(item, "type", "data_type", "value_type", default="string")
        )
        if data_type not in _ALLOWED_TYPES:
            data_type = "string"
        safe_value, _ = _safe_field_value(
            item,
            pii_detector=pii_detector,
            transformer=transformer,
        )
        item_payload: dict[str, Any] = {"linkId": link_id, "text": label}
        confidence = _value(item, "confidence", "score", default=None)
        try:
            confidence = (
                max(0.0, min(1.0, float(confidence)))
                if confidence is not None
                else None
            )
        except (TypeError, ValueError):
            confidence = None
        extensions: list[dict[str, Any]] = []
        if confidence is not None:
            extensions.append(
                {"url": OPENMED_CONFIDENCE_EXTENSION, "valueDecimal": confidence}
            )
        provenance = _safe_provenance(item, safe_value)
        extensions.append(
            {
                "url": OPENMED_PROVENANCE_EXTENSION,
                "valueString": json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        item_review = bool(
            _value(item, "review_required", "needs_review", default=False)
        )
        item_warnings = tuple(_value(item, "warnings", default=()) or ())
        if item_review:
            extensions.append({"url": OPENMED_REVIEW_EXTENSION, "valueBoolean": True})
        extensions.extend(
            {"url": OPENMED_WARNING_EXTENSION, "valueString": str(warning)}
            for warning in item_warnings
        )
        if extensions:
            item_payload["extension"] = extensions
        if data_type != "display":
            item_payload["answer"] = [
                {"value": None, **_answer_value(data_type, safe_value)}
            ]
            item_payload["answer"][0].pop("value", None)
        response["item"].append(item_payload)
    issues = validate_questionnaire_response(response, questionnaire=questionnaire)
    if issues:
        raise SDCValidationError(
            "generated QuestionnaireResponse failed supported validation"
        )
    return response


def _resource_profile(resource: Mapping[str, Any]) -> Sequence[str]:
    meta = resource.get("meta")
    if not isinstance(meta, Mapping):
        return ()
    profile = meta.get("profile")
    return (
        tuple(profile)
        if isinstance(profile, Sequence) and not isinstance(profile, str)
        else ()
    )


def validate_questionnaire(resource: Mapping[str, Any]) -> list[str]:
    """Validate the supported structural subset of a FHIR Questionnaire."""
    issues: list[str] = []
    if not isinstance(resource, Mapping):
        return ["Questionnaire must be a mapping"]
    if resource.get("resourceType") != "Questionnaire":
        issues.append("resourceType must be Questionnaire")
    if SDC_QUESTIONNAIRE_PROFILE not in _resource_profile(resource):
        issues.append("SDC Questionnaire profile is missing")
    if resource.get("status") not in {"draft", "active", "retired", "unknown"}:
        issues.append("Questionnaire status is invalid")
    items = resource.get("item")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        issues.append("Questionnaire item must be an array")
        return issues
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(f"Questionnaire item {index} is not a mapping")
            continue
        link_id = item.get("linkId")
        item_type = item.get("type")
        if not isinstance(link_id, str) or not link_id:
            issues.append(f"Questionnaire item {index} linkId is required")
        elif link_id in seen:
            issues.append(
                "Questionnaire linkIds must be unique in the supported subset"
            )
        else:
            seen.add(link_id)
        if item_type not in _ALLOWED_TYPES:
            issues.append(f"Questionnaire item {index} type is unsupported")
        if item_type != "display" and not isinstance(item.get("text"), str):
            issues.append(f"Questionnaire item {index} text is required")
    return issues


def _validate_extensions(item: Mapping[str, Any], issues: list[str]) -> None:
    extensions = item.get("extension", ())
    if not isinstance(extensions, Sequence) or isinstance(extensions, (str, bytes)):
        issues.append("extension must be an array")
        return
    for extension in extensions:
        if not isinstance(extension, Mapping) or not isinstance(
            extension.get("url"), str
        ):
            issues.append("extension entries require a URL")
            continue
        if extension["url"] == OPENMED_CONFIDENCE_EXTENSION:
            confidence = extension.get("valueDecimal")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                issues.append("confidence extension requires valueDecimal")
            elif not 0.0 <= float(confidence) <= 1.0:
                issues.append("confidence extension must be between zero and one")
        if extension["url"] == OPENMED_PROVENANCE_EXTENSION and not isinstance(
            extension.get("valueString"), str
        ):
            issues.append("provenance extension requires valueString")


def validate_questionnaire_response(
    resource: Mapping[str, Any],
    questionnaire: Mapping[str, Any] | str | None = None,
    *,
    original_identifiers: Iterable[str] = (),
) -> list[str]:
    """Validate a QuestionnaireResponse and optionally scan forbidden values."""
    issues: list[str] = []
    if not isinstance(resource, Mapping):
        return ["QuestionnaireResponse must be a mapping"]
    if resource.get("resourceType") != "QuestionnaireResponse":
        issues.append("resourceType must be QuestionnaireResponse")
    if SDC_QUESTIONNAIRE_RESPONSE_PROFILE not in _resource_profile(resource):
        issues.append("SDC QuestionnaireResponse profile is missing")
    if resource.get("status") not in {
        "in-progress",
        "completed",
        "amended",
        "entered-in-error",
        "stopped",
    }:
        issues.append("QuestionnaireResponse status is invalid")
    items = resource.get("item")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        issues.append("QuestionnaireResponse item must be an array")
        return issues
    allowed_link_ids: set[str] | None = None
    if isinstance(questionnaire, Mapping):
        allowed_link_ids = {
            str(item.get("linkId"))
            for item in questionnaire.get("item", ())
            if isinstance(item, Mapping) and item.get("linkId")
        }
    forbidden = tuple(
        str(identifier) for identifier in original_identifiers if identifier
    )
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(f"QuestionnaireResponse item {index} is not a mapping")
            continue
        link_id = item.get("linkId")
        if not isinstance(link_id, str) or not link_id:
            issues.append(f"QuestionnaireResponse item {index} linkId is required")
        elif allowed_link_ids is not None and link_id not in allowed_link_ids:
            issues.append("QuestionnaireResponse contains an unknown linkId")
        _validate_extensions(item, issues)
        answers = item.get("answer", ())
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            issues.append(f"QuestionnaireResponse item {index} answer must be an array")
            continue
        for answer in answers:
            if not isinstance(answer, Mapping):
                issues.append("QuestionnaireResponse answer must be a mapping")
                continue
            value_keys = [key for key in answer if key.startswith("value")]
            if len(value_keys) != 1:
                issues.append("QuestionnaireResponse answer must contain one value[x]")
            serialized = json.dumps(answer, sort_keys=True)
            if any(identifier in serialized for identifier in forbidden):
                issues.append(
                    "QuestionnaireResponse contains a forbidden original identifier"
                )
    return issues


def validate_sdc_response(
    resource: Mapping[str, Any],
    questionnaire: Mapping[str, Any] | str | None = None,
    *,
    original_identifiers: Iterable[str] = (),
) -> list[str]:
    """Alias for supported-subset QuestionnaireResponse validation."""
    return validate_questionnaire_response(
        resource,
        questionnaire=questionnaire,
        original_identifiers=original_identifiers,
    )


def is_valid_sdc_response(
    resource: Mapping[str, Any],
    questionnaire: Mapping[str, Any] | str | None = None,
) -> bool:
    """Return whether a QuestionnaireResponse passes the local subset."""
    return not validate_questionnaire_response(resource, questionnaire=questionnaire)


def assert_valid_sdc_response(
    resource: Mapping[str, Any],
    questionnaire: Mapping[str, Any] | str | None = None,
) -> None:
    """Raise a non-sensitive validation error when a response is invalid."""
    issues = validate_questionnaire_response(resource, questionnaire=questionnaire)
    if issues:
        raise SDCValidationError("; ".join(issues))


def to_sdc_resources(
    fields: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a Questionnaire and its privacy-safe QuestionnaireResponse."""
    questionnaire = to_questionnaire(fields)
    response = to_questionnaire_response(fields, questionnaire=questionnaire, **kwargs)
    return questionnaire, response


def build_sdc_bundle(
    questionnaire: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    bundle_id: str = "openmed-sdc-bundle",
) -> dict[str, Any]:
    """Create a local transaction-neutral Bundle containing both SDC resources."""
    if questionnaire.get("resourceType") != "Questionnaire":
        raise ValueError("questionnaire must be a FHIR Questionnaire")
    if response.get("resourceType") != "QuestionnaireResponse":
        raise ValueError("response must be a FHIR QuestionnaireResponse")
    bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "entry": [
            {
                "fullUrl": f"urn:uuid:{questionnaire.get('id', 'questionnaire')}",
                "resource": deepcopy(dict(questionnaire)),
            },
            {
                "fullUrl": f"urn:uuid:{response.get('id', 'response')}",
                "resource": deepcopy(dict(response)),
            },
        ],
    }
    return bundle


form_to_questionnaire = to_questionnaire
form_to_questionnaire_response = to_questionnaire_response
build_questionnaire = to_questionnaire
build_questionnaire_response = to_questionnaire_response
questionnaire_from_form = to_questionnaire
questionnaire_response_from_form = to_questionnaire_response
serialize_questionnaire = to_questionnaire
serialize_questionnaire_response = to_questionnaire_response
validate_response = validate_questionnaire_response
validate_sdc = validate_questionnaire_response


__all__ = [
    "OPENMED_CONFIDENCE_EXTENSION",
    "OPENMED_PROVENANCE_EXTENSION",
    "OPENMED_REVIEW_EXTENSION",
    "SDC_QUESTIONNAIRE_PROFILE",
    "SDC_QUESTIONNAIRE_RESPONSE_PROFILE",
    "SDC_VERSION",
    "SDCValidationError",
    "SDCValidationReport",
    "assert_valid_sdc_response",
    "build_sdc_bundle",
    "build_questionnaire",
    "build_questionnaire_response",
    "form_to_questionnaire",
    "form_to_questionnaire_response",
    "is_valid_sdc_response",
    "questionnaire_from_form",
    "questionnaire_response_from_form",
    "serialize_questionnaire",
    "serialize_questionnaire_response",
    "to_questionnaire",
    "to_questionnaire_response",
    "to_sdc_resources",
    "validate_questionnaire",
    "validate_questionnaire_response",
    "validate_response",
    "validate_sdc",
    "validate_sdc_response",
]
