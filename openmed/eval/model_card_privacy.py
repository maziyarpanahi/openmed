"""Differential-privacy and federation disclosures for model cards.

The disclosure section states how a privately or federally trained model was
protected: mechanism, privacy unit, adjacency, epsilon, delta, accountant and
conversion, rounds or steps, sampling, clipping, secure-aggregation mode,
participant band, tuning-cost inclusion and the risks that are not covered.

Every input value is a closed code, a bounded count, a band or a digest. A
differential-privacy claim is rendered only when its privacy unit, adjacency,
epsilon, delta and accountant are present and the canonical accountant
statement matches the digest carried by injected promotion evidence. Federated
or centralized training without a DP mechanism renders an explicit statement
that no formal differential-privacy guarantee applies.

Errors carry only a controlled code and a field name; input values, unknown
keys and source paths are never echoed. Producing promotion evidence and
publishing a card are outside this module.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import stable_hash
from openmed.eval.model_card_builder import ModelCardBuilderError
from openmed.training.federated_metrics import FederatedParticipantCountBand

PRIVACY_DISCLOSURE_SCHEMA_VERSION: Final = "openmed.model_card.privacy_disclosure.v1"
PRIVACY_ACCOUNTANT_STATEMENT_VERSION: Final = "openmed.privacy_accountant_statement.v1"
PRIVACY_DISCLOSURE_HEADING: Final = "## Privacy Disclosure"
NO_FORMAL_DP_GUARANTEE_STATEMENT: Final = (
    "No formal differential-privacy guarantee applies to this model."
)
ACCOUNTANT_STATEMENT_EVIDENCE_KEY: Final = "accountant_statement_digest"

_NOT_APPLICABLE: Final = "not_applicable"
_NO_MECHANISM: Final = "none"
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_ITERATIONS: Final = 1_000_000_000

TRAINING_SETTINGS: Final = ("centralized", "federated")
PRIVACY_MECHANISMS: Final = (
    "dp_fedavg_gaussian",
    "dp_ftrl_tree",
    "dp_sgd_gaussian",
    _NO_MECHANISM,
)
PRIVACY_UNITS: Final = ("note", "patient", "site")
ADJACENCY_RELATIONS: Final = ("add_remove", "replace_one")
ACCOUNTANT_CONVERSIONS: Final = MappingProxyType(
    {
        "basic_composition": ("direct",),
        "gdp": ("gdp_to_approx_dp",),
        "pld": ("direct",),
        "prv": ("direct",),
        "rdp": ("rdp_to_approx_dp",),
    }
)
SAMPLING_DECLARATIONS: Final = (
    "fixed_size_without_replacement",
    "full_participation",
    "poisson",
    "shuffled",
)
CLIPPING_MODES: Final = (
    "none",
    "per_client_update_l2",
    "per_example_l2",
    "per_unit_l2",
)
SECURE_AGGREGATION_MODES: Final = (
    "cryptographic",
    "none",
    _NOT_APPLICABLE,
    "trusted_execution",
)
PARTICIPANT_BANDS: Final = (
    *(band.value for band in FederatedParticipantCountBand),
    _NOT_APPLICABLE,
)
LIMITATION_STATEMENTS: Final = MappingProxyType(
    {
        "group_privacy_not_covered": (
            "The guarantee does not extend to groups of related privacy units."
        ),
        "inference_inputs_not_covered": (
            "Inputs processed at inference time are not covered."
        ),
        "participation_metadata_not_covered": (
            "Participation timing and other training metadata are not covered."
        ),
        "population_inference_not_covered": (
            "Population-level attribute inference is not prevented."
        ),
        "public_pretraining_data_not_covered": (
            "Data used to pretrain the base model is not covered."
        ),
        "secure_aggregation_not_dp": (
            "Secure aggregation hides individual updates from the aggregator "
            "but is not a differential-privacy guarantee."
        ),
    }
)

_FEDERATED_ONLY_MECHANISMS: Final = frozenset({"dp_fedavg_gaussian"})
_AMPLIFYING_SAMPLING: Final = frozenset({"fixed_size_without_replacement", "poisson"})
_ITERATION_UNITS: Final = MappingProxyType(
    {"centralized": "steps", "federated": "rounds"}
)
_DP_FIELDS: Final = (
    "privacy_unit",
    "adjacency",
    "epsilon",
    "delta",
    "accountant",
    "conversion",
    "sampling",
    "amplification_claimed",
    "tuning_cost_included",
)
_DISCLOSURE_FIELDS: Final = frozenset(
    {
        "clipping",
        "iteration_count",
        "iteration_unit",
        "limitations",
        "mechanism",
        "participant_band",
        "schema_version",
        "secure_aggregation",
        "training_setting",
        *_DP_FIELDS,
    }
)
_STATEMENT_FIELDS: Final = (
    "accountant",
    "adjacency",
    "amplification_claimed",
    "clipping",
    "conversion",
    "delta",
    "epsilon",
    "iteration_count",
    "iteration_unit",
    "mechanism",
    "privacy_unit",
    "sampling",
    "training_setting",
    "tuning_cost_included",
)


class PrivacyDisclosureError(ModelCardBuilderError):
    """Raised when a privacy disclosure cannot be rendered safely.

    Attributes:
        code: Controlled reason code, for example ``"delta_missing"``.
        field: Name of the disclosure field the code refers to.
    """

    def __init__(self, code: str, field: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"privacy disclosure rejected: {code} ({field})")


def build_privacy_disclosure(
    disclosure: Mapping[str, Any] | str | Path,
    *,
    promotion_evidence: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate a privacy disclosure and return its normalized card section.

    Args:
        disclosure: Disclosure mapping, or a path to a JSON object, using the
            ``openmed.model_card.privacy_disclosure.v1`` field set.
        promotion_evidence: Injected promotion-evidence digests, or a path to
            a JSON object. Every value must be a ``sha256:`` digest. A DP claim
            requires ``accountant_statement_digest`` to equal the digest of the
            canonical accountant statement derived from *disclosure*.

    Returns:
        A deterministic mapping holding only codes, bands, counts, the fixed
        guarantee statement, limitation statements and digests.

    Raises:
        PrivacyDisclosureError: If a field is missing, unknown or invalid, if a
            DP claim lacks accountant evidence, or if the claim disagrees with
            the evidence digest.
    """

    evidence = _evidence_digests(promotion_evidence)
    normalized = _normalize_disclosure(_load_mapping(disclosure, "privacy_disclosure"))
    if normalized["mechanism"] == _NO_MECHANISM:
        evidence_section = None
    else:
        expected = evidence.get(ACCOUNTANT_STATEMENT_EVIDENCE_KEY)
        if expected is None:
            raise PrivacyDisclosureError(
                "accountant_evidence_missing", ACCOUNTANT_STATEMENT_EVIDENCE_KEY
            )
        if _statement_digest(normalized) != expected:
            raise PrivacyDisclosureError(
                "accountant_evidence_mismatch", ACCOUNTANT_STATEMENT_EVIDENCE_KEY
            )
        evidence_section = {ACCOUNTANT_STATEMENT_EVIDENCE_KEY: expected}

    section: dict[str, Any] = {
        "schema_version": PRIVACY_DISCLOSURE_SCHEMA_VERSION,
        "formal_guarantee": (
            "none"
            if normalized["mechanism"] == _NO_MECHANISM
            else "differential_privacy"
        ),
        "guarantee_statement": _guarantee_statement(normalized),
        **{key: normalized[key] for key in sorted(normalized) if key != "limitations"},
        "limitations": [
            {"code": code, "statement": LIMITATION_STATEMENTS[code]}
            for code in normalized["limitations"]
        ],
        "evidence": evidence_section,
    }
    section = {key: section[key] for key in sorted(section)}
    section["disclosure_digest"] = stable_hash(section)
    return {key: section[key] for key in sorted(section)}


def privacy_accountant_statement(
    disclosure: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return the canonical accountant statement bound by promotion evidence.

    Args:
        disclosure: Disclosure mapping or JSON path with a DP mechanism.

    Returns:
        The statement fields that the accountant evidence digest covers.

    Raises:
        PrivacyDisclosureError: If the disclosure is invalid or declares no DP
            mechanism.
    """

    normalized = _normalize_disclosure(_load_mapping(disclosure, "privacy_disclosure"))
    if normalized["mechanism"] == _NO_MECHANISM:
        raise PrivacyDisclosureError("dp_mechanism_missing", "mechanism")
    return _statement(normalized)


def privacy_accountant_statement_digest(
    disclosure: Mapping[str, Any] | str | Path,
) -> str:
    """Return the ``sha256:`` digest of the canonical accountant statement.

    Args:
        disclosure: Disclosure mapping or JSON path with a DP mechanism.

    Returns:
        Digest of :func:`privacy_accountant_statement` as canonical JSON.

    Raises:
        PrivacyDisclosureError: If the disclosure is invalid or declares no DP
            mechanism.
    """

    return stable_hash(privacy_accountant_statement(disclosure))


def validate_privacy_disclosure_section(
    section: Mapping[str, Any],
    *,
    promotion_evidence: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Re-derive a rendered section and fail if it was altered.

    Args:
        section: Section previously returned by :func:`build_privacy_disclosure`.
        promotion_evidence: Optional injected evidence to verify against. When
            omitted, the digest recorded in *section* is re-checked against the
            disclosed fields.

    Returns:
        The re-derived section, equal to *section*.

    Raises:
        PrivacyDisclosureError: If the section is not internally consistent or
            disagrees with *promotion_evidence*.
    """

    if not isinstance(section, Mapping):
        raise PrivacyDisclosureError("privacy_disclosure_invalid", "datasheet")
    disclosure = {
        key: section[key]
        for key in _DISCLOSURE_FIELDS
        if key in section and key != "limitations"
    }
    limitations = section.get("limitations")
    if isinstance(limitations, Sequence) and not isinstance(limitations, str):
        disclosure["limitations"] = [
            item.get("code") if isinstance(item, Mapping) else item
            for item in limitations
        ]
    if promotion_evidence is None:
        recorded = section.get("evidence")
        promotion_evidence = dict(recorded) if isinstance(recorded, Mapping) else {}
    rebuilt = build_privacy_disclosure(
        disclosure,
        promotion_evidence=promotion_evidence,
    )
    if _plain(section) != rebuilt:
        raise PrivacyDisclosureError("privacy_disclosure_tampered", "datasheet")
    return rebuilt


def render_privacy_disclosure_markdown(section: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown privacy-disclosure section.

    Args:
        section: Section returned by :func:`build_privacy_disclosure`.

    Returns:
        Markdown text ending in a newline.
    """

    evidence = section.get("evidence")
    statement_digest = (
        evidence.get(ACCOUNTANT_STATEMENT_EVIDENCE_KEY)
        if isinstance(evidence, Mapping)
        else None
    )
    lines = [
        PRIVACY_DISCLOSURE_HEADING,
        "",
        f"> {section.get('guarantee_statement')}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Schema version | {_code(section.get('schema_version'))} |",
        f"| Formal guarantee | {_code(section.get('formal_guarantee'))} |",
        f"| Training setting | {_code(section.get('training_setting'))} |",
        f"| Mechanism | {_code(section.get('mechanism'))} |",
        f"| Privacy unit | {_code(section.get('privacy_unit'))} |",
        f"| Adjacency | {_code(section.get('adjacency'))} |",
        f"| Epsilon | {_number(section.get('epsilon'))} |",
        f"| Delta | {_number(section.get('delta'))} |",
        f"| Accountant | {_code(section.get('accountant'))} |",
        f"| Conversion | {_code(section.get('conversion'))} |",
        f"| Iterations | {section.get('iteration_count')} "
        f"{_code(section.get('iteration_unit'))} |",
        f"| Sampling | {_code(section.get('sampling'))} |",
        "| Amplification by sampling claimed | "
        f"{_flag(section.get('amplification_claimed'))} |",
        f"| Clipping | {_code(section.get('clipping'))} |",
        f"| Secure aggregation | {_code(section.get('secure_aggregation'))} |",
        f"| Participant band | {_code(section.get('participant_band'))} |",
        f"| Tuning cost included | {_flag(section.get('tuning_cost_included'))} |",
        f"| Accountant statement digest | {_code(statement_digest)} |",
        f"| Disclosure digest | {_code(section.get('disclosure_digest'))} |",
        "",
        "### Risks Not Covered",
        "",
    ]
    limitations = section.get("limitations")
    if isinstance(limitations, Sequence) and not isinstance(limitations, str):
        lines.extend(
            f"- `{item.get('code')}`: {item.get('statement')}"
            for item in limitations
            if isinstance(item, Mapping)
        )
    return "\n".join(lines) + "\n"


def privacy_disclosure_json(section: Mapping[str, Any]) -> str:
    """Return *section* as byte-stable, ASCII-only JSON.

    Args:
        section: Section returned by :func:`build_privacy_disclosure`.

    Returns:
        Indented JSON with sorted keys and a trailing newline.
    """

    return (
        json.dumps(
            _plain(section),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _normalize_disclosure(payload: Mapping[str, Any]) -> dict[str, Any]:
    if any(
        not isinstance(key, str) or key not in _DISCLOSURE_FIELDS for key in payload
    ):
        raise PrivacyDisclosureError("unknown_field", "disclosure")

    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise PrivacyDisclosureError("schema_version_missing", "schema_version")
    if schema_version != PRIVACY_DISCLOSURE_SCHEMA_VERSION:
        raise PrivacyDisclosureError("schema_version_unsupported", "schema_version")

    setting = _choice(payload, "training_setting", TRAINING_SETTINGS)
    mechanism = _choice(payload, "mechanism", PRIVACY_MECHANISMS)
    if mechanism in _FEDERATED_ONLY_MECHANISMS and setting != "federated":
        raise PrivacyDisclosureError("mechanism_setting_mismatch", "mechanism")

    iteration_unit = _choice(payload, "iteration_unit", ("rounds", "steps"))
    if iteration_unit != _ITERATION_UNITS[setting]:
        raise PrivacyDisclosureError("iteration_unit_mismatch", "iteration_unit")
    iteration_count = payload.get("iteration_count")
    if iteration_count is None:
        raise PrivacyDisclosureError("iteration_count_missing", "iteration_count")
    if (
        isinstance(iteration_count, bool)
        or not isinstance(iteration_count, int)
        or not 1 <= iteration_count <= _MAX_ITERATIONS
    ):
        raise PrivacyDisclosureError("iteration_count_invalid", "iteration_count")

    clipping = _choice(payload, "clipping", CLIPPING_MODES)
    secure_aggregation = _choice(
        payload, "secure_aggregation", SECURE_AGGREGATION_MODES
    )
    if (secure_aggregation == _NOT_APPLICABLE) != (setting == "centralized"):
        raise PrivacyDisclosureError(
            "secure_aggregation_mismatch", "secure_aggregation"
        )
    participant_band = _choice(payload, "participant_band", PARTICIPANT_BANDS)
    if (participant_band == _NOT_APPLICABLE) != (setting == "centralized"):
        raise PrivacyDisclosureError("participant_band_mismatch", "participant_band")

    limitations = _limitations(payload.get("limitations"))

    normalized: dict[str, Any] = {
        "clipping": clipping,
        "iteration_count": iteration_count,
        "iteration_unit": iteration_unit,
        "limitations": limitations,
        "mechanism": mechanism,
        "participant_band": participant_band,
        "secure_aggregation": secure_aggregation,
        "training_setting": setting,
    }
    if mechanism == _NO_MECHANISM:
        for field in _DP_FIELDS:
            if payload.get(field) is not None:
                raise PrivacyDisclosureError("dp_field_without_mechanism", field)
        normalized.update({field: None for field in _DP_FIELDS})
        return normalized

    for field in _DP_FIELDS:
        if payload.get(field) is None:
            raise PrivacyDisclosureError(f"{field}_missing", field)
    normalized["privacy_unit"] = _choice(payload, "privacy_unit", PRIVACY_UNITS)
    normalized["adjacency"] = _choice(payload, "adjacency", ADJACENCY_RELATIONS)
    normalized["epsilon"] = _epsilon(payload["epsilon"])
    normalized["delta"] = _delta(payload["delta"])
    accountant = _choice(payload, "accountant", tuple(ACCOUNTANT_CONVERSIONS))
    conversion = payload["conversion"]
    if not isinstance(conversion, str) or conversion not in {
        item for items in ACCOUNTANT_CONVERSIONS.values() for item in items
    }:
        raise PrivacyDisclosureError("conversion_invalid", "conversion")
    if conversion not in ACCOUNTANT_CONVERSIONS[accountant]:
        raise PrivacyDisclosureError("accountant_conversion_mismatch", "conversion")
    normalized["accountant"] = accountant
    normalized["conversion"] = conversion
    sampling = _choice(payload, "sampling", SAMPLING_DECLARATIONS)
    amplification = _flag_value(payload, "amplification_claimed")
    if amplification and sampling not in _AMPLIFYING_SAMPLING:
        raise PrivacyDisclosureError("sampling_amplification_unsupported", "sampling")
    normalized["sampling"] = sampling
    normalized["amplification_claimed"] = amplification
    normalized["tuning_cost_included"] = _flag_value(payload, "tuning_cost_included")
    if clipping == "none":
        raise PrivacyDisclosureError("clipping_required", "clipping")
    return normalized


def _statement(normalized: Mapping[str, Any]) -> dict[str, Any]:
    statement = {field: normalized[field] for field in _STATEMENT_FIELDS}
    statement["statement_version"] = PRIVACY_ACCOUNTANT_STATEMENT_VERSION
    return {key: statement[key] for key in sorted(statement)}


def _statement_digest(normalized: Mapping[str, Any]) -> str:
    return stable_hash(_statement(normalized))


def _guarantee_statement(normalized: Mapping[str, Any]) -> str:
    if normalized["mechanism"] == _NO_MECHANISM:
        return NO_FORMAL_DP_GUARANTEE_STATEMENT
    return (
        f"Differential privacy: epsilon={_number(normalized['epsilon'])}, "
        f"delta={_number(normalized['delta'])} per `{normalized['privacy_unit']}` "
        f"under `{normalized['adjacency']}` adjacency, accounted with "
        f"`{normalized['accountant']}` (`{normalized['conversion']}`)."
    )


def _evidence_digests(
    source: Mapping[str, Any] | str | Path | None,
) -> dict[str, str]:
    if source is None:
        return {}
    payload = _load_mapping(source, "privacy_evidence")
    digests: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PrivacyDisclosureError(
                "promotion_evidence_not_digest", "privacy_evidence"
            )
        if not _DIGEST.fullmatch(value):
            raise PrivacyDisclosureError(
                "promotion_evidence_not_digest", "privacy_evidence"
            )
        digests[key] = value
    return digests


def _load_mapping(source: Mapping[str, Any] | str | Path, name: str) -> Mapping:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, (str, Path)):
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            raise PrivacyDisclosureError(f"{name}_unreadable", name) from None
        if isinstance(payload, Mapping):
            return payload
    raise PrivacyDisclosureError(f"{name}_invalid", name)


def _choice(payload: Mapping[str, Any], field: str, allowed: Sequence[str]) -> str:
    value = payload.get(field)
    if value is None:
        raise PrivacyDisclosureError(f"{field}_missing", field)
    if not isinstance(value, str) or value not in allowed:
        raise PrivacyDisclosureError(f"{field}_invalid", field)
    return value


def _flag_value(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise PrivacyDisclosureError(f"{field}_invalid", field)
    return value


def _epsilon(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise PrivacyDisclosureError("epsilon_invalid", "epsilon")
    return float(value)


def _delta(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise PrivacyDisclosureError("delta_invalid", "delta")
    return float(value)


def _limitations(value: Any) -> list[str]:
    if value is None or (
        isinstance(value, Sequence) and not isinstance(value, str) and not value
    ):
        raise PrivacyDisclosureError("limitations_missing", "limitations")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PrivacyDisclosureError("limitations_invalid", "limitations")
    codes: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item not in LIMITATION_STATEMENTS:
            raise PrivacyDisclosureError("limitation_invalid", "limitations")
        codes.add(item)
    return sorted(codes)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _code(value: Any) -> str:
    return "Not applicable" if value is None else f"`{value}`"


def _number(value: Any) -> str:
    return "Not applicable" if value is None else json.dumps(value)


def _flag(value: Any) -> str:
    if value is None:
        return "Not applicable"
    return "Yes" if value else "No"


__all__ = [
    "ACCOUNTANT_CONVERSIONS",
    "ACCOUNTANT_STATEMENT_EVIDENCE_KEY",
    "ADJACENCY_RELATIONS",
    "CLIPPING_MODES",
    "LIMITATION_STATEMENTS",
    "NO_FORMAL_DP_GUARANTEE_STATEMENT",
    "PARTICIPANT_BANDS",
    "PRIVACY_ACCOUNTANT_STATEMENT_VERSION",
    "PRIVACY_DISCLOSURE_HEADING",
    "PRIVACY_DISCLOSURE_SCHEMA_VERSION",
    "PRIVACY_MECHANISMS",
    "PRIVACY_UNITS",
    "PrivacyDisclosureError",
    "SAMPLING_DECLARATIONS",
    "SECURE_AGGREGATION_MODES",
    "TRAINING_SETTINGS",
    "build_privacy_disclosure",
    "privacy_accountant_statement",
    "privacy_accountant_statement_digest",
    "privacy_disclosure_json",
    "render_privacy_disclosure_markdown",
    "validate_privacy_disclosure_section",
]
