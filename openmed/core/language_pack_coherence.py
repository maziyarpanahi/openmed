"""Language-pack coherence validation and capability coverage (#1584).

This module extends the OM-135 locale-coherence contract
(:func:`openmed.core.anonymizer.locales.locale_coherence_report`) into an
*atomic* validator over registered :class:`~openmed.core.language_pack.LanguagePack`
declarations, plus a JSON-friendly per-pack capability-coverage report.

It deliberately lives outside :mod:`openmed.core.language_pack` (which stays free
of PII/anonymizer/policy/threshold imports to avoid cycles): the registry is the
source of pack declarations, and this module resolves each declaration against
the existing surrogate-locale, national-ID, policy, and threshold contracts.

Capability coverage tracks five slots per pack -- ``script``, ``segmenter``,
``recognizers``, ``surrogate_locale``, and ``policy`` -- each classified as
``filled``, ``approximated``, or ``missing``. Coherence fails loudly when a
*declared* capability cannot be resolved: an unresolvable segmenter, a
surrogate locale that is neither a real Faker locale nor a documented
approximation, a national-ID provider whose surrogates do not round-trip its
registered validator, a policy profile the threshold matrix does not define, or
a recall-floor override keyed by a non-canonical label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping, Optional

from .language_pack import LANGUAGE_PACK_REGISTRY, LanguagePackRegistry
from .language_pack_catalog import is_registered_segmenter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .language_pack import LanguagePack

# Capability slots counted by the coverage report, in stable display order.
CAPABILITY_SLOTS: tuple[str, ...] = (
    "script",
    "segmenter",
    "recognizers",
    "surrogate_locale",
    "policy",
)

# Slot-status vocabulary.
FILLED = "filled"
APPROXIMATED = "approximated"
MISSING = "missing"

# Number of surrogate samples drawn per national-ID provider during round-trip
# validation. A mis-wired provider (wrong locale/method) fails deterministically
# well within this budget; kept modest so the report stays fast.
_NATIONAL_ID_SAMPLE_SIZE = 16


class LanguagePackCoherenceError(RuntimeError):
    """Raised when one or more registered packs fail coherence validation."""

    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows
        codes = ", ".join(str(row["language"]) for row in rows)
        super().__init__(f"incoherent language pack(s): {codes}")


def _segmenter_resolves(segmenter_id: str) -> bool:
    """Return whether ``segmenter_id`` resolves through the shared catalog."""

    return is_registered_segmenter(segmenter_id)


def _surrogate_locale_status(pack: "LanguagePack") -> tuple[str, dict[str, object]]:
    """Classify a pack's surrogate locale against the Faker/approximation contract."""
    from faker.config import AVAILABLE_LOCALES

    from .anonymizer.locales import (
        _APPROXIMATE_LOCALES,
        resolve_faker_backend_locale,
    )
    from .language_pack_catalog import LANG_TO_LOCALE

    backend = resolve_faker_backend_locale(pack.surrogate_locale)
    detail: dict[str, object] = {
        "locale": pack.surrogate_locale,
        "backend": backend,
    }
    if backend not in AVAILABLE_LOCALES:
        return MISSING, detail
    if (
        pack.code in _APPROXIMATE_LOCALES
        and LANG_TO_LOCALE.get(pack.code) == pack.surrogate_locale
    ):
        return APPROXIMATED, detail
    return FILLED, detail


def _national_id_validators(lang: str) -> list:
    """Registered national-ID checksum validators for ``lang``.

    Mirrors the OM-135 helper: prefer the language pack's own ``national_id``
    validators, falling back to the shared base SSN/national-ID validators for
    languages whose national ID is validated by the base pattern set.
    """
    from .pii_entity_merger import PII_PATTERNS
    from .pii_i18n import LANGUAGE_PII_PATTERNS

    lang_validators = [
        pattern.validator
        for pattern in LANGUAGE_PII_PATTERNS.get(lang, [])
        if pattern.validator is not None and pattern.entity_type == "national_id"
    ]
    if lang_validators:
        return lang_validators
    return [
        pattern.validator
        for pattern in PII_PATTERNS
        if pattern.validator is not None
        and pattern.entity_type in ("national_id", "ssn")
    ]


def _national_id_issues(pack: "LanguagePack") -> list[str]:
    """Return human-readable coherence issues for a pack's national-ID providers.

    Each declared ``method -> locale`` pairing must (a) agree with the
    anonymizer registry's locale dispatch and (b) produce surrogates that
    round-trip the pack language's registered validator.
    """
    if not pack.national_id_providers:
        return []

    from .anonymizer.registry import _LOCALE_ID_METHODS

    validators = _national_id_validators(pack.code)
    issues: list[str] = []
    for method, locale in sorted(pack.national_id_providers.items()):
        dispatch = _LOCALE_ID_METHODS.get(locale)
        if dispatch != method:
            issues.append(
                f"national-ID provider {method!r} disagrees with registry "
                f"dispatch for locale {locale!r} (expected {dispatch!r})"
            )
            continue
        if not validators:
            issues.append(
                f"national-ID provider {method!r} has no registered validator "
                f"for language {pack.code!r}"
            )
            continue
        if not _provider_round_trips(pack.code, locale, validators):
            issues.append(
                f"national-ID provider {method!r} ({locale}) surrogates do not "
                f"round-trip the registered validator for {pack.code!r}"
            )
    return issues


def _provider_round_trips(lang: str, locale: str, validators: list) -> bool:
    """Return whether generated surrogates pass any registered validator."""
    import warnings

    from .anonymizer import Anonymizer

    for seed in range(_NATIONAL_ID_SAMPLE_SIZE):
        anonymizer = Anonymizer(lang=lang, consistent=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = anonymizer.surrogate("123456789", "national_id", locale=locale)
        if not any(validator(surrogate) for validator in validators):
            return False
    return True


def _policy_status(pack: "LanguagePack") -> tuple[str, list[str], dict[str, object]]:
    """Classify a pack's policy/recall-floor overrides against threshold contracts.

    Returns ``(status, issues, detail)``. The slot is ``filled`` when the pack
    declares a valid policy profile and/or valid recall-floor overrides,
    ``missing`` when it declares neither, and any *invalid* declaration produces
    a coherence issue regardless of status.
    """
    from .labels import CANONICAL_LABELS
    from .thresholds import load_thresholds

    detail: dict[str, object] = {
        "profile": pack.policy_overrides.get("profile"),
        "recall_floor_labels": sorted(pack.recall_floor_overrides),
    }
    issues: list[str] = []

    unknown_policy_keys = sorted(set(pack.policy_overrides) - {"profile"})
    if unknown_policy_keys:
        issues.append(
            f"unsupported policy override keys: {unknown_policy_keys}; "
            "only 'profile' is defined"
        )

    declared_profile = pack.policy_overrides.get("profile")
    if declared_profile is not None:
        valid_profiles = set(load_thresholds()["profiles"])
        if declared_profile not in valid_profiles:
            issues.append(
                f"policy profile {declared_profile!r} is not defined in the "
                f"threshold matrix (known: {sorted(valid_profiles)})"
            )

    unknown_labels = sorted(set(pack.recall_floor_overrides) - CANONICAL_LABELS)
    if unknown_labels:
        issues.append(
            f"recall-floor overrides reference non-canonical labels: {unknown_labels}"
        )

    declared = declared_profile is not None or bool(pack.recall_floor_overrides)
    status = FILLED if declared and not issues else MISSING
    return status, issues, detail


def _pack_row(pack: "LanguagePack") -> dict[str, object]:
    """Build the JSON-friendly coherence + coverage row for a single pack."""
    issues: list[str] = []

    # script / recognizers are non-empty by the LanguagePack contract.
    script_status = FILLED if pack.scripts else MISSING
    recognizer_status = FILLED if pack.recognizers else MISSING

    segmenter_status = FILLED if _segmenter_resolves(pack.segmenter_id) else MISSING
    if segmenter_status == MISSING:
        issues.append(f"segmenter {pack.segmenter_id!r} does not resolve")

    surrogate_status, surrogate_detail = _surrogate_locale_status(pack)
    if surrogate_status == MISSING:
        issues.append(
            f"surrogate locale {pack.surrogate_locale!r} is neither a real Faker "
            "locale nor a documented approximation"
        )

    policy_status, policy_issues, policy_detail = _policy_status(pack)
    issues.extend(policy_issues)

    national_id_issues = _national_id_issues(pack)
    issues.extend(national_id_issues)

    slots = {
        "script": script_status,
        "segmenter": segmenter_status,
        "recognizers": recognizer_status,
        "surrogate_locale": surrogate_status,
        "policy": policy_status,
    }
    coverage = {
        "slots": slots,
        "filled": sum(1 for status in slots.values() if status == FILLED),
        "approximated": sum(1 for status in slots.values() if status == APPROXIMATED),
        "missing": sum(1 for status in slots.values() if status == MISSING),
    }

    return {
        "language": pack.code,
        "scripts": list(pack.scripts),
        "segmenter": {"id": pack.segmenter_id, "status": segmenter_status},
        "recognizers": list(pack.recognizers),
        "surrogate_locale": {**surrogate_detail, "status": surrogate_status},
        "national_id": {
            "providers": sorted(pack.national_id_providers),
            "status": MISSING
            if national_id_issues
            else (FILLED if pack.national_id_providers else "absent"),
        },
        "policy": {**policy_detail, "status": policy_status},
        "coverage": coverage,
        "coherent": not issues,
        "issues": issues,
    }


def _resolve_registry(
    registry: Optional[LanguagePackRegistry],
) -> LanguagePackRegistry:
    return registry if registry is not None else LANGUAGE_PACK_REGISTRY


def pack_coherence_report(
    *,
    registry: Optional[LanguagePackRegistry] = None,
) -> list[dict[str, object]]:
    """Return one JSON-serializable coherence + coverage row per registered pack.

    Rows are ordered by language code (via the registry's sorted snapshot) so
    the report is deterministic. Each row carries per-slot capability coverage,
    a ``coherent`` boolean, and a list of human-readable ``issues``.
    """
    resolved = _resolve_registry(registry)
    return [_pack_row(resolved.get(code)) for code in resolved.iter_codes()]


def incoherent_packs(
    *,
    registry: Optional[LanguagePackRegistry] = None,
    rows: Optional[Iterable[Mapping[str, object]]] = None,
) -> list[dict[str, object]]:
    """Return the coherence rows for packs that failed validation."""
    source = rows if rows is not None else pack_coherence_report(registry=registry)
    return [dict(row) for row in source if not row["coherent"]]


def check_language_pack_coherence(
    *,
    registry: Optional[LanguagePackRegistry] = None,
) -> int:
    """Return the number of incoherent packs (``0`` when every pack is coherent).

    A non-zero return is the fail signal for callers that want an exit-code-style
    gate; :func:`require_language_pack_coherence` is the raising variant.
    """
    return len(incoherent_packs(registry=registry))


def require_language_pack_coherence(
    *,
    registry: Optional[LanguagePackRegistry] = None,
) -> None:
    """Raise :class:`LanguagePackCoherenceError` if any registered pack is incoherent."""
    failures = incoherent_packs(registry=registry)
    if failures:
        raise LanguagePackCoherenceError(failures)


__all__ = [
    "CAPABILITY_SLOTS",
    "FILLED",
    "APPROXIMATED",
    "MISSING",
    "LanguagePackCoherenceError",
    "pack_coherence_report",
    "incoherent_packs",
    "check_language_pack_coherence",
    "require_language_pack_coherence",
]
