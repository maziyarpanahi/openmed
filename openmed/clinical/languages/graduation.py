"""Initial low-resource clinical language cohort and its graduation contract.

The cohort names a small set of target locales chosen from evidence already in
the repository: registered language packs, user-supplied-model and
national-ID-only routes, committed synthetic golden fixtures, model-manifest
language declarations and surrogate locales. Each locale records its
license/lineage, a review role code (never a person), a synthetic fixture plan
and an explicit ``experimental`` or ``supported`` status.

The graduation contract turns aggregate, content-free evaluation evidence into
a deterministic decision. Critical-identifier recall is non-compensable: every
critical label must pass on its own point estimate and on its confidence lower
bound, and an average can never hide a failing label. Cross-lingual
regressions on reference languages are bounded per cell, so a gain in one cell
cannot offset a loss in another.

All inputs are plain data (codes, counts and 64-character SHA-256 hex
references) so the same evidence can later feed the clinical language readiness
inventory without importing it. Nothing here loads models, reads clinical text,
touches the network or changes the routing of ``deidentify()``. A ``supported``
status is a release claim for the listed tasks only; it is not clinical
validation, certification or cultural validation, and every other task fails
closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Any

from openmed.training.synthetic.spdx_identifier import normalize_spdx_identifier

SCHEMA_VERSION = "1.0.0"
COHORT_ID = "low_resource_initial"

EXPERIMENTAL = "experimental"
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
STATUSES: tuple[str, ...] = (EXPERIMENTAL, SUPPORTED)

# Closed task vocabulary for the fail-closed gate. Only tasks listed in
# ``GRADUATION_TASKS`` have a graduation contract; the others always refuse.
TASKS: tuple[str, ...] = (
    "assertion_detection",
    "clinical_entity_extraction",
    "deidentification",
    "summarization",
    "surrogate_generation",
    "terminology_normalization",
)
GRADUATION_TASKS: tuple[str, ...] = ("deidentification",)

ROUTES: tuple[str, ...] = (
    "dedicated_checkpoint",
    "default_model_placeholder",
    "multilingual_fallback",
    "national_id_only",
    "user_supplied_model",
)
# Routes that ship no bundled weights cannot graduate a model-backed task.
_ROUTES_WITHOUT_BUNDLED_MODEL = frozenset({"national_id_only", "user_supplied_model"})
_ROUTE_MODEL_ORIGIN = {
    "dedicated_checkpoint": "dedicated_checkpoint",
    "default_model_placeholder": "multilingual_fallback",
    "multilingual_fallback": "multilingual_fallback",
    "national_id_only": "no_model",
    "user_supplied_model": "user_supplied",
}
_ROUTE_EVIDENCE = {
    "dedicated_checkpoint": "language_pack",
    "default_model_placeholder": "language_pack",
    "multilingual_fallback": "language_pack",
    "national_id_only": "national_id_only_route",
    "user_supplied_model": "user_supplied_model_route",
}

REVIEW_ROLES: tuple[str, ...] = (
    "clinical_linguist",
    "language_pack_maintainer",
    "native_clinical_reviewer",
    "privacy_reviewer",
)
REVIEWER_COMMITMENTS: tuple[str, ...] = ("active", "committed", "unfilled")
# Same vocabulary as the clinical language readiness reviewer status.
REVIEW_STATUSES: tuple[str, ...] = ("approved", "not_requested", "pending", "rejected")

MODEL_ORIGINS: tuple[str, ...] = (
    "dedicated_checkpoint",
    "multilingual_fallback",
    "no_model",
    "user_supplied",
)
FIXTURE_ORIGINS: tuple[str, ...] = ("none", "synthetic_repository_authored")
SURROGATE_ORIGINS: tuple[str, ...] = ("faker_approximation", "faker_native")
FIXTURE_AUTHORING: tuple[str, ...] = (
    "machine_translation",
    "native_authored_synthetic",
    "native_reviewed_translation",
)
# Translation alone never establishes cultural or clinical validity.
_GRADUATION_AUTHORING = frozenset(
    {"native_authored_synthetic", "native_reviewed_translation"}
)

REGIONS: tuple[str, ...] = (
    "east_africa",
    "horn_of_africa",
    "south_asia",
    "southern_africa",
    "west_africa",
)

REPOSITORY_EVIDENCE: tuple[str, ...] = (
    "golden_i18n_fixture",
    "language_pack",
    "locale_phi_generator",
    "manifest_model_language",
    "national_id_only_route",
    "national_id_validator",
    "on_device_mlx_variant",
    "user_supplied_model_route",
)

STRESSORS: tuple[str, ...] = (
    "attached_affixes",
    "code_switch_english",
    "conjunct_clusters",
    "decomposed_unicode",
    "ethiopic_numerals",
    "honorific_names",
    "native_digits",
    "noun_class_prefixes",
    "rtl_bidi",
    "tone_diacritics",
)

LIMITATIONS: tuple[str, ...] = (
    "identifier_only_patterns",
    "no_bundled_model",
    "no_golden_fixture",
    "no_locale_phi_generator",
    "placeholder_model_route",
    "surrogate_locale_approximation",
    "tokenizer_script_unclaimed",
)

CRITICAL_IDENTIFIER_LABELS: tuple[str, ...] = (
    "DATE",
    "ID_NUM",
    "PERSON",
    "PHONE",
    "STREET_ADDRESS",
    "ZIPCODE",
)

# Policy floors: a contract may be stricter than these, never looser.
MIN_DOCUMENTS_FLOOR = 200
MIN_LABEL_SUPPORT_FLOOR = 100
CRITICAL_RECALL_FLOOR = 0.99
CRITICAL_RECALL_LOWER_BOUND_FLOOR = 0.95
MIN_TERMINOLOGY_TERMS_FLOOR = 100
MIN_TERMINOLOGY_COVERAGE_FLOOR = 0.80
MIN_ABSTENTION_PROBES_FLOOR = 20
MIN_CROSS_LINGUAL_TRIALS_FLOOR = 50
MAX_CROSS_LINGUAL_REGRESSION_CAP = 0.01
MAX_PACKAGE_BYTES_CAP = 2 * 1024**3
MAX_REVIEW_INTERVAL_DAYS_CAP = 365

# Two-sided 95% normal quantile for the Wilson score lower bound.
_Z = 1.959963984540054
_MAX_COUNT = 10_000_000
_MAX_ITEMS = 4096
_MAX_COHORT = 256

_LANGUAGE = re.compile(r"[a-z]{2,3}")
_LANGUAGE_REQUEST = re.compile(r"([A-Za-z]{2,3})(?:[-_]([A-Za-z]{2}))?")
_LOCALE = re.compile(r"[a-z]{2,3}(?:_[A-Z]{2})?")
_SCRIPT = re.compile(r"[A-Z][A-Za-z/]{1,31}")
_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_DIGEST = re.compile(r"[0-9a-f]{64}")

_GOLDEN_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "eval" / "golden" / "fixtures" / "i18n"
)
_MAX_FIXTURE_BYTES = 4 * 1024 * 1024
_MULTILINGUAL_FALLBACK_MODEL = "OpenMed/privacy-filter-multilingual"


class LanguageGraduationError(ValueError):
    """Raised with a controlled code when a cohort gate refuses a request.

    The message is the code itself and never echoes caller input.

    Attributes:
        code: Stable machine-readable refusal reason.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _bounded(values: Any, limit: int = _MAX_ITEMS) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError("expected a bounded collection")
    try:
        result = tuple(islice(values, limit + 1))
    except TypeError:
        raise ValueError("expected a bounded collection") from None
    if len(result) > limit:
        raise ValueError("collection exceeds bound")
    return result


def _codes(
    values: Any,
    allowed: Iterable[str] | None = None,
    pattern: re.Pattern[str] = _TOKEN,
) -> tuple[str, ...]:
    items = _bounded(values, 256)
    allowed_set = None if allowed is None else frozenset(allowed)
    for item in items:
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise ValueError("invalid controlled code")
        if allowed_set is not None and item not in allowed_set:
            raise ValueError("unknown controlled code")
    if len(set(items)) != len(items):
        raise ValueError("duplicate controlled code")
    return tuple(sorted(items))


def _choice(value: Any, allowed: Iterable[str]) -> str:
    if not isinstance(value, str) or value not in frozenset(allowed):
        raise ValueError("unknown controlled code")
    return value


def _count(value: Any, *, minimum: int = 0, maximum: int = _MAX_COUNT) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid count")
    if not minimum <= value <= maximum:
        raise ValueError("count out of range")
    return value


def _ratio(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid ratio")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("ratio out of range")
    return value


def _digest_ref(value: Any) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError("invalid SHA-256 reference")
    return value


def _language_code(value: Any) -> str:
    if not isinstance(value, str) or _LANGUAGE.fullmatch(value) is None:
        raise ValueError("invalid language code")
    return value


def _permissive_license(value: Any) -> str:
    result = normalize_spdx_identifier(value)
    if not result.is_permissive or result.normalized is None:
        raise ValueError("license must be a permissive SPDX identifier")
    return result.normalized


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def wilson_lower_bound(successes: int, trials: int) -> float:
    """Return the two-sided 95% Wilson score lower bound of a proportion.

    Args:
        successes: Number of detected items.
        trials: Number of gold items; zero yields ``0.0``.

    Returns:
        The lower confidence bound in ``[0, 1]``.
    """
    successes = _count(successes)
    trials = _count(trials)
    if successes > trials:
        raise ValueError("successes exceed trials")
    if trials == 0:
        return 0.0
    p = successes / trials
    z2 = _Z * _Z
    centre = p + z2 / (2 * trials)
    margin = _Z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, (centre - margin) / (1 + z2 / trials))


@dataclass(frozen=True, slots=True)
class LicenseLineage:
    """License and lineage record for one cohort locale.

    Attributes:
        model_origin: Where the default weights come from (closed code).
        model_license: SPDX identifier of bundled weights, or ``None`` when the
            route ships no weights.
        fixture_origin: Lineage of the committed fixtures (closed code).
        fixture_license: SPDX identifier of committed fixtures, or ``None``.
        surrogate_origin: Whether surrogates use a native Faker locale.
        bundles_restricted_data: Must stay ``False``; restricted corpora and
            DUA-gated data are never bundled.
    """

    model_origin: str
    model_license: str | None
    fixture_origin: str
    fixture_license: str | None
    surrogate_origin: str
    bundles_restricted_data: bool = False

    def __post_init__(self) -> None:
        _choice(self.model_origin, MODEL_ORIGINS)
        _choice(self.fixture_origin, FIXTURE_ORIGINS)
        _choice(self.surrogate_origin, SURROGATE_ORIGINS)
        if self.bundles_restricted_data is not False:
            raise ValueError("restricted data cannot be bundled with a cohort locale")
        if self.model_origin in {"no_model", "user_supplied"}:
            if self.model_license is not None:
                raise ValueError("routes without bundled weights carry no license")
        else:
            object.__setattr__(
                self, "model_license", _permissive_license(self.model_license)
            )
        if self.fixture_origin == "none":
            if self.fixture_license is not None:
                raise ValueError("absent fixtures carry no license")
        else:
            object.__setattr__(
                self, "fixture_license", _permissive_license(self.fixture_license)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable license/lineage schema."""
        return {
            "bundles_restricted_data": self.bundles_restricted_data,
            "fixture_license": self.fixture_license,
            "fixture_origin": self.fixture_origin,
            "model_license": self.model_license,
            "model_origin": self.model_origin,
            "surrogate_origin": self.surrogate_origin,
        }


@dataclass(frozen=True, slots=True)
class SyntheticFixturePlan:
    """Synthetic fixture plan for one cohort locale.

    Attributes:
        golden_records: Committed synthetic golden records today.
        golden_labels: Critical identifier labels those records already cover.
        target_documents: Synthetic evaluation documents to author.
        authoring: How the documents are authored; machine translation alone
            is rejected.
        stressors: Script and morphology stressors the fixtures must cover.
    """

    golden_records: int
    golden_labels: tuple[str, ...]
    target_documents: int
    authoring: str
    stressors: tuple[str, ...]

    def __post_init__(self) -> None:
        _count(self.golden_records, maximum=_MAX_ITEMS)
        labels = _codes(self.golden_labels, CRITICAL_IDENTIFIER_LABELS, _LABEL)
        object.__setattr__(self, "golden_labels", labels)
        if self.golden_records == 0 and labels:
            raise ValueError("labels require committed golden records")
        _count(self.target_documents, minimum=MIN_DOCUMENTS_FLOOR)
        if _choice(self.authoring, FIXTURE_AUTHORING) not in _GRADUATION_AUTHORING:
            raise ValueError("fixture plan cannot rely on machine translation alone")
        stressors = _codes(self.stressors, STRESSORS)
        if not stressors:
            raise ValueError("fixture plan requires script or morphology stressors")
        object.__setattr__(self, "stressors", stressors)

    @property
    def missing_labels(self) -> tuple[str, ...]:
        """Critical labels the committed golden records do not cover yet."""
        return tuple(
            label
            for label in CRITICAL_IDENTIFIER_LABELS
            if label not in self.golden_labels
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable fixture-plan schema."""
        return {
            "authoring": self.authoring,
            "golden_labels": list(self.golden_labels),
            "golden_records": self.golden_records,
            "missing_labels": list(self.missing_labels),
            "stressors": list(self.stressors),
            "target_documents": self.target_documents,
        }


@dataclass(frozen=True, slots=True)
class CohortLocale:
    """One target locale with explicit status and graduation bindings.

    ``status`` is declared, not inferred: ``supported`` is accepted only with a
    graduation report digest for every supported task, an active reviewer
    commitment and a route that ships bundled weights.

    Attributes:
        language: Primary ISO 639 code.
        locale: Surrogate locale recorded by the language catalog.
        scripts: Unicode script names routed to this language.
        region: Deployment region code anchoring the clinical need.
        route: How the repository serves the language today.
        repository_evidence: Evidence codes that justify the selection.
        license_lineage: License and lineage record.
        review_role: Review role code; never a person's name.
        reviewer_commitment: Whether that role is filled.
        fixture_plan: Synthetic fixture plan.
        candidate_tasks: Tasks that may graduate under the contract.
        status: ``experimental`` or ``supported``.
        graduations: ``(task, report_digest)`` pairs for supported tasks.
        limitations: Controlled limitation codes.
    """

    language: str
    locale: str
    scripts: tuple[str, ...]
    region: str
    route: str
    repository_evidence: tuple[str, ...]
    license_lineage: LicenseLineage
    review_role: str
    reviewer_commitment: str
    fixture_plan: SyntheticFixturePlan
    candidate_tasks: tuple[str, ...] = GRADUATION_TASKS
    status: str = EXPERIMENTAL
    graduations: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _language_code(self.language)
        if not isinstance(self.locale, str) or _LOCALE.fullmatch(self.locale) is None:
            raise ValueError("invalid surrogate locale")
        scripts = _codes(self.scripts, pattern=_SCRIPT)
        if not scripts:
            raise ValueError("a cohort locale requires at least one script")
        object.__setattr__(self, "scripts", scripts)
        _choice(self.region, REGIONS)
        _choice(self.route, ROUTES)
        evidence = _codes(self.repository_evidence, REPOSITORY_EVIDENCE)
        object.__setattr__(self, "repository_evidence", evidence)
        if _ROUTE_EVIDENCE[self.route] not in evidence:
            raise ValueError("route is not backed by repository evidence")
        if not isinstance(self.license_lineage, LicenseLineage):
            raise ValueError("invalid license lineage record")
        if self.license_lineage.model_origin != _ROUTE_MODEL_ORIGIN[self.route]:
            raise ValueError("license lineage contradicts the route")
        if not isinstance(self.fixture_plan, SyntheticFixturePlan):
            raise ValueError("invalid synthetic fixture plan")
        has_golden = self.fixture_plan.golden_records > 0
        if has_golden != ("golden_i18n_fixture" in evidence) or has_golden != (
            self.license_lineage.fixture_origin != "none"
        ):
            raise ValueError("fixture plan contradicts fixture evidence")
        _choice(self.review_role, REVIEW_ROLES)
        _choice(self.reviewer_commitment, REVIEWER_COMMITMENTS)
        candidates = _codes(self.candidate_tasks, GRADUATION_TASKS)
        if not candidates:
            raise ValueError("a cohort locale requires a candidate task")
        object.__setattr__(self, "candidate_tasks", candidates)
        self._validate_status()
        limitations = _codes(self.limitations, LIMITATIONS)
        if set(limitations) - {"tokenizer_script_unclaimed"} != set(
            self._derived_limitations()
        ):
            raise ValueError("limitations do not match the recorded evidence")
        object.__setattr__(self, "limitations", limitations)

    def _validate_status(self) -> None:
        _choice(self.status, STATUSES)
        pairs = _bounded(self.graduations, len(GRADUATION_TASKS))
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("graduations must be (task, digest) pairs")
            if pair[0] not in self.candidate_tasks:
                raise ValueError("graduated task is not a candidate task")
            _digest_ref(pair[1])
        if len({task for task, _ in pairs}) != len(pairs):
            raise ValueError("duplicate graduated task")
        object.__setattr__(self, "graduations", tuple(sorted(pairs)))
        if (self.status == SUPPORTED) != bool(pairs):
            raise ValueError("supported status requires graduation evidence")
        if pairs and self.reviewer_commitment != "active":
            raise ValueError("supported status requires an active reviewer")
        if pairs and self.route in _ROUTES_WITHOUT_BUNDLED_MODEL:
            raise ValueError("supported status requires bundled weights")

    def _derived_limitations(self) -> tuple[str, ...]:
        derived = set()
        if self.route in _ROUTES_WITHOUT_BUNDLED_MODEL:
            derived.add("no_bundled_model")
        if self.route == "national_id_only":
            derived.add("identifier_only_patterns")
        if self.route == "default_model_placeholder":
            derived.add("placeholder_model_route")
        if self.fixture_plan.golden_records == 0:
            derived.add("no_golden_fixture")
        if "locale_phi_generator" not in self.repository_evidence:
            derived.add("no_locale_phi_generator")
        if self.license_lineage.surrogate_origin == "faker_approximation":
            derived.add("surrogate_locale_approximation")
        return tuple(sorted(derived))

    @property
    def supported_tasks(self) -> tuple[str, ...]:
        """Tasks with a recorded passing graduation report."""
        return tuple(task for task, _digest in self.graduations)

    def task_status(self, task: str) -> str:
        """Return ``supported``, ``experimental`` or ``unsupported`` for a task."""
        if task in self.supported_tasks:
            return SUPPORTED
        if task in self.candidate_tasks:
            return EXPERIMENTAL
        return UNSUPPORTED

    def to_dict(self) -> dict[str, Any]:
        """Return the stable cohort-locale schema without people or text."""
        return {
            "candidate_tasks": list(self.candidate_tasks),
            "fixture_plan": self.fixture_plan.to_dict(),
            "graduations": [
                {"report_digest": digest, "task": task}
                for task, digest in self.graduations
            ],
            "language": self.language,
            "license_lineage": self.license_lineage.to_dict(),
            "limitations": list(self.limitations),
            "locale": self.locale,
            "region": self.region,
            "repository_evidence": list(self.repository_evidence),
            "review_role": self.review_role,
            "reviewer_commitment": self.reviewer_commitment,
            "route": self.route,
            "scripts": list(self.scripts),
            "status": self.status,
            "supported_tasks": list(self.supported_tasks),
        }


@dataclass(frozen=True, slots=True)
class GraduationContract:
    """Thresholds a cohort locale must meet to graduate a task.

    Every field has a policy floor (or cap) that a contract may tighten but
    never loosen, and the critical label set may grow but never shrink.

    Attributes:
        min_documents: Minimum synthetic evaluation documents.
        min_label_support: Minimum gold items per critical label.
        critical_labels: Labels evaluated as non-compensable gates.
        critical_recall_floor: Per-label point-recall floor.
        critical_recall_lower_bound: Per-label Wilson 95% lower-bound floor.
        min_terminology_terms: Minimum reviewed clinical terms.
        min_terminology_coverage: Minimum covered share of reviewed terms.
        min_abstention_probes: Minimum unsupported-input probes; any failure
            to abstain blocks graduation.
        max_cross_lingual_regression: Largest allowed per-cell recall drop on
            a reference language.
        min_cross_lingual_trials: Minimum trials per cross-lingual cell.
        max_package_bytes: Offline package size budget.
        max_review_interval_days: Longest maintenance review interval.
    """

    min_documents: int = MIN_DOCUMENTS_FLOOR
    min_label_support: int = MIN_LABEL_SUPPORT_FLOOR
    critical_labels: tuple[str, ...] = CRITICAL_IDENTIFIER_LABELS
    critical_recall_floor: float = CRITICAL_RECALL_FLOOR
    critical_recall_lower_bound: float = CRITICAL_RECALL_LOWER_BOUND_FLOOR
    min_terminology_terms: int = 200
    min_terminology_coverage: float = 0.90
    min_abstention_probes: int = 50
    max_cross_lingual_regression: float = 0.0
    min_cross_lingual_trials: int = 100
    max_package_bytes: int = 1024**3
    max_review_interval_days: int = 180

    def __post_init__(self) -> None:
        _count(self.min_documents, minimum=MIN_DOCUMENTS_FLOOR)
        _count(self.min_label_support, minimum=MIN_LABEL_SUPPORT_FLOOR)
        labels = _codes(self.critical_labels, pattern=_LABEL)
        if not set(CRITICAL_IDENTIFIER_LABELS) <= set(labels):
            raise ValueError("critical identifier labels cannot be removed")
        object.__setattr__(self, "critical_labels", labels)
        floor = _ratio(
            self.critical_recall_floor, minimum=CRITICAL_RECALL_FLOOR, maximum=1.0
        )
        object.__setattr__(self, "critical_recall_floor", floor)
        object.__setattr__(
            self,
            "critical_recall_lower_bound",
            _ratio(
                self.critical_recall_lower_bound,
                minimum=CRITICAL_RECALL_LOWER_BOUND_FLOOR,
                maximum=floor,
            ),
        )
        _count(self.min_terminology_terms, minimum=MIN_TERMINOLOGY_TERMS_FLOOR)
        object.__setattr__(
            self,
            "min_terminology_coverage",
            _ratio(
                self.min_terminology_coverage,
                minimum=MIN_TERMINOLOGY_COVERAGE_FLOOR,
                maximum=1.0,
            ),
        )
        _count(self.min_abstention_probes, minimum=MIN_ABSTENTION_PROBES_FLOOR)
        object.__setattr__(
            self,
            "max_cross_lingual_regression",
            _ratio(
                self.max_cross_lingual_regression,
                minimum=0.0,
                maximum=MAX_CROSS_LINGUAL_REGRESSION_CAP,
            ),
        )
        _count(self.min_cross_lingual_trials, minimum=MIN_CROSS_LINGUAL_TRIALS_FLOOR)
        _count(self.max_package_bytes, minimum=1, maximum=MAX_PACKAGE_BYTES_CAP)
        _count(
            self.max_review_interval_days,
            minimum=1,
            maximum=MAX_REVIEW_INTERVAL_DAYS_CAP,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable contract schema."""
        return {
            "critical_labels": list(self.critical_labels),
            "critical_recall_floor": self.critical_recall_floor,
            "critical_recall_lower_bound": self.critical_recall_lower_bound,
            "max_cross_lingual_regression": self.max_cross_lingual_regression,
            "max_package_bytes": self.max_package_bytes,
            "max_review_interval_days": self.max_review_interval_days,
            "min_abstention_probes": self.min_abstention_probes,
            "min_cross_lingual_trials": self.min_cross_lingual_trials,
            "min_documents": self.min_documents,
            "min_label_support": self.min_label_support,
            "min_terminology_coverage": self.min_terminology_coverage,
            "min_terminology_terms": self.min_terminology_terms,
        }

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of the canonical contract."""
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class IdentifierRecall:
    """Aggregate recall counts for one critical identifier label."""

    label: str
    detected: int
    missed: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or _LABEL.fullmatch(self.label) is None:
            raise ValueError("invalid identifier label")
        _count(self.detected)
        _count(self.missed)

    @property
    def support(self) -> int:
        """Number of gold items for the label."""
        return self.detected + self.missed

    @property
    def recall(self) -> float:
        """Point recall, or ``0.0`` without support."""
        return self.detected / self.support if self.support else 0.0

    @property
    def lower_bound(self) -> float:
        """Wilson 95% lower bound of recall."""
        return wilson_lower_bound(self.detected, self.support)


@dataclass(frozen=True, slots=True)
class CrossLingualCell:
    """Paired baseline and candidate detections on one reference cell."""

    language: str
    label: str
    trials: int
    baseline_detected: int
    candidate_detected: int

    def __post_init__(self) -> None:
        _language_code(self.language)
        if not isinstance(self.label, str) or _LABEL.fullmatch(self.label) is None:
            raise ValueError("invalid identifier label")
        _count(self.trials)
        _count(self.baseline_detected, maximum=self.trials)
        _count(self.candidate_detected, maximum=self.trials)

    @property
    def delta(self) -> float:
        """Candidate minus baseline recall on the same trials."""
        if not self.trials:
            return 0.0
        return (self.candidate_detected - self.baseline_detected) / self.trials


@dataclass(frozen=True, slots=True)
class GraduationEvidence:
    """Content-free aggregate evidence for one locale and task.

    Digests are 64-character SHA-256 hex references to evidence held
    elsewhere; reviewer identities, notes, spans and source text never appear.
    """

    language: str
    task: str
    documents: int
    identifier_recall: tuple[IdentifierRecall, ...]
    terminology_terms: int
    terminology_covered: int
    abstention_probes: int
    abstention_failures: int
    cross_lingual: tuple[CrossLingualCell, ...]
    fixture_authoring: str
    review_role: str
    review_status: str
    model_tiers: tuple[str, ...]
    package_bytes: int
    offline_verified: bool
    maintainer_role: str
    review_interval_days: int
    lexicon_digest: str
    fixture_digest: str
    evaluation_digest: str
    native_review_digest: str
    package_digest: str

    def __post_init__(self) -> None:
        _language_code(self.language)
        _choice(self.task, TASKS)
        _count(self.documents)
        recalls = _bounded(self.identifier_recall)
        if not all(isinstance(item, IdentifierRecall) for item in recalls):
            raise ValueError("invalid identifier recall evidence")
        if len({item.label for item in recalls}) != len(recalls):
            raise ValueError("duplicate identifier recall label")
        object.__setattr__(
            self, "identifier_recall", tuple(sorted(recalls, key=lambda r: r.label))
        )
        _count(self.terminology_terms)
        _count(self.terminology_covered, maximum=self.terminology_terms)
        _count(self.abstention_probes)
        _count(self.abstention_failures, maximum=self.abstention_probes)
        cells = _bounded(self.cross_lingual)
        if not all(isinstance(cell, CrossLingualCell) for cell in cells):
            raise ValueError("invalid cross-lingual evidence")
        keys = {(cell.language, cell.label) for cell in cells}
        if len(keys) != len(cells):
            raise ValueError("duplicate cross-lingual cell")
        object.__setattr__(
            self,
            "cross_lingual",
            tuple(sorted(cells, key=lambda c: (c.language, c.label))),
        )
        _choice(self.fixture_authoring, FIXTURE_AUTHORING)
        _choice(self.review_role, REVIEW_ROLES)
        _choice(self.review_status, REVIEW_STATUSES)
        object.__setattr__(self, "model_tiers", _codes(self.model_tiers))
        _count(self.package_bytes, maximum=2**53)
        if not isinstance(self.offline_verified, bool):
            raise ValueError("offline verification must be an explicit boolean")
        _choice(self.maintainer_role, REVIEW_ROLES)
        _count(self.review_interval_days, maximum=100_000)
        for name in (
            "lexicon_digest",
            "fixture_digest",
            "evaluation_digest",
            "native_review_digest",
            "package_digest",
        ):
            _digest_ref(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable aggregate evidence schema."""
        return {
            "abstention_failures": self.abstention_failures,
            "abstention_probes": self.abstention_probes,
            "cross_lingual": [
                {
                    "baseline_detected": cell.baseline_detected,
                    "candidate_detected": cell.candidate_detected,
                    "label": cell.label,
                    "language": cell.language,
                    "trials": cell.trials,
                }
                for cell in self.cross_lingual
            ],
            "documents": self.documents,
            "evaluation_digest": self.evaluation_digest,
            "fixture_authoring": self.fixture_authoring,
            "fixture_digest": self.fixture_digest,
            "identifier_recall": [
                {"detected": r.detected, "label": r.label, "missed": r.missed}
                for r in self.identifier_recall
            ],
            "language": self.language,
            "lexicon_digest": self.lexicon_digest,
            "maintainer_role": self.maintainer_role,
            "model_tiers": list(self.model_tiers),
            "native_review_digest": self.native_review_digest,
            "offline_verified": self.offline_verified,
            "package_bytes": self.package_bytes,
            "package_digest": self.package_digest,
            "review_interval_days": self.review_interval_days,
            "review_role": self.review_role,
            "review_status": self.review_status,
            "task": self.task,
            "terminology_covered": self.terminology_covered,
            "terminology_terms": self.terminology_terms,
        }

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of the canonical evidence."""
        return _sha256(self.to_dict())

    def to_readiness_evidence(self) -> tuple[dict[str, str | None], ...]:
        """Project digests onto readiness evidence ``kind/task/digest/model_tier``.

        The rows match the field names of the clinical language readiness
        inventory, so they can be passed to its evidence type without this
        module importing it.
        """
        rows: list[dict[str, str | None]] = [
            {"kind": kind, "task": self.task, "digest": digest, "model_tier": None}
            for kind, digest in (
                ("evaluation", self.evaluation_digest),
                ("fixture", self.fixture_digest),
                ("lexicon", self.lexicon_digest),
                ("native_review", self.native_review_digest),
            )
        ]
        rows.extend(
            {
                "kind": "device",
                "task": self.task,
                "digest": self.package_digest,
                "model_tier": tier,
            }
            for tier in self.model_tiers
        )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class GraduationReport:
    """Deterministic, content-free outcome of one graduation evaluation."""

    language: str
    task: str
    passed: bool
    failures: tuple[str, ...]
    critical_identifiers: tuple[dict[str, Any], ...]
    cross_lingual: tuple[dict[str, Any], ...]
    reference_languages: tuple[str, ...]
    micro_recall: float
    contract_digest: str
    evidence_digest: str

    @property
    def status(self) -> str:
        """``supported`` when every gate passed, otherwise ``experimental``."""
        return SUPPORTED if self.passed else EXPERIMENTAL

    def to_dict(self) -> dict[str, Any]:
        """Return the stable report schema."""
        return {
            "contract_digest": self.contract_digest,
            "critical_identifiers": [dict(row) for row in self.critical_identifiers],
            "cross_lingual": [dict(row) for row in self.cross_lingual],
            # The micro average is informational; decisions use per-label gates.
            "decision_basis": "per_label_non_compensable",
            "evidence_digest": self.evidence_digest,
            "failures": list(self.failures),
            "language": self.language,
            "micro_recall": self.micro_recall,
            "passed": self.passed,
            "reference_languages": list(self.reference_languages),
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "task": self.task,
        }

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of the canonical report."""
        return _sha256(self.to_dict())


def _reference_set(values: Any, candidate: str) -> tuple[str, ...]:
    languages = _codes(values, pattern=_LANGUAGE)
    if not languages:
        raise ValueError("graduation requires at least one reference language")
    if candidate in languages:
        raise ValueError("candidate language cannot be its own reference")
    return languages


def evaluate_graduation(
    entry: CohortLocale,
    evidence: GraduationEvidence,
    *,
    reference_languages: Iterable[str],
    contract: GraduationContract | None = None,
) -> GraduationReport:
    """Evaluate every graduation gate for one cohort locale and task.

    Gates are independent and non-compensable: each critical label must meet
    the point-recall and Wilson lower-bound floors on its own, every
    ``(reference language, critical label)`` cell must stay within the
    regression bound, and slice size, terminology, abstention, review,
    packaging and maintenance gates all apply. The micro-averaged recall is
    reported for context only.

    Args:
        entry: Cohort locale being evaluated.
        evidence: Aggregate evidence for the locale and one task.
        reference_languages: Already-supported languages whose critical recall
            must not regress; must be non-empty and exclude ``entry.language``.
        contract: Thresholds; defaults to :class:`GraduationContract`.

    Returns:
        A deterministic report with controlled failure codes.
    """
    if not isinstance(entry, CohortLocale) or not isinstance(
        evidence, GraduationEvidence
    ):
        raise ValueError("invalid graduation inputs")
    contract = GraduationContract() if contract is None else contract
    if not isinstance(contract, GraduationContract):
        raise ValueError("invalid graduation contract")
    references = _reference_set(reference_languages, entry.language)
    failures: set[str] = set()

    if evidence.language != entry.language:
        failures.add("language_mismatch")
    if evidence.task not in entry.candidate_tasks:
        failures.add("task_not_candidate")
    if entry.route in _ROUTES_WITHOUT_BUNDLED_MODEL:
        failures.add("route_without_bundled_model")
    if entry.reviewer_commitment != "active":
        failures.add("reviewer_commitment_not_active")
    if evidence.review_role != entry.review_role:
        failures.add("review_role_mismatch")
    if evidence.review_status != "approved":
        failures.add("native_review_not_approved")
    if evidence.fixture_authoring not in _GRADUATION_AUTHORING:
        failures.add("translation_only_evidence")
    if evidence.documents < contract.min_documents:
        failures.add("slice_too_small")

    recalls = {item.label: item for item in evidence.identifier_recall}
    label_rows = []
    for label in contract.critical_labels:
        item = recalls.get(label)
        if item is None:
            failures.add(f"critical_label_missing:{label}")
            label_rows.append(
                {
                    "label": label,
                    "passed": False,
                    "recall": None,
                    "recall_lower_bound": None,
                    "support": 0,
                }
            )
            continue
        label_failures = set()
        if item.support < contract.min_label_support:
            label_failures.add(f"critical_label_support_low:{label}")
        if item.recall < contract.critical_recall_floor:
            label_failures.add(f"critical_recall_below_floor:{label}")
        if item.lower_bound < contract.critical_recall_lower_bound:
            label_failures.add(f"critical_recall_uncertain:{label}")
        failures |= label_failures
        label_rows.append(
            {
                "label": label,
                "passed": not label_failures,
                "recall": round(item.recall, 6),
                "recall_lower_bound": round(item.lower_bound, 6),
                "support": item.support,
            }
        )
    critical = [
        recalls[label] for label in contract.critical_labels if label in recalls
    ]
    support = sum(item.support for item in critical)
    micro = sum(item.detected for item in critical) / support if support else 0.0

    if evidence.terminology_terms < contract.min_terminology_terms:
        failures.add("terminology_slice_too_small")
    coverage = (
        evidence.terminology_covered / evidence.terminology_terms
        if evidence.terminology_terms
        else 0.0
    )
    if coverage < contract.min_terminology_coverage:
        failures.add("terminology_coverage_below_floor")
    if evidence.abstention_probes < contract.min_abstention_probes:
        failures.add("abstention_probes_insufficient")
    if evidence.abstention_failures:
        failures.add("abstention_failures")

    cells = {(cell.language, cell.label): cell for cell in evidence.cross_lingual}
    expected = {
        (lang, label) for lang in references for label in contract.critical_labels
    }
    cell_rows = []
    for key in sorted(expected | set(cells)):
        language, label = key
        cell = cells.get(key)
        if cell is None:
            failures.add(f"cross_lingual_cell_missing:{language}:{label}")
            continue
        cell_failures = set()
        if key not in expected:
            cell_failures.add(f"cross_lingual_cell_unexpected:{language}:{label}")
        if cell.trials < contract.min_cross_lingual_trials:
            cell_failures.add(f"cross_lingual_slice_too_small:{language}:{label}")
        if cell.delta < -contract.max_cross_lingual_regression:
            cell_failures.add(f"cross_lingual_regression:{language}:{label}")
        failures |= cell_failures
        cell_rows.append(
            {
                "delta": round(cell.delta, 6),
                "label": label,
                "language": language,
                "passed": not cell_failures,
                "trials": cell.trials,
            }
        )

    if not evidence.model_tiers:
        failures.add("device_tiers_missing")
    if not evidence.offline_verified:
        failures.add("package_not_offline")
    if evidence.package_bytes > contract.max_package_bytes:
        failures.add("package_over_budget")
    if evidence.review_interval_days > contract.max_review_interval_days:
        failures.add("maintenance_interval_too_long")

    return GraduationReport(
        language=entry.language,
        task=evidence.task,
        passed=not failures,
        failures=tuple(sorted(failures)),
        critical_identifiers=tuple(label_rows),
        cross_lingual=tuple(cell_rows),
        reference_languages=references,
        micro_recall=round(micro, 6),
        contract_digest=contract.digest,
        evidence_digest=evidence.digest,
    )


def graduate(
    entry: CohortLocale,
    evidence: GraduationEvidence,
    *,
    reference_languages: Iterable[str],
    contract: GraduationContract | None = None,
) -> tuple[CohortLocale, GraduationReport]:
    """Re-evaluate the contract and return the promoted locale and its report.

    Promotion always recomputes the report, so a hand-built report cannot be
    used to mark a locale supported.

    Raises:
        LanguageGraduationError: ``graduation_failed`` when any gate fails.
    """
    report = evaluate_graduation(
        entry, evidence, reference_languages=reference_languages, contract=contract
    )
    if not report.passed:
        raise LanguageGraduationError("graduation_failed")
    graduations = dict(entry.graduations)
    graduations[report.task] = report.digest
    promoted = replace(
        entry, status=SUPPORTED, graduations=tuple(sorted(graduations.items()))
    )
    return promoted, report


@dataclass(frozen=True)
class LanguageCohort:
    """An immutable, explicitly declared cohort with a fail-closed task gate.

    Attributes:
        entries: Cohort locales; languages must be unique.
        cohort_id: Stable cohort identifier.
        contract: Graduation contract that applies to the cohort.
    """

    entries: tuple[CohortLocale, ...]
    cohort_id: str = COHORT_ID
    contract: GraduationContract = field(default_factory=GraduationContract)
    _index: Mapping[str, CohortLocale] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = _bounded(self.entries, _MAX_COHORT)
        if not entries or not all(isinstance(e, CohortLocale) for e in entries):
            raise ValueError("a cohort requires cohort locale entries")
        index = {entry.language: entry for entry in entries}
        if len(index) != len(entries):
            raise ValueError("duplicate cohort language")
        if not isinstance(self.cohort_id, str) or not _TOKEN.fullmatch(self.cohort_id):
            raise ValueError("invalid cohort identifier")
        if not isinstance(self.contract, GraduationContract):
            raise ValueError("invalid graduation contract")
        ordered = tuple(index[code] for code in sorted(index))
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "_index", index)

    @property
    def languages(self) -> tuple[str, ...]:
        """Sorted cohort language codes."""
        return tuple(entry.language for entry in self.entries)

    def get(self, language: str) -> CohortLocale:
        """Return the cohort locale for a language or locale request.

        Raises:
            LanguageGraduationError: For malformed codes, languages outside
                the cohort, or a region that differs from the recorded locale.
        """
        if not isinstance(language, str) or len(language) > 16:
            raise LanguageGraduationError("invalid_language")
        match = _LANGUAGE_REQUEST.fullmatch(language.strip())
        if match is None:
            raise LanguageGraduationError("invalid_language")
        entry = self._index.get(match[1].lower())
        if entry is None:
            raise LanguageGraduationError("language_not_in_cohort")
        if match[2] is not None:
            recorded = entry.locale.partition("_")[2]
            if match[2].upper() != recorded:
                raise LanguageGraduationError("locale_not_in_cohort")
        return entry

    def task_status(self, language: str, task: str) -> str:
        """Return a task's status, treating anything unknown as unsupported."""
        try:
            entry = self.get(language)
        except LanguageGraduationError:
            return UNSUPPORTED
        return entry.task_status(task) if task in TASKS else UNSUPPORTED

    def require_task(
        self, language: str, task: str, *, allow_experimental: bool = False
    ) -> str:
        """Return the status a task may run under, or fail closed.

        Args:
            language: Language or ``language-REGION`` code.
            task: Task code from :data:`TASKS`.
            allow_experimental: Accept a candidate task that has not graduated
                yet. Its output stays experimental and assistive.

        Returns:
            ``supported`` or, with ``allow_experimental``, ``experimental``.

        Raises:
            LanguageGraduationError: ``unknown_task``, ``task_not_supported``
                or a lookup refusal from :meth:`get`.
        """
        entry = self.get(language)
        if not isinstance(task, str) or task not in TASKS:
            raise LanguageGraduationError("unknown_task")
        status = entry.task_status(task)
        if status == SUPPORTED:
            return SUPPORTED
        if status == EXPERIMENTAL and allow_experimental is True:
            return EXPERIMENTAL
        raise LanguageGraduationError("task_not_supported")

    def with_entry(self, entry: CohortLocale) -> LanguageCohort:
        """Return a new cohort with one existing locale replaced."""
        if not isinstance(entry, CohortLocale) or entry.language not in self._index:
            raise LanguageGraduationError("language_not_in_cohort")
        updated = dict(self._index)
        updated[entry.language] = entry
        return LanguageCohort(tuple(updated.values()), self.cohort_id, self.contract)

    def report(self) -> dict[str, Any]:
        """Return the deterministic cohort report ordered by language."""
        return {
            "cohort_id": self.cohort_id,
            "contract": self.contract.to_dict(),
            "contract_digest": self.contract.digest,
            "locales": [entry.to_dict() for entry in self.entries],
            "schema_version": SCHEMA_VERSION,
        }

    def to_json(self) -> str:
        """Serialize the cohort report as canonical JSON."""
        return _canonical_json(self.report()) + "\n"


def _golden_summary(language: str, root: Path) -> tuple[int, tuple[str, ...]]:
    path = root / f"{language}.jsonl"
    if not path.is_file():
        return 0, ()
    try:
        if path.stat().st_size > _MAX_FIXTURE_BYTES:
            raise ValueError("golden fixture exceeds bound")
        records = 0
        labels: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in islice(handle, _MAX_ITEMS + 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata") or {}
                if (
                    row.get("language") != language
                    or metadata.get("synthetic") is not True
                ):
                    raise ValueError("golden fixture is not a synthetic record")
                records += 1
                labels.update(
                    span.get("label")
                    for span in row.get("gold_spans") or ()
                    if isinstance(span, Mapping)
                )
    except (OSError, ValueError, AttributeError, TypeError):
        raise ValueError("golden fixture metadata is unreadable") from None
    if records > _MAX_ITEMS:
        raise ValueError("golden fixture exceeds bound")
    return records, tuple(
        label for label in CRITICAL_IDENTIFIER_LABELS if label in labels
    )


def repository_language_inventory(
    languages: Iterable[str],
    *,
    fixture_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Derive selection evidence for languages from the local repository.

    Reads the language-pack catalog, the committed model manifest, the
    surrogate-locale coherence report, the synthetic locale generators and the
    golden i18n fixture metadata. Only codes, counts and labels are returned;
    fixture text is never copied. No network or model access is involved.

    Args:
        languages: Primary language codes to describe.
        fixture_root: Optional golden i18n fixture directory override.

    Returns:
        A mapping of language code to inventory fields consumed by
        :func:`cohort_inventory_findings`.
    """
    from openmed.core.anonymizer.locales import locale_coherence_report
    from openmed.core.language_pack import LANGUAGE_PACK_REGISTRY
    from openmed.core.language_pack_catalog import (
        DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
        DEFAULT_PII_MODELS,
        LANG_TO_LOCALE,
        NATIONAL_ID_ONLY_LANGUAGES,
        NATIONAL_ID_PROVIDERS,
        SCRIPT_LANGUAGE_HINTS,
        SUPPORTED_LANGUAGES,
        USER_SUPPLIED_MODEL_LANGUAGES,
    )
    from openmed.core.model_registry import load_manifest_rows
    from openmed.training.synthetic.locale_phi import SUPPORTED_LOCALE_PHI_LANGUAGES

    codes = _codes(languages, pattern=_LANGUAGE)
    root = _GOLDEN_FIXTURE_ROOT if fixture_root is None else Path(fixture_root)
    rows = [
        row
        for row in load_manifest_rows()
        if str(row.get("family") or "").casefold() == "pii"
    ]
    approximate = {
        (row["language"], row["locale"]): bool(row["approximate"])
        for row in locale_coherence_report()
    }
    inventory: dict[str, dict[str, Any]] = {}
    for code in codes:
        evidence = set()
        if code in USER_SUPPLIED_MODEL_LANGUAGES:
            route = "user_supplied_model"
            evidence.add("user_supplied_model_route")
        elif code in NATIONAL_ID_ONLY_LANGUAGES:
            route = "national_id_only"
            evidence.add("national_id_only_route")
        elif code in SUPPORTED_LANGUAGES:
            evidence.add("language_pack")
            if code in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
                route = "default_model_placeholder"
            elif DEFAULT_PII_MODELS.get(code) == _MULTILINGUAL_FALLBACK_MODEL:
                route = "multilingual_fallback"
            else:
                route = "dedicated_checkpoint"
        else:
            route = None
        if code in SUPPORTED_LANGUAGES:
            scripts = tuple(LANGUAGE_PACK_REGISTRY.get(code).scripts)
        else:
            scripts = tuple(
                script
                for script, hints in SCRIPT_LANGUAGE_HINTS.items()
                if code in hints
            )
        declaring = [row for row in rows if code in (row.get("languages") or ())]
        if declaring:
            evidence.add("manifest_model_language")
        if any(
            any("mlx" in str(fmt) for fmt in row.get("formats") or ())
            for row in declaring
        ):
            evidence.add("on_device_mlx_variant")
        if code in NATIONAL_ID_PROVIDERS:
            evidence.add("national_id_validator")
        if code in SUPPORTED_LOCALE_PHI_LANGUAGES:
            evidence.add("locale_phi_generator")
        records, labels = _golden_summary(code, root)
        if records:
            evidence.add("golden_i18n_fixture")
        model_license = None
        tokenizer_unclaimed = False
        if route in {
            "dedicated_checkpoint",
            "default_model_placeholder",
            "multilingual_fallback",
        }:
            model = DEFAULT_PII_MODELS.get(code)
            for row in rows:
                if row.get("repo_id") == model:
                    model_license = normalize_spdx_identifier(
                        row.get("license")
                    ).normalized
                    coverage = row.get("script_coverage") or {}
                    tokenizer_unclaimed = any(
                        (coverage.get(script.casefold()) or {}).get("verdict")
                        == "unclaimed"
                        for script in scripts
                    )
                    break
        locale = LANG_TO_LOCALE.get(code)
        inventory[code] = {
            "golden_labels": labels,
            "golden_records": records,
            "locale": locale,
            "model_license": model_license,
            "repository_evidence": tuple(sorted(evidence)),
            "route": route,
            "scripts": tuple(sorted(scripts)),
            "surrogate_origin": (
                "faker_approximation"
                if approximate.get((code, locale))
                else "faker_native"
            ),
            "tokenizer_unclaimed": tokenizer_unclaimed,
        }
    return inventory


def cohort_inventory_findings(
    cohort: LanguageCohort,
    inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Report cohort declarations that the repository evidence does not back.

    Route, locale, scripts and model license must match exactly. Evidence
    codes, golden record counts and golden labels must not exceed what the
    repository shows, a native surrogate locale must not be claimed for an
    approximation, and an unclaimed tokenizer script must be declared as a
    limitation. Evidence added later (a new fixture, for example) leaves the
    record conservative rather than wrong, so it is not a finding.

    Args:
        cohort: Cohort whose declarations are checked.
        inventory: Mapping shaped like :func:`repository_language_inventory`.

    Returns:
        Sorted ``(language, finding_code)`` pairs; empty when nothing is
        overstated or contradicted.
    """
    if not isinstance(cohort, LanguageCohort) or not isinstance(inventory, Mapping):
        raise ValueError("invalid cohort inventory inputs")
    findings: set[tuple[str, str]] = set()
    for entry in cohort.entries:
        row = inventory.get(entry.language)
        if not isinstance(row, Mapping):
            findings.add((entry.language, "inventory_missing"))
            continue
        records = row.get("golden_records")
        checks = {
            "route_mismatch": entry.route != row.get("route"),
            "locale_mismatch": entry.locale != row.get("locale"),
            "scripts_mismatch": entry.scripts != tuple(row.get("scripts") or ()),
            "model_license_mismatch": (
                entry.license_lineage.model_license != row.get("model_license")
            ),
            "evidence_unbacked": not set(entry.repository_evidence)
            <= set(row.get("repository_evidence") or ()),
            "golden_records_overstated": not isinstance(records, int)
            or entry.fixture_plan.golden_records > records,
            "golden_labels_overstated": not set(entry.fixture_plan.golden_labels)
            <= set(row.get("golden_labels") or ()),
            "surrogate_origin_overstated": (
                entry.license_lineage.surrogate_origin == "faker_native"
                and row.get("surrogate_origin") != "faker_native"
            ),
            "tokenizer_limitation_missing": (
                row.get("tokenizer_unclaimed") is True
                and "tokenizer_script_unclaimed" not in entry.limitations
            ),
        }
        findings.update((entry.language, code) for code, bad in checks.items() if bad)
    return tuple(sorted(findings))


_ALL_CRITICAL = CRITICAL_IDENTIFIER_LABELS
_REPOSITORY_FIXTURES = "synthetic_repository_authored"


def _model_lineage(surrogate: str) -> LicenseLineage:
    return LicenseLineage(
        model_origin="multilingual_fallback",
        model_license="Apache-2.0",
        fixture_origin=_REPOSITORY_FIXTURES,
        fixture_license="Apache-2.0",
        surrogate_origin=surrogate,
    )


def _plan(
    records: int, labels: tuple[str, ...], *stressors: str
) -> SyntheticFixturePlan:
    return SyntheticFixturePlan(
        golden_records=records,
        golden_labels=labels,
        target_documents=MIN_DOCUMENTS_FLOOR,
        authoring="native_authored_synthetic",
        stressors=stressors,
    )


_PACK_EVIDENCE = (
    "golden_i18n_fixture",
    "language_pack",
    "locale_phi_generator",
    "national_id_validator",
)
_MLX_EVIDENCE = _PACK_EVIDENCE + ("manifest_model_language", "on_device_mlx_variant")

# The first cohort, recorded from the repository as of its selection.
# ``cohort_inventory_findings(cohort, repository_language_inventory(...))``
# fails if any declaration overstates the current repository evidence.
# No locale has native-review or evaluation evidence yet, so every locale is
# experimental and every review role is unfilled.
INITIAL_COHORT_LOCALES: tuple[CohortLocale, ...] = (
    CohortLocale(
        language="am",
        locale="am_ET",
        scripts=("Ethiopic",),
        region="horn_of_africa",
        route="multilingual_fallback",
        repository_evidence=_MLX_EVIDENCE,
        license_lineage=_model_lineage("faker_approximation"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            2, _ALL_CRITICAL, "code_switch_english", "ethiopic_numerals"
        ),
        limitations=("surrogate_locale_approximation",),
    ),
    CohortLocale(
        language="sw",
        locale="sw",
        scripts=("Latin",),
        region="east_africa",
        route="multilingual_fallback",
        repository_evidence=_MLX_EVIDENCE,
        license_lineage=_model_lineage("faker_native"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            3, _ALL_CRITICAL, "code_switch_english", "noun_class_prefixes"
        ),
    ),
    CohortLocale(
        language="zu",
        locale="zu_ZA",
        scripts=("Latin",),
        region="southern_africa",
        route="multilingual_fallback",
        repository_evidence=_MLX_EVIDENCE,
        license_lineage=_model_lineage("faker_native"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(1, _ALL_CRITICAL, "attached_affixes", "noun_class_prefixes"),
    ),
    CohortLocale(
        language="or",
        locale="or_IN",
        scripts=("Odia",),
        region="south_asia",
        route="multilingual_fallback",
        repository_evidence=_MLX_EVIDENCE,
        license_lineage=_model_lineage("faker_native"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            2,
            _ALL_CRITICAL,
            "code_switch_english",
            "conjunct_clusters",
            "honorific_names",
            "native_digits",
        ),
    ),
    CohortLocale(
        language="pa",
        locale="pa_IN",
        scripts=("Gurmukhi",),
        region="south_asia",
        route="default_model_placeholder",
        repository_evidence=_PACK_EVIDENCE,
        license_lineage=_model_lineage("faker_approximation"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            2,
            _ALL_CRITICAL,
            "code_switch_english",
            "honorific_names",
            "native_digits",
        ),
        limitations=(
            "placeholder_model_route",
            "surrogate_locale_approximation",
            "tokenizer_script_unclaimed",
        ),
    ),
    CohortLocale(
        language="ur",
        locale="ur_IN",
        scripts=("Arabic",),
        region="south_asia",
        route="default_model_placeholder",
        repository_evidence=_PACK_EVIDENCE,
        license_lineage=_model_lineage("faker_approximation"),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            3,
            _ALL_CRITICAL,
            "code_switch_english",
            "honorific_names",
            "native_digits",
            "rtl_bidi",
        ),
        limitations=("placeholder_model_route", "surrogate_locale_approximation"),
    ),
    CohortLocale(
        language="ne",
        locale="ne_NP",
        scripts=("Devanagari",),
        region="south_asia",
        route="user_supplied_model",
        repository_evidence=("user_supplied_model_route",),
        license_lineage=LicenseLineage(
            model_origin="user_supplied",
            model_license=None,
            fixture_origin="none",
            fixture_license=None,
            surrogate_origin="faker_native",
        ),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            0, (), "attached_affixes", "conjunct_clusters", "native_digits"
        ),
        limitations=(
            "no_bundled_model",
            "no_golden_fixture",
            "no_locale_phi_generator",
        ),
    ),
    CohortLocale(
        language="yo",
        locale="yo_NG",
        scripts=("Latin",),
        region="west_africa",
        route="national_id_only",
        repository_evidence=(
            "golden_i18n_fixture",
            "national_id_only_route",
            "national_id_validator",
        ),
        license_lineage=LicenseLineage(
            model_origin="no_model",
            model_license=None,
            fixture_origin=_REPOSITORY_FIXTURES,
            fixture_license="Apache-2.0",
            surrogate_origin="faker_native",
        ),
        review_role="native_clinical_reviewer",
        reviewer_commitment="unfilled",
        fixture_plan=_plan(
            3,
            ("DATE", "ID_NUM", "PHONE", "STREET_ADDRESS", "ZIPCODE"),
            "decomposed_unicode",
            "tone_diacritics",
        ),
        limitations=(
            "identifier_only_patterns",
            "no_bundled_model",
            "no_locale_phi_generator",
        ),
    ),
)

INITIAL_LOW_RESOURCE_COHORT = LanguageCohort(INITIAL_COHORT_LOCALES)


__all__ = [
    "COHORT_ID",
    "CRITICAL_IDENTIFIER_LABELS",
    "EXPERIMENTAL",
    "GRADUATION_TASKS",
    "INITIAL_COHORT_LOCALES",
    "INITIAL_LOW_RESOURCE_COHORT",
    "REVIEW_ROLES",
    "STATUSES",
    "SUPPORTED",
    "TASKS",
    "UNSUPPORTED",
    "CohortLocale",
    "CrossLingualCell",
    "GraduationContract",
    "GraduationEvidence",
    "GraduationReport",
    "IdentifierRecall",
    "LanguageCohort",
    "LanguageGraduationError",
    "LicenseLineage",
    "SyntheticFixturePlan",
    "cohort_inventory_findings",
    "evaluate_graduation",
    "graduate",
    "repository_language_inventory",
    "wilson_lower_bound",
]
