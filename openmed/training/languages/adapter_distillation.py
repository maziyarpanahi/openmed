"""Language-family adapter distillation plans and edge promotion gates.

Family-transfer adapters (:mod:`openmed.training.adapters`) let one clinical
backbone serve many languages, but loading several adapters side by side can
exceed a phone's memory and latency budget. This module plans the
distillation of selected family adapters into one compact student and decides,
from measured metrics, whether that student may replace its parents.

The plan builds on the Mode-A contract in :mod:`openmed.training.distill`. The
student backbone, critical labels, critical-label loss weight and seed come
from the ``tiny_distill`` preset. Teacher soft labels come from
:func:`~openmed.training.distill.soft_label_distributions`, and per-language
losses go through :func:`~openmed.training.distill.compute_kd_loss`. The gate
reads measured metric records as data and reuses
:func:`~openmed.training.distill.build_distillation_report` for per-label
recall evidence. Device budgets come from :data:`openmed.eval.tiers.TIERS`.

Nothing here trains a model, downloads an artifact or opens a network
connection. Plans and reports carry identifiers, language and label codes,
digests and numbers only, never source text or local paths.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.core.repro_hash import compute_canonical_payload_hash
from openmed.eval.tiers import TIERS
from openmed.training.adapters.config import (
    CLINICAL_ADAPTER_DISCLAIMER,
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    PERMISSIVE_ADAPTER_LICENSES,
    AdapterMetadata,
    FamilyTransferConfig,
    normalize_language_code,
)
from openmed.training.adapters.family_transfer import (
    FamilyTransferAdapterUnavailableError,
    FamilyTransferRouter,
    UnsupportedFamilyTransferLanguageError,
)
from openmed.training.corpus import GATED_DAPT_SOURCES
from openmed.training.distill import (
    DistillationReport,
    EntityTypeWeights,
    KDLossBreakdown,
    build_distillation_report,
    compute_kd_loss,
    soft_label_distributions,
    student_backbone_from_tiny_distill_preset,
)
from openmed.training.recipe import TrainingRecipeConfig, config_hash, load_preset
from openmed.training.synthetic.spdx_identifier import normalize_spdx_identifier

ADAPTER_DISTILLATION_PLAN_SCHEMA_VERSION = (
    "openmed.training.languages.adapter_distillation_plan.v1"
)
ADAPTER_DISTILLATION_GATE_SCHEMA_VERSION = (
    "openmed.training.languages.adapter_distillation_gate.v1"
)

SOURCE_KINDS = ("public", "synthetic")
"""Training-source categories a distillation plan accepts."""

OPEN_DATA_LICENSES = frozenset({"CC-BY-4.0", "CC0-1.0"})
"""Open-data licenses accepted for training sources besides permissive SPDX."""

ROUTE_TARGET_ADAPTER = "target_adapter"
ROUTE_DONOR_FALLBACK = "zero_shot_fallback"
TEACHER_ROLE_PARENT = "parent"
TEACHER_ROLE_DONOR = "donor"

DEFAULT_LANGUAGE_SAMPLING_EXPONENT = 0.3
DEFAULT_MAX_F1_DROP = 0.02

GATE_CHECKS = (
    "measurement_integrity",
    "critical_label_recall",
    "cross_language_interference",
    "artifact_size_vs_parent",
    "peak_memory_budget",
    "latency_p50_budget",
    "latency_p95_budget",
    "latency_p95_vs_parent",
)
"""Gate check codes, in report order."""

_CHECK_ORDER = {check: index for index, check in enumerate(GATE_CHECKS)}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]{0,199}\Z")
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_LANGUAGE = re.compile(r"[a-z]{2,3}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_RESTRICTED_SOURCE_TOKENS = frozenset(GATED_DAPT_SOURCES) | frozenset(
    {"cpt", "physionet", "snomed", "umls"}
)
_MAX_SOURCES = 256
_MAX_LANGUAGES = 64
_MAX_LABELS = 512
_MAX_RECORD_COUNT = 1_000_000_000
_MAX_METRIC_RECORDS = 10_000
_EPSILON = 1e-12
_METRIC_FIELDS = (
    "model_id",
    "language",
    "eval_set_digest",
    "device_profile",
    "label_recall",
    "f1",
    "size_bytes",
    "peak_memory_mb",
    "latency_p50_ms",
    "latency_p95_ms",
)


class AdapterDistillationError(ValueError):
    """Raised when a distillation plan, teacher output or metric is invalid."""


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value.strip()):
        raise AdapterDistillationError(
            f"{field_name} must be an opaque identifier without spaces or schemes"
        )
    return value.strip()


def _label_code(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterDistillationError(f"{field_name} must be a label code")
    normalized = value.strip().upper()
    if not _LABEL.fullmatch(normalized):
        raise AdapterDistillationError(f"{field_name} must be a controlled label code")
    return normalized


def _language(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise AdapterDistillationError(f"{field_name} must be a language code")
    try:
        code = normalize_language_code(value)
    except (TypeError, ValueError):
        raise AdapterDistillationError(
            f"{field_name} must be a language code"
        ) from None
    if not _LANGUAGE.fullmatch(code):
        raise AdapterDistillationError(f"{field_name} must be an ISO 639 code")
    return code


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise AdapterDistillationError(f"{field_name} must be a sha256: digest")
    return value


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterDistillationError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AdapterDistillationError(f"{field_name} must be finite")
    return number


def _fraction(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise AdapterDistillationError(f"{field_name} must be within [0, 1]")
    return number


def _positive(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if number <= 0.0:
        raise AdapterDistillationError(f"{field_name} must be greater than 0")
    return number


def _non_negative(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if number < 0.0:
        raise AdapterDistillationError(f"{field_name} must not be negative")
    return number


def _count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterDistillationError(f"{field_name} must be an integer")
    if not 1 <= value <= _MAX_RECORD_COUNT:
        raise AdapterDistillationError(
            f"{field_name} must be between 1 and {_MAX_RECORD_COUNT}"
        )
    return value


def _training_license(value: object) -> str:
    spdx = normalize_spdx_identifier(value)
    if spdx.is_permissive and spdx.normalized is not None:
        return spdx.normalized
    if isinstance(value, str):
        folded = value.strip().casefold()
        for allowed in sorted(OPEN_DATA_LICENSES):
            if allowed.casefold() == folded:
                return allowed
    raise AdapterDistillationError(
        "training source license must be a permissive SPDX or open-data identifier"
    )


def _references_restricted_corpus(source_id: str) -> bool:
    tokens = set(_TOKEN_SPLIT.split(source_id.casefold()))
    return bool(tokens & _RESTRICTED_SOURCE_TOKENS)


def _is_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


@dataclass(frozen=True)
class LanguageAdapterRoute:
    """Parent adapter selected to serve one language of a plan.

    Attributes:
        language: Normalized language served by the parent adapter.
        adapter_id: Parent adapter identity; the student is gated against it.
        adapter_language: Language the parent adapter was trained for.
        backbone_model_id: Backbone the parent adapter attaches to.
        route_mode: ``"target_adapter"`` for a language-specific adapter or
            ``"zero_shot_fallback"`` for a family donor adapter.
    """

    language: str
    adapter_id: str
    adapter_language: str
    backbone_model_id: str
    route_mode: str

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic, path-free route payload."""

        return {
            "adapter_id": self.adapter_id,
            "adapter_language": self.adapter_language,
            "backbone_model_id": self.backbone_model_id,
            "language": self.language,
            "route_mode": self.route_mode,
        }


@dataclass(frozen=True)
class FamilyAdapterSelection:
    """Parent adapters chosen from one language family for distillation."""

    family_id: str
    routes: tuple[LanguageAdapterRoute, ...]

    def __post_init__(self) -> None:
        """Order routes by language and reject duplicate languages."""

        routes = tuple(sorted(self.routes, key=lambda route: route.language))
        if not routes:
            raise AdapterDistillationError("selection must contain a language")
        languages = [route.language for route in routes]
        if len(set(languages)) != len(languages):
            raise AdapterDistillationError("selection languages must be unique")
        object.__setattr__(self, "routes", routes)

    @property
    def languages(self) -> tuple[str, ...]:
        """Return the selected languages in sorted order."""

        return tuple(route.language for route in self.routes)

    @property
    def parent_adapter_ids(self) -> tuple[str, ...]:
        """Return unique parent adapter identities in sorted order."""

        return tuple(sorted({route.adapter_id for route in self.routes}))

    def route_for(self, language: str) -> LanguageAdapterRoute:
        """Return the parent route for ``language``.

        Raises:
            AdapterDistillationError: If the language is not selected.
        """

        code = _language(language, "language")
        for route in self.routes:
            if route.language == code:
                return route
        raise AdapterDistillationError("language is not part of the selection")

    def languages_for_parent(self, adapter_id: str) -> tuple[str, ...]:
        """Return the languages a parent adapter serves in this selection."""

        return tuple(
            route.language for route in self.routes if route.adapter_id == adapter_id
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic selection metadata."""

        return {
            "family_id": self.family_id,
            "parent_adapter_ids": list(self.parent_adapter_ids),
            "routes": [route.to_dict() for route in self.routes],
        }


def select_family_adapters(
    family_id: str,
    available_adapters: Mapping[str, AdapterMetadata],
    *,
    languages: Sequence[str] | None = None,
    allow_donor_fallback: bool = False,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> FamilyAdapterSelection:
    """Select the parent adapters of one language family.

    Routing reuses :class:`~openmed.training.adapters.FamilyTransferRouter`,
    so a language is served either by its own installed adapter or, when
    ``allow_donor_fallback`` is true, by the configured family donor.

    Args:
        family_id: Family from the transfer taxonomy, for example ``"indic"``.
        available_adapters: Installed adapter metadata keyed by language.
        languages: Languages to distill. Defaults to every family language
            with its own installed adapter.
        allow_donor_fallback: Allow explicitly requested languages without
            their own adapter to be served by a family donor adapter.
        config: Family taxonomy and transfer graph.

    Returns:
        Parent routes ordered by language.

    Raises:
        AdapterDistillationError: If the family is unknown, a language is
            outside the family or cannot be served, or a parent adapter is
            not permissively licensed and offline-runnable.
    """

    if not isinstance(config, FamilyTransferConfig):
        raise AdapterDistillationError("config must be a FamilyTransferConfig")
    if not isinstance(family_id, str):
        raise AdapterDistillationError("family_id must be a string")
    family = config.families.get(family_id.strip().casefold())
    if family is None:
        raise AdapterDistillationError("family_id is not in the transfer taxonomy")
    try:
        router = FamilyTransferRouter(available_adapters, config=config)
    except (TypeError, ValueError) as exc:
        raise AdapterDistillationError(
            "available_adapters must map unique languages to AdapterMetadata"
        ) from exc

    if languages is None:
        installed = set(router.available_languages)
        requested = tuple(lang for lang in family.languages if lang in installed)
        if not requested:
            raise AdapterDistillationError(
                "no installed adapter serves a language of the selected family"
            )
    else:
        if isinstance(languages, (str, bytes)):
            raise AdapterDistillationError("languages must be a sequence of codes")
        requested = tuple(_language(lang, "languages") for lang in languages)
        if not requested:
            raise AdapterDistillationError("languages must not be empty")
        if len(set(requested)) != len(requested):
            raise AdapterDistillationError("languages must not contain duplicates")
        if len(requested) > _MAX_LANGUAGES:
            raise AdapterDistillationError("too many languages for one student")
        for lang in requested:
            if lang not in family.languages:
                raise AdapterDistillationError(
                    f"language {lang!r} is outside family {family.family_id!r}"
                )

    routes = []
    for lang in sorted(requested):
        try:
            route = router.route(lang)
        except (
            FamilyTransferAdapterUnavailableError,
            UnsupportedFamilyTransferLanguageError,
        ):
            raise AdapterDistillationError(
                f"no installed or donor adapter can serve language {lang!r}"
            ) from None
        if route.mode != ROUTE_TARGET_ADAPTER and not allow_donor_fallback:
            raise AdapterDistillationError(
                f"language {lang!r} has no installed adapter; pass "
                "allow_donor_fallback=True to distill from its family donor"
            )
        _require_distillable_adapter(route.adapter)
        routes.append(
            LanguageAdapterRoute(
                language=lang,
                adapter_id=_identifier(route.adapter.adapter_id, "adapter_id"),
                adapter_language=route.adapter_language,
                backbone_model_id=route.backbone_model_id,
                route_mode=route.mode,
            )
        )
    return FamilyAdapterSelection(family_id=family.family_id, routes=tuple(routes))


def _require_distillable_adapter(adapter: AdapterMetadata) -> None:
    if adapter.license not in PERMISSIVE_ADAPTER_LICENSES:
        raise AdapterDistillationError(
            "teacher adapters must carry a permissive adapter license"
        )
    if not adapter.offline_runnable:
        raise AdapterDistillationError("teacher adapters must be offline-runnable")


@dataclass(frozen=True)
class TeacherAssignment:
    """One teacher of a language's ensemble with its normalized weight."""

    language: str
    adapter_id: str
    role: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic teacher metadata."""

        return {
            "adapter_id": self.adapter_id,
            "language": self.language,
            "role": self.role,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class DistillationSource:
    """A local synthetic or public training source, described by counts only.

    The source is referenced by an opaque identity and a manifest digest. No
    path, URL or record content enters the plan.

    Attributes:
        source_id: Opaque source identity. Identities naming restricted
            corpora such as MIMIC, i2b2, n2c2, UMLS or SNOMED are refused.
        kind: ``"synthetic"`` or ``"public"``.
        license: Permissive SPDX identifier, or ``CC0-1.0`` / ``CC-BY-4.0``.
        content_digest: ``sha256:`` digest of the local source manifest.
        record_counts: Record count per language code.
    """

    source_id: str
    kind: str
    license: str
    content_digest: str
    record_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate identity, provenance and per-language counts."""

        source_id = _identifier(self.source_id, "source_id")
        if _references_restricted_corpus(source_id):
            raise AdapterDistillationError(
                "restricted or DUA-gated corpora cannot be distillation sources"
            )
        if self.kind not in SOURCE_KINDS:
            raise AdapterDistillationError("kind must be 'synthetic' or 'public'")
        if not isinstance(self.record_counts, Mapping) or not self.record_counts:
            raise AdapterDistillationError("record_counts must be a non-empty mapping")
        counts: dict[str, int] = {}
        for raw_language, raw_count in self.record_counts.items():
            language = _language(raw_language, "record_counts language")
            if language in counts:
                raise AdapterDistillationError(
                    "record_counts must not repeat a language"
                )
            counts[language] = _count(raw_count, "record_counts value")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "license", _training_license(self.license))
        object.__setattr__(
            self, "content_digest", _digest(self.content_digest, "content_digest")
        )
        object.__setattr__(
            self, "record_counts", MappingProxyType(dict(sorted(counts.items())))
        )

    @property
    def languages(self) -> tuple[str, ...]:
        """Return the languages this source contributes records for."""

        return tuple(self.record_counts)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, path-free source metadata."""

        return {
            "content_digest": self.content_digest,
            "kind": self.kind,
            "license": self.license,
            "record_counts": dict(self.record_counts),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class DistillationShard:
    """One source-by-language slice of the student's training mixture."""

    source_id: str
    kind: str
    language: str
    record_count: int
    sampling_probability: float
    teacher_adapter_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic shard metadata."""

        return {
            "kind": self.kind,
            "language": self.language,
            "record_count": self.record_count,
            "sampling_probability": self.sampling_probability,
            "source_id": self.source_id,
            "teacher_adapter_ids": list(self.teacher_adapter_ids),
        }


@dataclass(frozen=True)
class DistillationGatePolicy:
    """Promotion thresholds pinned into a plan before any student is trained.

    Device budgets come from :data:`openmed.eval.tiers.TIERS`; build policies
    with :meth:`from_tier` instead of restating those numbers.

    Attributes:
        tier: Device tier name, optionally ``"Tier/SubTier"``.
        max_peak_memory_mb: Resident memory budget for the student.
        max_latency_p50_ms: Student p50 latency budget per language.
        max_latency_p95_ms: Student p95 latency budget per language.
        critical_recall_tolerance: Allowed per-label critical recall drop
            relative to the parent adapter. Zero by default.
        max_f1_drop: Allowed per-language F1 drop relative to the parent.
        max_size_ratio: Student artifact size limit as a ratio of each
            parent's deployable size.
        max_latency_ratio: Student p95 latency limit as a ratio of the
            parent's p95 latency on the same language.
    """

    tier: str
    max_peak_memory_mb: float
    max_latency_p50_ms: float
    max_latency_p95_ms: float
    critical_recall_tolerance: float = 0.0
    max_f1_drop: float = DEFAULT_MAX_F1_DROP
    max_size_ratio: float = 1.0
    max_latency_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Validate finite, non-negative thresholds."""

        if not isinstance(self.tier, str) or not self.tier.strip():
            raise AdapterDistillationError("tier must be a non-empty string")
        for name in ("max_peak_memory_mb", "max_latency_p50_ms", "max_latency_p95_ms"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.max_latency_p95_ms < self.max_latency_p50_ms:
            raise AdapterDistillationError(
                "max_latency_p95_ms must not be below max_latency_p50_ms"
            )
        object.__setattr__(
            self,
            "critical_recall_tolerance",
            _fraction(self.critical_recall_tolerance, "critical_recall_tolerance"),
        )
        object.__setattr__(
            self, "max_f1_drop", _fraction(self.max_f1_drop, "max_f1_drop")
        )
        for name in ("max_size_ratio", "max_latency_ratio"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    @classmethod
    def from_tier(
        cls,
        tier: str = "Tiny",
        *,
        sub_tier: str | None = None,
        critical_recall_tolerance: float = 0.0,
        max_f1_drop: float = DEFAULT_MAX_F1_DROP,
        max_size_ratio: float = 1.0,
        max_latency_ratio: float = 1.0,
    ) -> DistillationGatePolicy:
        """Build a policy whose device budgets come from the tier table.

        Args:
            tier: Tier name from :data:`openmed.eval.tiers.TIERS`, matched
                case-insensitively (``"tiny"`` resolves to ``"Tiny"``).
            sub_tier: Optional sub-tier such as ``"Nano"``.
            critical_recall_tolerance: Allowed critical recall drop.
            max_f1_drop: Allowed per-language F1 drop.
            max_size_ratio: Student-to-parent artifact size ratio limit.
            max_latency_ratio: Student-to-parent p95 latency ratio limit.

        Returns:
            A validated gate policy.

        Raises:
            AdapterDistillationError: If the tier or sub-tier is unknown.
        """

        if not isinstance(tier, str):
            raise AdapterDistillationError("tier must be a string")
        matches = [name for name in TIERS if name.casefold() == tier.casefold()]
        if not matches:
            raise AdapterDistillationError("tier is not in the device tier table")
        name = matches[0]
        row: Mapping[str, Any] = TIERS[name]
        if sub_tier is not None:
            sub_rows = row.get("sub_tiers")
            if not isinstance(sub_tier, str) or not isinstance(sub_rows, Mapping):
                raise AdapterDistillationError("sub_tier is not defined for the tier")
            sub_matches = [
                sub for sub in sub_rows if sub.casefold() == sub_tier.casefold()
            ]
            if not sub_matches:
                raise AdapterDistillationError("sub_tier is not defined for the tier")
            name = f"{name}/{sub_matches[0]}"
            row = sub_rows[sub_matches[0]]
        return cls(
            tier=name,
            max_peak_memory_mb=float(row["ram_mb_max"]),
            max_latency_p50_ms=float(row["p50_ms_max"]),
            max_latency_p95_ms=float(row["p95_ms_max"]),
            critical_recall_tolerance=critical_recall_tolerance,
            max_f1_drop=max_f1_drop,
            max_size_ratio=max_size_ratio,
            max_latency_ratio=max_latency_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic gate thresholds."""

        return {
            "critical_recall_tolerance": self.critical_recall_tolerance,
            "max_f1_drop": self.max_f1_drop,
            "max_latency_p50_ms": self.max_latency_p50_ms,
            "max_latency_p95_ms": self.max_latency_p95_ms,
            "max_latency_ratio": self.max_latency_ratio,
            "max_peak_memory_mb": self.max_peak_memory_mb,
            "max_size_ratio": self.max_size_ratio,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class AdapterDistillationPlan:
    """Teacher-ensemble to student plan for one language family.

    Build plans with :func:`build_adapter_distillation_plan`. The plan pins
    the gate policy, so promotion thresholds are fixed before training.
    """

    plan_id: str
    selection: FamilyAdapterSelection
    teachers: tuple[TeacherAssignment, ...]
    sources: tuple[DistillationSource, ...]
    student_id: str
    student_backbone: str
    student_license: str
    recipe_preset: str
    recipe_hash: str
    seed: int
    temperature: float
    alpha: float
    span_loss_weight: float
    critical_labels: tuple[str, ...]
    critical_label_weight: float
    language_sampling_exponent: float
    gate: DistillationGatePolicy
    disclaimer: str = CLINICAL_ADAPTER_DISCLAIMER
    schema_version: str = ADAPTER_DISTILLATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Check teacher and source coverage for every selected language."""

        if not isinstance(self.selection, FamilyAdapterSelection):
            raise AdapterDistillationError("selection must be FamilyAdapterSelection")
        if not isinstance(self.gate, DistillationGatePolicy):
            raise AdapterDistillationError("gate must be DistillationGatePolicy")
        if any(
            not isinstance(teacher, TeacherAssignment)
            or teacher.role not in (TEACHER_ROLE_PARENT, TEACHER_ROLE_DONOR)
            or _fraction(teacher.weight, "teacher weight") != teacher.weight
            for teacher in self.teachers
        ):
            raise AdapterDistillationError("teachers must be TeacherAssignment records")
        if any(not isinstance(source, DistillationSource) for source in self.sources):
            raise AdapterDistillationError("sources must be DistillationSource records")
        languages = set(self.selection.languages)
        for language in self.selection.languages:
            parents = [
                teacher
                for teacher in self.teachers
                if teacher.language == language and teacher.role == TEACHER_ROLE_PARENT
            ]
            if [teacher.adapter_id for teacher in parents] != [
                self.selection.route_for(language).adapter_id
            ]:
                raise AdapterDistillationError(
                    f"language {language!r} must have exactly its parent teacher"
                )
            weight = sum(t.weight for t in self.teachers if t.language == language)
            if not math.isclose(weight, 1.0, abs_tol=1e-9):
                raise AdapterDistillationError(
                    f"teacher weights for {language!r} must sum to 1"
                )
        if any(teacher.language not in languages for teacher in self.teachers):
            raise AdapterDistillationError("teachers must serve selected languages")
        if not self.sources:
            raise AdapterDistillationError("a plan needs at least one source")
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise AdapterDistillationError("source_id values must be unique")
        for source in self.sources:
            if not set(source.languages) <= languages:
                raise AdapterDistillationError(
                    f"source {source.source_id!r} declares records for a "
                    "language outside the plan"
                )
        for language in self.selection.languages:
            if not any(
                source.kind == "synthetic" and language in source.record_counts
                for source in self.sources
            ):
                raise AdapterDistillationError(
                    f"language {language!r} needs synthetic training coverage"
                )
        if self.student_id in self.selection.parent_adapter_ids:
            raise AdapterDistillationError("student_id must differ from every parent")
        if not self.critical_labels:
            raise AdapterDistillationError("critical_labels must not be empty")

    @property
    def family_id(self) -> str:
        """Return the distilled language family."""

        return self.selection.family_id

    @property
    def languages(self) -> tuple[str, ...]:
        """Return the languages the student must serve."""

        return self.selection.languages

    @property
    def parent_adapter_ids(self) -> tuple[str, ...]:
        """Return the parent adapters the student is gated against."""

        return self.selection.parent_adapter_ids

    def parent_for(self, language: str) -> str:
        """Return the parent adapter that serves ``language`` today."""

        return self.selection.route_for(language).adapter_id

    def teachers_for(self, language: str) -> tuple[TeacherAssignment, ...]:
        """Return the ensemble that labels ``language`` records."""

        code = _language(language, "language")
        if code not in self.languages:
            raise AdapterDistillationError("language is not part of the plan")
        return tuple(teacher for teacher in self.teachers if teacher.language == code)

    def entity_type_weights(self) -> EntityTypeWeights:
        """Return Mode-A loss weights that up-weight the critical labels."""

        return EntityTypeWeights(
            weights={
                label: self.critical_label_weight for label in self.critical_labels
            }
        )

    def language_record_counts(self) -> dict[str, int]:
        """Return total planned records per language across all sources."""

        totals = {language: 0 for language in self.languages}
        for source in self.sources:
            for language, count in source.record_counts.items():
                totals[language] += count
        return totals

    def language_sampling_weights(self) -> dict[str, float]:
        """Return exponent-smoothed language sampling probabilities.

        Each language is sampled with probability proportional to
        ``records ** language_sampling_exponent``. Exponents below one shift
        mass toward low-resource languages so high-resource parents do not
        dominate the shared student.
        """

        totals = self.language_record_counts()
        scaled = {
            language: float(count) ** self.language_sampling_exponent
            for language, count in totals.items()
        }
        denominator = math.fsum(scaled.values())
        return {language: value / denominator for language, value in scaled.items()}

    def shards(self) -> tuple[DistillationShard, ...]:
        """Return the source-by-language training mixture.

        Within a language, a source's share is proportional to its records.
        Shard probabilities sum to one across the whole plan.
        """

        totals = self.language_record_counts()
        weights = self.language_sampling_weights()
        shards = []
        for source in self.sources:
            for language, count in source.record_counts.items():
                shards.append(
                    DistillationShard(
                        source_id=source.source_id,
                        kind=source.kind,
                        language=language,
                        record_count=count,
                        sampling_probability=weights[language]
                        * count
                        / totals[language],
                        teacher_adapter_ids=tuple(
                            teacher.adapter_id
                            for teacher in self.teachers_for(language)
                        ),
                    )
                )
        return tuple(sorted(shards, key=lambda s: (s.language, s.source_id)))

    def _payload(self) -> dict[str, Any]:
        totals = self.language_record_counts()
        synthetic = {language: 0 for language in self.languages}
        for source in self.sources:
            if source.kind == "synthetic":
                for language, count in source.record_counts.items():
                    synthetic[language] += count
        return {
            "disclaimer": self.disclaimer,
            "family_id": self.family_id,
            "gate": self.gate.to_dict(),
            "kd": {
                "alpha": self.alpha,
                "critical_label_weight": self.critical_label_weight,
                "critical_labels": list(self.critical_labels),
                "span_loss_weight": self.span_loss_weight,
                "temperature": self.temperature,
            },
            "language_sampling": {
                "exponent": self.language_sampling_exponent,
                "record_counts": totals,
                "synthetic_share": {
                    language: synthetic[language] / totals[language]
                    for language in self.languages
                },
                "weights": self.language_sampling_weights(),
            },
            "languages": list(self.languages),
            "network_accessed": False,
            "plan_id": self.plan_id,
            "schema_version": self.schema_version,
            "selection": self.selection.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards()],
            "sources": [source.to_dict() for source in self.sources],
            "student": {
                "backbone": self.student_backbone,
                "license": self.student_license,
                "recipe_hash": self.recipe_hash,
                "recipe_preset": self.recipe_preset,
                "seed": self.seed,
                "student_id": self.student_id,
            },
            "teachers": [teacher.to_dict() for teacher in self.teachers],
            "training_started": False,
        }

    @property
    def plan_digest(self) -> str:
        """Return the canonical ``sha256:`` digest of the plan payload."""

        return compute_canonical_payload_hash(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic plan payload with its digest."""

        payload = self._payload()
        payload["plan_digest"] = compute_canonical_payload_hash(payload)
        return payload

    def to_json(self) -> str:
        """Return deterministic plan JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write_json(self, path: str | Path) -> Path:
        """Write deterministic plan JSON to ``path``."""

        return _write_text(path, self.to_json())


def build_adapter_distillation_plan(
    *,
    plan_id: str,
    family_id: str,
    available_adapters: Mapping[str, AdapterMetadata],
    sources: Sequence[DistillationSource],
    student_id: str,
    languages: Sequence[str] | None = None,
    allow_donor_fallback: bool = False,
    donor_teacher_weight: float = 0.0,
    recipe: TrainingRecipeConfig | None = None,
    student_backbone: str | None = None,
    student_license: str = "apache-2.0",
    temperature: float = 2.0,
    alpha: float = 0.5,
    span_loss_weight: float = 1.0,
    critical_labels: Sequence[str] | None = None,
    critical_label_weight: float | None = None,
    language_sampling_exponent: float = DEFAULT_LANGUAGE_SAMPLING_EXPONENT,
    gate: DistillationGatePolicy | None = None,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> AdapterDistillationPlan:
    """Plan the distillation of family adapters into one compact student.

    Each selected language is labeled by its parent adapter. With a positive
    ``donor_teacher_weight``, installed family donors from the transfer graph
    join that language's ensemble as auxiliary teachers. Defaults for the
    student backbone, critical labels, critical-label weight, seed and device
    tier come from the ``tiny_distill`` preset.

    Args:
        plan_id: Stable identity for this plan.
        family_id: Language family to distill.
        available_adapters: Installed adapter metadata keyed by language.
        sources: Synthetic and public training sources. Every language needs
            at least one synthetic source.
        student_id: Identity of the planned student artifact.
        languages: Languages to distill; see :func:`select_family_adapters`.
        allow_donor_fallback: Allow donor adapters to serve requested
            languages that lack their own adapter.
        donor_teacher_weight: Relative weight of auxiliary donor teachers.
            Zero keeps one teacher per language.
        recipe: Mode-A recipe; defaults to the ``tiny_distill`` preset.
        student_backbone: Student backbone; defaults to the preset backbone.
        student_license: Permissive license of the planned student.
        temperature: KD temperature.
        alpha: Teacher-loss mixing weight in ``[0, 1]``.
        span_loss_weight: Span-transfer weight inside the teacher loss.
        critical_labels: Labels gated non-compensably. Defaults to the
            preset's critical labels.
        critical_label_weight: Loss weight for critical labels, at least 1.
            Defaults to the preset's value.
        language_sampling_exponent: Exponent in ``(0, 1]`` for language
            sampling; lower values favor low-resource languages.
        gate: Promotion policy; defaults to the preset's output tier.
        config: Family taxonomy and transfer graph.

    Returns:
        A validated, deterministic distillation plan.

    Raises:
        AdapterDistillationError: If any selection, source, teacher or
            hyperparameter constraint fails.
    """

    plan_id = _identifier(plan_id, "plan_id")
    student_id = _identifier(student_id, "student_id")
    selection = select_family_adapters(
        family_id,
        available_adapters,
        languages=languages,
        allow_donor_fallback=allow_donor_fallback,
        config=config,
    )
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise AdapterDistillationError("sources must be a sequence")
    if len(sources) > _MAX_SOURCES:
        raise AdapterDistillationError("too many training sources")
    if any(not isinstance(source, DistillationSource) for source in sources):
        raise AdapterDistillationError("sources must be DistillationSource records")

    recipe = recipe if recipe is not None else load_preset("tiny_distill")
    if not isinstance(recipe, TrainingRecipeConfig):
        raise AdapterDistillationError("recipe must be a TrainingRecipeConfig")
    if student_backbone is None:
        try:
            student_backbone = student_backbone_from_tiny_distill_preset(recipe)
        except ValueError:
            raise AdapterDistillationError(
                "student_backbone is required unless the recipe is the Mode-A "
                "tiny_distill preset"
            ) from None
    student_backbone = _identifier(student_backbone, "student_backbone")
    if not isinstance(student_license, str) or (
        student_license.strip().casefold() not in PERMISSIVE_ADAPTER_LICENSES
    ):
        raise AdapterDistillationError("student_license must be permissive")

    labels = tuple(
        sorted(
            {
                _label_code(label, "critical_labels")
                for label in (
                    recipe.loss.critical_labels
                    if critical_labels is None
                    else critical_labels
                )
            }
        )
    )
    if len(labels) > _MAX_LABELS:
        raise AdapterDistillationError("too many critical labels")
    label_weight = _number(
        recipe.loss.critical_label_weight
        if critical_label_weight is None
        else critical_label_weight,
        "critical_label_weight",
    )
    if label_weight < 1.0:
        raise AdapterDistillationError("critical_label_weight must be at least 1")
    temperature = _positive(temperature, "temperature")
    alpha = _fraction(alpha, "alpha")
    span_loss_weight = _non_negative(span_loss_weight, "span_loss_weight")
    exponent = _fraction(language_sampling_exponent, "language_sampling_exponent")
    if exponent == 0.0:
        raise AdapterDistillationError("language_sampling_exponent must be above 0")
    donor_weight = _non_negative(donor_teacher_weight, "donor_teacher_weight")
    if gate is None:
        gate = DistillationGatePolicy.from_tier(recipe.output_tier)

    return AdapterDistillationPlan(
        plan_id=plan_id,
        selection=selection,
        teachers=_assign_teachers(
            selection,
            available_adapters,
            donor_weight=donor_weight,
            config=config,
        ),
        sources=tuple(sorted(sources, key=lambda source: source.source_id)),
        student_id=student_id,
        student_backbone=student_backbone,
        student_license=student_license.strip().casefold(),
        recipe_preset=recipe.preset_name,
        recipe_hash=config_hash(recipe),
        seed=recipe.seed,
        temperature=temperature,
        alpha=alpha,
        span_loss_weight=span_loss_weight,
        critical_labels=labels,
        critical_label_weight=label_weight,
        language_sampling_exponent=exponent,
        gate=gate,
    )


def _assign_teachers(
    selection: FamilyAdapterSelection,
    available_adapters: Mapping[str, AdapterMetadata],
    *,
    donor_weight: float,
    config: FamilyTransferConfig,
) -> tuple[TeacherAssignment, ...]:
    installed = {
        normalize_language_code(language): adapter
        for language, adapter in available_adapters.items()
    }
    teachers: list[TeacherAssignment] = []
    for route in selection.routes:
        donors: list[str] = []
        if donor_weight > 0.0:
            for edge in config.donor_edges_for(route.language):
                adapter = installed.get(edge.donor_language)
                if adapter is None:
                    continue
                if adapter.backbone_model_id != edge.adapter.backbone_model_id:
                    continue
                adapter_id = _identifier(adapter.adapter_id, "adapter_id")
                if adapter_id == route.adapter_id or adapter_id in donors:
                    continue
                _require_distillable_adapter(adapter)
                donors.append(adapter_id)
        total = 1.0 + donor_weight * len(donors)
        teachers.append(
            TeacherAssignment(
                language=route.language,
                adapter_id=route.adapter_id,
                role=TEACHER_ROLE_PARENT,
                weight=1.0 / total,
            )
        )
        teachers.extend(
            TeacherAssignment(
                language=route.language,
                adapter_id=adapter_id,
                role=TEACHER_ROLE_DONOR,
                weight=donor_weight / total,
            )
            for adapter_id in donors
        )
    return tuple(teachers)


def ensemble_teacher_soft_labels(
    teacher_logits: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
    temperature: float,
) -> Any:
    """Average temperature-scaled teacher distributions with ensemble weights.

    Args:
        teacher_logits: Token logits per teacher adapter identity, each with
            shape ``batch x seq x labels``.
        weights: Non-negative weight per teacher; normalized internally.
        temperature: KD temperature applied before averaging.

    Returns:
        Ensemble soft labels with the teachers' shape. Nested tuples for
        Python inputs, or a tensor when every teacher returns tensors.

    Raises:
        AdapterDistillationError: If teachers and weights differ, weights are
            invalid, or teacher outputs disagree on shape.
    """

    if not isinstance(teacher_logits, Mapping) or not teacher_logits:
        raise AdapterDistillationError("teacher_logits must be a non-empty mapping")
    if not isinstance(weights, Mapping) or set(weights) != set(teacher_logits):
        raise AdapterDistillationError("weights must cover exactly the teachers")
    normalized = {
        teacher: _non_negative(weight, "teacher weight")
        for teacher, weight in weights.items()
    }
    total = math.fsum(normalized.values())
    if total <= 0.0:
        raise AdapterDistillationError("teacher weights must not all be zero")
    order = sorted(teacher_logits)
    distributions = [
        (
            soft_label_distributions(teacher_logits[teacher], temperature=temperature),
            normalized[teacher] / total,
        )
        for teacher in order
    ]
    if all(_is_tensor(distribution) for distribution, _ in distributions):
        return sum(  # pragma: no cover - optional torch path
            (distribution * weight for distribution, weight in distributions[1:]),
            distributions[0][0] * distributions[0][1],
        )
    if any(_is_tensor(distribution) for distribution, _ in distributions):
        raise AdapterDistillationError("teacher outputs must share one array type")
    return _weighted_sum(distributions)


def _weighted_sum(
    distributions: Sequence[tuple[Any, float]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    reference = distributions[0][0]
    shape = [len(sample) for sample in reference]
    widths = {len(token) for sample in reference for token in sample}
    for distribution, _ in distributions:
        if [len(sample) for sample in distribution] != shape or {
            len(token) for sample in distribution for token in sample
        } != widths:
            raise AdapterDistillationError(
                "teacher outputs must share the student label space and shape"
            )
    if len(widths) > 1:
        raise AdapterDistillationError("teacher outputs must have one label width")
    return tuple(
        tuple(
            tuple(
                math.fsum(
                    float(distribution[sample][token][label]) * weight
                    for distribution, weight in distributions
                )
                for label in range(len(reference[sample][token]))
            )
            for token in range(len(reference[sample]))
        )
        for sample in range(len(reference))
    )


def compute_language_kd_loss(
    plan: AdapterDistillationPlan,
    *,
    language: str,
    student_logits: Any,
    hard_labels: Any,
    teacher_logits: Mapping[str, Any],
    id2label: Mapping[int | str, str] | None = None,
    mask: Any | None = None,
) -> KDLossBreakdown:
    """Compute the Mode-A KD loss for one language batch of a plan.

    The batch must be labeled by exactly the language's planned teachers. A
    teacher from another language, or a missing teacher, is refused so one
    language's adapter never silently supervises another language.

    Args:
        plan: Distillation plan.
        language: Language of the batch.
        student_logits: Student token logits, ``batch x seq x labels``.
        hard_labels: Integer gold labels, ``batch x seq``.
        teacher_logits: Token logits per planned teacher adapter identity.
        id2label: Student label map; needed to apply the critical-label
            weight to the right rows.
        mask: Optional boolean token mask.

    Returns:
        Loss components from :func:`openmed.training.distill.compute_kd_loss`.

    Raises:
        AdapterDistillationError: If the language or teachers do not match
            the plan, or teacher outputs disagree on shape.
    """

    if not isinstance(plan, AdapterDistillationPlan):
        raise AdapterDistillationError("plan must be an AdapterDistillationPlan")
    assignments = plan.teachers_for(language)
    weights = {teacher.adapter_id: teacher.weight for teacher in assignments}
    if not isinstance(teacher_logits, Mapping) or set(teacher_logits) != set(weights):
        raise AdapterDistillationError(
            "teacher outputs must come from exactly the planned teachers for "
            f"{assignments[0].language!r}"
        )
    soft_labels = ensemble_teacher_soft_labels(
        teacher_logits,
        weights=weights,
        temperature=plan.temperature,
    )
    return compute_kd_loss(
        student_logits=student_logits,
        hard_labels=hard_labels,
        teacher_soft_labels=soft_labels,
        mask=mask,
        temperature=plan.temperature,
        alpha=plan.alpha,
        span_loss_weight=plan.span_loss_weight,
        entity_type_weights=plan.entity_type_weights(),
        id2label=id2label,
    )


@dataclass(frozen=True)
class AdapterMetricRecord:
    """Measured evaluation of one model on one language.

    Records hold aggregate numbers only. Parents and the student must be
    measured on the same evaluation set per language and on one device
    profile, or the gate fails closed.

    Attributes:
        model_id: Parent adapter identity or the plan's student identity.
        language: Evaluated language.
        eval_set_digest: ``sha256:`` digest of the local evaluation set.
        device_profile: Opaque identity of the measurement device.
        label_recall: Recall per label code; must include every critical
            label for both the parent and the student.
        f1: Overall span F1 on the evaluation set.
        size_bytes: Deployable artifact size. For a parent, the backbone plus
            its adapter.
        peak_memory_mb: Peak resident memory while serving the language.
        latency_p50_ms: Median latency for approximately one page.
        latency_p95_ms: 95th-percentile latency for approximately one page.
    """

    model_id: str
    language: str
    eval_set_digest: str
    device_profile: str
    label_recall: Mapping[str, float]
    f1: float
    size_bytes: int
    peak_memory_mb: float
    latency_p50_ms: float
    latency_p95_ms: float

    def __post_init__(self) -> None:
        """Validate identities, digests and finite metric ranges."""

        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(self, "language", _language(self.language, "language"))
        object.__setattr__(
            self, "eval_set_digest", _digest(self.eval_set_digest, "eval_set_digest")
        )
        object.__setattr__(
            self, "device_profile", _identifier(self.device_profile, "device_profile")
        )
        if not isinstance(self.label_recall, Mapping):
            raise AdapterDistillationError("label_recall must be a mapping")
        if len(self.label_recall) > _MAX_LABELS:
            raise AdapterDistillationError("label_recall has too many labels")
        recall: dict[str, float] = {}
        for raw_label, raw_value in self.label_recall.items():
            label = _label_code(raw_label, "label_recall label")
            if label in recall:
                raise AdapterDistillationError("label_recall must not repeat a label")
            recall[label] = _fraction(raw_value, "label_recall value")
        object.__setattr__(
            self, "label_recall", MappingProxyType(dict(sorted(recall.items())))
        )
        object.__setattr__(self, "f1", _fraction(self.f1, "f1"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise AdapterDistillationError("size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise AdapterDistillationError("size_bytes must be greater than 0")
        for name in ("peak_memory_mb", "latency_p50_ms", "latency_p95_ms"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.latency_p95_ms < self.latency_p50_ms:
            raise AdapterDistillationError(
                "latency_p95_ms must not be below latency_p50_ms"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdapterMetricRecord:
        """Build a record from a JSON-style mapping with exactly the fields.

        Raises:
            AdapterDistillationError: If fields are missing or unexpected.
        """

        if not isinstance(payload, Mapping):
            raise AdapterDistillationError("metric record must be a mapping")
        unexpected = set(payload) - set(_METRIC_FIELDS)
        if unexpected:
            raise AdapterDistillationError(
                f"metric record has {len(unexpected)} unexpected field(s)"
            )
        missing = [name for name in _METRIC_FIELDS if name not in payload]
        if missing:
            raise AdapterDistillationError(
                f"metric record is missing field(s): {', '.join(missing)}"
            )
        return cls(**{name: payload[name] for name in _METRIC_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic aggregate metric payload."""

        return {
            "device_profile": self.device_profile,
            "eval_set_digest": self.eval_set_digest,
            "f1": self.f1,
            "label_recall": dict(self.label_recall),
            "language": self.language,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "model_id": self.model_id,
            "peak_memory_mb": self.peak_memory_mb,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class DistillationGateCheck:
    """One gate comparison with its observed value, limit and reason code."""

    check: str
    passed: bool
    reason: str
    language: str | None = None
    parent_adapter_id: str | None = None
    label: str | None = None
    observed: float | None = None
    threshold: float | None = None

    def sort_key(self) -> tuple[str, int, str, str, str]:
        """Return the deterministic report ordering key."""

        return (
            self.language or "",
            _CHECK_ORDER[self.check],
            self.parent_adapter_id or "",
            self.label or "",
            self.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, code-only check payload."""

        return {
            "check": self.check,
            "label": self.label,
            "language": self.language,
            "observed": self.observed,
            "parent_adapter_id": self.parent_adapter_id,
            "passed": self.passed,
            "reason": self.reason,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class AdapterDistillationGateReport:
    """Promotion decision for a distilled student against its parents.

    A promotion only means the student met the pinned evidence thresholds
    for replacing its parent adapters on edge devices. It is not a clinical
    validation or certification.
    """

    plan_id: str
    plan_digest: str
    family_id: str
    student_id: str
    tier: str
    checks: tuple[DistillationGateCheck, ...]
    evidence: tuple[tuple[str, DistillationReport], ...]
    disclaimer: str = CLINICAL_ADAPTER_DISCLAIMER
    schema_version: str = ADAPTER_DISTILLATION_GATE_SCHEMA_VERSION

    @property
    def failed_checks(self) -> tuple[DistillationGateCheck, ...]:
        """Return the checks that blocked promotion."""

        return tuple(check for check in self.checks if not check.passed)

    @property
    def promoted(self) -> bool:
        """Return whether every check passed; any failure rejects the student."""

        return bool(self.checks) and not self.failed_checks

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        """Return the sorted unique reason codes of failed checks."""

        return tuple(sorted({check.reason for check in self.failed_checks}))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, code-only gate evidence."""

        return {
            "blocking_reasons": list(self.blocking_reasons),
            "check_counts": {
                "failed": len(self.failed_checks),
                "total": len(self.checks),
            },
            "checks": [check.to_dict() for check in self.checks],
            "decision": "promote" if self.promoted else "reject",
            "disclaimer": self.disclaimer,
            "distillation_evidence": [
                {"language": language, "report": report.to_dict()}
                for language, report in self.evidence
            ],
            "family_id": self.family_id,
            "network_accessed": False,
            "plan_digest": self.plan_digest,
            "plan_id": self.plan_id,
            "promoted": self.promoted,
            "schema_version": self.schema_version,
            "student_id": self.student_id,
            "tier": self.tier,
        }

    def to_json(self) -> str:
        """Return deterministic gate JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write_json(self, path: str | Path) -> Path:
        """Write deterministic gate JSON to ``path``."""

        return _write_text(path, self.to_json())


def evaluate_adapter_distillation_gate(
    plan: AdapterDistillationPlan,
    records: Iterable[AdapterMetricRecord | Mapping[str, Any]],
) -> AdapterDistillationGateReport:
    """Decide whether a distilled student may replace its parent adapters.

    Every comparison is made per language against the parent that serves the
    language today, and every check must pass:

    * ``critical_label_recall``: for each critical label, student recall must
      not fall below the parent's by more than the pinned tolerance. Checks
      are non-compensable: a gain on another label or language never offsets
      a loss, and an unmeasured critical label fails.
    * ``cross_language_interference``: the student's F1 on each language may
      drop at most ``max_f1_drop`` below that language's parent.
    * ``artifact_size_vs_parent``: the student artifact must fit within
      ``max_size_ratio`` of every parent's deployable size.
    * ``peak_memory_budget`` and ``latency_p50_budget`` /
      ``latency_p95_budget``: the student must meet the device tier budgets.
    * ``latency_p95_vs_parent``: student p95 latency must stay within
      ``max_latency_ratio`` of the parent on the same language.
    * ``measurement_integrity``: missing records, a different evaluation set
      for student and parent, mixed device profiles, or inconsistent size
      measurements fail closed.

    Args:
        plan: Plan whose pinned policy is applied.
        records: Measured metric records, as records or JSON-style mappings.

    Returns:
        A deterministic report; ``promoted`` is true only if all checks pass.

    Raises:
        AdapterDistillationError: If a record is malformed, duplicated, or
            refers to a model or language outside the plan.
    """

    if not isinstance(plan, AdapterDistillationPlan):
        raise AdapterDistillationError("plan must be an AdapterDistillationPlan")
    if isinstance(records, (str, bytes, Mapping)):
        raise AdapterDistillationError("records must be an iterable of records")
    parsed: list[AdapterMetricRecord] = []
    for record in records:
        if len(parsed) >= _MAX_METRIC_RECORDS:
            raise AdapterDistillationError("too many metric records")
        parsed.append(
            record
            if isinstance(record, AdapterMetricRecord)
            else AdapterMetricRecord.from_dict(record)
        )

    by_key: dict[tuple[str, str], AdapterMetricRecord] = {}
    for record in parsed:
        if record.language not in plan.languages:
            raise AdapterDistillationError("metric record language is outside the plan")
        if record.model_id in plan.parent_adapter_ids:
            if record.language not in plan.selection.languages_for_parent(
                record.model_id
            ):
                raise AdapterDistillationError(
                    "parent metric record is for a language the parent does "
                    "not serve in the plan"
                )
        elif record.model_id != plan.student_id:
            raise AdapterDistillationError(
                "metric record references a model outside the plan"
            )
        key = (record.model_id, record.language)
        if key in by_key:
            raise AdapterDistillationError("duplicate metric record for a model")
        by_key[key] = record

    policy = plan.gate
    checks: list[DistillationGateCheck] = []
    evidence: list[tuple[str, DistillationReport]] = []

    profiles = {record.device_profile for record in parsed}
    if len(profiles) > 1:
        checks.append(
            DistillationGateCheck(
                check="measurement_integrity",
                passed=False,
                reason="device_profile_mismatch",
                observed=float(len(profiles)),
                threshold=1.0,
            )
        )
    for model_id in (*plan.parent_adapter_ids, plan.student_id):
        sizes = {r.size_bytes for r in parsed if r.model_id == model_id}
        if len(sizes) > 1:
            checks.append(
                DistillationGateCheck(
                    check="measurement_integrity",
                    passed=False,
                    reason="inconsistent_size_measurement",
                    parent_adapter_id=(
                        None if model_id == plan.student_id else model_id
                    ),
                    observed=float(len(sizes)),
                    threshold=1.0,
                )
            )

    for language in plan.languages:
        parent_id = plan.parent_for(language)
        parent = by_key.get((parent_id, language))
        student = by_key.get((plan.student_id, language))
        if parent is None:
            checks.append(
                _integrity_failure("missing_parent_record", language, parent_id)
            )
        if student is None:
            checks.append(
                _integrity_failure("missing_student_record", language, parent_id)
            )
            continue
        checks.extend(_student_budget_checks(student, policy))
        if parent is None:
            continue
        if parent.eval_set_digest != student.eval_set_digest:
            checks.append(_integrity_failure("eval_set_mismatch", language, parent_id))
            continue
        report = build_distillation_report(
            teacher_id=parent_id,
            student_backbone=plan.student_backbone,
            temperature=plan.temperature,
            alpha=plan.alpha,
            teacher_recall_by_label=parent.label_recall,
            student_recall_by_label=student.label_recall,
            critical_labels=plan.critical_labels,
            critical_drop_tolerance=policy.critical_recall_tolerance,
        )
        evidence.append((language, report))
        checks.extend(
            _critical_recall_checks(
                plan, report, parent=parent, student=student, policy=policy
            )
        )
        f1_delta = student.f1 - parent.f1
        checks.append(
            DistillationGateCheck(
                check="cross_language_interference",
                passed=f1_delta >= -policy.max_f1_drop - _EPSILON,
                reason=(
                    "pass"
                    if f1_delta >= -policy.max_f1_drop - _EPSILON
                    else "f1_drop_exceeds_budget"
                ),
                language=language,
                parent_adapter_id=parent_id,
                observed=f1_delta,
                threshold=-policy.max_f1_drop,
            )
        )
        latency_limit = parent.latency_p95_ms * policy.max_latency_ratio
        latency_ok = student.latency_p95_ms <= latency_limit + _EPSILON
        checks.append(
            DistillationGateCheck(
                check="latency_p95_vs_parent",
                passed=latency_ok,
                reason="pass" if latency_ok else "latency_exceeds_parent",
                language=language,
                parent_adapter_id=parent_id,
                observed=student.latency_p95_ms,
                threshold=latency_limit,
            )
        )

    student_sizes = [r.size_bytes for r in parsed if r.model_id == plan.student_id]
    for parent_id in plan.parent_adapter_ids:
        parent_sizes = [r.size_bytes for r in parsed if r.model_id == parent_id]
        if not parent_sizes or not student_sizes:
            continue
        size_limit = min(parent_sizes) * policy.max_size_ratio
        student_size = max(student_sizes)
        size_ok = student_size <= size_limit + _EPSILON
        checks.append(
            DistillationGateCheck(
                check="artifact_size_vs_parent",
                passed=size_ok,
                reason="pass" if size_ok else "size_exceeds_parent",
                parent_adapter_id=parent_id,
                observed=float(student_size),
                threshold=size_limit,
            )
        )

    return AdapterDistillationGateReport(
        plan_id=plan.plan_id,
        plan_digest=plan.plan_digest,
        family_id=plan.family_id,
        student_id=plan.student_id,
        tier=policy.tier,
        checks=tuple(sorted(checks, key=DistillationGateCheck.sort_key)),
        evidence=tuple(evidence),
        disclaimer=plan.disclaimer,
    )


def _integrity_failure(
    reason: str,
    language: str,
    parent_id: str,
) -> DistillationGateCheck:
    return DistillationGateCheck(
        check="measurement_integrity",
        passed=False,
        reason=reason,
        language=language,
        parent_adapter_id=parent_id,
    )


def _student_budget_checks(
    student: AdapterMetricRecord,
    policy: DistillationGatePolicy,
) -> list[DistillationGateCheck]:
    budgets = (
        (
            "peak_memory_budget",
            student.peak_memory_mb,
            policy.max_peak_memory_mb,
            "memory_exceeds_tier_budget",
        ),
        (
            "latency_p50_budget",
            student.latency_p50_ms,
            policy.max_latency_p50_ms,
            "latency_exceeds_tier_budget",
        ),
        (
            "latency_p95_budget",
            student.latency_p95_ms,
            policy.max_latency_p95_ms,
            "latency_exceeds_tier_budget",
        ),
    )
    checks = []
    for check, observed, limit, failure in budgets:
        passed = observed <= limit + _EPSILON
        checks.append(
            DistillationGateCheck(
                check=check,
                passed=passed,
                reason="pass" if passed else failure,
                language=student.language,
                observed=observed,
                threshold=limit,
            )
        )
    return checks


def _critical_recall_checks(
    plan: AdapterDistillationPlan,
    report: DistillationReport,
    *,
    parent: AdapterMetricRecord,
    student: AdapterMetricRecord,
    policy: DistillationGatePolicy,
) -> list[DistillationGateCheck]:
    deltas = {delta.label: delta for delta in report.per_label_recall_delta}
    checks = []
    for label in plan.critical_labels:
        if label not in parent.label_recall or label not in student.label_recall:
            checks.append(
                DistillationGateCheck(
                    check="critical_label_recall",
                    passed=False,
                    reason="critical_label_unmeasured",
                    language=student.language,
                    parent_adapter_id=parent.model_id,
                    label=label,
                )
            )
            continue
        delta = deltas[label]
        checks.append(
            DistillationGateCheck(
                check="critical_label_recall",
                passed=not delta.critical_drop,
                reason="critical_recall_drop" if delta.critical_drop else "pass",
                language=student.language,
                parent_adapter_id=parent.model_id,
                label=label,
                observed=delta.student_recall,
                threshold=delta.teacher_recall - policy.critical_recall_tolerance,
            )
        )
    return checks


def _write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


__all__ = [
    "ADAPTER_DISTILLATION_GATE_SCHEMA_VERSION",
    "ADAPTER_DISTILLATION_PLAN_SCHEMA_VERSION",
    "DEFAULT_LANGUAGE_SAMPLING_EXPONENT",
    "DEFAULT_MAX_F1_DROP",
    "GATE_CHECKS",
    "OPEN_DATA_LICENSES",
    "ROUTE_DONOR_FALLBACK",
    "ROUTE_TARGET_ADAPTER",
    "SOURCE_KINDS",
    "TEACHER_ROLE_DONOR",
    "TEACHER_ROLE_PARENT",
    "AdapterDistillationError",
    "AdapterDistillationGateReport",
    "AdapterDistillationPlan",
    "AdapterMetricRecord",
    "DistillationGateCheck",
    "DistillationGatePolicy",
    "DistillationShard",
    "DistillationSource",
    "FamilyAdapterSelection",
    "LanguageAdapterRoute",
    "TeacherAssignment",
    "build_adapter_distillation_plan",
    "compute_language_kd_loss",
    "ensemble_teacher_soft_labels",
    "evaluate_adapter_distillation_gate",
    "select_family_adapters",
]
