"""Reproducible five-source synthetic Journey conformance scenario.

The runner composes the same public contracts used by applications.  It does
not emulate service logs or call a network endpoint.  Source values remain in
the explicitly synthetic input fixture; the generated report contains only
synthetic values, opaque identifiers, coordinates, versions, and digests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Final, cast

from openmed.clinical.journey import JourneyQuery, query_journey
from openmed.clinical.journey_contracts import (
    ClinicalFact,
    ConflictSet,
    ResolutionEvent,
    canonical_digest,
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.clinical.units import parse_measurement
from openmed.core.date_shift import stable_offset_for
from openmed.core.surrogate_vault import SurrogateVault
from openmed.interop.identity import (
    CompositeIdentityResolver,
    ExactIdentityResolver,
    IdentityResolutionRequest,
    IdentityResolutionStore,
    ProbabilisticIdentityCandidate,
    SourceIdentityKey,
)
from openmed.interop.ingest import (
    PIPELINE_COMPONENT_STAGES,
    PIPELINE_STAGES,
    CallablePipelineComponent,
    DelimitedTableEvidenceAdapter,
    DICOMEvidenceAdapter,
    DICOMEvidenceInput,
    EvidenceAdapterContext,
    FHIRR4EvidenceAdapter,
    HL7V2EvidenceAdapter,
    IngestionToFactPipeline,
    PipelineComponent,
    PipelineStageContext,
    PipelineStageProduct,
    SourceManifest,
    SQLiteIngestionStore,
    TextEvidenceAdapter,
)
from openmed.interop.omop import (
    OmopConceptMapping,
    OmopFactProjectionInput,
    OmopVocabularySnapshot,
    project_clinical_facts_to_omop,
    validate_omop_fact_projection,
)
from openmed.multimodal import ExtractedDocument, SourceSpan
from openmed.service.journey_resources import (
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceQuery,
    JourneyResourceRecord,
    JourneyResourceState,
)
from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.datasets import (
    DatasetBuildSpec,
    DatasetExportFormat,
    DatasetLicenseConstraint,
    DatasetRecord,
    DatasetSelection,
    RedistributionPolicy,
    build_dataset_snapshot,
)
from openmed.structured.facts import MappingFactAdapter
from openmed.structured.store import StoreResult, StoreState

GOLDEN_JOURNEY_SCHEMA_VERSION: Final = "1.0.0"
GOLDEN_JOURNEY_COMPATIBILITY_POLICY: Final = "same_major"
GOLDEN_JOURNEY_SCHEMA_NAME: Final = "golden_journey"
GOLDEN_JOURNEY_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
GOLDEN_JOURNEY_REGENERATION_COMMAND: Final = (
    ".venv/bin/python scripts/regenerate_v3_golden_journey.py --write"
)

_ADAPTERS: Final[Mapping[str, Callable[[], Any]]] = {
    "text": TextEvidenceAdapter,
    "fhir_r4": FHIRR4EvidenceAdapter,
    "hl7v2": HL7V2EvidenceAdapter,
    "csv": DelimitedTableEvidenceAdapter,
    "dicom_sr": DICOMEvidenceAdapter,
}


class GoldenJourneyError(ValueError):
    """Raised when the synthetic scenario cannot satisfy its pinned contract."""


@dataclass(frozen=True)
class JourneyPrivacySource:
    """An in-memory source surface; its representation omits protected text.

    Only the five golden source formats are supported. DICOM SR means the
    already extracted document text, not a DICOM binary or header profile.
    """

    format: str = field(repr=False)
    text: str = field(repr=False)


@dataclass(frozen=True)
class JourneyPrivacySpan:
    """Half-open source and replacement offsets for one confirmed witness.

    Date witnesses must contain complete ISO or compact calendar days.
    Identifier witnesses must belong to the confirmed patient and use the
    supplied subject vault. Detection and subject reconciliation are the
    caller's responsibility.
    """

    start: int
    end: int
    replacement_start: int
    replacement_end: int
    label: str = field(default="ID_NUM", repr=False)

    def __post_init__(self) -> None:
        if (
            any(
                type(item) is not int or item < 0
                for item in (
                    self.start,
                    self.end,
                    self.replacement_start,
                    self.replacement_end,
                )
            )
            or self.end <= self.start
            or self.replacement_end <= self.replacement_start
            or type(self.label) is not str
            or self.label not in {"ID_NUM", "PERSON"}
        ):
            raise ValueError("invalid privacy witness")


@dataclass(frozen=True)
class JourneyPrivacyTransformation:
    """Protected output and witnesses from a trusted local format processor.

    ``patient_keyed`` must explicitly assert that the processor applied the
    supplied patient key. Missing witnesses never prove consistency.
    """

    text: str = field(repr=False)
    patient_keyed: bool = field(default=False, repr=False)
    dates: tuple[JourneyPrivacySpan, ...] = field(default=(), repr=False)
    identifiers: tuple[JourneyPrivacySpan, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class JourneyPrivacyContext:
    """Shared, memory-only inputs passed to every local format processor."""

    patient_key: str = field(repr=False)
    date_shift_secret: bytes = field(repr=False)
    date_shift_max_days: int = field(repr=False)
    vault: SurrogateVault = field(repr=False)


@dataclass(frozen=True)
class JourneyPrivacyFormatResult:
    """Counts and keyed digests for one controlled source-format name."""

    format: str
    state: str
    date_witness_count: int = 0
    surrogate_witness_count: int = 0
    date_shift_inconsistency_count: int = 0
    surrogate_inconsistency_count: int = 0
    source_digest: str = ""
    date_shift_digest: str = ""
    shifted_interval_digest: str = ""
    surrogate_set_digest: str = ""

    def __post_init__(self) -> None:
        counts = (
            self.date_witness_count,
            self.surrogate_witness_count,
            self.date_shift_inconsistency_count,
            self.surrogate_inconsistency_count,
        )
        proofs = (
            self.source_digest,
            self.date_shift_digest,
            self.shifted_interval_digest,
            self.surrogate_set_digest,
        )
        if (
            type(self.format) is not str
            or self.format not in _ADAPTERS
            or type(self.state) is not str
            or self.state not in {"success", "unsupported", "failure"}
            or any(type(item) is not int or not 0 <= item <= 512 for item in counts)
            or any(
                type(item) is not str
                or (item and not re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", item))
                for item in proofs
            )
            or not self.source_digest
        ):
            raise ValueError("invalid privacy format evidence")

    def to_dict(self) -> dict[str, Any]:
        """Return only controlled states, counts and HMAC digests."""

        return {
            "format": self.format,
            "state": self.state,
            "date_witness_count": self.date_witness_count,
            "surrogate_witness_count": self.surrogate_witness_count,
            "date_shift_inconsistency_count": self.date_shift_inconsistency_count,
            "surrogate_inconsistency_count": self.surrogate_inconsistency_count,
            "source_digest": self.source_digest,
            "date_shift_digest": self.date_shift_digest,
            "shifted_interval_digest": self.shifted_interval_digest,
            "surrogate_set_digest": self.surrogate_set_digest,
        }


@dataclass(frozen=True)
class JourneyPrivacyConsistency:
    """Value-free five-format consistency evidence for one confirmed patient."""

    patient_digest: str
    formats: tuple[JourneyPrivacyFormatResult, ...]

    def __post_init__(self) -> None:
        if (
            type(self.patient_digest) is not str
            or not re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", self.patient_digest)
            or type(self.formats) is not tuple
            or any(
                type(item) is not JourneyPrivacyFormatResult for item in self.formats
            )
            or tuple(item.format for item in self.formats) != tuple(_ADAPTERS)
        ):
            raise ValueError("invalid privacy consistency evidence")

    @property
    def state(self) -> str:
        """Return failure, partial, unsupported or success without upgrading gaps."""

        states = {item.state for item in self.formats}
        if "failure" in states:
            return "failure"
        if states == {"unsupported"}:
            return "unsupported"
        return "partial" if "unsupported" in states else "success"

    def lane_metrics(self) -> dict[str, int]:
        """Return the four additional v1.1 privacy-lane counts.

        Failed processors count as unsupported coverage as well as retaining
        their failure state. These counts supplement, never replace, leakage
        and raw-value evidence.
        """

        return {
            "date_shift_inconsistency_count": sum(
                item.date_shift_inconsistency_count for item in self.formats
            ),
            "surrogate_inconsistency_count": sum(
                item.surrogate_inconsistency_count for item in self.formats
            ),
            "consistency_checked_format_count": sum(
                item.state != "unsupported" and bool(item.shifted_interval_digest)
                for item in self.formats
            ),
            "consistency_unsupported_format_count": sum(
                item.state == "unsupported" or not item.shifted_interval_digest
                for item in self.formats
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a separate evidence report without source or replacement values."""

        return {
            "schema_version": "1.1.0",
            "compatibility_policy": "same_major",
            "state": self.state,
            "patient_digest": self.patient_digest,
            "metrics": self.lane_metrics(),
            "formats": [item.to_dict() for item in self.formats],
        }


JourneyPrivacyProcessor = Callable[
    [JourneyPrivacySource, JourneyPrivacyContext], JourneyPrivacyTransformation
]
_PRIVACY_MAX_TEXT: Final = 1_048_576
_PRIVACY_MAX_WITNESSES: Final = 512
_PRIVACY_IDENTIFIER_LABELS: Final = frozenset({"PERSON", "ID_NUM"})


def verify_journey_privacy_consistency(
    sources: Sequence[JourneyPrivacySource],
    *,
    patient_key: str,
    date_shift_secret: bytes,
    proof_secret: bytes,
    processors: Mapping[str, JourneyPrivacyProcessor],
    date_shift_max_days: int = 365,
) -> JourneyPrivacyConsistency:
    """De-identify and verify five patient-linked surfaces using local processors.

    Each injected processor receives the same patient key, shift secret and
    fresh memory-only vault. It must return actual transformed text and source
    and output offsets for confirmed patient dates and identifiers. The
    verifier reads those surfaces itself, checks the existing HMAC-derived
    offset, preserves day intervals, and compares emitted surrogates to the
    one subject surrogate established before processing. It never infers
    missing format support, invokes a model loader, or writes protected data.

    Args:
        sources: Exactly one source for each of the five golden formats.
        patient_key: Application-confirmed, non-empty patient key.
        date_shift_secret: At least 32 bytes for the existing shift algorithm.
        proof_secret: Independent, private key of at least 32 bytes for evidence.
        processors: Trusted local callables; absent formats are unsupported.
        date_shift_max_days: Positive bound for the existing date shift.

    Returns:
        Counts and domain-separated HMAC digests, without protected surfaces.

    Raises:
        GoldenJourneyError: The bounded input contract is invalid.
    """

    if (
        type(sources) not in {list, tuple}
        or len(sources) != len(_ADAPTERS)
        or any(type(item) is not JourneyPrivacySource for item in sources)
        or tuple(item.format for item in sources) != tuple(_ADAPTERS)
        or any(not _privacy_text(item.text) for item in sources)
        or type(patient_key) is not str
        or not patient_key
        or len(patient_key) > 4096
        or type(date_shift_secret) is not bytes
        or len(date_shift_secret) < 32
        or type(proof_secret) is not bytes
        or len(proof_secret) < 32
        or hmac.compare_digest(date_shift_secret, proof_secret)
        or type(date_shift_max_days) is not int
        or not 1 <= date_shift_max_days <= 3650
        or not isinstance(processors, Mapping)
        or any(
            key not in _ADAPTERS or not callable(value)
            for key, value in processors.items()
        )
    ):
        raise GoldenJourneyError("invalid Journey privacy consistency inputs")
    try:
        patient_bytes = patient_key.encode("utf-8")
        expected_offset = stable_offset_for(
            patient_key, max_days=date_shift_max_days, secret=date_shift_secret
        )
        vault = SurrogateVault.in_memory(proof_secret)
        expected_surrogate = vault.resolve_subject(patient_key)
    except (UnicodeError, ValueError, TypeError):
        raise GoldenJourneyError("invalid Journey privacy consistency inputs") from None
    context = JourneyPrivacyContext(
        patient_key, date_shift_secret, date_shift_max_days, vault
    )
    results: list[JourneyPrivacyFormatResult] = []
    for source in sources:
        source_digest = _privacy_proof(
            proof_secret, "source", [patient_bytes.hex(), source.format, source.text]
        )
        processor = processors.get(source.format)
        if processor is None:
            results.append(
                JourneyPrivacyFormatResult(
                    source.format, "unsupported", source_digest=source_digest
                )
            )
            continue
        try:
            transformed = processor(source, context)
            result = _verify_privacy_transformation(
                source,
                transformed,
                context,
                expected_offset,
                expected_surrogate,
                proof_secret,
                source_digest,
            )
        except Exception:
            # Trusted processors may raise with source values in their messages.
            # Neither those messages nor chained exceptions enter evidence.
            result = JourneyPrivacyFormatResult(
                source.format, "failure", source_digest=source_digest
            )
        results.append(result)
    return JourneyPrivacyConsistency(
        _privacy_proof(proof_secret, "patient", patient_key), tuple(results)
    )


def verify_golden_journey_privacy(
    scenario: Mapping[str, Any],
    *,
    patient_key: str,
    date_shift_secret: bytes,
    proof_secret: bytes,
    processors: Mapping[str, JourneyPrivacyProcessor],
    date_shift_max_days: int = 365,
) -> JourneyPrivacyConsistency:
    """Verify the unchanged golden payloads without fabricating missing witnesses.

    The frozen scenario has no shared patient dates and identifiers in every
    payload. Absent processors or actual witnesses therefore remain unsupported.
    DICOM SR verification covers ``document_text`` only.
    """

    _validate_scenario(scenario)
    sources = tuple(
        JourneyPrivacySource(
            item["format"],
            item["document_text"] if item["format"] == "dicom_sr" else item["payload"],
        )
        for item in scenario["sources"]
    )
    return verify_journey_privacy_consistency(
        sources,
        patient_key=patient_key,
        date_shift_secret=date_shift_secret,
        proof_secret=proof_secret,
        processors=processors,
        date_shift_max_days=date_shift_max_days,
    )


def _privacy_text(value: Any) -> bool:
    if type(value) is not str or not value or len(value) > _PRIVACY_MAX_TEXT:
        return False
    try:
        return len(value.encode("utf-8")) <= _PRIVACY_MAX_TEXT
    except UnicodeError:
        return False


def _privacy_proof(secret: bytes, domain: str, value: Any) -> str:
    material = canonical_json(["journey-privacy-v1", domain, value]).encode("utf-8")
    return "hmac-sha256:" + hmac.new(secret, material, hashlib.sha256).hexdigest()


def _privacy_spans(
    spans: Any,
    source: str,
    transformed: str,
    *,
    identifiers: bool,
) -> list[tuple[str, str, str]]:
    if type(spans) is not tuple or not 1 <= len(spans) <= _PRIVACY_MAX_WITNESSES:
        raise ValueError("invalid privacy witnesses")
    pairs: list[tuple[str, str, str]] = []
    source_ranges: list[tuple[int, int]] = []
    output_ranges: list[tuple[int, int]] = []
    for span in spans:
        if type(span) is not JourneyPrivacySpan:
            raise ValueError("invalid privacy witnesses")
        offsets = (span.start, span.end, span.replacement_start, span.replacement_end)
        if (
            any(type(item) is not int for item in offsets)
            or not 0 <= span.start < span.end <= len(source)
            or not 0
            <= span.replacement_start
            < span.replacement_end
            <= len(transformed)
            or (identifiers and span.label not in _PRIVACY_IDENTIFIER_LABELS)
        ):
            raise ValueError("invalid privacy witnesses")
        source_ranges.append((span.start, span.end))
        output_ranges.append((span.replacement_start, span.replacement_end))
        pairs.append(
            (
                source[span.start : span.end],
                transformed[span.replacement_start : span.replacement_end],
                span.label,
            )
        )
    for ranges in (source_ranges, output_ranges):
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("invalid privacy witnesses")
    return pairs


def _privacy_day(value: str) -> date | None:
    if not re.fullmatch(r"[0-9]{4}(?:-[0-9]{2}-[0-9]{2}|[0-9]{4})", value):
        return None
    try:
        return date(
            int(value[:4]),
            int(value[-4:-2]) if "-" not in value else int(value[5:7]),
            int(value[-2:]),
        )
    except ValueError:
        return None


def _verify_privacy_transformation(
    source: JourneyPrivacySource,
    transformed: JourneyPrivacyTransformation,
    context: JourneyPrivacyContext,
    expected_offset: int,
    expected_surrogate: str,
    proof_secret: bytes,
    source_digest: str,
) -> JourneyPrivacyFormatResult:
    if (
        type(transformed) is not JourneyPrivacyTransformation
        or not _privacy_text(transformed.text)
        or type(transformed.patient_keyed) is not bool
    ):
        raise ValueError("invalid privacy transformation")
    if (
        not transformed.patient_keyed
        or not transformed.dates
        or not transformed.identifiers
    ):
        return JourneyPrivacyFormatResult(
            source.format, "unsupported", source_digest=source_digest
        )
    dates = _privacy_spans(
        transformed.dates, source.text, transformed.text, identifiers=False
    )
    identifiers = _privacy_spans(
        transformed.identifiers, source.text, transformed.text, identifiers=True
    )
    for side in ("source", "replacement"):
        all_spans = transformed.dates + transformed.identifiers
        ranges = sorted(
            (span.start, span.end)
            if side == "source"
            else (span.replacement_start, span.replacement_end)
            for span in all_spans
        )
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("invalid privacy witnesses")
    parsed_dates = [
        (_privacy_day(original), _privacy_day(shifted))
        for original, shifted, _label in dates
    ]
    if any(original is None or shifted is None for original, shifted in parsed_dates):
        return JourneyPrivacyFormatResult(
            source.format, "unsupported", source_digest=source_digest
        )
    day_pairs = sorted(
        (cast(date, original), cast(date, shifted))
        for original, shifted in parsed_dates
    )
    date_mismatch = any(
        (shifted - original).days != expected_offset for original, shifted in day_pairs
    )
    intervals = [
        (original.toordinal(), shifted.toordinal()) for original, shifted in day_pairs
    ]
    # Relative intervals can expose small gaps; protect them with a keyed proof.
    origin, shifted_origin = intervals[0]
    relative_intervals = [
        (original - origin, shifted - shifted_origin) for original, shifted in intervals
    ]
    date_mismatch = date_mismatch or any(
        left != right for left, right in relative_intervals
    )
    surrogate_mismatch = False
    emitted: set[str] = set()
    for original, replacement, label in identifiers:
        expected = context.vault.get(original, label=label, lang="en")
        surrogate_mismatch = surrogate_mismatch or (
            expected != expected_surrogate
            or replacement != expected_surrogate
            or replacement == original
        )
        emitted.add(replacement)
    return JourneyPrivacyFormatResult(
        source.format,
        "failure" if date_mismatch or surrogate_mismatch else "success",
        len(dates),
        len(identifiers),
        int(date_mismatch),
        int(surrogate_mismatch),
        source_digest,
        _privacy_proof(
            proof_secret,
            "shift",
            [
                context.patient_key,
                sorted({(shifted - original).days for original, shifted in day_pairs}),
            ],
        ),
        _privacy_proof(proof_secret, "intervals", [context.patient_key, intervals]),
        _privacy_proof(
            proof_secret, "surrogates", [context.patient_key, sorted(emitted)]
        ),
    )


class _SyntheticIdentityPlugin:
    """Deterministic review-only candidates for the synthetic identity lane."""

    def __init__(self, candidates: tuple[ProbabilisticIdentityCandidate, ...]) -> None:
        self._candidates = candidates

    def candidates(
        self, request: IdentityResolutionRequest
    ) -> StoreResult[tuple[ProbabilisticIdentityCandidate, ...]]:
        del request
        return StoreResult.success(self._candidates)


def load_golden_journey_schema() -> dict[str, Any]:
    """Load the bundled report schema."""

    resource = resources.files(GOLDEN_JOURNEY_SCHEMA_PACKAGE).joinpath(
        f"{GOLDEN_JOURNEY_SCHEMA_NAME}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_golden_journey_scenario(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one explicitly synthetic scenario."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenJourneyError("golden Journey scenario is unreadable") from exc
    _validate_scenario(payload)
    return payload


def _validate_scenario(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise GoldenJourneyError("golden Journey scenario must be an object")
    expected = {
        "compatibility_policy",
        "cohort_definition",
        "encounter_id",
        "recorded_at",
        "scenario_id",
        "schema_version",
        "sources",
        "subject_id",
        "synthetic",
        "versions",
        "worker_id",
    }
    if set(payload) != expected:
        raise GoldenJourneyError("golden Journey scenario fields differ")
    if payload["schema_version"] != GOLDEN_JOURNEY_SCHEMA_VERSION:
        raise GoldenJourneyError("golden Journey scenario version is unsupported")
    if payload["compatibility_policy"] != GOLDEN_JOURNEY_COMPATIBILITY_POLICY:
        raise GoldenJourneyError("golden Journey compatibility policy is unsupported")
    if payload["synthetic"] is not True:
        raise GoldenJourneyError("golden Journey inputs must assert synthetic origin")
    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != 5:
        raise GoldenJourneyError("golden Journey requires exactly five sources")
    formats = [item.get("format") for item in sources if isinstance(item, dict)]
    if formats != ["text", "fhir_r4", "hl7v2", "csv", "dicom_sr"]:
        raise GoldenJourneyError("golden Journey source order or formats differ")


def run_golden_journey(
    scenario: Mapping[str, Any],
    *,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Execute the offline five-source scenario and return its stable report."""

    _validate_scenario(scenario)
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    journey_path = _fresh_database_path(root, "journey.sqlite3")
    identity_path = _fresh_database_path(root, "identity.sqlite3")
    store = SQLiteIngestionStore(journey_path)
    identity_store = IdentityResolutionStore(identity_path)
    try:
        components = cast(Sequence[PipelineComponent], _pipeline_components(scenario))
        pipeline = IngestionToFactPipeline(store, components)
        sources, runs, facts_by_source = _run_sources(scenario, pipeline, store)
        replay = _replay_first_source(scenario, pipeline)
        corrected, conversion = _correct_laboratory_fact(
            store,
            facts_by_source["table"],
            committed_at="2026-01-02T04:04:05Z",
        )
        conflict, resolution = _record_conflict_and_resolution(
            store,
            affirmed=facts_by_source["note"],
            negated=facts_by_source["message"],
        )
        identity = _resolve_ambiguous_identity(scenario, identity_store)
        journey = query_journey(
            store,
            JourneyQuery(subject_id=str(scenario["subject_id"])),
        )
        if journey.value is None:
            raise GoldenJourneyError(
                f"journey query failed: {journey.code or journey.state.value}"
            )
        projection = _project_omop(
            scenario,
            (
                facts_by_source["note"],
                corrected,
                facts_by_source["image"],
            ),
            sources,
        )
        cohort = _build_cohort(
            scenario,
            condition=facts_by_source["note"],
            laboratory=corrected,
            journey_digest=canonical_digest(journey.value.to_dict()),
        )
        dataset = _build_dataset(
            scenario,
            cohort,
            condition=facts_by_source["note"],
            laboratory=corrected,
        )
        api = _build_api_results(
            tuple(fact for name, fact in facts_by_source.items() if name != "table")
            + (corrected,)
        )
        report = {
            "api_results": api,
            "cohort": cohort.to_dict(),
            "compatibility_policy": GOLDEN_JOURNEY_COMPATIBILITY_POLICY,
            "conflicts": [conflict.to_dict()],
            "dataset": dataset.manifest.to_dict(),
            "facts": [
                fact.to_dict()
                for fact in sorted(
                    (*facts_by_source.values(), corrected),
                    key=lambda item: item.fact_id,
                )
            ],
            "identity": identity.to_dict(),
            "journey": journey.value.to_dict(),
            "omop": projection,
            "pipeline": {
                "replay": _pipeline_run_payload(replay),
                "runs": [_pipeline_run_payload(item) for item in runs],
                "stage_order": list(PIPELINE_STAGES),
            },
            "provenance": {
                "classification": "synthetic",
                "model_versions": dict(scenario["versions"]["models"]),
                "policy_versions": dict(scenario["versions"]["policies"]),
                "regeneration_command": GOLDEN_JOURNEY_REGENERATION_COMMAND,
                "scenario_digest": canonical_digest(scenario),
                "vocabulary_versions": dict(scenario["versions"]["vocabularies"]),
            },
            "resolutions": [resolution.to_dict()],
            "scenario_id": scenario["scenario_id"],
            "schema_version": GOLDEN_JOURNEY_SCHEMA_VERSION,
            "sources": sources,
            "state_matrix": _state_matrix(api, identity, projection),
            "synthetic": True,
            "unit_conversion": conversion,
        }
        return json.loads(canonical_json(report))
    finally:
        identity_store.close()
        store.close()


def semantic_diff(expected: Any, actual: Any) -> list[dict[str, Any]]:
    """Return a deterministic, JSON-Pointer-like semantic difference list."""

    differences: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}/{_pointer_token(str(key))}"
                if key not in left:
                    differences.append(
                        {"actual": right[key], "expected": None, "path": child}
                    )
                elif key not in right:
                    differences.append(
                        {"actual": None, "expected": left[key], "path": child}
                    )
                else:
                    walk(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                child = f"{path}/{index}"
                if index >= len(left):
                    differences.append(
                        {"actual": right[index], "expected": None, "path": child}
                    )
                elif index >= len(right):
                    differences.append(
                        {"actual": None, "expected": left[index], "path": child}
                    )
                else:
                    walk(left[index], right[index], child)
            return
        if left != right:
            differences.append({"actual": right, "expected": left, "path": path})

    walk(expected, actual, "")
    return differences


def render_semantic_diff(expected: Any, actual: Any) -> str:
    """Render semantic differences as stable JSON Lines."""

    return "\n".join(canonical_json(item) for item in semantic_diff(expected, actual))


def _pipeline_components(
    scenario: Mapping[str, Any],
) -> tuple[CallablePipelineComponent, ...]:
    components: list[CallablePipelineComponent] = []
    for stage in PIPELINE_COMPONENT_STAGES:
        version = str(scenario["versions"]["models"].get(stage, "1.0.0"))
        components.append(
            CallablePipelineComponent(
                stage=stage,
                component=f"openmed.golden.{stage}",
                component_version=version,
                policy_digest=canonical_digest(
                    {"mode": "offline", "stage": stage, "version": version}
                ),
                operation=_stage_operation(stage, version, scenario),
            )
        )
    return tuple(components)


def _stage_operation(
    stage: str,
    version: str,
    scenario: Mapping[str, Any],
) -> Callable[[PipelineStageContext], PipelineStageProduct]:
    facts = {str(item["format"]): dict(item["fact"]) for item in scenario["sources"]}

    def run(context: PipelineStageContext) -> PipelineStageProduct:
        if context.evidence is None:
            raise GoldenJourneyError("pipeline evidence is missing")
        if stage != "extraction":
            return PipelineStageProduct(
                output_digest=canonical_digest(
                    {
                        "source_digest": context.evidence.source_digest,
                        "stage": stage,
                        "version": version,
                    }
                )
            )
        fact = facts[context.evidence.source_format]
        field_paths = {name: name for name in fact}
        adapted = MappingFactAdapter(
            component="openmed.golden.synthetic_extractor",
            component_version=version,
            output_schema=f"synthetic.golden.{context.fact_profile}",
            output_schema_version=GOLDEN_JOURNEY_SCHEMA_VERSION,
            kind="extraction",
            field_paths=field_paths,
        ).adapt(
            fact,
            evidence_ids=(context.evidence.locators[0].locator_id,),
            field_states={name: "known" for name in fact},
        )
        if adapted.value is None:
            raise GoldenJourneyError("synthetic extraction contract failed")
        fragment = adapted.value
        return PipelineStageProduct(
            output_digest=canonical_digest(
                {"fragment_id": fragment.fragment_id, "version": version}
            ),
            fragments=(fragment,),
            output_record_ids=(fragment.fragment_id,),
        )

    return run


def _run_sources(
    scenario: Mapping[str, Any],
    pipeline: IngestionToFactPipeline,
    store: SQLiteIngestionStore,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, ClinicalFact]]:
    receipts: list[dict[str, Any]] = []
    runs: list[Any] = []
    facts: dict[str, ClinicalFact] = {}
    for item in scenario["sources"]:
        adapter, source, source_bytes = _source_input(item)
        source_id = str(item["source_id"])
        manifest = _manifest(
            scenario,
            pipeline,
            adapter,
            source_id=source_id,
            source_bytes=source_bytes,
        )
        result = pipeline.run(
            manifest=manifest,
            source=source,
            adapter=adapter,
            adapter_context=EvidenceAdapterContext(
                source_id=source_id,
                subject_id=str(scenario["subject_id"]),
                encounter_id=str(scenario["encounter_id"]),
                recorded_at=str(scenario["recorded_at"]),
            ),
            subject_id=str(scenario["subject_id"]),
            encounter_id=str(scenario["encounter_id"]),
            fact_profile=str(item["fact_profile"]),
            recorded_at=str(scenario["recorded_at"]),
            worker_id=str(scenario["worker_id"]),
        )
        if not result.ok or result.value is None:
            raise GoldenJourneyError(
                f"source pipeline failed: {result.code or result.state.value}"
            )
        run = result.value
        fact_result = store.get_fact(run.fact_ids[0])
        if fact_result.value is None:
            raise GoldenJourneyError("persisted synthetic fact is missing")
        fact = fact_result.value
        locator_result = store.get_evidence(fact.evidence_ids[0])
        if locator_result.value is None:
            raise GoldenJourneyError("persisted synthetic evidence is missing")
        artifact_result = store.get_artifact(locator_result.value.artifact_id)
        if artifact_result.value is None:
            raise GoldenJourneyError("persisted synthetic artifact is missing")
        facts[str(item["name"])] = fact
        runs.append(run)
        receipts.append(
            {
                "artifact": artifact_result.value.to_dict(),
                "evidence": [locator_result.value.to_dict()],
                "format": item["format"],
                "name": item["name"],
                "source_digest": sha256_digest(source_bytes),
                "source_id": source_id,
                "source_size": len(source_bytes),
            }
        )
    return receipts, runs, facts


def _replay_first_source(
    scenario: Mapping[str, Any], pipeline: IngestionToFactPipeline
) -> Any:
    item = scenario["sources"][0]
    adapter, source, source_bytes = _source_input(item)
    source_id = str(item["source_id"])
    result = pipeline.run(
        manifest=_manifest(
            scenario,
            pipeline,
            adapter,
            source_id=source_id,
            source_bytes=source_bytes,
        ),
        source=source,
        adapter=adapter,
        adapter_context=EvidenceAdapterContext(
            source_id=source_id,
            subject_id=str(scenario["subject_id"]),
            encounter_id=str(scenario["encounter_id"]),
            recorded_at=str(scenario["recorded_at"]),
        ),
        subject_id=str(scenario["subject_id"]),
        encounter_id=str(scenario["encounter_id"]),
        fact_profile=str(item["fact_profile"]),
        recorded_at=str(scenario["recorded_at"]),
        worker_id=str(scenario["worker_id"]),
    )
    if not result.ok or result.value is None or not result.value.replayed:
        raise GoldenJourneyError("synthetic replay was not idempotent")
    return result.value


def _source_input(item: Mapping[str, Any]) -> tuple[Any, Any, bytes]:
    source_format = str(item["format"])
    adapter = _ADAPTERS[source_format]()
    if source_format != "dicom_sr":
        payload = str(item["payload"])
        return adapter, payload, payload.encode("utf-8")
    raw = str(item["payload"]).encode("utf-8")
    text = str(item["document_text"])
    document = ExtractedDocument(
        text=text,
        spans=(
            SourceSpan(
                start=0,
                end=len(text),
                metadata={"node_path": str(item["node_path"])},
            ),
        ),
        metadata={"format": "dicom_sr", "synthetic": True},
    )
    return (
        adapter,
        DICOMEvidenceInput.from_sr_document(
            document,
            source_bytes=raw,
            study_uid=str(item["study_uid"]),
            series_uid=str(item["series_uid"]),
            instance_uid=str(item["instance_uid"]),
        ),
        raw,
    )


def _manifest(
    scenario: Mapping[str, Any],
    pipeline: IngestionToFactPipeline,
    adapter: Any,
    *,
    source_id: str,
    source_bytes: bytes,
) -> SourceManifest:
    return SourceManifest(
        manifest_id=derived_opaque_id("manifest", scenario["scenario_id"], source_id),
        source_id=source_id,
        artifact_digests=(sha256_digest(source_bytes),),
        policy_digest=pipeline.policy_digest,
        pipeline_digest=pipeline.pipeline_digest(adapter),
        created_at=str(scenario["recorded_at"]),
    )


def _correct_laboratory_fact(
    store: SQLiteIngestionStore,
    original: ClinicalFact,
    *,
    committed_at: str,
) -> tuple[ClinicalFact, dict[str, Any]]:
    if not isinstance(original.value, Mapping):
        raise GoldenJourneyError("synthetic laboratory value is invalid")
    normalized = parse_measurement(original.value["numeric"], original.unit)
    if normalized["status"] != "ok":
        raise GoldenJourneyError("synthetic unit conversion failed")
    corrected_value = dict(original.value)
    corrected_value["numeric"] = normalized["canonical_magnitude"]
    corrected = replace(
        original,
        fact_id=derived_opaque_id("fact", original.fact_id, "unit-correction"),
        value=corrected_value,
        status="corrected",
        parent_fact_ids=(original.fact_id,),
        derivation_hash=canonical_digest(
            {"operation": "unit_correction", "parent_fact_id": original.fact_id}
        ),
        unit=str(normalized["canonical_unit"]),
        attributes={
            **dict(original.attributes),
            "correction_reason": "unit_normalized",
        },
    )
    persisted = store.put_fact(corrected, committed_at=committed_at)
    if not persisted.ok:
        raise GoldenJourneyError("corrected synthetic fact was not persisted")
    return corrected, {
        "canonical_magnitude": normalized["canonical_magnitude"],
        "canonical_unit": normalized["canonical_unit"],
        "original_fact_id": original.fact_id,
        "original_unit": original.unit,
        "status": normalized["status"],
    }


def _record_conflict_and_resolution(
    store: SQLiteIngestionStore,
    *,
    affirmed: ClinicalFact,
    negated: ClinicalFact,
) -> tuple[ConflictSet, ResolutionEvent]:
    conflict = ConflictSet(
        conflict_id=derived_opaque_id("conflict", affirmed.fact_id, negated.fact_id),
        subject_id=affirmed.subject_id,
        conflict_type="assertion_mismatch",
        fact_ids=(affirmed.fact_id, negated.fact_id),
        status="open",
        detected_by="openmed.golden.conflict_policy",
        derivation_hash=canonical_digest(
            {"affirmed": affirmed.fact_id, "negated": negated.fact_id}
        ),
        evidence_ids=tuple(sorted((*affirmed.evidence_ids, *negated.evidence_ids))),
        attributes={"review_required": True},
    )
    if not store.put_conflict(conflict, committed_at="2026-01-02T05:04:05Z").ok:
        raise GoldenJourneyError("synthetic conflict was not persisted")
    resolution = ResolutionEvent(
        resolution_id=derived_opaque_id("resolution", conflict.conflict_id, "select"),
        conflict_id=conflict.conflict_id,
        action="select",
        actor_type="policy",
        policy_id="openmed.golden.conflict_review",
        policy_version="1.0.0",
        occurred_at="2026-01-02T06:04:05Z",
        rationale_code="source_reviewed",
        derivation_hash=canonical_digest(
            {"conflict_id": conflict.conflict_id, "selected": affirmed.fact_id}
        ),
        selected_fact_ids=(affirmed.fact_id,),
        rejected_fact_ids=(negated.fact_id,),
    )
    if not store.put_resolution(resolution, committed_at="2026-01-02T06:04:05Z").ok:
        raise GoldenJourneyError("synthetic conflict resolution was not persisted")
    return conflict, resolution


def _resolve_ambiguous_identity(
    scenario: Mapping[str, Any], store: IdentityResolutionStore
) -> Any:
    source_key = SourceIdentityKey(
        entity_type="patient",
        source_id=str(scenario["sources"][0]["source_id"]),
        local_key="local_synthetic0000001",
    )
    candidates = tuple(
        ProbabilisticIdentityCandidate(
            canonical_key=f"patient_candidate000000{index}",
            score_basis_points=score,
            evidence_digest=canonical_digest(
                {"candidate": index, "scenario": scenario["scenario_id"]}
            ),
            plugin_id="openmed.golden.identity",
            plugin_version="1.0.0",
        )
        for index, score in ((1, 9400), (2, 9100))
    )
    request = IdentityResolutionRequest(
        request_id="request_synthetic0000001",
        entity_type="patient",
        source_keys=(source_key,),
        purpose="care",
        role="clinician",
        attributes=("identified_access",),
        policy_id="openmed.identity.default",
        policy_version="1.0.0",
        requested_at=str(scenario["recorded_at"]),
    )
    result = CompositeIdentityResolver(
        ExactIdentityResolver(store), _SyntheticIdentityPlugin(candidates)
    ).resolve(request)
    if result.value is None or result.value.state != "ambiguous":
        raise GoldenJourneyError("identity ambiguity was not preserved")
    return result.value


def _project_omop(
    scenario: Mapping[str, Any],
    facts: Sequence[ClinicalFact],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    vocabulary = OmopVocabularySnapshot(
        snapshot_id="openmed.synthetic.v1",
        version=str(scenario["versions"]["vocabularies"]["synthetic"]),
        digest=canonical_digest(scenario["versions"]["vocabularies"]),
        license="apache-2.0",
        usage_lane="redistributable",
        bundled=False,
    )
    source_by_name = {str(item["name"]): item for item in sources}
    names = ("note", "table", "image")
    inputs = []
    for index, (name, fact) in enumerate(zip(names, facts, strict=True), start=1):
        value = fact.value if isinstance(fact.value, Mapping) else {}
        source_code = str(value.get("code") or value.get("category"))
        inputs.append(
            OmopFactProjectionInput(
                fact=fact,
                source_key=str(source_by_name[name]["source_id"]),
                source_revision=str(source_by_name[name]["source_digest"]),
                mapping=OmopConceptMapping(
                    state="mapped",
                    source_system="synthetic",
                    source_code=source_code,
                    source_concept_id=1100 + index,
                    standard_concept_id=2100 + index,
                    standard_vocabulary="synthetic",
                    standard_code=f"golden-{fact.fact_type}",
                    reason_code="mapped",
                    snapshot_digest=vocabulary.digest,
                    valid_start_date="2026-01-01",
                    valid_end_date="2099-12-31",
                ),
                dataset_split="holdout",
            )
        )
    result = project_clinical_facts_to_omop(
        inputs,
        vocabulary_snapshot=vocabulary,
        etl_version="3.0.0",
        occurred_at="2026-01-02T07:04:05Z",
    )
    if result.value is None:
        raise GoldenJourneyError(
            f"OMOP projection failed: {result.code or result.state.value}"
        )
    violations = validate_omop_fact_projection(result.value)
    if violations:
        raise GoldenJourneyError("OMOP projection contains referential violations")
    return {
        "projection": result.value.to_dict(),
        "reason_code": result.code,
        "state": result.state.value,
        "violations": [],
    }


def _build_cohort(
    scenario: Mapping[str, Any],
    *,
    condition: ClinicalFact,
    laboratory: ClinicalFact,
    journey_digest: str,
) -> Any:
    definition = save_cohort_definition(
        PhenotypeDefinition.from_dict(dict(scenario["cohort_definition"]))
    )
    membership = CohortMembership(
        patient_key=str(scenario["subject_id"]),
        state=MembershipState.MET,
        criteria=(
            CriterionMembership(
                criterion_id="condition-present",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id=condition.evidence_ids[0],
                        fact_id=condition.fact_id,
                        time_window_id="window_condition0000001",
                    ),
                ),
            ),
            CriterionMembership(
                criterion_id="corrected-lab-present",
                state=MembershipState.MET,
                evidence=(
                    MembershipEvidence(
                        evidence_id=laboratory.evidence_ids[0],
                        fact_id=laboratory.fact_id,
                        time_window_id="window_laboratory0000001",
                    ),
                ),
            ),
        ),
    )
    result = build_cohort_execution(
        definition,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_goldenjourney001",
            digest=journey_digest,
            schema_version="journey-v1",
            license_tags=("synthetic",),
        ),
        vocabulary_digest=canonical_digest(scenario["versions"]["vocabularies"]),
        policy_digest=canonical_digest(scenario["versions"]["policies"]),
        evaluator_version="3.0.0",
        memberships=(membership,),
    )
    if result.value is None:
        raise GoldenJourneyError(
            f"cohort execution failed: {result.code or result.state.value}"
        )
    return result.value


def _build_dataset(
    scenario: Mapping[str, Any],
    cohort: Any,
    *,
    condition: ClinicalFact,
    laboratory: ClinicalFact,
) -> Any:
    selection = DatasetSelection.from_cohort_execution(cohort)
    record = DatasetRecord(
        record_id="record_goldenjourney001",
        patient_key=str(scenario["subject_id"]),
        split="holdout",
        source_fact_ids=(condition.fact_id, laboratory.fact_id),
        evidence_ids=(condition.evidence_ids[0], laboratory.evidence_ids[0]),
        labels=("Condition", "Laboratory"),
        values={"classification": "synthetic"},
    )
    result = build_dataset_snapshot(
        DatasetBuildSpec(
            dataset_id="dataset_goldenjourney001",
            created_at="2026-01-02T08:04:05Z",
            selection=selection,
            query_digest=canonical_digest(scenario["cohort_definition"]),
            policy_digest=canonical_digest(scenario["versions"]["policies"]),
            schema_digest=canonical_digest(load_golden_journey_schema()),
            vocabulary_digest=canonical_digest(scenario["versions"]["vocabularies"]),
            component_versions={"dataset_builder": "3.0.0"},
            model_versions=dict(scenario["versions"]["models"]),
            licenses=(
                DatasetLicenseConstraint(
                    source_id="synthetic_fixture",
                    license_id="Apache-2.0",
                    terms_digest=canonical_digest(
                        {"license": "Apache-2.0", "synthetic": True}
                    ),
                    redistribution=RedistributionPolicy.PERMITTED,
                ),
            ),
            formats=(
                DatasetExportFormat.JSONL,
                DatasetExportFormat.ANNOTATION_JSONL,
            ),
        ),
        (record,),
    )
    if result.value is None or result.state is not StoreState.SUCCESS:
        raise GoldenJourneyError(
            f"dataset build failed: {result.code or result.state.value}"
        )
    return result.value


def _build_api_results(facts: Sequence[ClinicalFact]) -> dict[str, Any]:
    records = [
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.FACT,
            resource_id=fact.fact_id,
            namespace="default",
            data={
                "assertion": fact.attributes.get("assertion", "unknown"),
                "concept": (
                    fact.value.get("code") or fact.value.get("category")
                    if isinstance(fact.value, Mapping)
                    else "unknown"
                ),
                "confidence": fact.confidence,
                "evidence_ids": list(fact.evidence_ids),
                "subject_id": fact.subject_id,
            },
        )
        for fact in facts
    ]
    terminal_states = (
        JourneyResourceState.PARTIAL,
        JourneyResourceState.UNKNOWN,
        JourneyResourceState.CONFLICT,
        JourneyResourceState.UNSUPPORTED,
        JourneyResourceState.FAILURE,
    )
    records.extend(
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.FACT,
            resource_id=f"fact_{state.value + '0' * 16}"[:21],
            namespace=state.value,
            data={},
            state=state,
        )
        for state in terminal_states
    )
    catalog = JourneyResourceCatalog(records)
    policy = JourneyAccessPolicy(
        allowed_namespaces=frozenset(
            {"default", "empty", *(state.value for state in terminal_states)}
        )
    )
    pages = {
        "success": catalog.list_resources(
            JourneyResourceQuery(
                resource_type=JourneyResourceKind.FACT,
                namespace="default",
                fields=("subject_id", "concept", "assertion", "evidence_ids"),
            ),
            policy=policy,
        ).to_dict(),
        "empty": catalog.list_resources(
            JourneyResourceQuery(
                resource_type=JourneyResourceKind.FACT, namespace="empty"
            ),
            policy=policy,
        ).to_dict(),
        "denied": catalog.list_resources(
            JourneyResourceQuery(
                resource_type=JourneyResourceKind.FACT,
                consent_state="withdrawn",
            ),
            policy=policy,
        ).to_dict(),
    }
    for state in terminal_states:
        pages[state.value] = catalog.list_resources(
            JourneyResourceQuery(
                resource_type=JourneyResourceKind.FACT,
                namespace=state.value,
            ),
            policy=policy,
        ).to_dict()
    return dict(sorted(pages.items()))


def _state_matrix(
    api: Mapping[str, Any], identity: Any, projection: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "ambiguous_identity": identity.state,
        "conflict": str(api["conflict"]["state"]),
        "denied_consent": str(api["denied"]["state"]),
        "empty_query": str(api["empty"]["state"]),
        "failure": str(api["failure"]["state"]),
        "partial_projection": str(projection["state"]),
        "unknown": str(api["unknown"]["state"]),
        "unsupported": str(api["unsupported"]["state"]),
    }


def _pipeline_run_payload(run: Any) -> dict[str, Any]:
    return {
        "fact_ids": list(run.fact_ids),
        "job_id": run.job_id,
        "manifest_digest": run.manifest_digest,
        "replayed": run.replayed,
        "stage_states": [
            {"stage": item.stage, "state": item.state} for item in run.stage_manifests
        ],
        "state": run.state.value,
    }


def _fresh_database_path(root: Path, name: str) -> Path:
    path = root / name
    if path.exists():
        raise GoldenJourneyError(f"refusing to reuse existing database: {name}")
    return path


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


__all__ = [
    "GOLDEN_JOURNEY_COMPATIBILITY_POLICY",
    "GOLDEN_JOURNEY_REGENERATION_COMMAND",
    "GOLDEN_JOURNEY_SCHEMA_NAME",
    "GOLDEN_JOURNEY_SCHEMA_VERSION",
    "GoldenJourneyError",
    "JourneyPrivacyConsistency",
    "JourneyPrivacyContext",
    "JourneyPrivacyFormatResult",
    "JourneyPrivacyProcessor",
    "JourneyPrivacySource",
    "JourneyPrivacySpan",
    "JourneyPrivacyTransformation",
    "load_golden_journey_scenario",
    "load_golden_journey_schema",
    "render_semantic_diff",
    "run_golden_journey",
    "semantic_diff",
    "verify_golden_journey_privacy",
    "verify_journey_privacy_consistency",
]
