"""Reproducible five-source synthetic Journey conformance scenario.

The runner composes the same public contracts used by applications.  It does
not emulate service logs or call a network endpoint.  Source values remain in
the explicitly synthetic input fixture; the generated report contains only
synthetic values, opaque identifiers, coordinates, versions, and digests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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
    "load_golden_journey_scenario",
    "load_golden_journey_schema",
    "render_semantic_diff",
    "run_golden_journey",
    "semantic_diff",
]
