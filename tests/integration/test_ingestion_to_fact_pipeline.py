"""Synthetic end-to-end tests for the ingestion-to-fact orchestrator."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.grounding.vocab import VocabConcept, VocabularyIndex
from openmed.clinical.journey_contracts import (
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
)
from openmed.clinical.terminology import (
    SQLiteTerminologyMappingStore,
    TerminologyCoverageSummary,
    TerminologyQuery,
    TerminologyRelationshipRule,
    TerminologyResolver,
    TerminologySnapshot,
)
from openmed.interop.ingest import (
    PIPELINE_COMPONENT_STAGES,
    CallablePipelineComponent,
    CDAEvidenceAdapter,
    DelimitedTableEvidenceAdapter,
    EvidenceAdapterContext,
    FHIRR4EvidenceAdapter,
    HL7V2EvidenceAdapter,
    IngestionToFactPipeline,
    PipelineStageContext,
    PipelineStageProduct,
    SourceManifest,
    SQLiteIngestionStore,
    TextEvidenceAdapter,
)
from openmed.structured.facts import (
    FactReconciler,
    FactReconciliationInput,
    FactReconciliationPolicy,
    MappingFactAdapter,
    persist_fact_reconciliation,
)
from openmed.structured.store import StoreState

pytestmark = pytest.mark.integration

T0 = "2026-01-02T03:04:05Z"
SUBJECT_ID = "subject_aaaaaaaaaaaaaaaa"
ENCOUNTER_ID = "encounter_aaaaaaaaaaaaaaaa"
WORKER_ID = "worker_aaaaaaaaaaaaaaaa"
FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "interop"
    / "ingestion_pipeline_five_sources.json"
)

ADAPTERS: dict[str, Callable[[], Any]] = {
    "text": TextEvidenceAdapter,
    "fhir_r4": FHIRR4EvidenceAdapter,
    "hl7v2": HL7V2EvidenceAdapter,
    "cda": CDAEvidenceAdapter,
    "csv": DelimitedTableEvidenceAdapter,
}


def _fixture_sources() -> tuple[dict[str, str], ...]:
    return tuple(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _components(
    *,
    extraction_version: str = "1.0.0",
    fail_stage: str | None = None,
) -> tuple[CallablePipelineComponent, ...]:
    components: list[CallablePipelineComponent] = []
    for stage in PIPELINE_COMPONENT_STAGES:
        version = extraction_version if stage == "extraction" else "1.0.0"
        policy_digest = canonical_digest(
            {"mode": "local_only", "stage": stage, "version": version}
        )
        components.append(
            CallablePipelineComponent(
                stage=stage,
                component=f"openmed.pipeline.synthetic.{stage}",
                component_version=version,
                policy_digest=policy_digest,
                operation=_operation(
                    stage,
                    version=version,
                    fail=stage == fail_stage,
                ),
            )
        )
    return tuple(components)


def _operation(
    stage: str,
    *,
    version: str,
    fail: bool,
) -> Callable[[PipelineStageContext], PipelineStageProduct]:
    def run(context: PipelineStageContext) -> PipelineStageProduct:
        assert context.evidence is not None
        if fail:
            return PipelineStageProduct.outcome(
                StoreState.FAILURE,
                "synthetic_stage_failure",
                output_digest=canonical_digest(
                    {"stage": stage, "state": "failure", "version": version}
                ),
            )
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
        adapter = MappingFactAdapter(
            component="openmed.pipeline.synthetic.extractor",
            component_version=version,
            output_schema="synthetic.condition",
            output_schema_version="1.0.0",
            kind="extraction",
            field_paths={
                "assertion": "assertion",
                "certainty": "certainty",
                "experiencer": "experiencer",
                "status": "status",
                "value": "value",
            },
        )
        adapted = adapter.adapt(
            {
                "assertion": "affirmed",
                "certainty": "certain",
                "experiencer": "patient",
                "status": "active",
                "value": {
                    "code": f"synthetic-{context.evidence.source_format}",
                    "system": "synthetic",
                },
            },
            evidence_ids=(context.evidence.locators[0].locator_id,),
        )
        assert adapted.ok and adapted.value is not None
        fragment = adapted.value
        return PipelineStageProduct(
            output_digest=canonical_digest(
                {"fragment_id": fragment.fragment_id, "version": version}
            ),
            fragments=(fragment,),
            output_record_ids=(fragment.fragment_id,),
        )

    return run


def _manifest(
    pipeline: IngestionToFactPipeline,
    *,
    adapter: Any,
    source: str,
    source_id: str,
    suffix: str,
) -> SourceManifest:
    return SourceManifest(
        manifest_id=derived_opaque_id("manifest", source_id, suffix),
        source_id=source_id,
        artifact_digests=(sha256_digest(source),),
        policy_digest=pipeline.policy_digest,
        pipeline_digest=pipeline.pipeline_digest(adapter),
        created_at=T0,
    )


@pytest.mark.parametrize(
    "source_fixture", _fixture_sources(), ids=lambda item: item["format"]
)
def test_five_source_pipeline_is_durable_and_idempotent(
    tmp_path: Path,
    source_fixture: dict[str, str],
) -> None:
    store = SQLiteIngestionStore(
        tmp_path / f"{source_fixture['format']}-journey.sqlite3"
    )
    pipeline = IngestionToFactPipeline(store, _components())
    adapter = ADAPTERS[source_fixture["format"]]()
    source_id = derived_opaque_id("source", source_fixture["format"])
    manifest = _manifest(
        pipeline,
        adapter=adapter,
        source=source_fixture["payload"],
        source_id=source_id,
        suffix="initial",
    )
    context = EvidenceAdapterContext(
        source_id=source_id,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        recorded_at=T0,
    )

    first = pipeline.run(
        manifest=manifest,
        source=source_fixture["payload"],
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )
    replay = pipeline.run(
        manifest=manifest,
        source=source_fixture["payload"],
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )

    assert first.ok and first.created and first.value is not None
    assert first.value.state is StoreState.SUCCESS
    assert len(first.value.stage_manifests) == 11
    assert tuple(item.stage for item in first.value.stage_manifests) == (
        "source_adaptation",
        "privacy_policy",
        "document_routing",
        "extraction",
        "assertion",
        "temporality",
        "relation_extraction",
        "grounding",
        "fact_normalization",
        "validation",
        "durable_writes",
    )
    assert replay.ok and not replay.created and replay.value is not None
    assert replay.value.replayed
    assert replay.value.fact_ids == first.value.fact_ids
    facts = store.list_facts(SUBJECT_ID)
    assert facts.ok and facts.value is not None and len(facts.value) == 1
    assert store.ingestion_integrity_check().ok
    store.close()


def test_golden_journey_fact_code_resolves_with_versioned_mapping(
    tmp_path: Path,
) -> None:
    """Carry one synthetic Journey fact through the public mapping contract."""

    journey_store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    pipeline = IngestionToFactPipeline(journey_store, _components())
    adapter = TextEvidenceAdapter()
    source = "Synthetic terminology journey fixture."
    source_id = derived_opaque_id("source", "terminology-golden")
    manifest = _manifest(
        pipeline,
        adapter=adapter,
        source=source,
        source_id=source_id,
        suffix="mapping",
    )
    context = EvidenceAdapterContext(
        source_id=source_id,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        recorded_at=T0,
    )

    ingested = pipeline.run(
        manifest=manifest,
        source=source,
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )
    assert ingested.ok
    facts = journey_store.list_facts(SUBJECT_ID)
    assert facts.ok and facts.value is not None
    fact = facts.value[0]
    assert isinstance(fact.value, Mapping)

    target_index = VocabularyIndex(
        "loinc",
        (
            VocabConcept(
                system="loinc",
                code="SYN-100",
                preferred_term="Synthetic target concept",
            ),
        ),
    )
    snapshot = TerminologySnapshot(
        vocabulary="synthetic-loinc",
        version="2026.1",
        index=target_index,
    )
    resolver = TerminologyResolver(
        snapshot,
        hmac_secret="synthetic-journey-secret-material",
        relationships=(
            TerminologyRelationshipRule(
                source_system=str(fact.value["system"]),
                source_code=str(fact.value["code"]),
                target_code="SYN-100",
                relationship="equivalent",
                mapping_rule="explicit_code",
            ),
        ),
    )
    mapped = resolver.resolve(
        TerminologyQuery(
            source_system=str(fact.value["system"]),
            source_code=str(fact.value["code"]),
        )
    )
    assert mapped.ok and mapped.value is not None
    assert mapped.value.selected_candidate is not None
    assert mapped.value.selected_candidate.code == "SYN-100"
    assert mapped.value.selected_candidate.vocabulary_version == "2026.1"

    with SQLiteTerminologyMappingStore(tmp_path / "terminology.sqlite3") as store:
        recorded = store.record(mapped.value, recorded_at=T0)
        history = store.history(mapped.value.source_digest)
        review = store.review_queue()
    coverage = TerminologyCoverageSummary.from_results((mapped.value,)).to_dict()

    assert recorded.ok and recorded.created
    assert history.ok and history.value is not None and len(history.value) == 1
    assert review.ok and review.value == ()
    assert coverage["mapped_rate"] == 1.0
    assert coverage["unmapped_rate"] == 0.0
    assert source not in mapped.value.to_json()
    journey_store.close()


def test_golden_journey_correction_appends_reconciliation_history(
    tmp_path: Path,
) -> None:
    """Carry a normalized fact through correction and canonical replay."""

    store = SQLiteIngestionStore(tmp_path / "correction-journey.sqlite3")
    pipeline = IngestionToFactPipeline(store, _components())
    adapter = TextEvidenceAdapter()
    source = "Synthetic correction journey fixture."
    source_id = derived_opaque_id("source", "correction-golden")
    manifest = _manifest(
        pipeline,
        adapter=adapter,
        source=source,
        source_id=source_id,
        suffix="correction",
    )
    context = EvidenceAdapterContext(
        source_id=source_id,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        recorded_at=T0,
    )
    ingested = pipeline.run(
        manifest=manifest,
        source=source,
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )
    assert ingested.ok
    facts = store.list_facts(SUBJECT_ID)
    assert facts.value is not None
    original = facts.value[0]
    corrected = replace(
        original,
        fact_id=derived_opaque_id("fact", original.fact_id, "correction"),
        value={"code": "synthetic-text-corrected", "system": "synthetic"},
        status="corrected",
        parent_fact_ids=(original.fact_id,),
        derivation_hash=canonical_digest(
            {"original_fact_id": original.fact_id, "operation": "correction"}
        ),
    )
    written = store.put_fact(corrected, committed_at="2026-01-02T04:04:05Z")
    assert written.ok and written.created

    reconciliation_id = derived_opaque_id("canonical", SUBJECT_ID, "synthetic-text")
    plan = FactReconciler(
        FactReconciliationPolicy(prefer_valid_amendment=True)
    ).reconcile(
        SUBJECT_ID,
        (
            FactReconciliationInput(
                fact=original,
                reconciliation_id=reconciliation_id,
                source="source.synthetic",
            ),
            FactReconciliationInput(
                fact=corrected,
                reconciliation_id=reconciliation_id,
                source="source.synthetic",
                amendment_of=original.fact_id,
            ),
        ),
        occurred_at="2026-01-02T05:04:05Z",
    )
    assert plan.ok and plan.value is not None
    persisted = persist_fact_reconciliation(
        plan.value,
        store,
        committed_at="2026-01-02T05:04:05Z",
    )

    assert persisted.ok and persisted.created
    canonical = store.get_canonical(reconciliation_id)
    assert canonical.ok and canonical.value is not None
    assert canonical.value.record.fact_id == corrected.fact_id
    assert canonical.value.record.metadata["policy_version"] == "1.0.0"
    history = store.list_facts(SUBJECT_ID)
    assert history.value is not None
    assert tuple(item.fact_id for item in history.value) == (
        original.fact_id,
        corrected.fact_id,
    )
    assert source not in plan.value.to_json()
    store.close()


def test_failed_stage_commits_no_fact_or_successful_job_and_no_raw_audit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.sqlite3"
    store = SQLiteIngestionStore(path)
    pipeline = IngestionToFactPipeline(store, _components(fail_stage="grounding"))
    adapter = TextEvidenceAdapter()
    raw_canary = "synthetic-raw-canary-must-not-enter-audit"
    source_id = "source_bbbbbbbbbbbbbbbb"
    manifest = _manifest(
        pipeline,
        adapter=adapter,
        source=raw_canary,
        source_id=source_id,
        suffix="failed",
    )

    result = pipeline.run(
        manifest=manifest,
        source=raw_canary,
        adapter=adapter,
        adapter_context=EvidenceAdapterContext(
            source_id=source_id,
            subject_id=SUBJECT_ID,
            encounter_id=ENCOUNTER_ID,
            recorded_at=T0,
        ),
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "synthetic_stage_failure"
    assert result.value is not None
    assert result.value.stage_manifests[-1].stage == "grounding"
    assert result.value.stage_manifests[-1].state == "failure"
    job = store.get_job(result.value.job_id)
    assert job.ok and job.value is not None and job.value.state == "quarantined"
    facts = store.list_facts(SUBJECT_ID)
    assert facts.ok and facts.value == ()
    store.close()
    assert raw_canary.encode("utf-8") not in path.read_bytes()


def test_malformed_source_stops_at_adaptation_and_is_quarantined(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "malformed.sqlite3")
    pipeline = IngestionToFactPipeline(store, _components())
    adapter = FHIRR4EvidenceAdapter()
    source = '{"resourceType":"Condition",'
    source_id = "source_dddddddddddddddd"
    manifest = _manifest(
        pipeline,
        adapter=adapter,
        source=source,
        source_id=source_id,
        suffix="malformed",
    )

    result = pipeline.run(
        manifest=manifest,
        source=source,
        adapter=adapter,
        adapter_context=EvidenceAdapterContext(
            source_id=source_id,
            subject_id=SUBJECT_ID,
            encounter_id=ENCOUNTER_ID,
            recorded_at=T0,
        ),
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "fhir_json_malformed"
    assert result.value is not None
    assert tuple(item.stage for item in result.value.stage_manifests) == (
        "source_adaptation",
    )
    assert store.get_job(result.value.job_id).value.state == "quarantined"
    assert store.list_facts(SUBJECT_ID).value == ()
    store.close()


def test_selected_stage_reprocessing_preserves_ancestors_and_appends_fact(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "reprocess.sqlite3")
    adapter = TextEvidenceAdapter()
    source = "Synthetic note: condition alpha remains active."
    source_id = "source_cccccccccccccccc"
    context = EvidenceAdapterContext(
        source_id=source_id,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        recorded_at=T0,
    )
    first_pipeline = IngestionToFactPipeline(store, _components())
    first_manifest = _manifest(
        first_pipeline,
        adapter=adapter,
        source=source,
        source_id=source_id,
        suffix="first",
    )
    first = first_pipeline.run(
        manifest=first_manifest,
        source=source,
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at=T0,
        worker_id=WORKER_ID,
    )
    assert first.ok and first.value is not None

    revised_pipeline = IngestionToFactPipeline(
        store,
        _components(extraction_version="1.1.0"),
    )
    revised_manifest = _manifest(
        revised_pipeline,
        adapter=adapter,
        source=source,
        source_id=source_id,
        suffix="revised",
    )
    revised = revised_pipeline.run(
        manifest=revised_manifest,
        source=source,
        adapter=adapter,
        adapter_context=context,
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        fact_profile="condition",
        recorded_at="2026-01-02T04:04:05Z",
        worker_id="worker_bbbbbbbbbbbbbbbb",
        reprocess_from="extraction",
        previous_job_id=first.value.job_id,
    )

    assert revised.ok and revised.value is not None
    assert revised.value.job_id != first.value.job_id
    assert revised_manifest.manifest_digest != first_manifest.manifest_digest
    assert revised.value.fact_ids != first.value.fact_ids
    assert revised.value.stage_manifests[:3] == first.value.stage_manifests[:3]
    invalidated_stages = {
        item.stage_manifest_id for item in revised.value.invalidations
    }
    assert invalidated_stages == {
        item.stage_manifest_id for item in first.value.stage_manifests[3:]
    }
    assert not invalidated_stages & {
        item.stage_manifest_id for item in first.value.stage_manifests[:3]
    }
    facts = store.list_facts(SUBJECT_ID)
    assert facts.ok and facts.value is not None and len(facts.value) == 2
    store.close()
