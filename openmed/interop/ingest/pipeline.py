"""Idempotent local ingestion-to-fact orchestration.

The orchestrator keeps clinical values in memory until the final Journey-store
transaction.  Its durable audit surface contains only opaque identifiers,
digests, versions, policy identities, typed states, and derivation edges.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
)
from openmed.structured.facts import (
    NORMALIZER_VERSION,
    ClinicalFactNormalizer,
    FactFragment,
    FactNormalizationRequest,
    NormalizedClinicalFact,
)
from openmed.structured.store import StoreResult, StoreState

from .contracts import SourceManifest
from .evidence_adapters import (
    EvidenceAdapterContext,
    StructuredEvidenceAdapter,
)
from .evidence_contracts import (
    StructuredEvidenceQuarantine,
    StructuredEvidenceResult,
)
from .pipeline_contracts import (
    PIPELINE_STAGES,
    PipelineStageInvalidation,
    PipelineStageManifest,
    build_stage_manifest,
)
from .runner import IngestionCoordinator, QuarantineRequest, StepOutput
from .store import IngestionLedger

PIPELINE_ORCHESTRATOR_VERSION = "1.0.0"
PIPELINE_COMPONENT_STAGES = PIPELINE_STAGES[1:8]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class PipelineStageProduct:
    """In-memory stage output plus value-free durable identities."""

    output_digest: str
    payload: Any = field(default=None, repr=False)
    fragments: tuple[FactFragment, ...] = ()
    normalized_facts: tuple[NormalizedClinicalFact, ...] = field(
        default=(),
        repr=False,
    )
    output_record_ids: tuple[str, ...] = ()
    state: StoreState = StoreState.SUCCESS
    reason_code: str | None = None
    committed_revision: int | None = None

    def __post_init__(self) -> None:
        _digest(self.output_digest, "output_digest")
        fragments = tuple(self.fragments)
        if not all(isinstance(item, FactFragment) for item in fragments):
            raise TypeError("fragments must contain FactFragment records")
        facts = tuple(self.normalized_facts)
        if not all(isinstance(item, NormalizedClinicalFact) for item in facts):
            raise TypeError("normalized_facts contains an invalid record")
        record_ids = _opaque_ids(self.output_record_ids, "output_record_ids")
        if self.state is StoreState.SUCCESS and self.reason_code is not None:
            raise ValueError("successful stage product cannot have a reason code")
        if self.state is not StoreState.SUCCESS:
            _controlled(self.reason_code, "reason_code")
        if self.committed_revision is not None and (
            type(self.committed_revision) is not int or self.committed_revision < 1
        ):
            raise ValueError("committed_revision must be positive")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "normalized_facts", facts)
        object.__setattr__(self, "output_record_ids", record_ids)

    @classmethod
    def outcome(
        cls,
        state: StoreState,
        reason_code: str,
        *,
        output_digest: str,
        payload: Any = None,
        fragments: Sequence[FactFragment] = (),
        normalized_facts: Sequence[NormalizedClinicalFact] = (),
        output_record_ids: Sequence[str] = (),
    ) -> "PipelineStageProduct":
        """Build a typed non-success product without changing its state."""

        if state is StoreState.SUCCESS:
            raise ValueError("use PipelineStageProduct directly for success")
        return cls(
            output_digest=output_digest,
            payload=payload,
            fragments=tuple(fragments),
            normalized_facts=tuple(normalized_facts),
            output_record_ids=tuple(output_record_ids),
            state=state,
            reason_code=reason_code,
        )


@dataclass(frozen=True, slots=True)
class PipelineStageContext:
    """Protected in-memory inputs available to one pipeline component."""

    stage: str
    source: Any = field(repr=False)
    source_manifest: SourceManifest
    evidence: StructuredEvidenceResult | None = field(default=None, repr=False)
    fragments: tuple[FactFragment, ...] = ()
    normalized_facts: tuple[NormalizedClinicalFact, ...] = field(
        default=(),
        repr=False,
    )
    products: Mapping[str, PipelineStageProduct] = field(
        default_factory=dict,
        repr=False,
    )
    subject_id: str = ""
    encounter_id: str | None = None
    fact_profile: str = ""
    parent_fact_ids: tuple[str, ...] = ()


@runtime_checkable
class PipelineComponent(Protocol):
    """One explicitly versioned, local pipeline stage component."""

    stage: str
    component: str
    component_version: str
    policy_digest: str

    def run(self, context: PipelineStageContext) -> PipelineStageProduct:
        """Execute one stage without external side effects."""


@dataclass(frozen=True, slots=True)
class CallablePipelineComponent:
    """Adapt a local callable to the versioned pipeline component contract."""

    stage: str
    component: str
    component_version: str
    policy_digest: str
    operation: Callable[[PipelineStageContext], PipelineStageProduct] = field(
        repr=False
    )

    def __post_init__(self) -> None:
        if self.stage not in PIPELINE_COMPONENT_STAGES:
            raise ValueError("component stage is unsupported")
        _controlled(self.component, "component")
        _version(self.component_version, "component_version")
        _digest(self.policy_digest, "policy_digest")
        if not callable(self.operation):
            raise TypeError("operation must be callable")

    def run(self, context: PipelineStageContext) -> PipelineStageProduct:
        """Run the wrapped local callable."""

        product = self.operation(context)
        if not isinstance(product, PipelineStageProduct):
            raise TypeError("pipeline component returned an invalid product")
        return product


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """Typed terminal result for one ingestion derivation."""

    job_id: str
    manifest_digest: str
    state: StoreState
    stage_manifests: tuple[PipelineStageManifest, ...]
    fact_ids: tuple[str, ...]
    replayed: bool = False
    reason_code: str | None = None
    invalidations: tuple[PipelineStageInvalidation, ...] = ()

    def __post_init__(self) -> None:
        _opaque(self.job_id, "job_id")
        _digest(self.manifest_digest, "manifest_digest")
        manifests = tuple(self.stage_manifests)
        if not manifests:
            raise ValueError("pipeline run requires at least one stage manifest")
        facts = _opaque_ids(self.fact_ids, "fact_ids")
        invalidations = tuple(self.invalidations)
        if self.state is StoreState.SUCCESS and self.reason_code is not None:
            raise ValueError("successful pipeline run cannot have a reason code")
        if self.state is not StoreState.SUCCESS:
            _controlled(self.reason_code, "reason_code")
        object.__setattr__(self, "stage_manifests", manifests)
        object.__setattr__(self, "fact_ids", facts)
        object.__setattr__(self, "invalidations", invalidations)


@runtime_checkable
class IngestionPipelineStore(IngestionLedger, Protocol):
    """Combined control-plane and Journey transaction surface."""

    def transaction(self, *, committed_at: str) -> Any:
        """Open one atomic Journey-store transaction."""

    def get_active_lease(self, job_id: str, *, at: str) -> StoreResult[Any]:
        """Read the job's current live lease."""

    def put_pipeline_stage(
        self,
        manifest: PipelineStageManifest,
    ) -> StoreResult[PipelineStageManifest]:
        """Persist one safe stage manifest and its derivation edges."""

    def list_pipeline_stages(
        self,
        job_id: str,
    ) -> StoreResult[tuple[PipelineStageManifest, ...]]:
        """List append-only stage manifests for one job."""

    def invalidate_pipeline_descendants(
        self,
        job_id: str,
        *,
        from_stage_manifest_id: str,
        replacement_job_id: str,
        recorded_at: str,
    ) -> StoreResult[tuple[PipelineStageInvalidation, ...]]:
        """Invalidate one replaced stage and only its graph descendants."""

    def list_pipeline_invalidations(
        self,
        job_id: str,
    ) -> StoreResult[tuple[PipelineStageInvalidation, ...]]:
        """List append-only stage invalidations for one job."""


class IngestionToFactPipeline:
    """Compose evidence adaptation, model stages, normalization, and storage."""

    def __init__(
        self,
        store: IngestionPipelineStore,
        components: Sequence[PipelineComponent],
        *,
        normalizer: ClinicalFactNormalizer | None = None,
    ) -> None:
        if not isinstance(store, IngestionLedger):
            raise TypeError("store does not satisfy ingestion contracts")
        component_map = {component.stage: component for component in components}
        if len(component_map) != len(tuple(components)):
            raise ValueError("pipeline component stages must be unique")
        if set(component_map) != set(PIPELINE_COMPONENT_STAGES):
            raise ValueError("pipeline requires every processing component stage")
        self.store = store
        self.coordinator = IngestionCoordinator(store)
        self.components = MappingProxyType(component_map)
        self.normalizer = normalizer or ClinicalFactNormalizer()

    def pipeline_digest(self, adapter: StructuredEvidenceAdapter) -> str:
        """Return the deterministic pipeline identity for one source adapter."""

        return canonical_digest(
            {
                "adapter": {
                    "component": adapter.parser_id,
                    "version": adapter.parser_version,
                },
                "components": [
                    {
                        "component": self.components[stage].component,
                        "policy_digest": self.components[stage].policy_digest,
                        "stage": stage,
                        "version": self.components[stage].component_version,
                    }
                    for stage in PIPELINE_COMPONENT_STAGES
                ],
                "normalizer": {
                    "component": "clinical.fact.normalizer",
                    "version": NORMALIZER_VERSION,
                },
                "orchestrator_version": PIPELINE_ORCHESTRATOR_VERSION,
                "stages": list(PIPELINE_STAGES),
                "validation": {
                    "component": "openmed.pipeline.evidence_validation",
                    "version": PIPELINE_ORCHESTRATOR_VERSION,
                },
                "writes": {
                    "component": "openmed.pipeline.journey_store",
                    "version": PIPELINE_ORCHESTRATOR_VERSION,
                },
            }
        )

    @property
    def policy_digest(self) -> str:
        """Return the combined identity of every configured stage policy."""

        return self._combined_policy_digest()

    def run(
        self,
        *,
        manifest: SourceManifest,
        source: Any,
        adapter: StructuredEvidenceAdapter,
        adapter_context: EvidenceAdapterContext,
        subject_id: str,
        fact_profile: str,
        recorded_at: str,
        worker_id: str,
        encounter_id: str | None = None,
        parent_fact_ids: Sequence[str] = (),
        attempt: int = 1,
        lease_seconds: int = 3600,
        reprocess_from: str | None = None,
        previous_job_id: str | None = None,
    ) -> StoreResult[PipelineRun]:
        """Run or replay one evidence-preserving ingestion derivation."""

        _opaque(subject_id, "subject_id")
        _opaque(worker_id, "worker_id")
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be positive")
        if encounter_id is not None:
            _opaque(encounter_id, "encounter_id")
        parents = _opaque_ids(tuple(parent_fact_ids), "parent_fact_ids")
        if manifest.pipeline_digest != self.pipeline_digest(adapter):
            return StoreResult.outcome(
                StoreState.CONFLICT,
                "pipeline_digest_mismatch",
            )
        if manifest.policy_digest != self._combined_policy_digest():
            return StoreResult.outcome(
                StoreState.CONFLICT,
                "pipeline_policy_mismatch",
            )
        reprocessing = reprocess_from is not None or previous_job_id is not None
        if reprocessing and (reprocess_from is None or previous_job_id is None):
            return StoreResult.outcome(
                StoreState.FAILURE,
                "reprocess_context_incomplete",
            )
        if reprocess_from is not None and reprocess_from not in PIPELINE_STAGES:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED,
                "reprocess_stage_unsupported",
            )

        registration = self.coordinator.register(manifest, recorded_at=recorded_at)
        if not registration.ok or registration.value is None:
            return StoreResult.outcome(
                registration.state,
                registration.code or "manifest_registration_failed",
            )
        job = registration.value.job
        if not registration.created:
            if reprocessing:
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "reprocess_manifest_unchanged",
                )
            if job.state == "completed":
                return self._completed_replay(job.job_id, manifest.manifest_digest)

        # Stage computation and reprocessing can have side effects. Claim the
        # job before either, not only when recording their checkpoints.
        lease = self.store.acquire_lease(
            job.job_id,
            worker_id,
            acquired_at=recorded_at,
            duration_seconds=lease_seconds,
        )
        if not lease.ok or lease.value is None:
            return StoreResult.outcome(
                lease.state, lease.code or "lease_acquisition_failed"
            )

        prior_manifests: tuple[PipelineStageManifest, ...] = ()
        invalidations: tuple[PipelineStageInvalidation, ...] = ()
        start_index = 0
        if reprocess_from is not None and previous_job_id is not None:
            prior = self.store.list_pipeline_stages(previous_job_id)
            if not prior.ok or prior.value is None:
                return StoreResult.outcome(
                    prior.state,
                    prior.code or "prior_lineage_unavailable",
                )
            prior_by_stage = {item.stage: item for item in prior.value}
            if reprocess_from not in prior_by_stage:
                return StoreResult.outcome(
                    StoreState.UNKNOWN,
                    "reprocess_stage_not_found",
                )
            start_index = PIPELINE_STAGES.index(reprocess_from)
            prior_manifests = tuple(
                prior_by_stage[stage]
                for stage in PIPELINE_STAGES[:start_index]
                if stage in prior_by_stage
            )
            if len(prior_manifests) != start_index:
                return StoreResult.outcome(
                    StoreState.PARTIAL,
                    "prior_lineage_incomplete",
                )
            invalidated = self.store.invalidate_pipeline_descendants(
                previous_job_id,
                from_stage_manifest_id=prior_by_stage[reprocess_from].stage_manifest_id,
                replacement_job_id=job.job_id,
                recorded_at=recorded_at,
            )
            if not invalidated.ok or invalidated.value is None:
                return StoreResult.outcome(
                    invalidated.state,
                    invalidated.code or "lineage_invalidation_failed",
                )
            invalidations = invalidated.value

        state = _MutablePipelineState(
            source=source,
            manifest=manifest,
            subject_id=subject_id,
            encounter_id=encounter_id,
            fact_profile=fact_profile,
            parent_fact_ids=parents,
        )
        if start_index:
            seeded = self._seed_context(
                state,
                adapter=adapter,
                adapter_context=adapter_context,
                stop_before=start_index,
            )
            if not seeded.ok:
                return StoreResult.outcome(
                    seeded.state,
                    seeded.code or "reprocess_seed_failed",
                )

        stage_manifests = list(prior_manifests)
        parent_stage_ids = (
            (prior_manifests[-1].stage_manifest_id,) if prior_manifests else ()
        )
        for index, stage in enumerate(PIPELINE_STAGES[start_index:], start=start_index):
            component, component_version, policy_digest = self._stage_identity(
                stage,
                adapter,
                manifest.policy_digest,
            )
            input_digests = self._stage_input_digests(state, stage)
            input_record_ids = self._stage_input_record_ids(state, stage)
            try:
                product = self._run_stage(
                    stage,
                    state,
                    adapter=adapter,
                    adapter_context=adapter_context,
                    recorded_at=recorded_at,
                )
            except Exception as error:
                execution = self.coordinator.execute_step(
                    job_id=job.job_id,
                    worker_id=worker_id,
                    step=stage,
                    input_digest=canonical_digest(list(input_digests)),
                    acquired_at=recorded_at,
                    lease_seconds=lease_seconds,
                    attempt=attempt,
                    operation=_failure_operation(error),
                )
                reason_code = execution.code or "stage_execution_failed"
                failure_digest = canonical_digest(
                    {"reason_code": reason_code, "stage": stage}
                )
                stage_manifest = build_stage_manifest(
                    job_id=job.job_id,
                    stage=stage,
                    state=StoreState.FAILURE.value,
                    input_digests=input_digests,
                    output_digests=(failure_digest,),
                    input_record_ids=input_record_ids,
                    output_record_ids=(),
                    component=component,
                    component_version=component_version,
                    policy_digest=policy_digest,
                    parent_stage_manifest_ids=parent_stage_ids,
                    recorded_at=recorded_at,
                    reason_code=reason_code,
                )
                persisted = self.store.put_pipeline_stage(stage_manifest)
                if persisted.ok:
                    stage_manifests.append(stage_manifest)
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    reason_code,
                    value=PipelineRun(
                        job_id=job.job_id,
                        manifest_digest=manifest.manifest_digest,
                        state=StoreState.FAILURE,
                        stage_manifests=tuple(stage_manifests),
                        fact_ids=(),
                        reason_code=reason_code,
                        invalidations=invalidations,
                    ),
                )

            operation: Callable[[], StepOutput | QuarantineRequest]
            if product.state is StoreState.SUCCESS:
                operation = lambda: StepOutput(
                    product.output_digest,
                    product.committed_revision,
                )
            else:
                operation = lambda: QuarantineRequest(
                    classification=_quarantine_classification(product.state),
                    reason_code=product.reason_code or "stage_not_successful",
                    candidate_count=len(product.output_record_ids),
                    failure_count=1,
                    output_digest=product.output_digest,
                )
            execution = self.coordinator.execute_step(
                job_id=job.job_id,
                worker_id=worker_id,
                step=stage,
                input_digest=canonical_digest(list(input_digests)),
                acquired_at=recorded_at,
                lease_seconds=lease_seconds,
                attempt=attempt,
                operation=operation,
            )
            stage_manifest = build_stage_manifest(
                job_id=job.job_id,
                stage=stage,
                state=product.state.value,
                input_digests=input_digests,
                output_digests=(product.output_digest,),
                input_record_ids=input_record_ids,
                output_record_ids=product.output_record_ids,
                component=component,
                component_version=component_version,
                policy_digest=policy_digest,
                parent_stage_manifest_ids=parent_stage_ids,
                recorded_at=recorded_at,
                reason_code=product.reason_code,
            )
            persisted = self.store.put_pipeline_stage(stage_manifest)
            if not persisted.ok:
                return StoreResult.outcome(
                    persisted.state,
                    persisted.code or "stage_manifest_write_failed",
                )
            stage_manifests.append(stage_manifest)
            parent_stage_ids = (stage_manifest.stage_manifest_id,)
            if product.state is not StoreState.SUCCESS or not execution.ok:
                state_value = (
                    product.state
                    if product.state is not StoreState.SUCCESS
                    else execution.state
                )
                reason_code = (
                    product.reason_code or execution.code or "stage_not_successful"
                )
                return StoreResult.outcome(
                    state_value,
                    reason_code,
                    value=PipelineRun(
                        job_id=job.job_id,
                        manifest_digest=manifest.manifest_digest,
                        state=state_value,
                        stage_manifests=tuple(stage_manifests),
                        fact_ids=(),
                        reason_code=reason_code,
                        invalidations=invalidations,
                    ),
                )
            state.accept(stage, product)

        active_lease = self.store.get_active_lease(job.job_id, at=recorded_at)
        if not active_lease.ok or active_lease.value is None:
            return StoreResult.outcome(
                active_lease.state,
                active_lease.code or "completion_lease_missing",
            )
        completed = self.coordinator.complete(
            job.job_id,
            lease_id=active_lease.value.lease_id,
            completed_at=recorded_at,
        )
        if not completed.ok:
            return StoreResult.outcome(
                completed.state,
                completed.code or "job_completion_failed",
            )
        facts = tuple(item.fact.fact_id for item in state.normalized_facts)
        run = PipelineRun(
            job_id=job.job_id,
            manifest_digest=manifest.manifest_digest,
            state=StoreState.SUCCESS,
            stage_manifests=tuple(stage_manifests),
            fact_ids=facts,
            replayed=False,
            invalidations=invalidations,
        )
        return StoreResult.success(run, created=True)

    def _combined_policy_digest(self) -> str:
        return canonical_digest(
            {
                stage: self.components[stage].policy_digest
                for stage in PIPELINE_COMPONENT_STAGES
            }
        )

    def _completed_replay(
        self,
        job_id: str,
        manifest_digest: str,
    ) -> StoreResult[PipelineRun]:
        stages = self.store.list_pipeline_stages(job_id)
        if not stages.ok or stages.value is None:
            return StoreResult.outcome(
                stages.state,
                stages.code or "completed_lineage_missing",
            )
        final = next(
            (item for item in stages.value if item.stage == "durable_writes"),
            None,
        )
        if final is None or final.state != StoreState.SUCCESS.value:
            return StoreResult.outcome(
                StoreState.FAILURE,
                "completed_lineage_invalid",
            )
        facts = tuple(
            record_id
            for record_id in final.output_record_ids
            if record_id.startswith("fact_")
        )
        return StoreResult.success(
            PipelineRun(
                job_id=job_id,
                manifest_digest=manifest_digest,
                state=StoreState.SUCCESS,
                stage_manifests=stages.value,
                fact_ids=facts,
                replayed=True,
            ),
            created=False,
        )

    def _seed_context(
        self,
        state: "_MutablePipelineState",
        *,
        adapter: StructuredEvidenceAdapter,
        adapter_context: EvidenceAdapterContext,
        stop_before: int,
    ) -> StoreResult[None]:
        for stage in PIPELINE_STAGES[:stop_before]:
            try:
                product = self._run_stage(
                    stage,
                    state,
                    adapter=adapter,
                    adapter_context=adapter_context,
                    recorded_at=state.manifest.created_at,
                    dry_run=True,
                )
            except Exception:
                return StoreResult.outcome(StoreState.FAILURE, "reprocess_seed_failed")
            if product.state is not StoreState.SUCCESS:
                return StoreResult.outcome(
                    product.state,
                    product.reason_code or "reprocess_seed_failed",
                )
            state.accept(stage, product)
        return StoreResult.success(None)

    def _run_stage(
        self,
        stage: str,
        state: "_MutablePipelineState",
        *,
        adapter: StructuredEvidenceAdapter,
        adapter_context: EvidenceAdapterContext,
        recorded_at: str,
        dry_run: bool = False,
    ) -> PipelineStageProduct:
        if stage == "source_adaptation":
            return _adapt_source(adapter, state.source, adapter_context)
        if stage in self.components:
            return self.components[stage].run(state.context(stage))
        if stage == "fact_normalization":
            return self._normalize(state)
        if stage == "validation":
            return _validate_facts(state)
        if stage == "durable_writes":
            if dry_run:
                return _dry_write_product(state)
            return self._write_journey(state, recorded_at)
        raise ValueError("pipeline stage is unsupported")

    def _normalize(self, state: "_MutablePipelineState") -> PipelineStageProduct:
        if state.evidence is None or not state.fragments:
            return PipelineStageProduct.outcome(
                StoreState.UNKNOWN,
                "fact_inputs_missing",
                output_digest=canonical_digest({"state": "fact_inputs_missing"}),
            )
        request = FactNormalizationRequest(
            subject_id=state.subject_id,
            encounter_id=state.encounter_id,
            profile=state.fact_profile,
            fragments=state.fragments,
            parent_fact_ids=state.parent_fact_ids,
        )
        evidence_by_id = {
            locator.locator_id: locator for locator in state.evidence.locators
        }
        result = self.normalizer.normalize(request, evidence_by_id=evidence_by_id)
        if result.value is None:
            return PipelineStageProduct.outcome(
                result.state,
                result.code or "fact_normalization_failed",
                output_digest=canonical_digest(
                    {
                        "code": result.code or "fact_normalization_failed",
                        "fragment_ids": [item.fragment_id for item in state.fragments],
                    }
                ),
            )
        normalized = result.value
        digest = canonical_digest(normalized.to_dict())
        if not result.ok:
            return PipelineStageProduct.outcome(
                result.state,
                result.code or "fact_normalization_failed",
                output_digest=digest,
                normalized_facts=(normalized,),
                output_record_ids=(normalized.fact.fact_id,),
            )
        return PipelineStageProduct(
            output_digest=digest,
            payload=normalized,
            normalized_facts=(normalized,),
            output_record_ids=(normalized.fact.fact_id,),
        )

    def _write_journey(
        self,
        state: "_MutablePipelineState",
        recorded_at: str,
    ) -> PipelineStageProduct:
        if state.evidence is None or not state.normalized_facts:
            return PipelineStageProduct.outcome(
                StoreState.UNKNOWN,
                "journey_write_inputs_missing",
                output_digest=canonical_digest(
                    {"state": "journey_write_inputs_missing"}
                ),
            )
        created = False
        revision: int | None = None
        try:
            with self.store.transaction(committed_at=recorded_at) as transaction:
                artifact_result = transaction.put_artifact(state.evidence.artifact)
                if not artifact_result.ok:
                    raise _WriteRejected(
                        artifact_result.state,
                        artifact_result.code or "artifact_write_failed",
                    )
                created = created or artifact_result.created
                for locator in state.evidence.locators:
                    evidence_result = transaction.put_evidence(locator)
                    if not evidence_result.ok:
                        raise _WriteRejected(
                            evidence_result.state,
                            evidence_result.code or "evidence_write_failed",
                        )
                    created = created or evidence_result.created
                for normalized in state.normalized_facts:
                    fact_result = transaction.put_fact(normalized.fact)
                    if not fact_result.ok:
                        raise _WriteRejected(
                            fact_result.state,
                            fact_result.code or "fact_write_failed",
                        )
                    created = created or fact_result.created
                if created:
                    revision = transaction.revision
        except _WriteRejected as rejected:
            return PipelineStageProduct.outcome(
                rejected.state,
                rejected.reason_code,
                output_digest=canonical_digest(
                    {"reason_code": rejected.reason_code, "state": rejected.state.value}
                ),
            )
        records = (
            state.evidence.artifact.artifact_id,
            *(item.locator_id for item in state.evidence.locators),
            *(item.fact.fact_id for item in state.normalized_facts),
        )
        return PipelineStageProduct(
            output_digest=canonical_digest(
                {"record_ids": sorted(records), "schema": "journey-1"}
            ),
            output_record_ids=records,
            committed_revision=revision,
        )

    def _stage_identity(
        self,
        stage: str,
        adapter: StructuredEvidenceAdapter,
        manifest_policy_digest: str,
    ) -> tuple[str, str, str]:
        if stage == "source_adaptation":
            return adapter.parser_id, adapter.parser_version, manifest_policy_digest
        if stage in self.components:
            component = self.components[stage]
            return (
                component.component,
                component.component_version,
                component.policy_digest,
            )
        if stage == "fact_normalization":
            return (
                "clinical.fact.normalizer",
                NORMALIZER_VERSION,
                manifest_policy_digest,
            )
        if stage == "validation":
            return (
                "openmed.pipeline.evidence_validation",
                PIPELINE_ORCHESTRATOR_VERSION,
                manifest_policy_digest,
            )
        return (
            "openmed.pipeline.journey_store",
            PIPELINE_ORCHESTRATOR_VERSION,
            manifest_policy_digest,
        )

    @staticmethod
    def _stage_input_digests(
        state: "_MutablePipelineState",
        stage: str,
    ) -> tuple[str, ...]:
        if stage == "source_adaptation":
            return state.manifest.artifact_digests
        previous = state.products.get(PIPELINE_STAGES[PIPELINE_STAGES.index(stage) - 1])
        if previous is None:
            return (state.manifest.manifest_digest,)
        return (previous.output_digest, state.manifest.manifest_digest)

    @staticmethod
    def _stage_input_record_ids(
        state: "_MutablePipelineState",
        stage: str,
    ) -> tuple[str, ...]:
        if stage == "source_adaptation":
            return (state.manifest.source_id,)
        previous = state.products.get(PIPELINE_STAGES[PIPELINE_STAGES.index(stage) - 1])
        return () if previous is None else previous.output_record_ids


@dataclass(slots=True)
class _MutablePipelineState:
    source: Any = field(repr=False)
    manifest: SourceManifest
    subject_id: str
    encounter_id: str | None
    fact_profile: str
    parent_fact_ids: tuple[str, ...]
    evidence: StructuredEvidenceResult | None = field(default=None, repr=False)
    fragments: tuple[FactFragment, ...] = ()
    normalized_facts: tuple[NormalizedClinicalFact, ...] = field(
        default=(),
        repr=False,
    )
    products: dict[str, PipelineStageProduct] = field(default_factory=dict, repr=False)

    def context(self, stage: str) -> PipelineStageContext:
        return PipelineStageContext(
            stage=stage,
            source=self.source,
            source_manifest=self.manifest,
            evidence=self.evidence,
            fragments=self.fragments,
            normalized_facts=self.normalized_facts,
            products=MappingProxyType(dict(self.products)),
            subject_id=self.subject_id,
            encounter_id=self.encounter_id,
            fact_profile=self.fact_profile,
            parent_fact_ids=self.parent_fact_ids,
        )

    def accept(self, stage: str, product: PipelineStageProduct) -> None:
        self.products[stage] = product
        if isinstance(product.payload, StructuredEvidenceResult):
            self.evidence = product.payload
        if product.fragments:
            self.fragments = (*self.fragments, *product.fragments)
        if product.normalized_facts:
            self.normalized_facts = product.normalized_facts


class _WriteRejected(RuntimeError):
    def __init__(self, state: StoreState, reason_code: str) -> None:
        super().__init__("Journey write rejected")
        self.state = state
        self.reason_code = reason_code


def _adapt_source(
    adapter: StructuredEvidenceAdapter,
    source: Any,
    context: EvidenceAdapterContext,
) -> PipelineStageProduct:
    result = adapter.adapt(source, context)
    if result.value is None:
        return PipelineStageProduct.outcome(
            result.state,
            result.code or "source_adaptation_failed",
            output_digest=canonical_digest(
                {"code": result.code or "source_adaptation_failed"}
            ),
        )
    output = result.value
    digest = canonical_digest(output.to_dict())
    if isinstance(output, StructuredEvidenceQuarantine):
        return PipelineStageProduct.outcome(
            result.state,
            output.reason_code,
            output_digest=digest,
            payload=output,
            output_record_ids=(output.quarantine_id,),
        )
    if not isinstance(output, StructuredEvidenceResult):
        raise TypeError("source adapter returned an invalid record")
    record_ids = (
        output.artifact.artifact_id,
        *(locator.locator_id for locator in output.locators),
    )
    if not result.ok:
        return PipelineStageProduct.outcome(
            result.state,
            result.code or "source_adaptation_failed",
            output_digest=digest,
            payload=output,
            output_record_ids=record_ids,
        )
    return PipelineStageProduct(
        output_digest=digest,
        payload=output,
        output_record_ids=record_ids,
    )


def _validate_facts(state: _MutablePipelineState) -> PipelineStageProduct:
    if state.evidence is None or not state.normalized_facts:
        return PipelineStageProduct.outcome(
            StoreState.UNKNOWN,
            "validation_inputs_missing",
            output_digest=canonical_digest({"state": "validation_inputs_missing"}),
        )
    evidence_ids = {item.locator_id for item in state.evidence.locators}
    facts: tuple[ClinicalFact, ...] = tuple(
        item.fact for item in state.normalized_facts
    )
    if any(
        item.normalization_state != StoreState.SUCCESS.value
        for item in state.normalized_facts
    ):
        return PipelineStageProduct.outcome(
            StoreState.PARTIAL,
            "fact_review_required",
            output_digest=canonical_digest(
                {"fact_ids": [item.fact_id for item in facts], "state": "partial"}
            ),
            normalized_facts=state.normalized_facts,
            output_record_ids=tuple(item.fact_id for item in facts),
        )
    if any(not set(item.evidence_ids) <= evidence_ids for item in facts):
        return PipelineStageProduct.outcome(
            StoreState.CONFLICT,
            "fact_evidence_missing",
            output_digest=canonical_digest(
                {"fact_ids": [item.fact_id for item in facts], "state": "conflict"}
            ),
            normalized_facts=state.normalized_facts,
            output_record_ids=tuple(item.fact_id for item in facts),
        )
    return PipelineStageProduct(
        output_digest=canonical_digest(
            {
                "derivations": [item.derivation_hash for item in facts],
                "fact_ids": [item.fact_id for item in facts],
                "state": "success",
            }
        ),
        normalized_facts=state.normalized_facts,
        output_record_ids=tuple(item.fact_id for item in facts),
    )


def _dry_write_product(state: _MutablePipelineState) -> PipelineStageProduct:
    if state.evidence is None or not state.normalized_facts:
        return PipelineStageProduct.outcome(
            StoreState.UNKNOWN,
            "journey_write_inputs_missing",
            output_digest=canonical_digest({"state": "journey_write_inputs_missing"}),
        )
    record_ids = (
        state.evidence.artifact.artifact_id,
        *(item.locator_id for item in state.evidence.locators),
        *(item.fact.fact_id for item in state.normalized_facts),
    )
    return PipelineStageProduct(
        output_digest=canonical_digest(
            {"record_ids": sorted(record_ids), "schema": "journey-1"}
        ),
        output_record_ids=record_ids,
    )


def _quarantine_classification(state: StoreState) -> str:
    return {
        StoreState.PARTIAL: "partial",
        StoreState.UNKNOWN: "partial",
        StoreState.CONFLICT: "ambiguous",
        StoreState.UNSUPPORTED: "unsupported",
        StoreState.DENIED: "policy_denied",
        StoreState.FAILURE: "unsafe",
    }[state]


def _failure_operation(error: Exception) -> Callable[[], StepOutput]:
    def fail() -> StepOutput:
        raise error

    return fail


def _opaque_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _opaque(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(values))


def _opaque(value: Any, name: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _controlled(value: Any, name: str) -> None:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a controlled identifier")


def _version(value: Any, name: str) -> None:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded version")


__all__ = [
    "PIPELINE_COMPONENT_STAGES",
    "PIPELINE_ORCHESTRATOR_VERSION",
    "CallablePipelineComponent",
    "IngestionPipelineStore",
    "IngestionToFactPipeline",
    "PipelineComponent",
    "PipelineRun",
    "PipelineStageContext",
    "PipelineStageProduct",
]
