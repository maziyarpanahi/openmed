"""Release certification for the DirectID Tiny checkpoint.

This module consumes aggregate, PHI-free outputs from the DirectID dataset,
distillation, safety-sweep, and quantization workflows. It does not load model
weights or source records. Releasable candidates receive a checkpoint manifest
and model card; failing candidates receive a signed quarantine decision only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import stable_hash
from openmed.core.manifest_schema import (
    ALLOWED_FORMATS,
    TRAINING_PROVENANCE_FIELDS,
    validate_manifest_row,
)
from openmed.core.repro_hash import load_training_provenance, verify_reproducibility
from openmed.core.safety_sweep import SAFETY_SWEEP_SOURCE
from openmed.eval import release_gates
from openmed.eval.directid import (
    DIRECTID_EVIDENCE_SCHEMA_VERSION,
    DirectIDEvidenceReport,
)
from openmed.eval.model_card_builder import (
    MODEL_DATASHEET_FILENAME,
    ModelCardBuildResult,
    build_model_card,
)
from openmed.eval.release_gates import (
    G1B_RECALL_FLOOR,
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
    ReleaseGate,
)
from openmed.eval.report import BenchmarkReport
from openmed.training.directid import (
    DIRECTID_CONTRACT_REF,
    DIRECTID_FAMILY,
    DIRECTID_TIER,
    DIRECTID_TINY_HEAD_CONTRACT,
    validate_directid_contract,
)
from openmed.training.directid_dataset import (
    DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION,
    DIRECTID_DATASET_MANIFEST_ID,
    directid_dataset_manifest_hash,
)

DIRECTID_RELEASE_SCHEMA_VERSION = "openmed.eval.directid_release.v1"
DIRECTID_RELEASE_GATE_MILESTONE = "v2.0"
DIRECTID_REQUIRED_GATES = ("G1b", "G3", "G4", "G5")
DIRECTID_MODEL_ID = "OpenMed/OpenMed-PII-DirectID-Tiny"
DIRECTID_CANDIDATE_SCHEMA_VERSION = "openmed.training.directid_candidate.v1"
DIRECTID_TRAINING_REPORT_SCHEMA_VERSION = "openmed.training.directid_training_report.v1"
DIRECTID_RUN_MANIFEST_SCHEMA_VERSION = "openmed.training.directid_run_manifest.v1"
DIRECTID_QUANTIZATION_SCHEMA_VERSION = "openmed.training.directid_quantization.v1"

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9._~:/?#@!$&()*+,;=%-]{1,2048}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DEFAULT_EVIDENCE_REFS: Mapping[str, str] = {
    "candidate_checkpoint": "release-evidence/candidate-checkpoint.json",
    "dataset_evidence": "release-evidence/dataset-evidence.json",
    "directid_evidence": "release-evidence/directid-evidence.json",
    "quantization_evidence": "release-evidence/quantization-evidence.json",
    "run_manifest": "release-evidence/run-manifest.json",
    "training_report": "release-evidence/training-report.json",
}

_DIRECTID_CARD_SECTION = """## DirectID Release Boundary

This checkpoint detects deterministic identifiers for assistive, local-first
de-identification. It must not diagnose, recommend treatment, or trigger a
clinical decision. The signed certification covers G1b structured-identifier
recall, G3 leakage, G4 quantization recall delta, and G5 Tiny-tier fit.

Release evidence contains aggregate counts, hashes, offsets, and provenance
references only. Source records, raw identifiers, restricted dataset payloads,
and model signing secrets are excluded.
"""


class DirectIDReleaseError(ValueError):
    """Raised when DirectID release evidence is incomplete or inconsistent."""


class DirectIDGateFailure(DirectIDReleaseError):
    """Raised when a caller requires publication for a quarantined candidate."""

    def __init__(self, release: "DirectIDRelease") -> None:
        failed = ", ".join(
            check.gate for check in release.gate_report.gate_results if not check.passed
        )
        super().__init__(f"DirectID release gates failed: {failed}")
        self.release = release
        self.report = release.gate_report


@dataclass(frozen=True)
class DirectIDReleasePaths:
    """Paths written for a DirectID release or quarantine evidence package."""

    gate_report: Path
    release_manifest: Path
    checkpoint_manifest: Path | None = None
    model_card: Path | None = None
    model_datasheet: Path | None = None


@dataclass(frozen=True)
class DirectIDRelease:
    """Signed DirectID release decision and optional publishable artifacts."""

    gate_report: GateReport
    release_manifest: Mapping[str, Any]
    checkpoint_manifest: Mapping[str, Any] | None = None
    model_card: ModelCardBuildResult | None = None

    @property
    def published(self) -> bool:
        """Return whether this package contains publishable artifacts."""

        return self.gate_report.decision == RELEASABLE

    def write(self, output_dir: str | Path) -> DirectIDReleasePaths:
        """Write deterministic artifacts beneath *output_dir*.

        Quarantined packages write only the signed gate report and release
        manifest. That makes it impossible for a failing candidate to acquire
        a publishable checkpoint manifest or model card by accident.
        """

        destination = Path(output_dir)
        publish_paths = (
            destination / "checkpoint-manifest.json",
            destination / "README.md",
            destination / MODEL_DATASHEET_FILENAME,
        )
        if not self.published and any(path.exists() for path in publish_paths):
            raise DirectIDReleaseError(
                "refusing to write quarantine evidence beside publishable artifacts"
            )
        destination.mkdir(parents=True, exist_ok=True)
        gate_path = destination / "gate-report.json"
        release_manifest_path = destination / "release-manifest.json"
        gate_path.write_text(self.gate_report.to_json() + "\n", encoding="utf-8")
        _write_json(release_manifest_path, self.release_manifest)

        checkpoint_path: Path | None = None
        card_path: Path | None = None
        datasheet_path: Path | None = None
        if self.published:
            if self.checkpoint_manifest is None or self.model_card is None:
                raise DirectIDReleaseError(
                    "releasable package is missing publishable artifacts"
                )
            checkpoint_path, card_path, datasheet_path = publish_paths
            _write_json(checkpoint_path, self.checkpoint_manifest)
            self.model_card.write_markdown(card_path)
            self.model_card.write_datasheet(datasheet_path)

        return DirectIDReleasePaths(
            gate_report=gate_path,
            release_manifest=release_manifest_path,
            checkpoint_manifest=checkpoint_path,
            model_card=card_path,
            model_datasheet=datasheet_path,
        )


@dataclass(frozen=True)
class _ValidatedInputs:
    candidate: Mapping[str, Any]
    training_report: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    dataset: Mapping[str, Any]
    directid: Mapping[str, Any]
    quantization: Mapping[str, Any]
    artifact: Mapping[str, Any]
    refs: Mapping[str, str]


def build_directid_gate_report(
    directid_evidence: DirectIDEvidenceReport | Mapping[str, Any] | str | Path,
    *,
    candidate_checkpoint: Mapping[str, Any] | str | Path,
    training_report: Mapping[str, Any] | str | Path,
    run_manifest: Mapping[str, Any] | str | Path,
    training_provenance: Mapping[str, Any] | str | Path,
    dataset_evidence: Mapping[str, Any] | str | Path,
    quantization_evidence: Mapping[str, Any] | str | Path,
    release_format: str = "mlx-8bit",
    evidence_refs: Mapping[str, str] | None = None,
    candidate_checkpoint_ref: str | None = None,
    training_report_ref: str | None = None,
    run_manifest_ref: str | None = None,
    dataset_evidence_ref: str | None = None,
    directid_evidence_ref: str | None = None,
    quantization_evidence_ref: str | None = None,
    signing_key: bytes | str,
    key_id: str = "directid-tiny-release-gate",
) -> GateReport:
    """Return a signed G1b/G3/G4/G5 report for one runtime format.

    The generic :class:`ReleaseGate` performs the underlying scoring. This
    wrapper selects the DirectID gates, strengthens their evidence-completeness
    requirements, and binds each result to the upstream aggregate artifacts.
    """

    _require_signing_key(signing_key)
    refs = _evidence_references(
        evidence_refs,
        candidate_checkpoint=candidate_checkpoint_ref,
        training_report=training_report_ref,
        run_manifest=run_manifest_ref,
        dataset_evidence=dataset_evidence_ref,
        directid_evidence=directid_evidence_ref,
        quantization_evidence=quantization_evidence_ref,
    )
    inputs = _validate_inputs(
        directid_evidence=directid_evidence,
        candidate_checkpoint=candidate_checkpoint,
        training_report=training_report,
        run_manifest=run_manifest,
        training_provenance=training_provenance,
        dataset_evidence=dataset_evidence,
        quantization_evidence=quantization_evidence,
        release_format=release_format,
        refs=refs,
    )
    scoring_report = _scoring_report(inputs, release_format=release_format)
    preview = ReleaseGate(
        milestone=DIRECTID_RELEASE_GATE_MILESTONE,
        policy="strict_no_leak",
        signing_key=signing_key,
        key_id=key_id,
    ).preview(scoring_report, baseline={})
    checks = _scoped_checks(preview, inputs=inputs)
    decision = RELEASABLE if all(check.passed for check in checks) else QUARANTINED
    blocked_formats = set(_string_list(inputs.quantization.get("quarantined_formats")))
    blocked_formats.update(
        check.blocking_format
        for check in checks
        if not check.passed and check.blocking_format is not None
    )
    report = GateReport(
        repo_id=preview.repo_id,
        family=preview.family,
        tier=preview.tier,
        param_count=preview.param_count,
        format=preview.format,
        per_label_recall=preview.per_label_recall,
        per_label_precision=preview.per_label_precision,
        critical_leakage_count=preview.critical_leakage_count,
        residual_leakage_rate=preview.residual_leakage_rate,
        quant_recall_delta=preview.quant_recall_delta,
        p50_ms=preview.p50_ms,
        p95_ms=preview.p95_ms,
        ram_mb=preview.ram_mb,
        eval_set_hash=preview.eval_set_hash,
        leakage_fixture_hash=preview.leakage_fixture_hash,
        decision=decision,
        gate_results=checks,
        policy=preview.policy,
        threshold_profile=preview.threshold_profile,
        target_leakage_rate=0.0,
        blocked_formats=tuple(sorted(blocked_formats)),
    )
    return report.sign(signing_key, key_id=key_id)


def build_directid_release(
    directid_evidence: DirectIDEvidenceReport | Mapping[str, Any] | str | Path,
    *,
    candidate_checkpoint: Mapping[str, Any] | str | Path,
    training_report: Mapping[str, Any] | str | Path,
    run_manifest: Mapping[str, Any] | str | Path,
    training_provenance: Mapping[str, Any] | str | Path,
    dataset_evidence: Mapping[str, Any] | str | Path,
    quantization_evidence: Mapping[str, Any] | str | Path,
    release_date: str,
    release_format: str = "mlx-8bit",
    evidence_refs: Mapping[str, str] | None = None,
    candidate_checkpoint_ref: str | None = None,
    training_report_ref: str | None = None,
    run_manifest_ref: str | None = None,
    dataset_evidence_ref: str | None = None,
    directid_evidence_ref: str | None = None,
    quantization_evidence_ref: str | None = None,
    signing_key: bytes | str,
    key_id: str = "directid-tiny-release-gate",
    require_releasable: bool = False,
) -> DirectIDRelease:
    """Build a signed DirectID release or quarantine evidence package."""

    if not isinstance(release_date, str) or _DATE.fullmatch(release_date) is None:
        raise DirectIDReleaseError("release_date must match YYYY-MM-DD")
    refs = _evidence_references(
        evidence_refs,
        candidate_checkpoint=candidate_checkpoint_ref,
        training_report=training_report_ref,
        run_manifest=run_manifest_ref,
        dataset_evidence=dataset_evidence_ref,
        directid_evidence=directid_evidence_ref,
        quantization_evidence=quantization_evidence_ref,
    )
    inputs = _validate_inputs(
        directid_evidence=directid_evidence,
        candidate_checkpoint=candidate_checkpoint,
        training_report=training_report,
        run_manifest=run_manifest,
        training_provenance=training_provenance,
        dataset_evidence=dataset_evidence,
        quantization_evidence=quantization_evidence,
        release_format=release_format,
        refs=refs,
    )
    gate_report = build_directid_gate_report(
        inputs.directid,
        candidate_checkpoint=inputs.candidate,
        training_report=inputs.training_report,
        run_manifest=inputs.run_manifest,
        training_provenance=inputs.provenance,
        dataset_evidence=inputs.dataset,
        quantization_evidence=inputs.quantization,
        release_format=release_format,
        evidence_refs=refs,
        signing_key=signing_key,
        key_id=key_id,
    )

    checkpoint_manifest: Mapping[str, Any] | None = None
    model_card: ModelCardBuildResult | None = None
    if gate_report.decision == RELEASABLE:
        checkpoint_manifest = _checkpoint_manifest(
            inputs,
            gate_report=gate_report,
            release_date=release_date,
        )
        model_card = build_model_card(
            checkpoint_manifest,
            gate_report,
            quant_delta=inputs.artifact.get("g4"),
            training_provenance=inputs.provenance,
        )
        model_card = ModelCardBuildResult(
            manifest_row=model_card.manifest_row,
            gate_report=model_card.gate_report,
            datasheet=model_card.datasheet,
            markdown=model_card.markdown.rstrip()
            + "\n\n"
            + _directid_card_section(inputs, gate_report),
        )

    release_manifest = _release_manifest(
        inputs,
        gate_report=gate_report,
        checkpoint_manifest=checkpoint_manifest,
        model_card=model_card,
    )
    release = DirectIDRelease(
        gate_report=gate_report,
        release_manifest=release_manifest,
        checkpoint_manifest=checkpoint_manifest,
        model_card=model_card,
    )
    if require_releasable and not release.published:
        raise DirectIDGateFailure(release)
    return release


def _validate_inputs(
    *,
    directid_evidence: DirectIDEvidenceReport | Mapping[str, Any] | str | Path,
    candidate_checkpoint: Mapping[str, Any] | str | Path,
    training_report: Mapping[str, Any] | str | Path,
    run_manifest: Mapping[str, Any] | str | Path,
    training_provenance: Mapping[str, Any] | str | Path,
    dataset_evidence: Mapping[str, Any] | str | Path,
    quantization_evidence: Mapping[str, Any] | str | Path,
    release_format: str,
    refs: Mapping[str, str],
) -> _ValidatedInputs:
    contract = validate_directid_contract()
    candidate = _load_mapping(candidate_checkpoint, "candidate_checkpoint")
    report = _load_mapping(training_report, "training_report")
    run = _load_mapping(run_manifest, "run_manifest")
    provenance = _load_provenance(training_provenance)
    dataset = _load_mapping(dataset_evidence, "dataset_evidence")
    directid = _load_mapping(directid_evidence, "directid_evidence")
    quantization = _load_mapping(quantization_evidence, "quantization_evidence")

    _require_schema(candidate, DIRECTID_CANDIDATE_SCHEMA_VERSION, "candidate")
    _require_identity(candidate, "candidate")
    if candidate.get("repo_id") != DIRECTID_MODEL_ID:
        raise DirectIDReleaseError("candidate does not identify DirectID Tiny")
    if candidate.get("ready_for_quantization") is not True:
        raise DirectIDReleaseError("candidate is not ready for quantization")
    if (
        candidate.get("certified") is not False
        or candidate.get("published") is not False
    ):
        raise DirectIDReleaseError("candidate must be uncertified and unpublished")
    _require_hash(candidate.get("artifact_hash"), "candidate artifact_hash")
    _require_hash(
        candidate.get("reproducibility_hash"), "candidate reproducibility_hash"
    )

    try:
        verified_hash = verify_reproducibility(provenance)
    except (OSError, TypeError, ValueError) as exc:
        raise DirectIDReleaseError(f"training provenance is invalid: {exc}") from exc
    if verified_hash != candidate["reproducibility_hash"]:
        raise DirectIDReleaseError(
            "candidate reproducibility_hash does not match training provenance"
        )
    if provenance.get("repo_id") != DIRECTID_MODEL_ID:
        raise DirectIDReleaseError("training provenance repo_id is not DirectID Tiny")
    if provenance.get("checkpoint_id") != candidate.get("checkpoint_ref"):
        raise DirectIDReleaseError(
            "training provenance checkpoint_id does not match candidate"
        )

    _require_schema(dataset, DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION, "dataset")
    _require_identity(dataset, "dataset")
    expected_dataset_hash = directid_dataset_manifest_hash()
    if dataset.get("manifest_id") != DIRECTID_DATASET_MANIFEST_ID:
        raise DirectIDReleaseError("dataset evidence has the wrong manifest_id")
    if dataset.get("manifest_hash") != expected_dataset_hash:
        raise DirectIDReleaseError(
            "dataset evidence is not bound to the committed DirectID manifest"
        )
    if dataset.get("raw_records_persisted") is not False:
        raise DirectIDReleaseError("dataset evidence must not persist raw records")
    if provenance.get("data_manifest_hash") != expected_dataset_hash:
        raise DirectIDReleaseError(
            "training provenance is not bound to the DirectID dataset manifest"
        )
    _validate_dataset_provenance(dataset)

    _require_schema(report, DIRECTID_TRAINING_REPORT_SCHEMA_VERSION, "training report")
    _require_identity(report, "training report")
    _require_safe_flags(report, "training report")
    if report.get("run_id") != candidate.get("run_id"):
        raise DirectIDReleaseError("training report run_id does not match candidate")
    if report.get("dataset_manifest_hash") != expected_dataset_hash:
        raise DirectIDReleaseError(
            "training report is not bound to the DirectID dataset manifest"
        )
    if _mapping(report.get("candidate_checkpoint")) != candidate:
        raise DirectIDReleaseError(
            "training report candidate_checkpoint does not match candidate"
        )
    training_eval_hash = _require_hash(
        report.get("eval_set_hash"), "training report eval_set_hash"
    )

    _require_schema(run, DIRECTID_RUN_MANIFEST_SCHEMA_VERSION, "run manifest")
    _require_identity(run, "run manifest")
    _require_safe_flags(run, "run manifest")
    if run.get("run_id") != candidate.get("run_id"):
        raise DirectIDReleaseError("run manifest run_id does not match candidate")
    if run.get("contract_ref") != contract.contract_ref:
        raise DirectIDReleaseError("run manifest contract_ref does not match")
    if _mapping(run.get("candidate_checkpoint")) != candidate:
        raise DirectIDReleaseError("run manifest candidate does not match")
    if run.get("training_report_hash") != stable_hash(report):
        raise DirectIDReleaseError("run manifest training_report_hash does not match")
    if _mapping(run.get("training_provenance")) != provenance:
        raise DirectIDReleaseError("run manifest training provenance does not match")
    run_dataset = _mapping(run.get("dataset"))
    if run_dataset.get("manifest_hash") != expected_dataset_hash:
        raise DirectIDReleaseError("run manifest dataset hash does not match")
    split_hashes = _mapping(run_dataset.get("split_hashes"))
    if split_hashes.get("test") != training_eval_hash:
        raise DirectIDReleaseError("run manifest test split hash does not match")
    if run.get("ready_for_gate_evaluation") is not True:
        raise DirectIDReleaseError("run manifest is not ready for gate evaluation")

    _require_schema(directid, DIRECTID_EVIDENCE_SCHEMA_VERSION, "DirectID evidence")
    _require_identity(directid, "DirectID evidence")
    directid_eval_hash = _require_hash(
        directid.get("eval_set_hash"), "DirectID evidence eval_set_hash"
    )
    _require_hash(
        directid.get("leakage_fixture_hash"),
        "DirectID evidence leakage_fixture_hash",
    )
    if directid_eval_hash != training_eval_hash:
        raise DirectIDReleaseError(
            "DirectID evidence eval_set_hash does not match training evidence"
        )
    _validate_directid_evidence(directid)

    _require_schema(
        quantization,
        DIRECTID_QUANTIZATION_SCHEMA_VERSION,
        "quantization evidence",
    )
    _require_identity(quantization, "quantization evidence")
    _require_safe_flags(quantization, "quantization evidence")
    if quantization.get("eval_set_hash") != directid_eval_hash:
        raise DirectIDReleaseError(
            "quantization eval_set_hash does not match DirectID evidence"
        )
    quant_candidate = _mapping(quantization.get("candidate"))
    for key in ("artifact_hash", "checkpoint_ref", "reproducibility_hash", "run_id"):
        if quant_candidate.get(key) != candidate.get(key):
            raise DirectIDReleaseError(
                f"quantization candidate {key} does not match training candidate"
            )
    parent_recall = _probability_map(
        quantization.get("fp_parent_per_label_recall"),
        "quantization fp_parent_per_label_recall",
    )
    _require_label_coverage(parent_recall, "quantization parent recall")
    _validate_quantization_inventory(quantization)
    artifact = _quantized_artifact(quantization, release_format=release_format)
    _validate_quantized_artifact(artifact, release_format=release_format)
    if artifact.get("eval_set_hash") != directid_eval_hash:
        raise DirectIDReleaseError(
            "selected quantized artifact eval_set_hash does not match DirectID evidence"
        )
    if _mapping(artifact.get("fp_parent_per_label_recall")) != parent_recall:
        raise DirectIDReleaseError(
            "selected quantized artifact parent recall does not match quantization evidence"
        )

    return _ValidatedInputs(
        candidate=candidate,
        training_report=report,
        run_manifest=run,
        provenance=provenance,
        dataset=dataset,
        directid=directid,
        quantization=quantization,
        artifact=artifact,
        refs=refs,
    )


def _validate_dataset_provenance(dataset: Mapping[str, Any]) -> None:
    sources = _sequence_of_mappings(dataset.get("source_provenance"))
    if not sources:
        raise DirectIDReleaseError("dataset evidence lacks source provenance")
    seen: set[str] = set()
    for source in sources:
        source_id = _non_empty_string(source.get("source_id"), "dataset source_id")
        if source_id in seen:
            raise DirectIDReleaseError("dataset source provenance ids must be unique")
        seen.add(source_id)
        _non_empty_string(source.get("license_id"), f"{source_id} license_id")
        _non_empty_string(source.get("revision"), f"{source_id} revision")
        _require_hash(
            source.get("source_manifest_hash"),
            f"{source_id} source_manifest_hash",
        )
        if source.get("content_hash_required") is not True:
            raise DirectIDReleaseError(
                f"{source_id} must require content hashes for release provenance"
            )
    synthetic = _mapping(dataset.get("synthetic_generation"))
    _require_hash(synthetic.get("settings_hash"), "synthetic settings_hash")
    settings = _mapping(synthetic.get("settings"))
    if settings.get("contains_real_phi") is not False:
        raise DirectIDReleaseError("synthetic settings must declare no real PHI")


def _validate_directid_evidence(evidence: Mapping[str, Any]) -> None:
    denominators = _integer_map(
        evidence.get("per_label_denominators"), "DirectID per_label_denominators"
    )
    if any(value <= 0 for value in denominators.values()):
        raise DirectIDReleaseError("DirectID label denominators must be positive")
    _require_label_coverage(denominators, "DirectID evaluation")
    combined = _mapping(evidence.get("combined"))
    recall = _probability_map(combined.get("per_label_recall"), "combined recall")
    precision = _probability_map(
        combined.get("per_label_precision"), "combined precision"
    )
    _require_label_coverage(recall, "combined recall")
    _require_label_coverage(precision, "combined precision")
    structured_recall = _probability(
        combined.get("structured_id_recall"), "structured_id_recall"
    )
    structured_labels = set(DIRECTID_TINY_HEAD_CONTRACT.structured_id_labels)
    structured_total = sum(denominators[label] for label in structured_labels)
    weighted_recall = (
        sum(recall[label] * denominators[label] for label in structured_labels)
        / structured_total
    )
    if not math.isclose(structured_recall, weighted_recall, abs_tol=1e-12):
        raise DirectIDReleaseError(
            "structured_id_recall does not match per-label DirectID evidence"
        )
    critical = evidence.get("critical_leakage_count")
    if isinstance(critical, bool) or not isinstance(critical, int) or critical < 0:
        raise DirectIDReleaseError(
            "DirectID critical_leakage_count must be a non-negative integer"
        )
    _probability(evidence.get("residual_leakage_rate"), "residual_leakage_rate")
    sweep = _mapping(evidence.get("safety_sweep"))
    if sweep.get("source") != SAFETY_SWEEP_SOURCE:
        raise DirectIDReleaseError("DirectID evidence lacks safety-sweep attribution")
    _non_empty_string(sweep.get("patterns_version"), "safety-sweep patterns_version")
    if sweep.get("patterns_version") != evidence.get("patterns_version"):
        raise DirectIDReleaseError("safety-sweep patterns_version is inconsistent")
    for key in ("spans_added", "recovered_model_misses", "structured_ids_recovered"):
        value = sweep.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DirectIDReleaseError(f"safety-sweep {key} must be non-negative")
    span_integrity = _mapping(evidence.get("span_integrity"))
    if span_integrity.get("passed") is not True:
        raise DirectIDReleaseError("DirectID evidence failed span integrity")
    gate_evidence = _mapping(evidence.get("gate_evidence"))
    g1b = _mapping(gate_evidence.get("G1b"))
    g3 = _mapping(gate_evidence.get("G3"))
    if not g1b or not g3:
        raise DirectIDReleaseError("DirectID evidence lacks G1b/G3 evidence blocks")
    if (
        _mapping(g1b.get("per_label_recall"))
        != _mapping(combined.get("per_label_recall"))
        or g1b.get("structured_id_recall") != combined.get("structured_id_recall")
        or g1b.get("eval_set_hash") != evidence.get("eval_set_hash")
    ):
        raise DirectIDReleaseError("DirectID G1b evidence is internally inconsistent")
    if (
        g3.get("critical_leakage_count") != evidence.get("critical_leakage_count")
        or g3.get("residual_leakage_rate") != evidence.get("residual_leakage_rate")
        or g3.get("leakage_fixture_hash") != evidence.get("leakage_fixture_hash")
    ):
        raise DirectIDReleaseError("DirectID G3 evidence is internally inconsistent")


def _validate_quantization_inventory(quantization: Mapping[str, Any]) -> None:
    artifacts = _sequence_of_mappings(quantization.get("artifacts"))
    if not artifacts:
        raise DirectIDReleaseError("quantization evidence contains no artifacts")
    formats = [str(artifact.get("format") or "") for artifact in artifacts]
    if any(not format_name for format_name in formats) or len(formats) != len(
        set(formats)
    ):
        raise DirectIDReleaseError("quantization artifact formats must be unique")
    accepted = sorted(
        str(artifact["format"])
        for artifact in artifacts
        if artifact.get("disposition") == "accepted"
    )
    quarantined = sorted(
        str(artifact["format"])
        for artifact in artifacts
        if artifact.get("disposition") == "quarantined"
    )
    unknown = sorted(
        str(artifact["format"])
        for artifact in artifacts
        if artifact.get("disposition") not in {"accepted", "quarantined"}
    )
    if unknown:
        raise DirectIDReleaseError(
            "quantization artifacts have invalid dispositions: " + ", ".join(unknown)
        )
    if sorted(_string_list(quantization.get("accepted_formats"))) != accepted:
        raise DirectIDReleaseError("quantization accepted_formats is inconsistent")
    if sorted(_string_list(quantization.get("quarantined_formats"))) != quarantined:
        raise DirectIDReleaseError("quantization quarantined_formats is inconsistent")


def _validate_quantized_artifact(
    artifact: Mapping[str, Any], *, release_format: str
) -> None:
    if artifact.get("format") != release_format:
        raise DirectIDReleaseError("selected quantized artifact format drifted")
    if release_format not in ALLOWED_FORMATS:
        raise DirectIDReleaseError(
            f"release format {release_format!r} is not publishable in models.jsonl"
        )
    bits = artifact.get("bits")
    if bits not in (4, 8):
        raise DirectIDReleaseError("quantized artifact bits must be 4 or 8")
    _require_hash(artifact.get("artifact_hash"), "quantized artifact_hash")
    size = artifact.get("artifact_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise DirectIDReleaseError("quantized artifact size must be positive")
    _require_hash(artifact.get("eval_set_hash"), "quantized eval_set_hash")
    _require_hash(artifact.get("performance_fixture_hash"), "performance_fixture_hash")
    sample_count = artifact.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise DirectIDReleaseError("quantized sample_count must be positive")
    param_count = artifact.get("param_count")
    if (
        isinstance(param_count, bool)
        or not isinstance(param_count, int)
        or param_count <= 0
    ):
        raise DirectIDReleaseError("quantized param_count must be positive")
    for key in ("p50_ms", "p95_ms", "ram_mb"):
        _positive_number(artifact.get(key), f"quantized {key}")
    recall = _probability_map(artifact.get("per_label_recall"), "quantized recall")
    parent = _probability_map(
        artifact.get("fp_parent_per_label_recall"), "quantized parent recall"
    )
    _require_label_coverage(recall, "quantized recall")
    _require_label_coverage(parent, "quantized parent recall")
    delta = artifact.get("quant_recall_delta")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or not 0.0 <= float(delta) <= 1.0
    ):
        raise DirectIDReleaseError("quant_recall_delta must be between 0 and 1")
    g4 = _mapping(artifact.get("g4"))
    tiny_fit = _mapping(artifact.get("tiny_tier_fit"))
    if not isinstance(g4.get("passed"), bool) or not isinstance(
        tiny_fit.get("passed"), bool
    ):
        raise DirectIDReleaseError("quantized artifact lacks G4/G5 verdicts")
    expected_disposition = (
        "accepted" if g4["passed"] and tiny_fit["passed"] else "quarantined"
    )
    if artifact.get("disposition") != expected_disposition:
        raise DirectIDReleaseError("quantized artifact disposition is inconsistent")


def _scoring_report(
    inputs: _ValidatedInputs, *, release_format: str
) -> BenchmarkReport:
    combined = _mapping(inputs.directid.get("combined"))
    artifact = inputs.artifact
    metadata = {
        "family": DIRECTID_FAMILY,
        "tier": "Tiny",
        "format": release_format,
        "repo_id": DIRECTID_MODEL_ID,
        "param_count": artifact["param_count"],
        "eval_set_hash": inputs.directid["eval_set_hash"],
        "leakage_fixture_hash": inputs.directid["leakage_fixture_hash"],
        "per_label_denominators": inputs.directid["per_label_denominators"],
        "quant_recall_delta": artifact["quant_recall_delta"],
        "fp_parent_per_label_recall": artifact["fp_parent_per_label_recall"],
        "p50_ms": artifact["p50_ms"],
        "p95_ms": artifact["p95_ms"],
        "ram_mb": artifact["ram_mb"],
    }
    metrics = {
        "per_label_recall": combined["per_label_recall"],
        "per_label_precision": combined["per_label_precision"],
        "structured_id_recall": combined["structured_id_recall"],
        "critical_leakage_count": inputs.directid["critical_leakage_count"],
        "residual_leakage_rate": inputs.directid["residual_leakage_rate"],
        "quant_recall_delta": artifact["quant_recall_delta"],
        "fp_parent_per_label_recall": artifact["fp_parent_per_label_recall"],
        "latency": {"p50_ms": artifact["p50_ms"], "p95_ms": artifact["p95_ms"]},
        "resources": {"ram_mb": artifact["ram_mb"]},
    }
    return BenchmarkReport(
        suite="directid-certified-evaluation",
        model_name=DIRECTID_MODEL_ID,
        device=str(artifact.get("device") or "offline"),
        fixture_count=int(artifact["sample_count"]),
        metrics=metrics,
        generated_at=None,
        metadata=metadata,
    )


def _scoped_checks(
    preview: GateReport, *, inputs: _ValidatedInputs
) -> tuple[GateCheck, ...]:
    checks = {check.gate: check for check in preview.gate_results}
    missing = [gate for gate in DIRECTID_REQUIRED_GATES if gate not in checks]
    if missing:
        raise DirectIDReleaseError(
            "release harness did not emit required gates: " + ", ".join(missing)
        )
    combined = _mapping(inputs.directid.get("combined"))
    recall = _probability_map(combined.get("per_label_recall"), "combined recall")
    denominators = _integer_map(
        inputs.directid.get("per_label_denominators"), "DirectID denominators"
    )
    structured_labels = set(DIRECTID_TINY_HEAD_CONTRACT.structured_id_labels)
    missing_structured = sorted(
        label
        for label in structured_labels
        if label not in recall or denominators.get(label, 0) <= 0
    )
    structured_recall = _probability(
        combined.get("structured_id_recall"), "structured_id_recall"
    )
    evidence = _gate_evidence_refs(inputs)

    scoped: list[GateCheck] = []
    for gate in DIRECTID_REQUIRED_GATES:
        original = checks[gate]
        passed = original.passed
        reason = original.reason
        details = dict(original.details)
        details["evidence"] = evidence
        blocking_format = original.blocking_format

        if gate == "G1b":
            passed = (
                passed
                and not missing_structured
                and structured_recall >= G1B_RECALL_FLOOR
            )
            details.update(
                {
                    "required_structured_labels": sorted(structured_labels),
                    "missing_structured_labels": missing_structured,
                    "structured_id_recall": structured_recall,
                    "floor": G1B_RECALL_FLOOR,
                    "safety_sweep_recovered_count": _mapping(
                        inputs.directid.get("safety_sweep")
                    ).get("recovered_model_misses"),
                }
            )
            if missing_structured:
                reason = "certified evaluation lacks structured-label coverage"
            elif structured_recall < G1B_RECALL_FLOOR:
                reason = "structured-id recall is below the G1b floor"
        elif gate == "G3":
            passed = (
                passed
                and preview.critical_leakage_count == 0
                and preview.residual_leakage_rate == 0.0
            )
            details.update(
                {
                    "critical_leakage_count": preview.critical_leakage_count,
                    "residual_leakage_rate": preview.residual_leakage_rate,
                    "residual_leakage_ceiling": 0.0,
                }
            )
            if preview.critical_leakage_count != 0:
                reason = "critical leakage must be exactly zero"
            elif preview.residual_leakage_rate != 0.0:
                reason = "DirectID residual leakage must be exactly zero"
        elif gate == "G4":
            artifact_g4 = _mapping(inputs.artifact.get("g4"))
            passed = passed and artifact_g4.get("passed") is True
            details["quantization_artifact"] = _safe_quant_summary(inputs.artifact)
            if artifact_g4.get("passed") is not True:
                reason = "selected format exceeds the G4 recall-delta limit"
                blocking_format = str(inputs.artifact["format"])
        elif gate == "G5":
            fit = _mapping(inputs.artifact.get("tiny_tier_fit"))
            passed = passed and fit.get("passed") is True
            details["quantization_artifact"] = _safe_quant_summary(inputs.artifact)
            if fit.get("passed") is not True:
                reason = "selected format exceeds the Tiny-tier budget"
                blocking_format = str(inputs.artifact["format"])

        scoped.append(
            GateCheck(
                gate=gate,
                passed=passed,
                reason=reason,
                details=details,
                blocking_format=blocking_format,
            )
        )
    return tuple(scoped)


def _gate_evidence_refs(inputs: _ValidatedInputs) -> dict[str, Any]:
    sweep = _mapping(inputs.directid.get("safety_sweep"))
    return {
        "candidate_checkpoint": {
            "artifact_hash": inputs.candidate["artifact_hash"],
            "ref": inputs.refs["candidate_checkpoint"],
            "reproducibility_hash": inputs.candidate["reproducibility_hash"],
        },
        "dataset": {
            "evidence_hash": stable_hash(inputs.dataset),
            "manifest_hash": inputs.dataset["manifest_hash"],
            "manifest_id": inputs.dataset["manifest_id"],
            "ref": inputs.refs["dataset_evidence"],
            "sources": _safe_dataset_sources(inputs.dataset),
        },
        "directid_safety_sweep": {
            "patterns_version": sweep["patterns_version"],
            "recovered_model_misses": sweep["recovered_model_misses"],
            "ref": inputs.refs["directid_evidence"],
            "report_hash": stable_hash(inputs.directid),
            "source": sweep["source"],
        },
        "quantization": {
            "artifact_hash": inputs.artifact["artifact_hash"],
            "evidence_hash": stable_hash(inputs.quantization),
            "format": inputs.artifact["format"],
            "quarantined_formats": _string_list(
                inputs.quantization.get("quarantined_formats")
            ),
            "ref": inputs.refs["quantization_evidence"],
        },
        "training": {
            "report_hash": stable_hash(inputs.training_report),
            "report_ref": inputs.refs["training_report"],
            "run_manifest_hash": stable_hash(inputs.run_manifest),
            "run_manifest_ref": inputs.refs["run_manifest"],
        },
    }


def _checkpoint_manifest(
    inputs: _ValidatedInputs,
    *,
    gate_report: GateReport,
    release_date: str,
) -> dict[str, Any]:
    provenance = inputs.provenance
    artifact = inputs.artifact
    row: dict[str, Any] = {
        "repo_id": DIRECTID_MODEL_ID,
        "family": DIRECTID_FAMILY,
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": artifact["param_count"],
        "architecture": "directid-token-classifier",
        "base_model": provenance["base_model"],
        "formats": [artifact["format"]],
        "canonical_labels": list(DIRECTID_TINY_HEAD_CONTRACT.labels),
        "benchmark": {
            "dataset": DIRECTID_DATASET_MANIFEST_ID,
            "micro_f1": None,
            "recall": _probability(
                _mapping(inputs.directid.get("combined")).get("structured_id_recall"),
                "structured_id_recall",
            ),
        },
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": provenance["reproducibility_hash"],
        "released": release_date,
        "download_mb": float(artifact["artifact_size_bytes"]) / 1_000_000.0,
        "disk_mb": float(artifact["artifact_size_bytes"]) / 1_000_000.0,
        "latency_ms": {
            "p50": float(artifact["p50_ms"]),
            "p95": float(artifact["p95_ms"]),
        },
        "peak_ram_mb": {"measured": float(artifact["ram_mb"])},
        "recommended_tier": "phone",
        "training_provenance": {
            key: _plain(provenance[key])
            for key in sorted(TRAINING_PROVENANCE_FIELDS - {"path"})
            if key in provenance
        },
    }
    violations = validate_manifest_row(row, 1)
    if violations:
        raise DirectIDReleaseError(
            "generated checkpoint manifest is invalid: "
            + "; ".join(item.message for item in violations)
        )
    mismatches = release_gates._manifest_row_mismatches(
        row,
        {
            "repo_id": gate_report.repo_id,
            "family": gate_report.family,
            "tier": gate_report.tier,
            "param_count": gate_report.param_count,
            "format": gate_report.format,
        },
        source="gate_report",
    )
    if mismatches:
        raise DirectIDReleaseError(
            "checkpoint manifest diverges from gate report: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return _plain_mapping(row)


def _release_manifest(
    inputs: _ValidatedInputs,
    *,
    gate_report: GateReport,
    checkpoint_manifest: Mapping[str, Any] | None,
    model_card: ModelCardBuildResult | None,
) -> dict[str, Any]:
    signature = gate_report.signature
    if signature is None:  # pragma: no cover - signed by the builder.
        raise DirectIDReleaseError("gate report must be signed")
    failed_gates = [
        check.gate for check in gate_report.gate_results if not check.passed
    ]
    selected_format = str(inputs.artifact["format"])
    quantized_quarantine = set(
        _string_list(inputs.quantization.get("quarantined_formats"))
    )
    if gate_report.decision != RELEASABLE:
        quantized_quarantine.add(selected_format)
    artifacts: dict[str, str] = {
        "gate_report": stable_hash(gate_report.to_dict()),
    }
    if checkpoint_manifest is not None and model_card is not None:
        artifacts.update(
            {
                "checkpoint_manifest": stable_hash(checkpoint_manifest),
                "model_card": stable_hash(model_card.markdown),
                "model_datasheet": stable_hash(model_card.datasheet),
            }
        )
    return {
        "schema_version": DIRECTID_RELEASE_SCHEMA_VERSION,
        "model_id": DIRECTID_MODEL_ID,
        "decision": gate_report.decision,
        "certified_gates": [
            check.gate for check in gate_report.gate_results if check.passed
        ],
        "failed_gates": failed_gates,
        "selected_format": selected_format,
        "published_formats": (
            [selected_format] if gate_report.decision == RELEASABLE else []
        ),
        "quarantined_formats": sorted(quantized_quarantine),
        "publication": {
            "publish_target": (
                DIRECTID_MODEL_ID if gate_report.decision == RELEASABLE else None
            ),
            "status": (
                "published" if gate_report.decision == RELEASABLE else "quarantined"
            ),
        },
        "gate_report": {
            "repro_hash": gate_report.repro_hash,
            "signature_algorithm": signature.algorithm,
            "signature_key_id": signature.key_id,
        },
        "evidence": _gate_evidence_refs(inputs),
        "artifacts": artifacts,
        "safety": {
            "assist_only": True,
            "clinical_decision_trigger": False,
            "local_offline_after_download": True,
            "raw_phi_in_artifacts": False,
            "restricted_dataset_payloads_in_artifacts": False,
        },
    }


def _directid_card_section(inputs: _ValidatedInputs, gate_report: GateReport) -> str:
    evidence = _gate_evidence_refs(inputs)
    dataset = _mapping(evidence.get("dataset"))
    sweep = _mapping(evidence.get("directid_safety_sweep"))
    quantization = _mapping(evidence.get("quantization"))
    sources = ", ".join(
        f"{source['source_id']} ({source['license_id']})"
        for source in _sequence_of_mappings(dataset.get("sources"))
    )
    lines = [
        _DIRECTID_CARD_SECTION.rstrip(),
        "",
        "### Dataset Provenance",
        "",
        f"- Dataset manifest: `{dataset.get('manifest_id')}` "
        f"(`{dataset.get('manifest_hash')}`)",
        f"- Dataset evidence: `{dataset.get('ref')}` "
        f"(`{dataset.get('evidence_hash')}`)",
        f"- Reference-only sources: {sources or 'Not reported'}",
        "",
        "### Safety-Sweep Evidence",
        "",
        f"- Evidence report: `{sweep.get('ref')}` (`{sweep.get('report_hash')}`)",
        f"- Patterns version: `{sweep.get('patterns_version')}`",
        f"- Recovered model misses: {sweep.get('recovered_model_misses')}",
        "",
        "### Quantization Evidence",
        "",
        f"- Selected format: `{gate_report.format}`",
        f"- Quantization report: `{quantization.get('ref')}` "
        f"(`{quantization.get('evidence_hash')}`)",
        f"- Quantized artifact: `{quantization.get('artifact_hash')}`",
        "- Independently quarantined formats: "
        + (
            ", ".join(
                f"`{item}`"
                for item in _string_list(quantization.get("quarantined_formats"))
            )
            or "None"
        ),
        "",
    ]
    return "\n".join(lines)


def _evidence_references(
    supplied: Mapping[str, str] | None,
    **overrides: str | None,
) -> dict[str, str]:
    refs = dict(_DEFAULT_EVIDENCE_REFS)
    if supplied is not None:
        unknown = sorted(set(supplied) - set(refs))
        if unknown:
            raise DirectIDReleaseError(
                "unknown evidence reference(s): " + ", ".join(unknown)
            )
        refs.update({str(key): value for key, value in supplied.items()})
    refs.update({key: value for key, value in overrides.items() if value is not None})
    return {
        key: _validate_reference(value, f"{key}_ref")
        for key, value in sorted(refs.items())
    }


def _load_mapping(source: Any, name: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload = source
    elif isinstance(source, (str, Path)):
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DirectIDReleaseError(f"{name} is invalid: {exc}") from exc
    elif hasattr(source, "to_dict") and callable(source.to_dict):
        payload = source.to_dict()
    else:
        raise DirectIDReleaseError(f"{name} must be a mapping, JSON path, or artifact")
    if not isinstance(payload, Mapping):
        raise DirectIDReleaseError(f"{name} must contain a JSON object")
    return _plain_mapping(payload)


def _load_provenance(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    try:
        payload = (
            load_training_provenance(source)
            if isinstance(source, (str, Path))
            else _plain_mapping(source)
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DirectIDReleaseError(f"training provenance is invalid: {exc}") from exc
    return payload


def _require_identity(payload: Mapping[str, Any], name: str) -> None:
    expected = {
        "contract_ref": DIRECTID_CONTRACT_REF,
        "family": DIRECTID_FAMILY,
        "tier": DIRECTID_TIER,
    }
    for key, value in expected.items():
        if key in payload and payload.get(key) != value:
            raise DirectIDReleaseError(f"{name} {key} does not match DirectID Tiny")


def _require_schema(payload: Mapping[str, Any], schema: str, name: str) -> None:
    if payload.get("schema_version") != schema:
        raise DirectIDReleaseError(f"{name} has an unsupported schema_version")


def _require_safe_flags(payload: Mapping[str, Any], name: str) -> None:
    for key in (
        "raw_phi_persisted",
        "raw_records_persisted",
        "restricted_dataset_payloads_persisted",
    ):
        if key in payload and payload.get(key) is not False:
            raise DirectIDReleaseError(f"{name} must declare {key}=false")


def _quantized_artifact(
    quantization: Mapping[str, Any], *, release_format: str
) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in _sequence_of_mappings(quantization.get("artifacts"))
        if artifact.get("format") == release_format
    ]
    if len(matches) != 1:
        raise DirectIDReleaseError(
            f"quantization evidence must contain exactly one {release_format!r} artifact"
        )
    return matches[0]


def _safe_dataset_sources(dataset: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "license_id": str(source["license_id"]),
            "revision": str(source["revision"]),
            "source_class": str(source.get("source_class") or ""),
            "source_id": str(source["source_id"]),
            "source_manifest_hash": str(source["source_manifest_hash"]),
        }
        for source in sorted(
            _sequence_of_mappings(dataset.get("source_provenance")),
            key=lambda item: str(item.get("source_id") or ""),
        )
    ]


def _safe_quant_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_hash": artifact["artifact_hash"],
        "artifact_size_bytes": artifact["artifact_size_bytes"],
        "disposition": artifact["disposition"],
        "format": artifact["format"],
        "p50_ms": artifact["p50_ms"],
        "p95_ms": artifact["p95_ms"],
        "quant_recall_delta": artifact["quant_recall_delta"],
        "ram_mb": artifact["ram_mb"],
    }


def _require_label_coverage(values: Mapping[str, Any], name: str) -> None:
    missing = sorted(set(DIRECTID_TINY_HEAD_CONTRACT.labels) - set(values))
    if missing:
        raise DirectIDReleaseError(
            f"{name} is missing DirectID label(s): " + ", ".join(missing)
        )


def _probability_map(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DirectIDReleaseError(f"{name} must be an object")
    return {
        str(key): _probability(item, f"{name}.{key}")
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _integer_map(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise DirectIDReleaseError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        if isinstance(item, bool) or not isinstance(item, int):
            raise DirectIDReleaseError(f"{name}.{key} must be an integer")
        result[str(key)] = item
    return result


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectIDReleaseError(f"{name} must be between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DirectIDReleaseError(f"{name} must be between 0 and 1")
    return number


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectIDReleaseError(f"{name} must be positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DirectIDReleaseError(f"{name} must be positive")
    return number


def _require_hash(value: Any, name: str) -> str:
    digest = str(value or "")
    if _SHA256.fullmatch(digest) is None:
        raise DirectIDReleaseError(
            f"{name} must match sha256:<64 lowercase hex characters>"
        )
    return digest


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DirectIDReleaseError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_reference(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_REFERENCE.fullmatch(value.strip()) is None:
        raise DirectIDReleaseError(
            f"{name} must be a non-empty reference without control characters"
        )
    return value.strip()


def _require_signing_key(key: bytes | str) -> None:
    if not isinstance(key, (bytes, str)) or not key:
        raise DirectIDReleaseError("an explicit non-empty signing key is required")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item)]


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DIRECTID_MODEL_ID",
    "DIRECTID_RELEASE_GATE_MILESTONE",
    "DIRECTID_RELEASE_SCHEMA_VERSION",
    "DIRECTID_REQUIRED_GATES",
    "DirectIDGateFailure",
    "DirectIDRelease",
    "DirectIDReleaseError",
    "DirectIDReleasePaths",
    "build_directid_gate_report",
    "build_directid_release",
]
