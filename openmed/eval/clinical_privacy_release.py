"""Release certification for the clinical PHI flagship checkpoint.

This module consumes aggregate, PHI-free evidence produced by the training and
SHIELD benchmark workflows. It deliberately does not load model weights or
corpus rows. Release artifacts contain metrics, hashes, offsets-independent
provenance, and stable references only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import stable_hash
from openmed.core.manifest_schema import (
    TRAINING_PROVENANCE_FIELDS,
    validate_manifest_row,
)
from openmed.core.repro_hash import (
    compute_file_digest,
    load_training_provenance,
    verify_reproducibility,
)
from openmed.eval import release_gates
from openmed.eval.datasets.clinical_phi import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PHI_MANIFEST_REF,
    CLINICAL_PRIVACY_MODEL_ID,
    clinical_phi_manifest_hash,
    load_clinical_phi_manifest,
)
from openmed.eval.model_card_builder import (
    MODEL_DATASHEET_FILENAME,
    ModelCardBuildResult,
    build_model_card,
)
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
    ReleaseGate,
)
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites.shield import SHIELD, SHIELD_LABEL_TO_CANONICAL
from openmed.training.recipe import CONFIG_DIR, load_preset

CLINICAL_PRIVACY_RELEASE_SCHEMA_VERSION = "openmed.eval.clinical_privacy_release.v1"
CLINICAL_PRIVACY_FAMILY = "ClinicalPrivacy"
CLINICAL_PRIVACY_GATE_MILESTONE = "v1.6"
CLINICAL_PRIVACY_REQUIRED_GATES = ("G1a", "G2", "G3")
CLINICAL_PRIVACY_HELD_OUT_SUITE = "clinical-phi-held-out"

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9._~:/?#@!$&()*+,;=%-]{1,2048}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_G1A_COVERAGE_GROUPS = (
    "names",
    "dates",
    "ages_over_89",
    "addresses",
    "ids",
    "contacts",
)
_G2_COVERAGE_GROUPS = ("names", "dates", "addresses")
_CLINICAL_SAFETY_SECTION = """## Clinical Privacy Release Boundary

This checkpoint is intended for assistive clinical-text de-identification with
human review. It must not diagnose, recommend treatment, or trigger clinical
decisions. Execution is local/offline after checkpoint acquisition.

The signed certification covers G1a, G2, and G3 on held-out evaluation and
leakage fixtures. SHIELD is linked as public comparison evidence only; it is
not used as the high-recall release gate. Published artifacts contain
aggregate metrics, hashes, and provenance references, never source notes or raw
identifiers.
"""


class ClinicalPrivacyReleaseError(ValueError):
    """Raised when clinical-privacy release evidence is incomplete or invalid."""


class ClinicalPrivacyGateFailure(ClinicalPrivacyReleaseError):
    """Raised when a signed clinical-privacy gate report quarantines a candidate."""

    def __init__(self, report: GateReport) -> None:
        failed = ", ".join(
            check.gate for check in report.gate_results if not check.passed
        )
        super().__init__(f"clinical privacy release gates failed: {failed}")
        self.report = report


@dataclass(frozen=True)
class ClinicalPrivacyReleasePaths:
    """Paths written for one clinical-privacy release evidence package."""

    gate_report: Path
    model_card: Path
    model_datasheet: Path
    model_manifest_entry: Path
    release_manifest: Path


@dataclass(frozen=True)
class ClinicalPrivacyRelease:
    """Validated, PHI-free artifacts for a releasable flagship checkpoint."""

    gate_report: GateReport
    model_card: ModelCardBuildResult
    model_manifest_entry: Mapping[str, Any]
    release_manifest: Mapping[str, Any]

    def write(self, output_dir: str | Path) -> ClinicalPrivacyReleasePaths:
        """Write deterministic release artifacts beneath *output_dir*."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        gate_path = destination / "gate-report.json"
        manifest_entry_path = destination / "model-manifest-entry.json"
        release_manifest_path = destination / "release-manifest.json"
        card_path = destination / "README.md"
        datasheet_path = destination / MODEL_DATASHEET_FILENAME

        gate_path.write_text(self.gate_report.to_json() + "\n", encoding="utf-8")
        _write_json(manifest_entry_path, self.model_manifest_entry)
        _write_json(release_manifest_path, self.release_manifest)
        self.model_card.write_markdown(card_path)
        self.model_card.write_datasheet(datasheet_path)
        return ClinicalPrivacyReleasePaths(
            gate_report=gate_path,
            model_card=card_path,
            model_datasheet=datasheet_path,
            model_manifest_entry=manifest_entry_path,
            release_manifest=release_manifest_path,
        )


def build_clinical_privacy_gate_report(
    held_out_report: BenchmarkReport | Mapping[str, Any] | str | Path,
    shield_report: BenchmarkReport | Mapping[str, Any] | str | Path,
    *,
    checkpoint_manifest: Mapping[str, Any] | str | Path,
    training_provenance: Mapping[str, Any] | str | Path,
    checkpoint_manifest_ref: str,
    held_out_report_ref: str,
    shield_report_ref: str,
    signing_key: bytes | str,
    key_id: str = "clinical-privacy-release-gate",
) -> GateReport:
    """Return a signed G1a/G2/G3 report for the named flagship checkpoint.

    The generic release harness performs the actual G1a, G2, and G3 scoring.
    This wrapper adds clinical-PHI coverage requirements and binds each signed
    check to the checkpoint, dataset manifest, held-out report, and SHIELD
    comparison report without copying source text or fixture identifiers.

    Args:
        held_out_report: Aggregate held-out and leakage-fixture benchmark report.
        shield_report: Manifest-linked SHIELD comparison report.
        checkpoint_manifest: Candidate model-manifest row, JSON, or JSONL file.
        training_provenance: Reproducible training provenance object or file.
        checkpoint_manifest_ref: Stable publication reference for the checkpoint.
        held_out_report_ref: Stable publication reference for held-out evidence.
        shield_report_ref: Stable publication reference for SHIELD evidence.
        signing_key: Explicit HMAC key used to sign the gate report.
        key_id: Non-secret identifier for the signing key.

    Returns:
        A signed, reproducible ``GateReport`` containing exactly G1a, G2, and G3.

    Raises:
        ClinicalPrivacyReleaseError: If evidence is malformed or inconsistent.
    """

    _require_signing_key(signing_key)
    references = {
        "checkpoint_manifest": _validate_reference(
            checkpoint_manifest_ref, "checkpoint_manifest_ref"
        ),
        "held_out_report": _validate_reference(
            held_out_report_ref, "held_out_report_ref"
        ),
        "shield_report": _validate_reference(shield_report_ref, "shield_report_ref"),
    }
    held_out = _load_benchmark_report(held_out_report, "held_out_report")
    shield = _load_benchmark_report(shield_report, "shield_report")
    checkpoint = _load_checkpoint_manifest(checkpoint_manifest)
    provenance = _load_and_validate_training_provenance(
        training_provenance,
        checkpoint=checkpoint,
    )
    _validate_checkpoint(checkpoint, provenance)
    _validate_shield_report(shield, checkpoint=checkpoint)

    scoring_report = _held_out_scoring_report(held_out, checkpoint=checkpoint)
    preview = ReleaseGate(
        milestone=CLINICAL_PRIVACY_GATE_MILESTONE,
        signing_key=signing_key,
        key_id=key_id,
    ).preview(scoring_report, baseline={})
    checks = _scoped_gate_checks(
        preview,
        scoring_report=scoring_report,
        checkpoint=checkpoint,
        shield_report=shield,
        references=references,
    )
    decision = RELEASABLE if all(check.passed for check in checks) else QUARANTINED
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
        quant_recall_delta=None,
        p50_ms=preview.p50_ms,
        p95_ms=preview.p95_ms,
        ram_mb=preview.ram_mb,
        eval_set_hash=preview.eval_set_hash,
        leakage_fixture_hash=preview.leakage_fixture_hash,
        decision=decision,
        gate_results=checks,
        policy=preview.policy,
        threshold_profile=preview.threshold_profile,
        target_leakage_rate=preview.target_leakage_rate,
    )
    return report.sign(signing_key, key_id=key_id)


def build_clinical_privacy_release(
    held_out_report: BenchmarkReport | Mapping[str, Any] | str | Path,
    shield_report: BenchmarkReport | Mapping[str, Any] | str | Path,
    *,
    checkpoint_manifest: Mapping[str, Any] | str | Path,
    training_provenance: Mapping[str, Any] | str | Path,
    checkpoint_manifest_ref: str,
    held_out_report_ref: str,
    shield_report_ref: str,
    release_date: str,
    signing_key: bytes | str,
    key_id: str = "clinical-privacy-release-gate",
) -> ClinicalPrivacyRelease:
    """Build the signed report, model card, and release manifest artifacts.

    A quarantined candidate never receives a model card or manifest entry. The
    raised :class:`ClinicalPrivacyGateFailure` retains the signed failure report
    so release automation can archive the fail-closed decision.
    """

    checkpoint = _load_checkpoint_manifest(checkpoint_manifest)
    provenance = _load_and_validate_training_provenance(
        training_provenance,
        checkpoint=checkpoint,
    )
    held_out = _load_benchmark_report(held_out_report, "held_out_report")
    shield = _load_benchmark_report(shield_report, "shield_report")
    gate_report = build_clinical_privacy_gate_report(
        held_out,
        shield,
        checkpoint_manifest=checkpoint,
        training_provenance=provenance,
        checkpoint_manifest_ref=checkpoint_manifest_ref,
        held_out_report_ref=held_out_report_ref,
        shield_report_ref=shield_report_ref,
        signing_key=signing_key,
        key_id=key_id,
    )
    if gate_report.decision != RELEASABLE:
        raise ClinicalPrivacyGateFailure(gate_report)

    manifest_entry = _build_model_manifest_entry(
        checkpoint,
        provenance=provenance,
        held_out_report=held_out,
        shield_report=shield,
        gate_report=gate_report,
        release_date=release_date,
    )
    card = build_model_card(
        manifest_entry,
        gate_report,
        training_provenance=provenance,
    )
    card = ModelCardBuildResult(
        manifest_row=card.manifest_row,
        gate_report=card.gate_report,
        datasheet=card.datasheet,
        markdown=(
            card.markdown.rstrip() + "\n\n" + _clinical_safety_section(gate_report)
        ),
    )
    release_manifest = _build_release_manifest(
        gate_report=gate_report,
        model_card=card,
        model_manifest_entry=manifest_entry,
    )
    return ClinicalPrivacyRelease(
        gate_report=gate_report,
        model_card=card,
        model_manifest_entry=manifest_entry,
        release_manifest=release_manifest,
    )


def _load_benchmark_report(
    source: BenchmarkReport | Mapping[str, Any] | str | Path,
    name: str,
) -> BenchmarkReport:
    if isinstance(source, BenchmarkReport):
        report = source
    elif isinstance(source, Mapping):
        try:
            report = BenchmarkReport.from_dict(source)
        except (KeyError, TypeError, ValueError) as exc:
            raise ClinicalPrivacyReleaseError(f"{name} is invalid: {exc}") from exc
    else:
        try:
            report = BenchmarkReport.read_json(source)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ClinicalPrivacyReleaseError(f"{name} is invalid: {exc}") from exc
    if report.fixture_count <= 0:
        raise ClinicalPrivacyReleaseError(f"{name} must contain evaluated fixtures")
    return report


def _load_checkpoint_manifest(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload: Any = source
    else:
        path = Path(source)
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ClinicalPrivacyReleaseError(
                f"failed to read checkpoint manifest: {path}"
            ) from exc
        try:
            payload = json.loads(contents)
        except json.JSONDecodeError:
            try:
                payload = [
                    json.loads(line) for line in contents.splitlines() if line.strip()
                ]
            except json.JSONDecodeError as exc:
                raise ClinicalPrivacyReleaseError(
                    f"checkpoint manifest is not valid JSON or JSONL: {path}"
                ) from exc

    rows: Sequence[Any]
    if isinstance(payload, Mapping):
        nested = payload.get("models") or payload.get("checkpoints")
        if isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            rows = nested
        else:
            rows = (payload,)
    elif isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        rows = payload
    else:
        raise ClinicalPrivacyReleaseError(
            "checkpoint manifest must contain model metadata"
        )

    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("model_id") or row.get("repo_id") or "")
        == CLINICAL_PRIVACY_MODEL_ID
    ]
    if len(matches) != 1:
        raise ClinicalPrivacyReleaseError(
            "checkpoint manifest must contain exactly one "
            f"{CLINICAL_PRIVACY_MODEL_ID!r} row; found {len(matches)}"
        )
    return _plain_mapping(matches[0])


def _load_and_validate_training_provenance(
    source: Mapping[str, Any] | str | Path,
    *,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        provenance = (
            load_training_provenance(source)
            if isinstance(source, (str, Path))
            else _plain_mapping(source)
        )
        recorded_hash = verify_reproducibility(provenance)
    except (OSError, TypeError, ValueError) as exc:
        raise ClinicalPrivacyReleaseError(
            f"training provenance is invalid: {exc}"
        ) from exc

    expected_data_hash = clinical_phi_manifest_hash()
    if provenance.get("data_manifest_hash") != expected_data_hash:
        raise ClinicalPrivacyReleaseError(
            "training provenance is not bound to the clinical PHI dataset manifest"
        )
    recipe = load_preset("C")
    if recipe.mode != "C" or recipe.dapt.corpus_ref != CLINICAL_PHI_MANIFEST_REF:
        raise ClinicalPrivacyReleaseError(
            "committed large-teacher preset is not recipe mode C for clinical PHI"
        )
    expected_recipe_hash = compute_file_digest(CONFIG_DIR / "large_teacher.yaml")
    if provenance.get("recipe_config_hash") != expected_recipe_hash:
        raise ClinicalPrivacyReleaseError(
            "training provenance is not bound to the committed mode-C recipe"
        )
    if provenance.get("repo_id") not in (None, CLINICAL_PRIVACY_MODEL_ID):
        raise ClinicalPrivacyReleaseError(
            "training provenance repo_id does not identify the clinical PHI flagship"
        )
    if checkpoint.get("reproducibility_hash") != recorded_hash:
        raise ClinicalPrivacyReleaseError(
            "checkpoint reproducibility_hash does not match training provenance"
        )
    return provenance


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if checkpoint.get("repo_id") != CLINICAL_PRIVACY_MODEL_ID:
        raise ClinicalPrivacyReleaseError(
            "checkpoint repo_id does not identify the named clinical PHI flagship"
        )
    if checkpoint.get("family") != CLINICAL_PRIVACY_FAMILY:
        raise ClinicalPrivacyReleaseError(
            f"checkpoint family must be {CLINICAL_PRIVACY_FAMILY!r}"
        )
    if checkpoint.get("task") != "token-classification":
        raise ClinicalPrivacyReleaseError(
            "checkpoint task must be 'token-classification'"
        )
    formats = checkpoint.get("formats")
    if (
        not isinstance(formats, Sequence)
        or isinstance(formats, (str, bytes, bytearray))
        or not formats
    ):
        raise ClinicalPrivacyReleaseError("checkpoint formats must be a non-empty list")
    required_labels = set(load_clinical_phi_manifest().required_labels())
    labels = checkpoint.get("canonical_labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        raise ClinicalPrivacyReleaseError("checkpoint canonical_labels must be a list")
    missing_labels = sorted(required_labels - {str(label) for label in labels})
    if missing_labels:
        raise ClinicalPrivacyReleaseError(
            "checkpoint is missing clinical PHI labels: " + ", ".join(missing_labels)
        )
    if checkpoint.get("reproducibility_hash") != provenance.get("reproducibility_hash"):
        raise ClinicalPrivacyReleaseError(
            "checkpoint and training provenance hashes do not match"
        )


def _validate_shield_report(
    report: BenchmarkReport,
    *,
    checkpoint: Mapping[str, Any],
) -> None:
    if report.suite != SHIELD or report.model_name != CLINICAL_PRIVACY_MODEL_ID:
        raise ClinicalPrivacyReleaseError(
            "SHIELD evidence must target the named clinical PHI flagship"
        )
    comparison = _mapping(report.metrics.get("shield_comparison"))
    if (
        comparison.get("evidence_role") != "comparison"
        or comparison.get("high_recall_release_gate") is not False
    ):
        raise ClinicalPrivacyReleaseError(
            "SHIELD evidence must be marked comparison-only, not a release gate"
        )
    aggregate = _mapping(comparison.get("aggregate"))
    for metric in ("exact_span_f1", "recall", "leakage"):
        _probability(aggregate.get(metric), f"SHIELD aggregate {metric}")
    by_label = _mapping(comparison.get("by_label"))
    if not by_label:
        raise ClinicalPrivacyReleaseError(
            "SHIELD evidence must include per-label recall and leakage"
        )
    expected_labels = set(SHIELD_LABEL_TO_CANONICAL.values())
    missing_labels = sorted(expected_labels - set(by_label))
    if missing_labels:
        raise ClinicalPrivacyReleaseError(
            "SHIELD evidence is missing canonical labels: " + ", ".join(missing_labels)
        )
    for label, values in by_label.items():
        label_metrics = _mapping(values)
        _probability(label_metrics.get("recall"), f"SHIELD {label} recall")
        _probability(label_metrics.get("leakage"), f"SHIELD {label} leakage")

    metadata = _mapping(report.metadata)
    if (
        metadata.get("comparison_evidence_only") is not True
        or metadata.get("gate_target") is not False
    ):
        raise ClinicalPrivacyReleaseError(
            "SHIELD report metadata must preserve the comparison-only boundary"
        )
    dataset = _mapping(metadata.get("dataset_manifest"))
    if (
        dataset.get("manifest_id") != CLINICAL_PHI_MANIFEST_ID
        or dataset.get("manifest_hash") != clinical_phi_manifest_hash()
        or dataset.get("manifest_ref") != CLINICAL_PHI_MANIFEST_REF
    ):
        raise ClinicalPrivacyReleaseError(
            "SHIELD report is not linked to the clinical PHI dataset manifest"
        )
    public_corpus = _mapping(metadata.get("public_corpus_reference"))
    if (
        public_corpus.get("dataset") != SHIELD
        or public_corpus.get("redistribution") != "reference-only"
        or not str(public_corpus.get("source_url") or "").startswith("https://")
    ):
        raise ClinicalPrivacyReleaseError(
            "SHIELD report is missing its public corpus reference"
        )
    shield_checkpoint = _mapping(metadata.get("checkpoint_manifest"))
    if shield_checkpoint.get("model_id") != CLINICAL_PRIVACY_MODEL_ID or (
        shield_checkpoint.get("reproducibility_hash")
        != checkpoint.get("reproducibility_hash")
    ):
        raise ClinicalPrivacyReleaseError(
            "SHIELD report checkpoint evidence does not match the candidate"
        )


def _held_out_scoring_report(
    report: BenchmarkReport,
    *,
    checkpoint: Mapping[str, Any],
) -> BenchmarkReport:
    if report.suite == "shield" or report.model_name != CLINICAL_PRIVACY_MODEL_ID:
        raise ClinicalPrivacyReleaseError(
            "held-out gate evidence must target the named flagship outside SHIELD"
        )
    metadata = _mapping(report.metadata)
    expected_identity = {
        "repo_id": CLINICAL_PRIVACY_MODEL_ID,
        "family": CLINICAL_PRIVACY_FAMILY,
        "tier": checkpoint.get("tier"),
        "param_count": checkpoint.get("param_count"),
    }
    for key, expected in expected_identity.items():
        observed = metadata.get(key)
        if observed is not None and observed != expected:
            raise ClinicalPrivacyReleaseError(
                f"held-out report {key} does not match checkpoint metadata"
            )

    formats = [str(item) for item in checkpoint.get("formats", ())]
    format_name = str(metadata.get("format") or metadata.get("model_format") or "")
    if not format_name:
        if len(formats) != 1:
            raise ClinicalPrivacyReleaseError(
                "held-out report must identify one checkpoint format"
            )
        format_name = formats[0]
    if format_name not in formats:
        raise ClinicalPrivacyReleaseError(
            "held-out report format is absent from the checkpoint manifest"
        )

    eval_set_hash = _require_hash(metadata.get("eval_set_hash"), "eval_set_hash")
    leakage_hash = _require_hash(
        metadata.get("leakage_fixture_hash"), "leakage_fixture_hash"
    )
    metadata.update(
        {
            **expected_identity,
            "eval_set_hash": eval_set_hash,
            "format": format_name,
            "leakage_fixture_hash": leakage_hash,
            "manifest": dict(checkpoint),
        }
    )
    return BenchmarkReport(
        suite=report.suite,
        model_name=report.model_name,
        device=report.device,
        fixture_count=report.fixture_count,
        metrics=report.metrics,
        generated_at=report.generated_at,
        metadata=metadata,
    )


def _scoped_gate_checks(
    preview: GateReport,
    *,
    scoring_report: BenchmarkReport,
    checkpoint: Mapping[str, Any],
    shield_report: BenchmarkReport,
    references: Mapping[str, str],
) -> tuple[GateCheck, ...]:
    checks_by_name = {check.gate: check for check in preview.gate_results}
    missing_checks = [
        gate for gate in CLINICAL_PRIVACY_REQUIRED_GATES if gate not in checks_by_name
    ]
    if missing_checks:
        raise ClinicalPrivacyReleaseError(
            "release harness did not emit required gates: " + ", ".join(missing_checks)
        )

    metrics = _mapping(scoring_report.metrics)
    metadata = _mapping(scoring_report.metadata)
    _, denominators = release_gates._per_label_recall(metrics, metadata)
    covered_labels = {
        label for label in preview.per_label_recall if denominators.get(label, 0) > 0
    }
    manifest = load_clinical_phi_manifest()
    coverage_groups = {
        name: sorted(covered_labels & set(manifest.label_groups[name]))
        for name in sorted(set(_G1A_COVERAGE_GROUPS) | set(_G2_COVERAGE_GROUPS))
    }
    explicit_g3 = "critical_leakage_count" in metrics or (
        "critical_leakage_count" in metadata
    )
    evidence = {
        "checkpoint_manifest": {
            "content_hash": stable_hash(checkpoint),
            "ref": references["checkpoint_manifest"],
            "reproducibility_hash": checkpoint["reproducibility_hash"],
        },
        "dataset_manifest": {
            "manifest_hash": clinical_phi_manifest_hash(manifest),
            "manifest_id": CLINICAL_PHI_MANIFEST_ID,
            "ref": CLINICAL_PHI_MANIFEST_REF,
        },
        "held_out_report": {
            "fixture_count": scoring_report.fixture_count,
            "ref": references["held_out_report"],
            "report_hash": stable_hash(scoring_report.to_dict()),
        },
        "shield_comparison": {
            "ref": references["shield_report"],
            "report_hash": stable_hash(shield_report.to_dict()),
            "role": "comparison_only",
        },
    }

    scoped: list[GateCheck] = []
    for gate in CLINICAL_PRIVACY_REQUIRED_GATES:
        original = checks_by_name[gate]
        required_groups: tuple[str, ...] = ()
        missing_groups: list[str] = []
        if gate == "G1a":
            required_groups = _G1A_COVERAGE_GROUPS
        elif gate == "G2":
            required_groups = _G2_COVERAGE_GROUPS
        if required_groups:
            missing_groups = [
                group for group in required_groups if not coverage_groups[group]
            ]

        passed = original.passed and not missing_groups
        reason = original.reason
        if missing_groups:
            reason = "held-out coverage missing: " + ", ".join(missing_groups)
        if gate == "G3" and not explicit_g3:
            passed = False
            reason = "critical leakage count is not explicitly reported"
        elif gate == "G3" and preview.residual_leakage_rate != 0.0:
            passed = False
            reason = "clinical PHI residual leakage must be exactly zero"

        details = dict(original.details)
        details["evidence"] = evidence
        if required_groups:
            details["coverage_groups"] = {
                group: coverage_groups[group] for group in required_groups
            }
            details["missing_coverage_groups"] = missing_groups
        if gate == "G3":
            details["explicit_critical_leakage_count"] = explicit_g3
            details["residual_leakage_ceiling"] = 0.0
            details["residual_leakage_rate"] = preview.residual_leakage_rate
        scoped.append(
            GateCheck(
                gate=gate,
                passed=passed,
                reason=reason,
                details=details,
            )
        )
    return tuple(scoped)


def _build_model_manifest_entry(
    checkpoint: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    held_out_report: BenchmarkReport,
    shield_report: BenchmarkReport,
    gate_report: GateReport,
    release_date: str,
) -> dict[str, Any]:
    row = dict(checkpoint)
    row.pop("model_id", None)
    row["released"] = release_date
    row["reproducibility_hash"] = provenance["reproducibility_hash"]
    row["training_provenance"] = {
        key: _plain(provenance[key])
        for key in sorted(TRAINING_PROVENANCE_FIELDS - {"path"})
        if key in provenance
    }
    held_out_metrics = _mapping(held_out_report.metrics)
    shield_comparison = _mapping(shield_report.metrics.get("shield_comparison"))
    shield_aggregate = _mapping(shield_comparison.get("aggregate"))
    row["benchmark"] = [
        {
            "suite": held_out_report.suite or CLINICAL_PRIVACY_HELD_OUT_SUITE,
            "dataset": CLINICAL_PHI_MANIFEST_ID,
            "micro_f1": _benchmark_f1(held_out_metrics),
            "recall": _certified_g1a_recall(gate_report),
            "leakage": gate_report.residual_leakage_rate,
        },
        {
            "suite": "shield",
            "dataset": "shield-public-comparison-by-reference",
            "micro_f1": _probability(
                shield_aggregate.get("exact_span_f1"),
                "SHIELD aggregate exact_span_f1",
            ),
            "recall": _probability(
                shield_aggregate.get("recall"), "SHIELD aggregate recall"
            ),
            "leakage": _probability(
                shield_aggregate.get("leakage"), "SHIELD aggregate leakage"
            ),
        },
    ]
    violations = validate_manifest_row(row, 1)
    if violations:
        raise ClinicalPrivacyReleaseError(
            "generated model manifest entry is invalid: "
            + "; ".join(violation.message for violation in violations)
        )
    return _plain_mapping(row)


def _build_release_manifest(
    *,
    gate_report: GateReport,
    model_card: ModelCardBuildResult,
    model_manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _mapping(gate_report.gate_results[0].details.get("evidence"))
    signature = gate_report.signature
    if signature is None:  # pragma: no cover - guarded by the builder contract.
        raise ClinicalPrivacyReleaseError("gate report must be signed")
    artifacts = {
        "gate_report": stable_hash(gate_report.to_dict()),
        "model_card": stable_hash(model_card.markdown),
        "model_datasheet": stable_hash(model_card.datasheet),
        "model_manifest_entry": stable_hash(model_manifest_entry),
    }
    return {
        "schema_version": CLINICAL_PRIVACY_RELEASE_SCHEMA_VERSION,
        "model_id": CLINICAL_PRIVACY_MODEL_ID,
        "decision": gate_report.decision,
        "certified_gates": list(CLINICAL_PRIVACY_REQUIRED_GATES),
        "gate_report": {
            "repro_hash": gate_report.repro_hash,
            "signature_algorithm": signature.algorithm,
            "signature_key_id": signature.key_id,
        },
        "evidence": evidence,
        "artifacts": artifacts,
        "safety": {
            "assist_only": True,
            "clinical_decision_trigger": False,
            "local_offline_after_download": True,
            "raw_phi_in_artifacts": False,
            "shield_role": "comparison_only",
        },
    }


def _benchmark_f1(metrics: Mapping[str, Any]) -> float:
    exact = _mapping(metrics.get("exact_span_f1"))
    return _probability(exact.get("f1"), "held-out exact_span_f1.f1")


def _certified_g1a_recall(report: GateReport) -> float:
    check = next(check for check in report.gate_results if check.gate == "G1a")
    labels = [str(label) for label in check.details.get("applicable_labels", ())]
    if not labels:
        raise ClinicalPrivacyReleaseError("G1a has no applicable held-out labels")
    return min(report.per_label_recall[label] for label in labels)


def _clinical_safety_section(report: GateReport) -> str:
    evidence = _mapping(report.gate_results[0].details.get("evidence"))
    checkpoint = _mapping(evidence.get("checkpoint_manifest"))
    dataset = _mapping(evidence.get("dataset_manifest"))
    held_out = _mapping(evidence.get("held_out_report"))
    shield = _mapping(evidence.get("shield_comparison"))
    links = "\n".join(
        (
            "### Release Evidence",
            "",
            f"- Checkpoint manifest: `{checkpoint.get('ref')}` "
            f"(`{checkpoint.get('reproducibility_hash')}`)",
            f"- Dataset manifest: `{dataset.get('ref')}` "
            f"(`{dataset.get('manifest_hash')}`)",
            f"- Held-out gate report: `{held_out.get('ref')}` "
            f"(`{held_out.get('report_hash')}`)",
            f"- SHIELD comparison report: `{shield.get('ref')}` "
            f"(`{shield.get('report_hash')}`)",
            "",
        )
    )
    return _CLINICAL_SAFETY_SECTION + "\n" + links


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClinicalPrivacyReleaseError(f"{name} must be a number between 0 and 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ClinicalPrivacyReleaseError(f"{name} must be a number between 0 and 1")
    return number


def _validate_reference(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ClinicalPrivacyReleaseError(f"{name} must be a string")
    reference = value.strip()
    if _SAFE_REFERENCE.fullmatch(reference) is None:
        raise ClinicalPrivacyReleaseError(
            f"{name} must be a non-empty reference without control characters"
        )
    return reference


def _require_hash(value: Any, name: str) -> str:
    digest = str(value or "")
    if _SHA256.fullmatch(digest) is None:
        raise ClinicalPrivacyReleaseError(
            f"{name} must match sha256:<64 lowercase hex characters>"
        )
    return digest


def _require_signing_key(key: bytes | str) -> None:
    if not isinstance(key, (bytes, str)) or not key:
        raise ClinicalPrivacyReleaseError(
            "an explicit non-empty signing key is required"
        )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(item) for key, item in sorted(value.items())}


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
    "CLINICAL_PRIVACY_FAMILY",
    "CLINICAL_PRIVACY_GATE_MILESTONE",
    "CLINICAL_PRIVACY_HELD_OUT_SUITE",
    "CLINICAL_PRIVACY_RELEASE_SCHEMA_VERSION",
    "CLINICAL_PRIVACY_REQUIRED_GATES",
    "ClinicalPrivacyGateFailure",
    "ClinicalPrivacyRelease",
    "ClinicalPrivacyReleaseError",
    "ClinicalPrivacyReleasePaths",
    "build_clinical_privacy_gate_report",
    "build_clinical_privacy_release",
]
