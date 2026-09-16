from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.repro_hash import compute_canonical_payload_hash
from openmed.training import (
    DIRECTID_INT4_EXPORT_FORMATS,
    DIRECTID_INT8_EXPORT_FORMATS,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDArtifactMeasurement,
    DirectIDEvaluationRequest,
    DirectIDExportRequest,
    DirectIDQuantizationError,
    hash_directid_artifact,
    run_directid_tiny_quantization,
)

EVAL_SET_HASH = "sha256:" + "c" * 64
PERFORMANCE_FIXTURE_HASH = "sha256:" + "d" * 64


def _candidate(checkpoint: Path) -> dict[str, object]:
    artifact_hash = hash_directid_artifact(checkpoint)
    return {
        "artifact_hash": artifact_hash,
        "certified": False,
        "checkpoint_ref": "openmed://candidate/directid/" + artifact_hash[7:],
        "family": DIRECTID_TINY_HEAD_CONTRACT.family,
        "format": "pytorch-fp32",
        "published": False,
        "ready_for_quantization": True,
        "reproducibility_hash": "sha256:" + "b" * 64,
        "run_id": "directid-tiny-synthetic-run",
        "schema_version": "openmed.training.directid_candidate.v1",
        "tier": DIRECTID_TINY_HEAD_CONTRACT.tier,
    }


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "candidate"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"fixture":"synthetic"}\n',
        encoding="utf-8",
    )
    (checkpoint / "model.bin").write_bytes(b"synthetic-checkpoint")
    return checkpoint


def _export(request: DirectIDExportRequest) -> Path:
    request.output_dir.mkdir(parents=True)
    artifact = request.output_dir / "model.bin"
    artifact.write_bytes(f"synthetic-{request.format}".encode())
    return artifact


def _recall(value: float = 1.0) -> dict[str, float]:
    return {label: value for label in DIRECTID_TINY_HEAD_CONTRACT.labels}


def _measurement(
    request: DirectIDEvaluationRequest,
) -> DirectIDArtifactMeasurement:
    recall = _recall(0.997)
    if request.format == "mlx-4bit":
        recall["SSN"] = 0.989
    elif request.bits == 4:
        recall = _recall(0.994)
    return DirectIDArtifactMeasurement(
        per_label_recall=recall,
        eval_set_hash=request.eval_set_hash,
        performance_fixture_hash=PERFORMANCE_FIXTURE_HASH,
        device="synthetic-phone-cpu",
        sample_count=32,
        p50_ms=20.0,
        p95_ms=50.0,
        ram_mb=128.0,
    )


def _record(evidence: dict[str, object], format_name: str) -> dict[str, object]:
    records = evidence["artifacts"]
    assert isinstance(records, list)
    return next(record for record in records if record["format"] == format_name)


def test_directid_quantization_accepts_int8_and_quarantines_only_failing_int4(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    result = run_directid_tiny_quantization(
        candidate=_candidate(checkpoint),
        checkpoint_path=checkpoint,
        fp_parent_per_label_recall=_recall(),
        eval_set_hash=EVAL_SET_HASH,
        param_count=44_000_000,
        exporter=_export,
        evaluator=_measurement,
        output_dir=tmp_path / "quantized",
    )

    evidence = dict(result.evidence)
    assert tuple(result.artifact_paths) == (
        *DIRECTID_INT8_EXPORT_FORMATS,
        *DIRECTID_INT4_EXPORT_FORMATS,
    )
    assert evidence["int8_g4_passed"] is True
    assert evidence["int8_tiny_fit_passed"] is True
    assert evidence["int8_ready_for_certification"] is True
    assert evidence["int4_status"] == "partially_quarantined"
    assert evidence["quarantined_formats"] == ["mlx-4bit"]
    assert evidence["final_certification_performed"] is False
    assert evidence["publishing_performed"] is False

    for format_name in DIRECTID_INT8_EXPORT_FORMATS:
        record = _record(evidence, format_name)
        assert record["disposition"] == "accepted"
        assert record["artifact_size_bytes"] > 0
        assert record["quant_recall_delta"] == pytest.approx(0.003)
        assert record["param_count"] == 44_000_000
        assert record["p50_ms"] == 20.0
        assert record["p95_ms"] == 50.0
        assert record["ram_mb"] == 128.0
        assert record["tiny_tier_fit"]["passed"] is True
        per_label = record["per_label_recall_delta"]
        assert set(per_label) == set(DIRECTID_TINY_HEAD_CONTRACT.labels)
        assert all(row["critical"] is True for row in per_label.values())

    rejected = _record(evidence, "mlx-4bit")
    assert rejected["g4"]["blocking_format"] == "mlx-4bit"
    assert rejected["g4"]["critical_offending_labels"] == ["SSN"]
    assert rejected["quarantine_reasons"] == ["G4_RECALL_DELTA"]
    assert _record(evidence, "coreml-int4")["disposition"] == "accepted"

    evidence_text = result.evidence_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in evidence_text
    assert "synthetic-checkpoint" not in evidence_text
    assert evidence["raw_phi_persisted"] is False
    assert result.manifest["evidence_hash"] == compute_canonical_payload_hash(evidence)
    assert json.loads(result.manifest_path.read_text(encoding="utf-8")) == dict(
        result.manifest
    )


def test_directid_quantization_quarantines_failing_int8_handoff(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)

    def measurement(
        request: DirectIDEvaluationRequest,
    ) -> DirectIDArtifactMeasurement:
        recall = _recall(0.997)
        if request.format == "onnx-int8":
            recall["API_KEY"] = 0.994
        return DirectIDArtifactMeasurement(
            per_label_recall=recall,
            eval_set_hash=request.eval_set_hash,
            performance_fixture_hash=PERFORMANCE_FIXTURE_HASH,
            device="synthetic-phone-cpu",
            sample_count=16,
            p50_ms=25.0,
            p95_ms=60.0,
            ram_mb=150.0,
        )

    result = run_directid_tiny_quantization(
        candidate=_candidate(checkpoint),
        checkpoint_path=checkpoint,
        fp_parent_per_label_recall=_recall(),
        eval_set_hash=EVAL_SET_HASH,
        param_count=44_000_000,
        exporter=_export,
        evaluator=measurement,
        output_dir=tmp_path / "quantized",
        include_int4=False,
    )

    evidence = dict(result.evidence)
    assert evidence["int8_g4_passed"] is False
    assert evidence["int8_tiny_fit_passed"] is True
    assert evidence["int8_ready_for_certification"] is False
    assert evidence["int4_status"] == "not_requested"
    assert evidence["quarantined_formats"] == ["onnx-int8"]
    assert _record(evidence, "onnx-int8")["quarantine_reasons"] == ["G4_RECALL_DELTA"]


def test_directid_quantization_requires_every_critical_label_before_export(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    parent_recall = _recall()
    del parent_recall["SSN"]
    called = False

    def exporter(request: DirectIDExportRequest) -> Path:
        nonlocal called
        called = True
        return _export(request)

    with pytest.raises(
        DirectIDQuantizationError,
        match="cover exactly the DirectID contract labels",
    ):
        run_directid_tiny_quantization(
            candidate=_candidate(checkpoint),
            checkpoint_path=checkpoint,
            fp_parent_per_label_recall=parent_recall,
            eval_set_hash=EVAL_SET_HASH,
            param_count=44_000_000,
            exporter=exporter,
            evaluator=_measurement,
            output_dir=tmp_path / "quantized",
        )

    assert called is False


def test_directid_quantization_records_tiny_tier_resource_quarantine(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)

    def measurement(
        request: DirectIDEvaluationRequest,
    ) -> DirectIDArtifactMeasurement:
        return DirectIDArtifactMeasurement(
            per_label_recall=_recall(0.997),
            eval_set_hash=request.eval_set_hash,
            performance_fixture_hash=PERFORMANCE_FIXTURE_HASH,
            device="synthetic-phone-cpu",
            sample_count=16,
            p50_ms=20.0,
            p95_ms=151.0 if request.format == "coreml-int8" else 50.0,
            ram_mb=128.0,
        )

    result = run_directid_tiny_quantization(
        candidate=_candidate(checkpoint),
        checkpoint_path=checkpoint,
        fp_parent_per_label_recall=_recall(),
        eval_set_hash=EVAL_SET_HASH,
        param_count=44_000_000,
        exporter=_export,
        evaluator=measurement,
        output_dir=tmp_path / "quantized",
        include_int4=False,
    )

    evidence = dict(result.evidence)
    coreml = _record(evidence, "coreml-int8")
    assert evidence["int8_g4_passed"] is True
    assert evidence["int8_tiny_fit_passed"] is False
    assert evidence["int8_ready_for_certification"] is False
    assert coreml["tiny_tier_fit"]["violations"] == {
        "p95_ms": {"limit": 150.0, "observed": 151.0}
    }
    assert coreml["quarantine_reasons"] == ["G5_TINY_TIER_FIT"]
