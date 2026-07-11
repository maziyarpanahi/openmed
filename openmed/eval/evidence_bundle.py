"""Bundle release-gate evidence artifacts into a deterministic evidence pack."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core import baseline as baseline_store
from openmed.core.audit import manifest_hash, stable_hash
from openmed.eval.quant_delta import is_quantized_format

SCHEMA_VERSION = "openmed.gate_evidence_bundle.v1"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.txt"

G1_G8 = ("G1a", "G1b", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
_GATE_ORDER = (
    "policy_profile",
    "thresholds_matrix",
    "manifest_coherence",
    "calibration_present",
    *G1_G8,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THRESHOLDS_MATRIX = _REPO_ROOT / "openmed" / "core" / "thresholds.json"
_DEFAULT_MODEL_MANIFEST = _REPO_ROOT / "models.jsonl"
_DEFAULT_POLICY_DIR = _REPO_ROOT / "openmed" / "core" / "policies"

_DEFAULT_GATES_BY_ARTIFACT: dict[str, tuple[str, ...]] = {
    "candidate_report": G1_G8,
    "model_manifest": ("manifest_coherence",),
    "model_card": ("manifest_coherence",),
    "readme": ("manifest_coherence",),
    "thresholds_matrix": ("thresholds_matrix", "G1a", "G2"),
    "policy_profile": ("policy_profile", "calibration_present"),
    "calibration_thresholds": ("calibration_present", "G1a", "G1b", "G2"),
    "calibration_report": ("calibration_present",),
    "eval_set": ("G1a", "G1b", "G2", "G7"),
    "leakage_fixtures": ("G3", "G7"),
    "quant_recall_delta": ("G4",),
    "performance_report": ("G5", "G6"),
    "resource_report": ("G5",),
    "baseline_store": ("G7",),
    "span_fixtures": ("G8",),
    "gate_report": (),
}

_PATH_ALIASES: dict[str, str] = {
    "baseline_path": "baseline_store",
    "baseline_store_path": "baseline_store",
    "benchmark_report_path": "candidate_report",
    "calibration_path": "calibration_report",
    "calibration_report_path": "calibration_report",
    "calibration_thresholds_path": "calibration_thresholds",
    "candidate_path": "candidate_report",
    "candidate_report_path": "candidate_report",
    "eval_fixture_path": "eval_set",
    "eval_fixtures_path": "eval_set",
    "eval_report_path": "candidate_report",
    "eval_set_path": "eval_set",
    "fixture_path": "span_fixtures",
    "fixtures_path": "span_fixtures",
    "g8_fixture_path": "span_fixtures",
    "g8_span_fixture_path": "span_fixtures",
    "last_green_baseline_path": "baseline_store",
    "leakage_fixture_path": "leakage_fixtures",
    "leakage_fixtures_path": "leakage_fixtures",
    "manifest_path": "model_manifest",
    "model_card_path": "model_card",
    "models_manifest_path": "model_manifest",
    "performance_report_path": "performance_report",
    "policy_path": "policy_profile",
    "policy_profile_path": "policy_profile",
    "quant_delta_path": "quant_recall_delta",
    "quant_recall_delta_path": "quant_recall_delta",
    "quantization_report_path": "quant_recall_delta",
    "readme_path": "readme",
    "resource_report_path": "resource_report",
    "span_fixture_path": "span_fixtures",
    "span_fixtures_path": "span_fixtures",
    "threshold_matrix_path": "thresholds_matrix",
    "thresholds_json_path": "calibration_thresholds",
    "thresholds_matrix_path": "thresholds_matrix",
    "thresholds_path": "calibration_thresholds",
}


@dataclass(frozen=True)
class EvidenceArtifactSpec:
    """A referenced evidence artifact before it is copied into the bundle."""

    artifact_id: str
    path: str | Path | None = None
    gates: tuple[str, ...] = ()
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class EvidenceBundleResult:
    """Result metadata for a written evidence bundle."""

    output_dir: Path
    manifest_path: Path
    summary_path: Path
    manifest: Mapping[str, Any]
    missing_required: tuple[Mapping[str, Any], ...]
    summary: str

    @property
    def has_missing_required(self) -> bool:
        """Return whether any required artifact is absent."""
        return bool(self.missing_required)


def bundle_gate_evidence(
    gate_report: Mapping[str, Any] | Any,
    output_dir: str | Path,
    *,
    evidence_root: str | Path | None = None,
    extra_artifacts: Mapping[str, str | Path] | Sequence[Mapping[str, Any]] = (),
    manifest_name: str = MANIFEST_FILENAME,
) -> EvidenceBundleResult:
    """Collect evidence referenced by *gate_report* into *output_dir*.

    The manifest is deterministic for identical inputs: artifact entries,
    gates, and summary fields are sorted, and no wall-clock timestamps are
    recorded. Missing required artifacts are represented as manifest entries
    with ``status: "missing"`` rather than being dropped.
    """

    payload = _gate_report_payload(gate_report)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    root = Path(evidence_root) if evidence_root is not None else None
    specs = _extract_specs(payload)
    specs.extend(_coerce_extra_artifacts(extra_artifacts))
    _add_inferred_required_specs(payload, specs)

    artifacts = _materialise_artifacts(specs, destination, root)
    manifest = _build_manifest(payload, artifacts)
    summary = _render_summary(manifest)

    manifest_path = destination / manifest_name
    summary_path = destination / SUMMARY_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(summary + "\n", encoding="utf-8")

    missing_required = tuple(
        entry
        for entry in manifest["artifacts"]
        if entry["required"] and entry["status"] == "missing"
    )
    return EvidenceBundleResult(
        output_dir=destination,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        missing_required=missing_required,
        summary=summary,
    )


def _gate_report_payload(gate_report: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(gate_report, (str, Path)):
        path = Path(gate_report)
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, Mapping):
                return dict(value)
        raise TypeError("gate_report path must point to a JSON object")
    if isinstance(gate_report, Mapping):
        return dict(gate_report)
    if hasattr(gate_report, "to_dict") and callable(gate_report.to_dict):
        value = gate_report.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("gate_report must be a mapping or expose to_dict()")


def _extract_specs(payload: Mapping[str, Any]) -> list[EvidenceArtifactSpec]:
    specs: list[EvidenceArtifactSpec] = []
    for key in ("evidence", "evidence_artifacts"):
        specs.extend(_coerce_evidence_value(payload.get(key)))
    _scan_path_aliases(payload, specs, gate=None)

    for check in _gate_checks(payload):
        gate = str(check.get("gate") or "")
        details = check.get("details")
        if not isinstance(details, Mapping):
            continue
        for key in ("evidence", "evidence_artifacts"):
            specs.extend(_coerce_evidence_value(details.get(key), default_gate=gate))
        _scan_path_aliases(details, specs, gate=gate)
    return specs


def _coerce_extra_artifacts(
    extra_artifacts: Mapping[str, str | Path] | Sequence[Mapping[str, Any]],
) -> list[EvidenceArtifactSpec]:
    if isinstance(extra_artifacts, Mapping):
        return [
            EvidenceArtifactSpec(
                artifact_id=str(artifact_id),
                path=path,
                gates=_DEFAULT_GATES_BY_ARTIFACT.get(str(artifact_id), ()),
            )
            for artifact_id, path in sorted(extra_artifacts.items())
        ]

    specs: list[EvidenceArtifactSpec] = []
    for item in extra_artifacts:
        if isinstance(item, Mapping):
            spec = _spec_from_mapping(item)
            if spec is not None:
                specs.append(spec)
    return specs


def _coerce_evidence_value(
    value: Any,
    *,
    default_gate: str | None = None,
    default_artifact_id: str | None = None,
) -> list[EvidenceArtifactSpec]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        artifact_id = default_artifact_id or Path(str(value)).stem or "artifact"
        return [
            EvidenceArtifactSpec(
                artifact_id=artifact_id,
                path=value,
                gates=_merge_gates(
                    _DEFAULT_GATES_BY_ARTIFACT.get(artifact_id, ()),
                    (default_gate,) if default_gate else (),
                ),
            )
        ]
    if isinstance(value, Mapping):
        direct = _spec_from_mapping(value, default_gate=default_gate)
        if direct is not None:
            return [direct]

        specs: list[EvidenceArtifactSpec] = []
        for key, item in value.items():
            artifact_id = str(key)
            if isinstance(item, Mapping):
                nested = dict(item)
                nested.setdefault("artifact_id", artifact_id)
                specs.extend(_coerce_evidence_value(nested, default_gate=default_gate))
            else:
                specs.extend(
                    _coerce_evidence_value(
                        item,
                        default_gate=default_gate,
                        default_artifact_id=artifact_id,
                    )
                )
        return specs
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        specs = []
        for item in value:
            specs.extend(_coerce_evidence_value(item, default_gate=default_gate))
        return specs
    return []


def _spec_from_mapping(
    value: Mapping[str, Any],
    *,
    default_gate: str | None = None,
) -> EvidenceArtifactSpec | None:
    artifact_id = str(
        value.get("artifact_id")
        or value.get("id")
        or value.get("name")
        or value.get("type")
        or ""
    )
    path = value.get("path") or value.get("source_path") or value.get("file")
    if not artifact_id and path:
        artifact_id = Path(str(path)).stem or "artifact"
    if not artifact_id:
        return None

    gates = _gate_tuple(value.get("gates") or value.get("gate"))
    if default_gate:
        gates = _merge_gates(gates, (default_gate,))
    if not gates:
        gates = _DEFAULT_GATES_BY_ARTIFACT.get(artifact_id, ())
    required = _boolish(value.get("required", True))
    return EvidenceArtifactSpec(
        artifact_id=artifact_id,
        path=path,
        gates=gates,
        required=required,
        description=str(value.get("description") or ""),
    )


def _scan_path_aliases(
    value: Any,
    specs: list[EvidenceArtifactSpec],
    *,
    gate: str | None,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            artifact_id = _PATH_ALIASES.get(key.lower())
            if artifact_id and isinstance(item, (str, Path)) and str(item):
                gates = _DEFAULT_GATES_BY_ARTIFACT.get(artifact_id, ())
                if gate:
                    gates = _merge_gates(gates, (gate,))
                specs.append(
                    EvidenceArtifactSpec(
                        artifact_id=artifact_id,
                        path=item,
                        gates=gates,
                    )
                )
            _scan_path_aliases(item, specs, gate=gate)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _scan_path_aliases(item, specs, gate=gate)


def _add_inferred_required_specs(
    payload: Mapping[str, Any],
    specs: list[EvidenceArtifactSpec],
) -> None:
    gates = set(_gate_names(payload))

    if payload.get("eval_set_hash") and not _has_artifact(specs, "eval_set"):
        specs.append(EvidenceArtifactSpec("eval_set", gates=("G1a", "G1b", "G2", "G7")))

    if payload.get("leakage_fixture_hash") and not _has_artifact(
        specs, "leakage_fixtures"
    ):
        specs.append(EvidenceArtifactSpec("leakage_fixtures", gates=("G3", "G7")))

    if "calibration_present" in gates:
        if not _has_artifact(specs, "calibration_thresholds"):
            specs.append(EvidenceArtifactSpec("calibration_thresholds"))
        if not _has_artifact(specs, "calibration_report"):
            specs.append(EvidenceArtifactSpec("calibration_report"))

    if "G8" in gates and not _has_artifact(specs, "span_fixtures"):
        specs.append(EvidenceArtifactSpec("span_fixtures"))

    if _quant_delta_required(payload) and not _has_artifact(
        specs, "quant_recall_delta"
    ):
        specs.append(EvidenceArtifactSpec("quant_recall_delta"))

    if "G7" in gates and not _has_artifact(specs, "baseline_store"):
        specs.append(
            EvidenceArtifactSpec("baseline_store", path=baseline_store.BASELINE_PATH)
        )

    if "thresholds_matrix" in gates and not _has_artifact(specs, "thresholds_matrix"):
        specs.append(
            EvidenceArtifactSpec("thresholds_matrix", path=_DEFAULT_THRESHOLDS_MATRIX)
        )

    if "manifest_coherence" in gates and not _has_artifact(specs, "model_manifest"):
        specs.append(
            EvidenceArtifactSpec("model_manifest", path=_DEFAULT_MODEL_MANIFEST)
        )

    if "policy_profile" in gates and not _has_artifact(specs, "policy_profile"):
        policy_path = _policy_profile_path(str(payload.get("policy") or ""))
        specs.append(EvidenceArtifactSpec("policy_profile", path=policy_path))


def _quant_delta_required(payload: Mapping[str, Any]) -> bool:
    format_name = str(payload.get("format") or "")
    if not is_quantized_format(format_name):
        return False
    for check in _gate_checks(payload):
        if check.get("gate") != "G4":
            continue
        details = check.get("details") or {}
        if isinstance(details, Mapping):
            source = str(details.get("source") or "")
            return source not in {"", "not_applicable"}
    return True


def _policy_profile_path(policy: str) -> Path | None:
    if not policy:
        return None
    candidate = _DEFAULT_POLICY_DIR / f"{policy}.json"
    return candidate


def _has_artifact(specs: Sequence[EvidenceArtifactSpec], artifact_id: str) -> bool:
    return any(spec.artifact_id == artifact_id for spec in specs)


def _materialise_artifacts(
    specs: Sequence[EvidenceArtifactSpec],
    output_dir: Path,
    evidence_root: Path | None,
) -> list[dict[str, Any]]:
    grouped = _merge_specs(specs, evidence_root)
    entries: list[dict[str, Any]] = []
    used_bundle_paths: set[str] = set()
    for item in grouped:
        artifact_id = item["artifact_id"]
        source_path = item["source_path"]
        gates = _sort_gates(item["gates"])
        required = bool(item["required"])
        entry: dict[str, Any] = {
            "artifact_id": artifact_id,
            "artifact_ids": sorted(item["artifact_ids"]),
            "description": item["description"],
            "gates": gates,
            "required": required,
            "source_path": str(source_path) if source_path is not None else None,
        }
        if source_path is None or not source_path.is_file():
            entry.update({"status": "missing", "bundle_path": None, "sha256": None})
        else:
            bundle_path = _copy_artifact(
                source_path,
                output_dir,
                artifact_id,
                used_bundle_paths,
            )
            entry.update(
                {
                    "status": "present",
                    "bundle_path": bundle_path,
                    "sha256": manifest_hash(source_path),
                }
            )
        entries.append(entry)

    return sorted(
        entries,
        key=lambda entry: (
            entry["status"] != "present",
            entry["artifact_id"],
            entry["source_path"] or "",
        ),
    )


def _merge_specs(
    specs: Sequence[EvidenceArtifactSpec],
    evidence_root: Path | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for spec in specs:
        artifact_id = _normalise_artifact_id(spec.artifact_id)
        path = _resolve_source_path(spec.path, evidence_root)
        key = str(path) if path is not None else f"missing:{artifact_id}"
        if key not in grouped:
            grouped[key] = {
                "artifact_id": artifact_id,
                "artifact_ids": {artifact_id},
                "description": spec.description,
                "gates": set(_gate_tuple(spec.gates)),
                "required": bool(spec.required),
                "source_path": path,
            }
        else:
            grouped[key]["artifact_ids"].add(artifact_id)
            grouped[key]["gates"].update(_gate_tuple(spec.gates))
            grouped[key]["required"] = grouped[key]["required"] or bool(spec.required)
            if spec.description and not grouped[key]["description"]:
                grouped[key]["description"] = spec.description

    for item in grouped.values():
        if not item["gates"]:
            gates: set[str] = set()
            for artifact_id in item["artifact_ids"]:
                gates.update(_DEFAULT_GATES_BY_ARTIFACT.get(artifact_id, ()))
            item["gates"] = gates

    return sorted(
        grouped.values(),
        key=lambda item: (item["artifact_id"], str(item["source_path"] or "")),
    )


def _resolve_source_path(
    path: str | Path | None, evidence_root: Path | None
) -> Path | None:
    if path is None or str(path) == "":
        return None
    source = Path(path)
    if source.is_absolute():
        return source
    if evidence_root is not None:
        return evidence_root / source
    return source


def _copy_artifact(
    source_path: Path,
    output_dir: Path,
    artifact_id: str,
    used_bundle_paths: set[str],
) -> str:
    artifact_dir = output_dir / "artifacts" / _safe_path_part(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(source_path.name)
    target = artifact_dir / filename
    relative = target.relative_to(output_dir).as_posix()
    if relative in used_bundle_paths:
        suffix = _path_suffix(source_path)
        target = artifact_dir / f"{target.stem}-{suffix}{target.suffix}"
        relative = target.relative_to(output_dir).as_posix()
    shutil.copy2(source_path, target)
    used_bundle_paths.add(relative)
    return relative


def _build_manifest(
    payload: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate_summary = _gate_summary(payload, artifacts)
    stability_summary = _stability_summary(payload)
    missing_required = [
        entry
        for entry in artifacts
        if entry["required"] and entry["status"] == "missing"
    ]
    covered_gates = [
        gate for gate, state in gate_summary.items() if state["status"] == "covered"
    ]
    missing_gates = [
        gate for gate, state in gate_summary.items() if state["status"] == "missing"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_report": {
            "repo_id": payload.get("repo_id"),
            "family": payload.get("family"),
            "tier": payload.get("tier"),
            "format": payload.get("format"),
            "decision": payload.get("decision"),
            "gate_results": [dict(check) for check in _gate_checks(payload)],
            "repro_hash": payload.get("repro_hash"),
            "stability_summary": stability_summary,
        },
        "summary": {
            "covered_gates": _sort_gates(covered_gates),
            "missing_gates": _sort_gates(missing_gates),
            "required_missing": len(missing_required),
            "present_artifacts": len(
                [entry for entry in artifacts if entry["status"] == "present"]
            ),
            "missing_artifacts": len(
                [entry for entry in artifacts if entry["status"] == "missing"]
            ),
            "quarantined_stability_gates": list(
                stability_summary.get("quarantined_gates", [])
            ),
            "stability_verdict": stability_summary.get("verdict"),
            "total_artifacts": len(artifacts),
        },
        "gates": gate_summary,
        "artifacts": [dict(entry) for entry in artifacts],
    }


def _gate_summary(
    payload: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    gates = set(_gate_names(payload))
    for artifact in artifacts:
        gates.update(str(gate) for gate in artifact.get("gates", ()))

    summary: dict[str, dict[str, Any]] = {}
    for gate in _sort_gates(gates):
        present = sorted(
            {
                str(artifact["artifact_id"])
                for artifact in artifacts
                if gate in artifact.get("gates", ())
                and artifact.get("status") == "present"
            }
        )
        missing = sorted(
            {
                str(artifact["artifact_id"])
                for artifact in artifacts
                if gate in artifact.get("gates", ())
                and artifact.get("status") == "missing"
                and artifact.get("required")
            }
        )
        if missing:
            status = "missing"
        elif present:
            status = "covered"
        else:
            status = "not_referenced"
        summary[gate] = {
            "status": status,
            "artifacts": present,
            "missing_artifacts": missing,
        }
    return summary


def _render_summary(manifest: Mapping[str, Any]) -> str:
    lines = [
        "OpenMed gate evidence bundle",
        f"Decision: {manifest['gate_report'].get('decision')}",
        f"Stability: {manifest['summary'].get('stability_verdict') or 'not_recorded'}",
        f"Present artifacts: {manifest['summary']['present_artifacts']}",
        f"Missing required artifacts: {manifest['summary']['required_missing']}",
        "Gate coverage:",
    ]
    for gate, state in manifest["gates"].items():
        artifacts = ", ".join(state["artifacts"]) if state["artifacts"] else "none"
        missing = (
            ", ".join(state["missing_artifacts"])
            if state["missing_artifacts"]
            else "none"
        )
        lines.append(
            f"- {gate}: {state['status']} (artifacts: {artifacts}; missing: {missing})"
        )
    return "\n".join(lines)


def _gate_checks(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    checks = payload.get("gate_results") or []
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes, bytearray)):
        return []
    return [check for check in checks if isinstance(check, Mapping)]


def _stability_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("stability_summary")
    return dict(value) if isinstance(value, Mapping) else {}


def _gate_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    gates = {
        str(check.get("gate")) for check in _gate_checks(payload) if check.get("gate")
    }
    gates.update(G1_G8)
    return tuple(_sort_gates(gates))


def _gate_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _merge_gates(*values: Sequence[str]) -> tuple[str, ...]:
    gates: set[str] = set()
    for value in values:
        gates.update(_gate_tuple(value))
    return tuple(_sort_gates(gates))


def _sort_gates(gates: Sequence[str] | set[str]) -> list[str]:
    order = {gate: index for index, gate in enumerate(_GATE_ORDER)}
    return sorted(
        {str(gate) for gate in gates if str(gate)},
        key=lambda gate: (order.get(gate, len(order)), gate),
    )


def _normalise_artifact_id(value: str) -> str:
    text = str(value).strip()
    return text if text else "artifact"


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _safe_path_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return text or "artifact"


def _safe_filename(value: str) -> str:
    text = _safe_path_part(value)
    if "." not in text and "." in value:
        suffix = Path(value).suffix
        if suffix:
            text = f"{text}{suffix}"
    return text or "artifact"


def _path_suffix(path: Path) -> str:
    return stable_hash(str(path.resolve())).split(":", 1)[1][:12]
