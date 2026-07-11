"""Build model cards and datasheets from signed release artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import manifest_hash, stable_hash
from openmed.core.model_card import append_model_card_sections, render_model_card
from openmed.core.repro_hash import load_training_provenance
from openmed.eval import release_gates
from openmed.eval.release_gates import GateReport

MODEL_CARD_DATASHEET_SCHEMA_VERSION = "openmed.model_card_datasheet.v1"
MODEL_DATASHEET_FILENAME = "model-datasheet.json"


class ModelCardBuilderError(ValueError):
    """Raised when artifact-backed model-card generation is inconsistent."""


@dataclass(frozen=True)
class ModelCardBuildResult:
    """Rendered model-card and datasheet payload for one model artifact."""

    manifest_row: Mapping[str, Any]
    gate_report: Mapping[str, Any]
    datasheet: Mapping[str, Any]
    markdown: str

    def datasheet_json(self) -> str:
        """Return the datasheet as byte-stable JSON."""

        return (
            json.dumps(
                _plain(self.datasheet),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def write_markdown(self, path: str | Path) -> Path:
        """Write the rendered model card to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.markdown, encoding="utf-8")
        return output_path

    def write_datasheet(self, path: str | Path) -> Path:
        """Write the rendered datasheet JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.datasheet_json(), encoding="utf-8")
        return output_path


@dataclass(frozen=True)
class _LoadedArtifact:
    artifact_id: str
    payload: Mapping[str, Any]
    source_path: str | None
    sha256: str


def build_model_card(
    manifest_row: Mapping[str, Any],
    gate_report: Mapping[str, Any] | GateReport | str | Path,
    *,
    calibration_report: Mapping[str, Any] | str | Path | None = None,
    calibration_thresholds: Mapping[str, Any] | str | Path | None = None,
    fairness_report: Mapping[str, Any] | str | Path | None = None,
    quant_delta: Mapping[str, Any] | str | Path | None = None,
    training_provenance: Mapping[str, Any] | str | Path | None = None,
) -> ModelCardBuildResult:
    """Build a model card plus datasheet from source artifacts.

    The generated datasheet is derived from the manifest row, release-gate
    report, and optional artifact payloads. Metrics copied from the release
    gate are validated field-for-field before the card is returned.
    """

    row = _plain_mapping(manifest_row, "manifest_row")
    gate_artifact = _load_artifact("gate_report", gate_report, required=True)
    gate_payload = GateReport.from_dict(gate_artifact.payload).to_dict()

    provenance_artifact = _load_training_provenance_artifact(
        training_provenance if training_provenance is not None else _row_provenance(row)
    )
    artifacts = [
        gate_artifact,
        *_optional_artifacts(
            {
                "calibration_report": calibration_report,
                "calibration_thresholds": calibration_thresholds,
                "fairness_report": fairness_report,
                "quant_recall_delta": quant_delta,
            }
        ),
    ]
    if provenance_artifact is not None:
        artifacts.append(provenance_artifact)

    _validate_gate_manifest_identity(row, gate_payload)
    datasheet = _build_datasheet(
        row,
        gate_payload,
        artifacts=artifacts,
        training_provenance=(
            provenance_artifact.payload if provenance_artifact is not None else {}
        ),
    )
    markdown = append_model_card_sections(
        render_model_card(row),
        [render_datasheet_markdown(datasheet)],
    )
    validate_model_card_consistency(
        markdown,
        row,
        gate_payload,
        datasheet=datasheet,
    )
    return ModelCardBuildResult(
        manifest_row=row,
        gate_report=gate_payload,
        datasheet=datasheet,
        markdown=markdown,
    )


def render_datasheet_markdown(datasheet: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown datasheet section."""

    payload = _plain_mapping(datasheet, "datasheet")
    model = _mapping(payload.get("model"))
    intended_use = _mapping(payload.get("intended_use"))
    lineage = _mapping(payload.get("training_data_lineage"))
    evaluation = _mapping(payload.get("evaluation"))
    residual_risk = _mapping(payload.get("residual_risk"))
    limitations = _sequence_of_mappings(payload.get("known_limitations"))
    gate_evidence = _mapping(payload.get("gate_evidence"))

    lines = [
        "## Datasheet",
        "",
        "### Intended Use",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Repository | `{_string(model.get('repo_id'), 'Not specified')}` |",
        f"| Task | {_string(intended_use.get('task'), 'Not specified')} |",
        f"| Family | {_string(intended_use.get('family'), 'Not specified')} |",
        f"| Languages | {_markdown_values(intended_use.get('languages'))} |",
        f"| Tier | {_string(intended_use.get('tier'), 'Not specified')} |",
        f"| Formats | {_markdown_values(intended_use.get('formats'))} |",
        "",
        "### Training Data Lineage",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Base model | `{_string(lineage.get('base_model'), 'Not reported')}` |",
        f"| Base model revision | `{_string(lineage.get('base_model_revision'), 'Not reported')}` |",
        f"| Data manifest hash | `{_string(lineage.get('data_manifest_hash'), 'Not reported')}` |",
        f"| Recipe config hash | `{_string(lineage.get('recipe_config_hash'), 'Not reported')}` |",
        f"| Environment lock digest | `{_string(lineage.get('env_lock_digest'), 'Not reported')}` |",
        f"| Git SHA | `{_string(lineage.get('git_sha'), 'Not reported')}` |",
        f"| RNG seeds | {_rng_seeds(lineage.get('rng_seeds'))} |",
        f"| License tags | {_markdown_values(lineage.get('license_tags'))} |",
        "",
        "### Evaluation Suites And Hashes",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Eval set hash | `{_string(evaluation.get('eval_set_hash'), 'Not reported')}` |",
        f"| Leakage fixture hash | `{_string(evaluation.get('leakage_fixture_hash'), 'Not reported')}` |",
        f"| Artifact hashes | {_artifact_hashes(evaluation.get('artifact_hashes'))} |",
        "",
        "### Residual Risk",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Decision | {_string(residual_risk.get('decision'), 'Not reported')} |",
        f"| Critical leakage count | {_number(residual_risk.get('critical_leakage_count'))} |",
        f"| Residual leakage rate | {_number(residual_risk.get('residual_leakage_rate'))} |",
        f"| Target leakage rate | {_number(residual_risk.get('target_leakage_rate'))} |",
        f"| Blocked formats | {_markdown_values(residual_risk.get('blocked_formats'))} |",
        "",
        "### Known Limitations",
        "",
        "| Gate | Reason |",
        "|---|---|",
    ]
    if limitations:
        lines.extend(
            f"| {_string(item.get('gate'), 'unknown')} | "
            f"{_string(item.get('reason'), 'Not reported')} |"
            for item in limitations
        )
    else:
        lines.append("| None | No failing release gates reported. |")

    lines.extend(
        [
            "",
            "### Gate Evidence",
            "",
            "```json",
            json.dumps(gate_evidence, ensure_ascii=True, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_model_card_consistency(
    markdown: str,
    manifest_row: Mapping[str, Any],
    gate_report: Mapping[str, Any] | GateReport,
    *,
    datasheet: Mapping[str, Any],
) -> None:
    """Fail if card metadata or datasheet metrics drift from source artifacts."""

    row = _plain_mapping(manifest_row, "manifest_row")
    gate_payload = _gate_payload(gate_report)
    card_metadata = release_gates._parse_model_card_front_matter(markdown)
    card_mismatches = release_gates._model_card_mismatches(card_metadata, row)
    if card_mismatches:
        raise ModelCardBuilderError(
            "generated model card front matter diverges from manifest: "
            + json.dumps(card_mismatches, sort_keys=True)
        )

    _validate_gate_manifest_identity(row, gate_payload)
    expected_gate_evidence = _gate_evidence(gate_payload)
    actual_gate_evidence = _mapping(datasheet.get("gate_evidence"))
    if actual_gate_evidence != expected_gate_evidence:
        raise ModelCardBuilderError(
            "datasheet gate evidence diverges from signed gate report: "
            + _mismatch_summary(actual_gate_evidence, expected_gate_evidence)
        )

    residual_risk = _mapping(datasheet.get("residual_risk"))
    expected_residual = _residual_risk(gate_payload)
    if residual_risk != expected_residual:
        raise ModelCardBuilderError(
            "datasheet residual risk diverges from signed gate report: "
            + _mismatch_summary(residual_risk, expected_residual)
        )

    evaluation = _mapping(datasheet.get("evaluation"))
    for key in ("eval_set_hash", "leakage_fixture_hash"):
        if evaluation.get(key) != gate_payload.get(key):
            raise ModelCardBuilderError(
                f"datasheet {key} diverges from signed gate report"
            )


def _build_datasheet(
    row: Mapping[str, Any],
    gate_payload: Mapping[str, Any],
    *,
    artifacts: Sequence[_LoadedArtifact],
    training_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_hashes = {
        artifact.artifact_id: artifact.sha256
        for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
    }
    artifact_summaries = {
        artifact.artifact_id: _artifact_summary(artifact)
        for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
    }
    return {
        "schema_version": MODEL_CARD_DATASHEET_SCHEMA_VERSION,
        "model": _model_summary(row),
        "intended_use": _intended_use(row),
        "training_data_lineage": _training_lineage(row, training_provenance),
        "evaluation": {
            "eval_set_hash": gate_payload.get("eval_set_hash"),
            "leakage_fixture_hash": gate_payload.get("leakage_fixture_hash"),
            "artifact_hashes": artifact_hashes,
            "artifacts": artifact_summaries,
        },
        "gate_evidence": _gate_evidence(gate_payload),
        "residual_risk": _residual_risk(gate_payload),
        "known_limitations": _known_limitations(gate_payload),
        "consistency": {
            "source": "signed_gate_report",
            "checked_fields": [
                "critical_leakage_count",
                "eval_set_hash",
                "leakage_fixture_hash",
                "per_label_recall",
                "quant_recall_delta",
                "residual_leakage_rate",
                "tier_fit",
            ],
        },
    }


def _gate_evidence(gate_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "per_label_recall": _sorted_mapping(gate_payload.get("per_label_recall")),
        "critical_leakage_count": gate_payload.get("critical_leakage_count"),
        "quant_recall_delta": gate_payload.get("quant_recall_delta"),
        "tier_fit": _gate_check_payload(gate_payload, "G5"),
        "eval_set_hash": gate_payload.get("eval_set_hash"),
        "leakage_fixture_hash": gate_payload.get("leakage_fixture_hash"),
    }


def _residual_risk(gate_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blocked_formats": _string_list(gate_payload.get("blocked_formats")),
        "critical_leakage_count": gate_payload.get("critical_leakage_count"),
        "decision": gate_payload.get("decision"),
        "residual_leakage_rate": gate_payload.get("residual_leakage_rate"),
        "target_leakage_rate": gate_payload.get("target_leakage_rate"),
    }


def _training_lineage(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    row_provenance = _mapping(row.get("training_provenance"))
    return {
        "base_model": provenance.get("base_model") or row.get("base_model"),
        "base_model_revision": provenance.get("base_model_revision")
        or row_provenance.get("base_model_revision"),
        "data_manifest_hash": provenance.get("data_manifest_hash")
        or row_provenance.get("data_manifest_hash"),
        "recipe_config_hash": provenance.get("recipe_config_hash")
        or row_provenance.get("recipe_config_hash"),
        "env_lock_digest": provenance.get("env_lock_digest")
        or row_provenance.get("env_lock_digest"),
        "git_sha": provenance.get("git_sha") or row_provenance.get("git_sha"),
        "rng_seeds": _sorted_mapping(
            provenance.get("rng_seeds") or row_provenance.get("rng_seeds")
        ),
        "manifest_reproducibility_hash": row.get("reproducibility_hash"),
        "provenance_reproducibility_hash": provenance.get("reproducibility_hash")
        or row_provenance.get("reproducibility_hash"),
        "license_tags": _license_tags(row, provenance),
    }


def _license_tags(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[str]:
    values: list[str] = []
    for source in (row, provenance):
        for key in ("license", "license_tag", "license_tags", "licenses"):
            values.extend(_string_list(source.get(key)))
        data_sources = source.get("data_sources")
        if isinstance(data_sources, Sequence) and not isinstance(
            data_sources,
            (str, bytes, bytearray),
        ):
            for data_source in data_sources:
                if isinstance(data_source, Mapping):
                    values.extend(_string_list(data_source.get("license")))
                    values.extend(_string_list(data_source.get("license_id")))
                    values.extend(_string_list(data_source.get("license_tags")))
    return _stable_unique(values)


def _model_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architecture": row.get("architecture"),
        "base_model": row.get("base_model"),
        "family": row.get("family"),
        "formats": _string_list(row.get("formats")),
        "languages": _string_list(row.get("languages")),
        "license": row.get("license"),
        "param_count": row.get("param_count"),
        "repo_id": row.get("repo_id"),
        "task": row.get("task"),
        "tier": row.get("tier"),
    }


def _intended_use(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": row.get("family"),
        "formats": _string_list(row.get("formats")),
        "languages": _string_list(row.get("languages")),
        "task": row.get("task"),
        "tier": row.get("tier"),
    }


def _known_limitations(gate_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for check in _gate_checks(gate_payload):
        if check.get("passed") is False:
            failures.append(
                {
                    "gate": str(check.get("gate") or "unknown"),
                    "reason": str(check.get("reason") or "failed"),
                    "details": _plain(check.get("details") or {}),
                }
            )
    return sorted(failures, key=lambda item: item["gate"])


def _artifact_summary(artifact: _LoadedArtifact) -> dict[str, Any]:
    payload = artifact.payload
    summary: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
        "source_path": artifact.source_path,
    }
    for key in (
        "artifact_type",
        "schema_version",
        "suite",
        "model_name",
        "model_id",
        "format",
        "passed",
        "max_delta",
        "leakage_disparity",
        "worst_group_leakage",
        "worst_group",
    ):
        if key in payload:
            summary[key] = _plain(payload[key])
    groups = payload.get("groups")
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray)):
        summary["group_count"] = len(groups)
    return summary


def _load_artifact(
    artifact_id: str,
    source: Mapping[str, Any] | GateReport | str | Path | None,
    *,
    required: bool = False,
) -> _LoadedArtifact | None:
    if source is None:
        if required:
            raise ModelCardBuilderError(f"{artifact_id} is required")
        return None

    source_path: str | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ModelCardBuilderError(f"{artifact_id} must be a JSON object")
        source_path = str(path)
        digest = manifest_hash(path)
    else:
        payload = _coerce_mapping(source, artifact_id)
        digest = stable_hash(_plain(payload))
    return _LoadedArtifact(
        artifact_id=artifact_id,
        payload=_plain_mapping(payload, artifact_id),
        source_path=source_path,
        sha256=digest,
    )


def _load_training_provenance_artifact(
    source: Mapping[str, Any] | str | Path | None,
) -> _LoadedArtifact | None:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source)
        payload = load_training_provenance(path)
        return _LoadedArtifact(
            artifact_id="training_provenance",
            payload=payload,
            source_path=str(path),
            sha256=manifest_hash(path),
        )
    return _load_artifact("training_provenance", source)


def _optional_artifacts(
    sources: Mapping[str, Mapping[str, Any] | str | Path | None],
) -> list[_LoadedArtifact]:
    artifacts: list[_LoadedArtifact] = []
    for artifact_id, source in sources.items():
        artifact = _load_artifact(artifact_id, source)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def _row_provenance(row: Mapping[str, Any]) -> Mapping[str, Any] | str | Path | None:
    value = row.get("training_provenance")
    if isinstance(value, Mapping):
        path = value.get("path")
        if isinstance(path, (str, Path)) and str(path):
            return path
        return value
    return None


def _gate_payload(gate_report: Mapping[str, Any] | GateReport) -> dict[str, Any]:
    if isinstance(gate_report, GateReport):
        return gate_report.to_dict()
    return GateReport.from_dict(_plain_mapping(gate_report, "gate_report")).to_dict()


def _validate_gate_manifest_identity(
    row: Mapping[str, Any],
    gate_payload: Mapping[str, Any],
) -> None:
    identity = {
        "repo_id": gate_payload.get("repo_id"),
        "family": gate_payload.get("family"),
        "tier": gate_payload.get("tier"),
        "param_count": gate_payload.get("param_count"),
        "format": gate_payload.get("format"),
    }
    mismatches = release_gates._manifest_row_mismatches(
        row,
        identity,
        source="gate_report",
    )
    if mismatches:
        raise ModelCardBuilderError(
            "signed gate report diverges from manifest row: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _gate_check_payload(
    gate_payload: Mapping[str, Any],
    gate_name: str,
) -> dict[str, Any]:
    for check in _gate_checks(gate_payload):
        if check.get("gate") == gate_name:
            return _plain_mapping(check, gate_name)
    return {}


def _gate_checks(gate_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = gate_payload.get("gate_results") or []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mismatch_summary(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    keys = sorted(set(actual) | set(expected))
    for key in keys:
        if actual.get(key) != expected.get(key):
            return f"{key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}"
    return "unknown mismatch"


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    payload = _coerce_mapping(value, name)
    return {str(key): _plain(payload[key]) for key in sorted(payload, key=str)}


def _coerce_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise ModelCardBuilderError(f"{name} must be a mapping")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sorted_mapping(value: Any) -> dict[str, Any]:
    return _plain(value) if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _stable_unique(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if value})


def _markdown_values(value: Any) -> str:
    values = _string_list(value)
    if not values:
        return "Not reported"
    return ", ".join(f"`{item}`" for item in values)


def _rng_seeds(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "Not reported"
    return ", ".join(f"`{key}`={value[key]}" for key in sorted(value, key=str))


def _artifact_hashes(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "Not reported"
    return ", ".join(f"`{key}`=`{value[key]}`" for key in sorted(value, key=str))


def _number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "Not reported"
    return str(value)


__all__ = [
    "MODEL_CARD_DATASHEET_SCHEMA_VERSION",
    "MODEL_DATASHEET_FILENAME",
    "ModelCardBuildResult",
    "ModelCardBuilderError",
    "build_model_card",
    "render_datasheet_markdown",
    "validate_model_card_consistency",
]
