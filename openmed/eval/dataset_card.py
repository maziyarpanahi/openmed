"""Dataset cards for registered evaluation suites."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.audit import manifest_hash, stable_hash
from openmed.eval.datasets.dua_stubs import DUA_GATED_CORPORA
from openmed.eval.datasets.licenses import DatasetLicense, license_for
from openmed.eval.golden import GOLDEN_CATEGORIES, load_golden_fixtures
from openmed.eval.suites import (
    DRUGPROT,
    GOLDEN,
    MASAKHANER,
    NAAMAPADAM,
    SHIELD,
    suite_metadata,
)

DATASET_CARD_SUITES: tuple[str, ...] = (
    GOLDEN,
    SHIELD,
    DRUGPROT,
    MASAKHANER,
    NAAMAPADAM,
)
_EXTERNAL_SUITES = {SHIELD, DRUGPROT, MASAKHANER}
MODEL_CARD_SCHEMA_VERSION = "openmed.eval.model_card.v1"
PROVENANCE_MANIFEST_SCHEMA_VERSION = "openmed.eval.model_card.provenance.v1"
DATA_PROVENANCE_ASSERTION = "synthetic_or_permissive_only"
MODEL_CARD_ARTIFACT_ROLES: tuple[str, ...] = (
    "gate_report",
    "fairness_report",
    "calibration_report",
    "coverage_report",
    "transfer_matrix",
)
_DUA_SOURCE_IDS = frozenset(
    {
        *DUA_GATED_CORPORA,
        "i2b2_eval_only",
        "n2c2_eval_only",
        "mimic_iii",
        "mimic-iii",
        "mimic_iv",
        "mimic-iv",
        "local-dua-required",
    }
)
_DUA_SOURCE_RE = re.compile(
    "|".join(
        rf"(?<![a-z0-9]){re.escape(source_id)}(?![a-z0-9])"
        for source_id in sorted(_DUA_SOURCE_IDS, key=len, reverse=True)
    )
)
_DUA_SCAN_IGNORED_KEYS = {
    "artifact_hash",
    "hash_algorithm",
    "path",
    "sha256",
    "source_path",
}


class ModelCardProvenanceError(ValueError):
    """Raised when eval model-card provenance cannot be trusted."""


@dataclass(frozen=True)
class DatasetCard:
    """Provenance, license, and fixture summary for one eval dataset."""

    dataset: str
    source_url: str
    license_id: str
    redistribution: str
    record_count: int
    labels: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = field(default_factory=tuple)
    splits: tuple[str, ...] = field(default_factory=tuple)
    provenance: str = ""
    source_hash: str = ""
    content_hash: str = ""
    notes: str = ""
    schema_version: str = "openmed.eval.dataset_card.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _stable_unique(self.labels))
        object.__setattr__(self, "languages", _stable_unique(self.languages))
        object.__setattr__(self, "splits", _stable_unique(self.splits))
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready representation with no row text."""

        return {
            "dataset": self.dataset,
            "content_hash": self.content_hash,
            "labels": list(self.labels),
            "languages": list(self.languages),
            "license_id": self.license_id,
            "notes": self.notes,
            "provenance": self.provenance,
            "record_count": self.record_count,
            "redistribution": self.redistribution,
            "schema_version": self.schema_version,
            "source_url": self.source_url,
            "source_hash": self.source_hash,
            "splits": list(self.splits),
        }

    def to_json(self) -> str:
        """Return byte-stable JSON for identical card contents."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render a deterministic Markdown dataset card."""

        rows = [
            ("Dataset", self.dataset),
            ("Source URL", self.source_url),
            ("License", self.license_id),
            ("Redistribution", self.redistribution),
            ("Record count", str(self.record_count)),
            ("Labels", _markdown_list(self.labels)),
            ("Languages", _markdown_list(self.languages)),
            ("Splits", _markdown_list(self.splits)),
            ("Provenance", self.provenance),
            ("Source hash", self.source_hash),
            ("Content hash", self.content_hash),
            ("Notes", self.notes),
        ]
        lines = [
            f"# Dataset Card: {self.dataset}",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
        lines.extend(f"| {field} | {_markdown_cell(value)} |" for field, value in rows)
        return "\n".join(lines) + "\n"


def build_dataset_card(suite: str, **loader_kwargs: Any) -> DatasetCard:
    """Build a dataset card for a concrete registered eval suite.

    External corpora are summarized from metadata by default so this function
    does not download rows. Pass suite loader keyword arguments, such as a
    DrugProt ``path`` or SHIELD ``rows_loader``, to include fixture counts from
    an explicitly supplied source.
    """

    if suite == GOLDEN:
        return _golden_card(**loader_kwargs)
    if suite == SHIELD:
        return _shield_card(**loader_kwargs)
    if suite == DRUGPROT:
        return _drugprot_card(**loader_kwargs)
    if suite == MASAKHANER:
        return _masakhaner_card(**loader_kwargs)
    if suite == NAAMAPADAM:
        return _naamapadam_card(**loader_kwargs)
    allowed = ", ".join(DATASET_CARD_SUITES)
    raise ValueError(
        f"unknown dataset card suite {suite!r}; expected one of: {allowed}"
    )


def build_all_dataset_cards(
    suites: Sequence[str] | None = None,
    *,
    suite_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[DatasetCard, ...]:
    """Build dataset cards for all concrete registered eval suites."""

    selected = DATASET_CARD_SUITES if suites is None else tuple(suites)
    kwargs_by_suite = suite_kwargs or {}
    return tuple(
        build_dataset_card(suite, **dict(kwargs_by_suite.get(suite, {})))
        for suite in selected
    )


def render_dataset_card_markdown(card: DatasetCard) -> str:
    """Render a dataset card as deterministic Markdown."""

    return card.to_markdown()


@dataclass(frozen=True)
class ModelCardArtifact:
    """One eval artifact cited by a generated eval model card."""

    artifact_id: str
    role: str
    payload: Mapping[str, Any]
    source_path: str | None = None
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        artifact_id = _normalise_artifact_id(self.artifact_id)
        role = _normalise_artifact_id(self.role)
        if not artifact_id:
            raise ValueError("artifact_id is required")
        if not role:
            raise ValueError("artifact role is required")
        if not isinstance(self.payload, Mapping):
            raise TypeError("model card artifact payload must be a mapping")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "payload", _plain(self.payload))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))

    @classmethod
    def from_path(
        cls,
        artifact_id: str,
        role: str,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "ModelCardArtifact":
        """Load a JSON object artifact from *path*."""

        artifact_path = Path(path)
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"model card artifact must be a JSON object: {path}")
        return cls(
            artifact_id=artifact_id,
            role=role,
            payload=payload,
            source_path=str(artifact_path),
            expected_sha256=expected_sha256,
        )

    @classmethod
    def from_payload(
        cls,
        artifact_id: str,
        role: str,
        payload: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
    ) -> "ModelCardArtifact":
        """Wrap an in-memory JSON-compatible artifact payload."""

        return cls(
            artifact_id=artifact_id,
            role=role,
            payload=payload,
            expected_sha256=expected_sha256,
        )


@dataclass(frozen=True)
class ModelCardClaim:
    """A quantitative card claim tied to one hashed source artifact."""

    claim_id: str
    label: str
    value: int | float
    artifact_id: str
    source_field: str
    artifact_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("model card quantitative claim value must be numeric")
        object.__setattr__(
            self, "artifact_id", _normalise_artifact_id(self.artifact_id)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic datasheet representation."""

        return {
            "artifact_hash": self.artifact_hash,
            "artifact_id": self.artifact_id,
            "claim_id": self.claim_id,
            "label": self.label,
            "source_field": self.source_field,
            "value": self.value,
        }


@dataclass(frozen=True)
class EvalModelCard:
    """Model card and JSON datasheet rendered from hashed eval artifacts."""

    model_name: str
    summary: Mapping[str, Any]
    quantitative_claims: tuple[ModelCardClaim, ...]
    provenance_manifest: Mapping[str, Any]
    data_provenance: Mapping[str, Any]
    licensing: Mapping[str, Any]
    schema_version: str = MODEL_CARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _plain(self.summary))
        object.__setattr__(
            self,
            "quantitative_claims",
            tuple(
                sorted(
                    self.quantitative_claims,
                    key=lambda claim: (claim.artifact_id, claim.claim_id),
                )
            ),
        )
        object.__setattr__(
            self, "provenance_manifest", _plain(self.provenance_manifest)
        )
        object.__setattr__(self, "data_provenance", _plain(self.data_provenance))
        object.__setattr__(self, "licensing", _plain(self.licensing))
        validate_eval_model_card_claims(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable, byte-stable datasheet payload."""

        return {
            "data_provenance": _plain(self.data_provenance),
            "licensing": _plain(self.licensing),
            "model_name": self.model_name,
            "provenance_manifest": _plain(self.provenance_manifest),
            "quantitative_claims": [
                claim.to_dict() for claim in self.quantitative_claims
            ],
            "schema_version": self.schema_version,
            "summary": _plain(self.summary),
        }

    def to_json(self) -> str:
        """Render the datasheet as deterministic JSON."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown eval model card."""

        return _render_eval_model_card_markdown(self)

    def write_json(self, path: str | Path) -> Path:
        """Write the JSON datasheet to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
        return output_path

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown model card to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def build_eval_model_card(
    artifacts: Sequence[ModelCardArtifact | Mapping[str, Any]],
    *,
    model_name: str | None = None,
    data_sources: Sequence[Mapping[str, Any] | str] = (),
    license_id: str = "Apache-2.0",
    verify: bool = False,
    provenance_manifest: Mapping[str, Any] | None = None,
) -> EvalModelCard:
    """Build an eval model card and datasheet from source artifacts.

    Args:
        artifacts: Eval artifacts with roles such as ``gate_report``,
            ``fairness_report``, ``calibration_report``, ``coverage_report``,
            or ``transfer_matrix``.
        model_name: Optional model name override. If omitted, the generator
            uses the gate report ``repo_id`` or ``model_name`` when present.
        data_sources: Synthetic/permissive source assertions to embed in the
            datasheet. DUA-gated source identifiers are rejected.
        license_id: License assertion for the emitted card and datasheet.
        verify: When true, recompute artifact hashes against
            ``provenance_manifest`` and refuse to render on mismatch.
        provenance_manifest: Previously emitted provenance manifest to verify.
    """

    resolved = _resolve_model_card_artifacts(artifacts)
    if verify:
        verify_eval_model_card_artifacts(
            [item.artifact for item in resolved],
            provenance_manifest=provenance_manifest,
        )

    _assert_no_dua_source_ids(
        {"artifacts": [item.artifact.payload for item in resolved]},
    )
    data_provenance = _build_data_provenance(data_sources)
    manifest = _build_model_card_provenance_manifest(resolved)
    summary = _build_model_card_summary(resolved, model_name=model_name)
    claims = _build_model_card_claims(resolved)
    licensing = {
        "license_id": str(license_id),
        "assertion": DATA_PROVENANCE_ASSERTION,
        "permissive_or_synthetic_only": True,
    }
    card = EvalModelCard(
        model_name=str(summary.get("model_name") or model_name or "unknown-model"),
        summary=summary,
        quantitative_claims=claims,
        provenance_manifest=manifest,
        data_provenance=data_provenance,
        licensing=licensing,
    )
    _assert_no_dua_source_ids(card.to_dict())
    return card


def render_eval_model_card_markdown(card: EvalModelCard) -> str:
    """Render a generated eval model card as deterministic Markdown."""

    return card.to_markdown()


def render_eval_model_datasheet_json(card: EvalModelCard) -> str:
    """Render a generated eval model-card datasheet as deterministic JSON."""

    return card.to_json()


def write_eval_model_card(
    card: EvalModelCard,
    path: str | Path,
) -> Path:
    """Write a generated Markdown eval model card."""

    return card.write_markdown(path)


def write_eval_model_datasheet(
    card: EvalModelCard,
    path: str | Path,
) -> Path:
    """Write a generated JSON eval model-card datasheet."""

    return card.write_json(path)


def validate_eval_model_card_claims(card: EvalModelCard) -> None:
    """Ensure every quantitative claim maps to a manifest artifact hash."""

    artifacts = card.provenance_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ModelCardProvenanceError("provenance manifest requires artifacts")

    for claim in card.quantitative_claims:
        entry = artifacts.get(claim.artifact_id)
        if not isinstance(entry, Mapping):
            raise ModelCardProvenanceError(
                f"claim {claim.claim_id} references missing artifact "
                f"{claim.artifact_id}"
            )
        if entry.get("sha256") != claim.artifact_hash:
            raise ModelCardProvenanceError(
                f"claim {claim.claim_id} hash does not match artifact "
                f"{claim.artifact_id}"
            )


def verify_eval_model_card_artifacts(
    artifacts: Sequence[ModelCardArtifact | Mapping[str, Any]],
    *,
    provenance_manifest: Mapping[str, Any] | None,
) -> None:
    """Recompute artifact hashes and fail on any provenance mismatch."""

    if provenance_manifest is None:
        raise ModelCardProvenanceError("verification requires a provenance_manifest")
    expected = _manifest_hashes(provenance_manifest)
    for item in _resolve_model_card_artifacts(artifacts):
        artifact_id = item.artifact.artifact_id
        expected_hash = expected.get(artifact_id)
        if expected_hash is None:
            raise ModelCardProvenanceError(
                f"artifact {artifact_id} is missing from provenance manifest"
            )
        if expected_hash != item.sha256:
            raise ModelCardProvenanceError(
                f"artifact {artifact_id} hash mismatch: expected "
                f"{expected_hash}, got {item.sha256}"
            )


@dataclass(frozen=True)
class _ResolvedModelCardArtifact:
    artifact: ModelCardArtifact
    sha256: str


def _resolve_model_card_artifacts(
    artifacts: Sequence[ModelCardArtifact | Mapping[str, Any]],
) -> tuple[_ResolvedModelCardArtifact, ...]:
    resolved: list[_ResolvedModelCardArtifact] = []
    seen: set[str] = set()
    for value in artifacts:
        artifact = _coerce_model_card_artifact(value)
        if artifact.artifact_id in seen:
            raise ValueError(f"duplicate model card artifact: {artifact.artifact_id}")
        seen.add(artifact.artifact_id)
        digest = _hash_model_card_artifact(artifact)
        if artifact.expected_sha256 and artifact.expected_sha256 != digest:
            raise ModelCardProvenanceError(
                f"artifact {artifact.artifact_id} hash mismatch: expected "
                f"{artifact.expected_sha256}, got {digest}"
            )
        resolved.append(_ResolvedModelCardArtifact(artifact=artifact, sha256=digest))
    return tuple(sorted(resolved, key=lambda item: item.artifact.artifact_id))


def _coerce_model_card_artifact(
    value: ModelCardArtifact | Mapping[str, Any],
) -> ModelCardArtifact:
    if isinstance(value, ModelCardArtifact):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("model card artifacts must be ModelCardArtifact or mapping")

    artifact_id = value.get("artifact_id") or value.get("id") or value.get("name")
    role = value.get("role") or value.get("artifact_role") or artifact_id
    path = value.get("path") or value.get("source_path") or value.get("file")
    expected_sha256 = value.get("expected_sha256") or value.get("sha256")
    payload = value.get("payload")

    if path is not None and payload is None:
        return ModelCardArtifact.from_path(
            str(artifact_id or Path(str(path)).stem),
            str(role or artifact_id or Path(str(path)).stem),
            path,
            expected_sha256=(
                str(expected_sha256) if expected_sha256 is not None else None
            ),
        )
    if artifact_id is None:
        raise ValueError("model card artifact mapping requires artifact_id")
    if payload is None:
        payload = {
            key: item
            for key, item in value.items()
            if key
            not in {
                "artifact_id",
                "artifact_role",
                "expected_sha256",
                "file",
                "id",
                "name",
                "path",
                "payload",
                "role",
                "sha256",
                "source_path",
            }
        }
    if not isinstance(payload, Mapping):
        raise TypeError("model card artifact payload must be a mapping")
    return ModelCardArtifact(
        artifact_id=str(artifact_id),
        role=str(role or artifact_id),
        payload=payload,
        source_path=str(path) if path is not None else None,
        expected_sha256=str(expected_sha256) if expected_sha256 is not None else None,
    )


def _hash_model_card_artifact(artifact: ModelCardArtifact) -> str:
    if artifact.source_path is not None:
        path = Path(artifact.source_path)
        if path.is_file():
            return manifest_hash(path)
    return stable_hash(_plain(artifact.payload))


def _build_model_card_provenance_manifest(
    artifacts: Sequence[_ResolvedModelCardArtifact],
) -> dict[str, Any]:
    entries = {
        item.artifact.artifact_id: {
            "artifact_id": item.artifact.artifact_id,
            "hash_algorithm": "sha256",
            "role": item.artifact.role,
            "sha256": item.sha256,
            "source_path": item.artifact.source_path,
        }
        for item in artifacts
    }
    return {
        "artifact_order": sorted(entries),
        "artifacts": entries,
        "schema_version": PROVENANCE_MANIFEST_SCHEMA_VERSION,
    }


def _build_model_card_summary(
    artifacts: Sequence[_ResolvedModelCardArtifact],
    *,
    model_name: str | None,
) -> dict[str, Any]:
    by_role = _artifacts_by_role(artifacts)
    gate = _first_artifact(by_role, "gate_report")
    fairness = _first_artifact(by_role, "fairness_report")
    calibration = _first_artifact(by_role, "calibration_report")
    coverage = _first_artifact(by_role, "coverage_report")
    transfer = _first_artifact(by_role, "transfer_matrix")

    gate_payload = gate.artifact.payload if gate is not None else {}
    resolved_model = (
        model_name
        or _string_value(gate_payload.get("repo_id"))
        or _string_value(gate_payload.get("model_name"))
        or _string_value(gate_payload.get("model_id"))
        or "unknown-model"
    )
    summary: dict[str, Any] = {
        "artifact_ids": [item.artifact.artifact_id for item in artifacts],
        "model_name": resolved_model,
    }

    if gate is not None:
        verdict = (
            _string_value(gate_payload.get("decision"))
            or _string_value(gate_payload.get("verdict"))
            or _string_value(gate_payload.get("gate_verdict"))
        )
        summary["gate"] = {
            "policy": _string_value(gate_payload.get("policy")),
            "source_artifact": gate.artifact.artifact_id,
            "threshold_profile": _string_value(gate_payload.get("threshold_profile")),
            "verdict": verdict,
        }
        languages = sorted(_extract_per_language_leakage(gate_payload))
        if languages:
            summary["per_language_leakage"] = {
                "languages": languages,
                "source_artifact": gate.artifact.artifact_id,
            }

    if fairness is not None:
        fairness_payload = fairness.artifact.payload
        summary["fairness"] = {
            "source_artifact": fairness.artifact.artifact_id,
            "worst_group": _string_value(fairness_payload.get("worst_group")),
        }

    if calibration is not None:
        summary["calibration"] = {
            "labels": _calibration_labels(calibration.artifact.payload),
            "languages": _calibration_languages(calibration.artifact.payload),
            "source_artifact": calibration.artifact.artifact_id,
        }

    if coverage is not None:
        coverage_payload = coverage.artifact.payload
        summary["coverage"] = {
            "covered_categories": _nested_string_list(
                coverage_payload,
                "categories",
                "covered",
            ),
            "covered_labels": _nested_string_list(
                coverage_payload, "labels", "covered"
            ),
            "covered_languages": _nested_string_list(
                coverage_payload,
                "languages",
                "covered",
            ),
            "source_artifact": coverage.artifact.artifact_id,
        }

    if transfer is not None:
        summary["transfer_matrix"] = {
            "source_artifact": transfer.artifact.artifact_id,
            "source_languages": _transfer_source_languages(transfer.artifact.payload),
            "target_languages": _transfer_target_languages(transfer.artifact.payload),
        }

    return summary


def _build_model_card_claims(
    artifacts: Sequence[_ResolvedModelCardArtifact],
) -> tuple[ModelCardClaim, ...]:
    claims: list[ModelCardClaim] = []
    for item in artifacts:
        role = item.artifact.role
        if role == "gate_report":
            claims.extend(_gate_report_claims(item))
        elif role == "fairness_report":
            claims.extend(_fairness_report_claims(item))
        elif role == "calibration_report":
            claims.extend(_calibration_report_claims(item))
        elif role == "coverage_report":
            claims.extend(_coverage_report_claims(item))
        elif role == "transfer_matrix":
            claims.extend(_transfer_matrix_claims(item))
        else:
            claims.extend(
                _numeric_claims(
                    item,
                    item.artifact.payload,
                    prefix=role,
                    label_prefix=role.replace("_", " ").title(),
                )
            )
    return tuple(claims)


def _gate_report_claims(
    item: _ResolvedModelCardArtifact,
) -> list[ModelCardClaim]:
    payload = item.artifact.payload
    claims: list[ModelCardClaim] = []
    fields = {
        "critical_leakage_count": "Critical leakage count",
        "param_count": "Parameter count",
        "p50_ms": "Latency p50 ms",
        "p95_ms": "Latency p95 ms",
        "quant_recall_delta": "Quantized recall delta",
        "ram_mb": "Peak RAM MB",
        "residual_leakage_rate": "Residual leakage rate",
        "target_leakage_rate": "Target leakage rate",
    }
    for source_field, label in fields.items():
        _append_claim(
            claims,
            item,
            claim_id=f"gate.{source_field}",
            label=label,
            source_field=source_field,
            value=payload.get(source_field),
        )

    for label_name, value in _mapping(payload.get("per_label_recall")).items():
        _append_claim(
            claims,
            item,
            claim_id=f"gate.per_label_recall.{label_name}",
            label=f"Recall for {label_name}",
            source_field=f"per_label_recall.{label_name}",
            value=value,
        )
    for label_name, value in _mapping(payload.get("per_label_precision")).items():
        _append_claim(
            claims,
            item,
            claim_id=f"gate.per_label_precision.{label_name}",
            label=f"Precision for {label_name}",
            source_field=f"per_label_precision.{label_name}",
            value=value,
        )
    for language, value in _extract_per_language_leakage(payload).items():
        _append_claim(
            claims,
            item,
            claim_id=f"gate.per_language_leakage.{language}",
            label=f"Leakage rate for {language}",
            source_field=f"leakage.by_language.{language}",
            value=value,
        )
    return claims


def _fairness_report_claims(
    item: _ResolvedModelCardArtifact,
) -> list[ModelCardClaim]:
    payload = item.artifact.payload
    claims: list[ModelCardClaim] = []
    for source_field, label in (
        ("fixture_count", "Fairness fixture count"),
        ("leakage_disparity", "Fairness leakage disparity"),
        ("worst_group_leakage", "Worst group leakage"),
    ):
        _append_claim(
            claims,
            item,
            claim_id=f"fairness.{source_field}",
            label=label,
            source_field=source_field,
            value=payload.get(source_field),
        )

    for group, metrics in _mapping(payload.get("per_group")).items():
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in (
            "covered_chars",
            "leakage_rate",
            "leaked_chars",
            "recall",
            "span_count",
            "total_chars",
        ):
            _append_claim(
                claims,
                item,
                claim_id=f"fairness.per_group.{group}.{metric_name}",
                label=f"{metric_name.replace('_', ' ').title()} for {group}",
                source_field=f"per_group.{group}.{metric_name}",
                value=metrics.get(metric_name),
            )
    return claims


def _calibration_report_claims(
    item: _ResolvedModelCardArtifact,
) -> list[ModelCardClaim]:
    payload = item.artifact.payload
    claims: list[ModelCardClaim] = []
    groups = _sequence(payload.get("groups"))
    for source_field, label, value in (
        ("groups", "Calibration group count", len(groups)),
        (
            "labels",
            "Calibration covered label count",
            len(_calibration_labels(payload)),
        ),
        (
            "languages",
            "Calibration covered language count",
            len(_calibration_languages(payload)),
        ),
        ("min_recall", "Calibration minimum recall", payload.get("min_recall")),
        (
            "target_leakage",
            "Calibration target leakage",
            payload.get("target_leakage"),
        ),
    ):
        _append_claim(
            claims,
            item,
            claim_id=f"calibration.{source_field}",
            label=label,
            source_field=source_field,
            value=value,
        )

    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            continue
        group_id = _calibration_group_id(group, index)
        for metric_name in (
            "chosen_threshold",
            "negative_weight",
            "over_redaction",
            "positive_weight",
            "precision",
            "recall",
            "resulting_leakage",
            "target_leakage",
        ):
            _append_claim(
                claims,
                item,
                claim_id=f"calibration.groups.{group_id}.{metric_name}",
                label=f"{metric_name.replace('_', ' ').title()} for {group_id}",
                source_field=f"groups.{index}.{metric_name}",
                value=group.get(metric_name),
            )
    return claims


def _coverage_report_claims(
    item: _ResolvedModelCardArtifact,
) -> list[ModelCardClaim]:
    payload = item.artifact.payload
    claims: list[ModelCardClaim] = []
    fields = {
        "fixture_count": payload.get("fixture_count"),
        "labels.covered": len(_nested_string_list(payload, "labels", "covered")),
        "labels.missing": len(_nested_string_list(payload, "labels", "missing")),
        "languages.covered": len(_nested_string_list(payload, "languages", "covered")),
        "languages.missing": len(_nested_string_list(payload, "languages", "missing")),
        "categories.covered": len(
            _nested_string_list(payload, "categories", "covered")
        ),
        "categories.missing": len(
            _nested_string_list(payload, "categories", "missing")
        ),
    }
    for source_field, value in fields.items():
        _append_claim(
            claims,
            item,
            claim_id=f"coverage.{source_field}",
            label=f"Coverage {source_field.replace('.', ' ')}",
            source_field=source_field,
            value=value,
        )
    for category, count in _mapping(payload.get("category_counts")).items():
        _append_claim(
            claims,
            item,
            claim_id=f"coverage.category_counts.{category}",
            label=f"Fixture count for {category}",
            source_field=f"category_counts.{category}",
            value=count,
        )
    return claims


def _transfer_matrix_claims(
    item: _ResolvedModelCardArtifact,
) -> list[ModelCardClaim]:
    payload = item.artifact.payload
    matrix = (
        _mapping(payload.get("transfer_matrix"))
        or _mapping(payload.get("matrix"))
        or payload
    )
    return _numeric_claims(
        item,
        matrix,
        prefix="transfer",
        label_prefix="Transfer",
    )


def _numeric_claims(
    item: _ResolvedModelCardArtifact,
    value: Any,
    *,
    prefix: str,
    label_prefix: str,
) -> list[ModelCardClaim]:
    claims: list[ModelCardClaim] = []
    for path, numeric_value in _flatten_numeric(value):
        claim_path = ".".join((prefix, path)) if path else prefix
        _append_claim(
            claims,
            item,
            claim_id=claim_path,
            label=f"{label_prefix} {path.replace('.', ' ')}".strip(),
            source_field=path,
            value=numeric_value,
        )
    return claims


def _append_claim(
    claims: list[ModelCardClaim],
    item: _ResolvedModelCardArtifact,
    *,
    claim_id: str,
    label: str,
    source_field: str,
    value: Any,
) -> None:
    if not _is_number(value):
        return
    claims.append(
        ModelCardClaim(
            claim_id=claim_id,
            label=label,
            value=value,
            artifact_id=item.artifact.artifact_id,
            source_field=source_field,
            artifact_hash=item.sha256,
        )
    )


def _extract_per_language_leakage(payload: Mapping[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("per_language_leakage"),
        payload.get("leakage_by_language"),
        _nested(payload, "leakage", "by_language"),
        _nested(payload, "leakage", "rate_by_language"),
        _nested(payload, "metrics", "leakage", "by_language"),
        _nested(payload, "metrics", "leakage", "rate_by_language"),
        _nested(payload, "metrics", "leakage", "overall_by_language"),
    ]
    for candidate in candidates:
        parsed = _float_mapping(candidate)
        if parsed:
            return parsed

    leaked = _float_mapping(
        _nested(payload, "metrics", "leakage", "leaked_chars_by_language")
    )
    total = _float_mapping(
        _nested(payload, "metrics", "leakage", "total_chars_by_language")
    )
    if not leaked or not total:
        leaked = _float_mapping(_nested(payload, "leakage", "leaked_chars_by_language"))
        total = _float_mapping(_nested(payload, "leakage", "total_chars_by_language"))
    result: dict[str, float] = {}
    for language in sorted(set(leaked) | set(total)):
        denominator = total.get(language, 0.0)
        if denominator > 0.0:
            result[language] = leaked.get(language, 0.0) / denominator
    return result


def _build_data_provenance(
    data_sources: Sequence[Mapping[str, Any] | str],
) -> dict[str, Any]:
    sources = [_plain(source) for source in data_sources]
    _assert_no_dua_source_ids({"data_sources": sources})
    return {
        "assertion": DATA_PROVENANCE_ASSERTION,
        "permissive_or_synthetic_only": True,
        "restricted_source_ids": [],
        "source_count": len(sources),
        "sources": sources,
    }


def _render_eval_model_card_markdown(card: EvalModelCard) -> str:
    summary = card.summary
    lines = [
        f"# Eval Model Card: {card.model_name}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | `{_markdown_cell(card.model_name)}` |",
    ]

    gate = _mapping(summary.get("gate"))
    if gate:
        lines.append(
            "| Gate verdict | "
            f"`{_markdown_cell(_display(gate.get('verdict')))}` "
            f"from `{_markdown_cell(_display(gate.get('source_artifact')))}` |"
        )
        if gate.get("policy"):
            lines.append(f"| Policy | `{_markdown_cell(_display(gate['policy']))}` |")
        if gate.get("threshold_profile"):
            lines.append(
                "| Threshold profile | "
                f"`{_markdown_cell(_display(gate['threshold_profile']))}` |"
            )

    language_summary = _mapping(summary.get("per_language_leakage"))
    if language_summary:
        languages = ", ".join(
            f"`{_markdown_cell(language)}`"
            for language in _string_list(language_summary.get("languages"))
        )
        lines.append(
            "| Per-language leakage | "
            f"{languages or 'not declared'}; values are source-backed below |"
        )

    fairness = _mapping(summary.get("fairness"))
    if fairness:
        lines.append(
            "| Fairness | "
            f"worst group `{_markdown_cell(_display(fairness.get('worst_group')))}`; "
            "values are source-backed below |"
        )

    calibration = _mapping(summary.get("calibration"))
    if calibration:
        lines.append(
            "| Calibration coverage | "
            f"labels {_markdown_inline_list(_string_list(calibration.get('labels')))}; "
            f"languages {_markdown_inline_list(_string_list(calibration.get('languages')))} |"
        )

    coverage = _mapping(summary.get("coverage"))
    if coverage:
        lines.append(
            "| Fixture coverage | "
            f"labels {_markdown_inline_list(_string_list(coverage.get('covered_labels')))}; "
            f"languages {_markdown_inline_list(_string_list(coverage.get('covered_languages')))} |"
        )

    transfer = _mapping(summary.get("transfer_matrix"))
    if transfer:
        lines.append(
            "| Transfer matrix | "
            f"sources {_markdown_inline_list(_string_list(transfer.get('source_languages')))}; "
            f"targets {_markdown_inline_list(_string_list(transfer.get('target_languages')))} |"
        )

    lines.extend(
        [
            "",
            "## Quantitative Claims",
            "",
            "| Claim | Value | Source Artifact | Source Field | Source Hash |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for claim in card.quantitative_claims:
        lines.append(
            "| "
            f"{_markdown_cell(claim.label)} | "
            f"{_format_claim_value(claim.value)} | "
            f"`{claim.artifact_id}` | "
            f"`{_markdown_cell(claim.source_field)}` | "
            f"`{claim.artifact_hash}` |"
        )

    artifacts = _mapping(card.provenance_manifest.get("artifacts"))
    lines.extend(
        [
            "",
            "## Provenance Manifest",
            "",
            "| Artifact | Role | SHA-256 |",
            "| --- | --- | --- |",
        ]
    )
    for artifact_id in sorted(artifacts):
        entry = _mapping(artifacts[artifact_id])
        lines.append(
            "| "
            f"`{artifact_id}` | "
            f"`{_markdown_cell(_display(entry.get('role')))}` | "
            f"`{_markdown_cell(_display(entry.get('sha256')))}` |"
        )

    lines.extend(
        [
            "",
            "## Data And Licensing",
            "",
            "| Field | Assertion |",
            "| --- | --- |",
            (
                "| Data provenance | "
                f"`{_markdown_cell(_display(card.data_provenance.get('assertion')))}` |"
            ),
            (
                "| Permissive or synthetic only | "
                f"{_format_bool(card.data_provenance.get('permissive_or_synthetic_only'))} |"
            ),
            f"| License | `{_markdown_cell(_display(card.licensing.get('license_id')))}` |",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ModelCardProvenanceError("provenance manifest requires artifacts")
    expected: dict[str, str] = {}
    for artifact_id, entry in artifacts.items():
        if isinstance(entry, Mapping) and entry.get("sha256"):
            expected[_normalise_artifact_id(str(artifact_id))] = str(entry["sha256"])
    return expected


def _artifacts_by_role(
    artifacts: Sequence[_ResolvedModelCardArtifact],
) -> dict[str, list[_ResolvedModelCardArtifact]]:
    by_role: dict[str, list[_ResolvedModelCardArtifact]] = {}
    for artifact in artifacts:
        by_role.setdefault(artifact.artifact.role, []).append(artifact)
    return by_role


def _first_artifact(
    by_role: Mapping[str, Sequence[_ResolvedModelCardArtifact]],
    role: str,
) -> _ResolvedModelCardArtifact | None:
    values = by_role.get(role) or ()
    return values[0] if values else None


def _normalise_artifact_id(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").split())


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_string_list(value: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    return tuple(_string_list(_nested(value, *keys)))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return sorted({str(item) for item in value if str(item)})
    return []


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if _is_number(item):
            result[str(key)] = float(item)
    return result


def _flatten_numeric(value: Any, prefix: str = "") -> list[tuple[str, int | float]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, int | float]] = []
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric(value[key], child))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = []
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_flatten_numeric(item, child))
        return rows
    if _is_number(value):
        return [(prefix, value)]
    return []


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _calibration_labels(payload: Mapping[str, Any]) -> list[str]:
    labels = {
        str(group["label"])
        for group in _sequence(payload.get("groups"))
        if isinstance(group, Mapping) and group.get("label") is not None
    }
    return sorted(labels)


def _calibration_languages(payload: Mapping[str, Any]) -> list[str]:
    languages = {
        str(group["language"])
        for group in _sequence(payload.get("groups"))
        if isinstance(group, Mapping) and group.get("language") is not None
    }
    return sorted(languages)


def _calibration_group_id(group: Mapping[str, Any], index: int) -> str:
    parts = [
        _string_value(group.get("model_id")),
        _string_value(group.get("label")),
        _string_value(group.get("language")),
    ]
    value = ".".join(part for part in parts if part)
    return value or str(index)


def _transfer_source_languages(payload: Mapping[str, Any]) -> list[str]:
    explicit = _string_list(payload.get("source_languages"))
    if explicit:
        return explicit
    matrix = _mapping(payload.get("transfer_matrix")) or _mapping(payload.get("matrix"))
    return sorted(str(key) for key in matrix)


def _transfer_target_languages(payload: Mapping[str, Any]) -> list[str]:
    explicit = _string_list(payload.get("target_languages"))
    if explicit:
        return explicit
    matrix = _mapping(payload.get("transfer_matrix")) or _mapping(payload.get("matrix"))
    targets: set[str] = set()
    for row in matrix.values():
        if isinstance(row, Mapping):
            targets.update(str(key) for key in row)
    return sorted(targets)


def _format_claim_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _display(value: Any) -> str:
    if value is None or value == "":
        return "not declared"
    return str(value)


def _markdown_inline_list(values: Sequence[str]) -> str:
    if not values:
        return "`not declared`"
    return ", ".join(f"`{_markdown_cell(value)}`" for value in values)


def _assert_no_dua_source_ids(value: Any) -> None:
    matches = sorted(_find_dua_source_ids(value))
    if matches:
        joined = ", ".join(matches)
        raise ModelCardProvenanceError(
            f"DUA-gated source ids cannot be rendered in an eval model card: {joined}"
        )


def _find_dua_source_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _DUA_SCAN_IGNORED_KEYS:
                continue
            found.update(_find_dua_source_ids(str(key)))
            found.update(_find_dua_source_ids(item))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_find_dua_source_ids(item))
        return found
    if isinstance(value, str):
        lowered = value.lower()
        for match in _DUA_SOURCE_RE.finditer(lowered):
            found.add(match.group(0))
    return found


def _golden_card(**loader_kwargs: Any) -> DatasetCard:
    fixtures = load_golden_fixtures(**loader_kwargs)
    license_metadata = license_for(GOLDEN)
    labels = {span.label for fixture in fixtures for span in fixture.gold_spans}
    languages = {fixture.language for fixture in fixtures}
    return _card_from_license(
        license_metadata,
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=GOLDEN_CATEGORIES,
        provenance="committed synthetic golden fixtures",
    )


def _shield_card(**loader_kwargs: Any) -> DatasetCard:
    metadata = suite_metadata(SHIELD, use_sample=loader_kwargs.get("use_sample", True))
    fixtures = _load_external_fixtures(SHIELD, loader_kwargs)
    labels = _labels_from_mapping(metadata.get("label_mapping"))
    languages = _fixture_languages(fixtures, fallback=("en",))
    splits = _fixture_splits(fixtures, fallback=(str(metadata["split"]),))
    return _card_from_license(
        license_for(SHIELD),
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=splits,
        provenance=str(metadata["source_url"]),
        notes=str(metadata["annotation"]),
    )


def _drugprot_card(**loader_kwargs: Any) -> DatasetCard:
    metadata = suite_metadata(DRUGPROT, task=loader_kwargs.get("task", "ner"))
    fixtures = _load_external_fixtures(DRUGPROT, loader_kwargs)
    labels = _labels_from_mapping(metadata.get("entity_label_mapping"))
    languages = _fixture_languages(fixtures, fallback=("en",))
    splits = _fixture_splits(
        fixtures,
        fallback=(str(loader_kwargs.get("split", "training")),),
    )
    return _card_from_license(
        license_for(DRUGPROT),
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=splits,
        provenance=str(metadata["doi"]),
        notes=str(metadata["redistribution"]),
    )


def _masakhaner_card(**loader_kwargs: Any) -> DatasetCard:
    normalized_kwargs = dict(loader_kwargs)
    language = normalized_kwargs.pop("language", None)
    if language is not None and "languages" not in normalized_kwargs:
        normalized_kwargs["languages"] = (language,)
    languages = tuple(normalized_kwargs.get("languages") or ()) or None
    if languages:
        metadata = suite_metadata(
            MASAKHANER,
            languages=languages,
            split=normalized_kwargs.get("split", "test"),
            version=normalized_kwargs.get("version", "2.0"),
        )
    else:
        metadata = suite_metadata(
            MASAKHANER,
            split=normalized_kwargs.get("split", "test"),
            version=normalized_kwargs.get("version", "2.0"),
        )
    fixtures = _load_external_fixtures(MASAKHANER, normalized_kwargs)
    content_hashes = {
        str(fixture.language): str(fixture.metadata.get("content_hash"))
        for fixture in fixtures
        if fixture.metadata.get("content_hash")
    }
    content_hash = (
        next(iter(content_hashes.values()))
        if len(content_hashes) == 1
        else stable_hash({"content_hashes": content_hashes})
        if content_hashes
        else ""
    )
    fallback_languages = tuple(str(item) for item in metadata["languages"])
    license_metadata = license_for(MASAKHANER)
    if str(metadata["version"]) == "1.0":
        first_source = next(iter(metadata["sources"].values()))
        license_metadata = DatasetLicense(
            dataset=MASAKHANER,
            license_id=str(metadata["license_id"]),
            source_url=str(first_source["source_url"]),
            redistribution=str(metadata["redistribution"]),
            notes=(
                "MasakhaNER 1.0 uses its dataset-card license declaration; "
                "callers must verify and explicitly accept that card."
            ),
        )
    return _card_from_license(
        license_metadata,
        record_count=len(fixtures),
        labels=_labels_from_mapping(metadata.get("canonical_label_mapping")),
        languages=_fixture_languages(fixtures, fallback=fallback_languages),
        splits=_fixture_splits(
            fixtures,
            fallback=(str(metadata["split"]),),
        ),
        provenance="user-supplied local corpus or pre-populated cache",
        source_hash=content_hash,
        content_hash=content_hash,
        notes=" ".join(
            item
            for item in (
                str(metadata["organization_handling"]),
                str(metadata.get("license_notice") or ""),
            )
            if item
        ),
    )


def _naamapadam_card(**loader_kwargs: Any) -> DatasetCard:
    from openmed.eval.suites.naamapadam import load_naamapadam_fixtures

    fixtures = load_naamapadam_fixtures(**loader_kwargs)
    labels = {span.label for fixture in fixtures for span in fixture.gold_spans}
    languages = {fixture.language for fixture in fixtures}
    return _card_from_license(
        license_for(NAAMAPADAM),
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=("synthetic",),
        provenance="committed synthetic Naamapadam-style fixtures",
    )


def _card_from_license(
    license_metadata: DatasetLicense,
    *,
    record_count: int,
    labels: Iterable[str],
    languages: Iterable[str],
    splits: Iterable[str],
    provenance: str,
    source_hash: str = "",
    content_hash: str = "",
    notes: str = "",
) -> DatasetCard:
    return DatasetCard(
        dataset=license_metadata.dataset,
        source_url=license_metadata.source_url,
        license_id=license_metadata.license_id,
        redistribution=license_metadata.redistribution,
        record_count=record_count,
        labels=tuple(labels),
        languages=tuple(languages),
        splits=tuple(splits),
        provenance=provenance,
        source_hash=source_hash,
        content_hash=content_hash,
        notes=notes or license_metadata.notes,
    )


def _load_external_fixtures(suite: str, loader_kwargs: Mapping[str, Any]) -> list[Any]:
    if not loader_kwargs:
        return []
    if suite in _EXTERNAL_SUITES and not _has_explicit_source(loader_kwargs):
        return []

    from openmed.eval.suites import load_suite_fixtures

    return list(load_suite_fixtures(suite, **dict(loader_kwargs)))


def _has_explicit_source(loader_kwargs: Mapping[str, Any]) -> bool:
    explicit_keys = {"cache_dir", "path", "paths", "rows_loader", "downloader"}
    return any(loader_kwargs.get(key) is not None for key in explicit_keys)


def _labels_from_mapping(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return _stable_unique(str(label) for label in value.values() if str(label))


def _fixture_languages(
    fixtures: Sequence[Any],
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    languages = {
        str(getattr(fixture, "language", ""))
        for fixture in fixtures
        if str(getattr(fixture, "language", ""))
    }
    return _stable_unique(languages or fallback)


def _fixture_splits(
    fixtures: Sequence[Any],
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    splits = set()
    for fixture in fixtures:
        metadata = getattr(fixture, "metadata", None)
        if isinstance(metadata, Mapping) and metadata.get("split") is not None:
            splits.add(str(metadata["split"]))
    return _stable_unique(splits or fallback)


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(value) for value in values if value is not None and str(value)})
    )


def _markdown_list(values: Sequence[str]) -> str:
    if not values:
        return ""
    return ", ".join(f"`{value}`" for value in values)


def _markdown_cell(value: str) -> str:
    cleaned = value.replace("\n", " ").replace("|", "\\|").strip()
    return cleaned or "not declared"


__all__ = [
    "DATASET_CARD_SUITES",
    "DATA_PROVENANCE_ASSERTION",
    "DatasetCard",
    "EvalModelCard",
    "MODEL_CARD_ARTIFACT_ROLES",
    "MODEL_CARD_SCHEMA_VERSION",
    "ModelCardArtifact",
    "ModelCardClaim",
    "ModelCardProvenanceError",
    "PROVENANCE_MANIFEST_SCHEMA_VERSION",
    "build_all_dataset_cards",
    "build_dataset_card",
    "build_eval_model_card",
    "render_eval_model_card_markdown",
    "render_eval_model_datasheet_json",
    "render_dataset_card_markdown",
    "validate_eval_model_card_claims",
    "verify_eval_model_card_artifacts",
    "write_eval_model_card",
    "write_eval_model_datasheet",
]
