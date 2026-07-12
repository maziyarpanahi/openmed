"""Render OpenMed model cards from manifest rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

DEFAULT_ARXIV = "2508.01630"


def render_model_card(row: dict[str, Any]) -> str:
    """Return a README.md model card for one manifest row."""

    repo_id = _string(row.get("repo_id"), "OpenMed/model")
    title = repo_id.rsplit("/", 1)[-1]
    benchmark = row.get("benchmark") if isinstance(row.get("benchmark"), dict) else {}
    formats = _list(row.get("formats"))
    languages = _list(row.get("languages"))
    labels = _list(row.get("canonical_labels"))
    arxiv = _string(row.get("arxiv"), DEFAULT_ARXIV)
    license_name = _string(row.get("license"), "Not specified")
    task = _string(row.get("task"), "unknown")

    lines = [
        "---",
        f"license: {license_name}",
        f"pipeline_tag: {task}",
        "library_name: openmed",
        "tags:",
        "- openmed",
        "- medical-nlp",
        "---",
        "",
        f"# {title}",
        "",
        "This model card is rendered from the OpenMed model manifest. Update `models.jsonl` and rerun the publish step instead of editing this file directly.",
        "",
        "## Manifest Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Repository | `{repo_id}` |",
        f"| Family | {_string(row.get('family'), 'Not specified')} |",
        f"| Task | {task} |",
        f"| Languages | {_comma_or_unspecified(languages)} |",
        f"| Tier | {_string(row.get('tier'), 'Not specified')} |",
        f"| Parameters | {_format_param_count(row.get('param_count'))} |",
        f"| Architecture | {_string(row.get('architecture'), 'Not specified')} |",
        f"| Base model | {_string(row.get('base_model'), 'Not specified')} |",
        f"| Formats | {_comma_or_unspecified(formats)} |",
        f"| License | {license_name} |",
        f"| arXiv | {_arxiv_link(arxiv)} |",
        f"| Reproducibility hash | `{_string(row.get('reproducibility_hash'), 'Not specified')}` |",
        f"| Released | {_string(row.get('released'), 'Not specified')} |",
        "",
        "## Benchmark",
        "",
        "| Dataset | Micro F1 | Recall |",
        "|---|---:|---:|",
        f"| {_string(benchmark.get('dataset'), 'Not reported')} | {_metric(benchmark.get('micro_f1'))} | {_metric(benchmark.get('recall'))} |",
        "",
        "## Canonical Labels",
        "",
        _labels_block(labels),
    ]
    artifact_lines = _artifact_format_block(formats)
    if artifact_lines:
        lines.extend(["", "## Artifact Format", "", *artifact_lines])
    distillation_lines = _distillation_block(row)
    if distillation_lines:
        lines.extend(["", "## Distillation Evidence", "", *distillation_lines])
    training_provenance_lines = _training_provenance_block(row)
    if training_provenance_lines:
        lines.extend(["", "## Training Provenance", "", *training_provenance_lines])
    return "\n".join(lines) + "\n"


def write_model_card(path: str | Path, row: dict[str, Any]) -> Path:
    """Write a rendered model card to *path* and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(row), encoding="utf-8")
    return path


def append_model_card_sections(card: str, sections: Sequence[str]) -> str:
    """Append generated Markdown sections to a rendered model card."""

    rendered = card if card.endswith("\n") else f"{card}\n"
    for section in sections:
        section_text = str(section).strip()
        if not section_text:
            continue
        rendered = f"{rendered.rstrip()}\n\n{section_text}\n"
    return rendered


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _comma_or_unspecified(values: list[str]) -> str:
    return ", ".join(values) if values else "Not specified"


def _format_param_count(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "Not specified"
    if value >= 1_000_000_000:
        compact = f"{value / 1_000_000_000:g}B"
    elif value >= 1_000_000:
        compact = f"{value / 1_000_000:g}M"
    elif value >= 1_000:
        compact = f"{value / 1_000:g}K"
    else:
        compact = str(value)
    return f"{compact} ({value:,})"


def _metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "Not reported"


def _arxiv_link(value: str) -> str:
    if value == "Not specified":
        return value
    return f"[arXiv:{value}](https://arxiv.org/abs/{value})"


def _labels_block(labels: list[str]) -> str:
    if not labels:
        return "Not specified."
    return ", ".join(f"`{label}`" for label in labels)


def _artifact_format_block(formats: list[str]) -> list[str]:
    runtime_formats = []
    quantization_formats = []
    for value in formats:
        if _normalize_format(value) in {
            "onnx",
            "onnx-android",
            "ort-android",
            "transformersjs",
            "webgpu",
        }:
            runtime_formats.append(value)
        quantization = _quantization_label(value)
        if quantization is not None:
            quantization_formats.append(quantization)
    if not runtime_formats and not quantization_formats:
        return []

    return [
        "| Field | Value |",
        "|---|---|",
        f"| Runtime artifacts | {_comma_or_unspecified(runtime_formats)} |",
        f"| Quantization | {_comma_or_unspecified(quantization_formats)} |",
    ]


def _normalize_format(value: str) -> str:
    return value.lower().replace("_", "-")


def _quantization_label(value: str) -> str | None:
    normalized = _normalize_format(value)
    if normalized in {"int8", "8bit", "8-bit", "onnx-int8"}:
        return "int8"
    if normalized in {"int4", "4bit", "4-bit", "onnx-int4"}:
        return "int4"
    if normalized in {"awq", "openmed-awq"}:
        return "awq"
    if normalized in {"gptq", "openmed-gptq"}:
        return "gptq"
    return None


def _distillation_block(row: dict[str, Any]) -> list[str]:
    payload = _distillation_payload(row)
    if not payload:
        return []

    critical_drops = _list(payload.get("critical_label_drops"))
    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Teacher | `{_string(payload.get('teacher_id'), 'Not reported')}` |",
        f"| Student backbone | `{_string(payload.get('student_backbone'), 'Not reported')}` |",
        f"| Temperature | {_metric(payload.get('temperature'))} |",
        f"| Alpha | {_metric(payload.get('alpha'))} |",
        f"| Recall gate | {_gate_status(payload.get('recall_gate_passed'))} |",
        f"| Critical drops | {_critical_drops(critical_drops)} |",
    ]

    deltas = payload.get("per_label_recall_delta")
    if isinstance(deltas, list) and deltas:
        lines.extend(
            [
                "",
                "| Label | Teacher recall | Student recall | Delta | Critical drop |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item in deltas:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {_string(item.get('label'), 'UNKNOWN')} | "
                f"{_metric(item.get('teacher_recall'))} | "
                f"{_metric(item.get('student_recall'))} | "
                f"{_metric(item.get('delta'))} | "
                f"{_yes_no(item.get('critical_drop'))} |"
            )
    return lines


def _distillation_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("distillation")
    if isinstance(payload, dict):
        return payload
    evidence = row.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("distillation"), dict):
        return evidence["distillation"]
    return {}


def _training_provenance_block(row: dict[str, Any]) -> list[str]:
    payload = row.get("training_provenance")
    if not isinstance(payload, dict):
        return []

    seeds = payload.get("rng_seeds")
    seed_text = "Not reported"
    if isinstance(seeds, dict) and seeds:
        seed_text = ", ".join(
            f"`{str(key)}`={value}" for key, value in sorted(seeds.items())
        )

    lines = [
        "| Field | Value |",
        "|---|---|",
        f"| Provenance file | `{_string(payload.get('path'), 'training_provenance.json')}` |",
        f"| Base model revision | `{_string(payload.get('base_model_revision'), 'Not reported')}` |",
        f"| Git SHA | `{_string(payload.get('git_sha'), 'Not reported')}` |",
        f"| RNG seeds | {seed_text} |",
        f"| Data manifest hash | `{_string(payload.get('data_manifest_hash'), 'Not reported')}` |",
        f"| Recipe config hash | `{_string(payload.get('recipe_config_hash'), 'Not reported')}` |",
        f"| Environment lock digest | `{_string(payload.get('env_lock_digest'), 'Not reported')}` |",
        f"| Provenance reproducibility hash | `{_string(payload.get('reproducibility_hash'), 'Not reported')}` |",
    ]
    return lines


def _gate_status(value: Any) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "Not reported"


def _critical_drops(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "None"


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "Not reported"
