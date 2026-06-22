"""Render OpenMed model cards from manifest rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    return "\n".join(lines) + "\n"


def write_model_card(path: str | Path, row: dict[str, Any]) -> Path:
    """Write a rendered model card to *path* and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(row), encoding="utf-8")
    return path


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
