"""Render HuggingFace model cards from a single ``models.jsonl`` manifest row.

This module is pure: it maps a manifest row (dict) to a Markdown string with no I/O
and no network access. Uploading the rendered card lives in
:mod:`openmed.core.hf_publish`. The manifest row is the single source of truth, so
cards are generated and never hand-edited.
"""

from __future__ import annotations

from typing import Any

# Extra front-matter tags keyed by manifest ``family``. Every card also gets the
# ``openmed`` / ``medical`` tags appended.
_FAMILY_TAGS: dict[str, list[str]] = {
    "NER": ["named-entity-recognition", "biomedical-nlp"],
}

# Section-title emoji keyed by ``family`` (purely cosmetic, matches OpenMed cards).
_FAMILY_EMOJI: dict[str, str] = {
    "NER": "🧬",
}
_DEFAULT_EMOJI = "🏥"

_GENERATED_BANNER = (
    "> ⚙️ This model card is generated automatically from the OpenMed model "
    "manifest. Do not edit by hand."
)


def render_model_card(row: dict[str, Any]) -> str:
    """Render a complete HF model card (front-matter + body) from a manifest row."""
    blocks = [
        _render_frontmatter(row),
        _render_header(row),
        _render_model_details(row),
        _render_labels(row),
        _render_quickstart(row),
        _render_benchmark(row),
        _render_reproducibility(row),
        _render_citation(row),
        _render_license(row),
    ]
    body = "\n\n".join(block for block in blocks if block)
    return body + "\n"


def _format_params(param_count: int | None) -> str | None:
    """Format a parameter count compactly: 135_000_000 -> '135M', 1_500_000_000 -> '1.5B'."""
    if param_count is None:
        return None
    if param_count >= 1_000_000_000:
        value, suffix = param_count / 1_000_000_000, "B"
    elif param_count >= 1_000_000:
        value, suffix = param_count / 1_000_000, "M"
    elif param_count >= 1_000:
        value, suffix = param_count / 1_000, "K"
    else:
        return str(param_count)
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _derive_tags(family: str) -> list[str]:
    tags = list(_FAMILY_TAGS.get(family, [])) + ["openmed", "medical"]
    return list(dict.fromkeys(tags))


def _render_frontmatter(row: dict[str, Any]) -> str:
    lines = ["---"]
    if row.get("license"):
        lines.append(f"license: {row['license']}")
    languages = row.get("languages") or []
    if languages:
        lines.append("language:")
        lines += [f"- {lang}" for lang in languages]
    if row.get("task"):
        lines.append(f"pipeline_tag: {row['task']}")
    tags = _derive_tags(row.get("family") or "")
    if tags:
        lines.append("tags:")
        lines += [f"- {tag}" for tag in tags]
    if row.get("base_model"):
        lines.append(f"base_model: {row['base_model']}")
    lines.append("library_name: transformers")
    lines.append("---")
    return "\n".join(lines)


def _render_header(row: dict[str, Any]) -> str:
    repo_id = row["repo_id"]
    name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
    emoji = _FAMILY_EMOJI.get(row.get("family") or "", _DEFAULT_EMOJI)
    url = f"https://huggingface.co/{repo_id}"
    family = row.get("family") or ""
    task = row.get("task") or ""

    lines = [
        f"# {emoji} [{name}]({url})",
        "",
        f"**OpenMed {family} model · {task}**",
        "",
    ]
    license_id = row.get("license")
    if license_id:
        lines.append(
            f"[![License](https://img.shields.io/badge/License-{license_id}-blue.svg)]({url})"
        )
    lines.append(
        "[![OpenMed](https://img.shields.io/badge/🏥-OpenMed-green)]"
        "(https://huggingface.co/OpenMed)"
    )
    arxiv = row.get("arxiv")
    if arxiv:
        lines.append(
            f"[![arXiv](https://img.shields.io/badge/arXiv-{arxiv}-b31b1b.svg)]"
            f"(https://arxiv.org/abs/{arxiv})"
        )
    lines.append("")
    lines.append(_GENERATED_BANNER)
    return "\n".join(lines)


def _render_model_details(row: dict[str, Any]) -> str:
    rows: list[tuple[str, str]] = []
    if row.get("tier"):
        rows.append(("Tier", row["tier"]))
    params = _format_params(row.get("param_count"))
    if params:
        rows.append(("Parameters", params))
    if row.get("architecture"):
        rows.append(("Architecture", row["architecture"]))
    if row.get("base_model"):
        rows.append(("Base model", row["base_model"]))
    formats = row.get("formats") or []
    if formats:
        rows.append(("Formats", ", ".join(f"`{fmt}`" for fmt in formats)))
    languages = row.get("languages") or []
    if languages:
        rows.append(("Languages", ", ".join(languages)))
    if row.get("license"):
        rows.append(("License", row["license"]))
    if row.get("released"):
        rows.append(("Released", row["released"]))

    lines = ["## 📋 Model Details", "", "| Field | Value |", "|-------|-------|"]
    lines += [f"| {field} | {value} |" for field, value in rows]
    return "\n".join(lines)


def _render_labels(row: dict[str, Any]) -> str:
    labels = row.get("canonical_labels") or []
    if not labels:
        return ""
    lines = ["## 🏷️ Entity Labels", ""]
    lines += [f"- `{label}`" for label in labels]
    return "\n".join(lines)


def _render_quickstart(row: dict[str, Any]) -> str:
    repo_id = row["repo_id"]
    task = row.get("task") or ""
    if task == "token-classification":
        snippet = (
            "from transformers import pipeline\n\n"
            "ner = pipeline(\n"
            '    "token-classification",\n'
            f'    model="{repo_id}",\n'
            '    aggregation_strategy="simple",\n'
            ")\n\n"
            'entities = ner("The liver showed signs of fatty infiltration.")\n'
            "print(entities)"
        )
    else:
        snippet = (
            "from transformers import pipeline\n\n"
            f'pipe = pipeline("{task}", model="{repo_id}")'
        )
    return "## 🚀 Quick Start\n\n```python\n" + snippet + "\n```"


def _render_benchmark(row: dict[str, Any]) -> str:
    benchmark = row.get("benchmark") or {}
    dataset = benchmark.get("dataset")
    micro_f1 = benchmark.get("micro_f1")
    recall = benchmark.get("recall")
    cell_dataset = dataset if dataset else "—"
    cell_f1 = f"{micro_f1:.4f}" if micro_f1 is not None else "—"
    cell_recall = f"{recall:.4f}" if recall is not None else "—"
    return (
        "## 📊 Benchmark\n\n"
        "| Dataset | Micro-F1 | Recall |\n"
        "|---------|----------|--------|\n"
        f"| {cell_dataset} | {cell_f1} | {cell_recall} |"
    )


def _render_reproducibility(row: dict[str, Any]) -> str:
    return (
        "## 🔁 Reproducibility\n\n"
        f"- **Manifest hash**: `{row['reproducibility_hash']}`"
    )


def _render_citation(row: dict[str, Any]) -> str:
    arxiv = row.get("arxiv")
    if not arxiv:
        return ""
    # The bibtex entry is the OpenMed NER paper. Every manifest row that carries an
    # ``arxiv`` currently references this single paper; if a future family ships its
    # own arXiv ref, key the citation off ``family`` rather than emitting this one.
    return (
        "## 📜 Citation\n\n"
        "```bibtex\n"
        "@misc{panahi2025openmedner,\n"
        "      title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art "
        "Transformers for Biomedical NER Across 12 Public Datasets},\n"
        "      author={Maziyar Panahi},\n"
        "      year={2025},\n"
        f"      eprint={{{arxiv}}},\n"
        "      archivePrefix={arXiv},\n"
        "      primaryClass={cs.CL},\n"
        f"      url={{https://arxiv.org/abs/{arxiv}}},\n"
        "}\n"
        "```"
    )


def _render_license(row: dict[str, Any]) -> str:
    license_id = row.get("license") or "unspecified"
    return f"## 📄 License\n\nLicensed under the **{license_id}** license."
