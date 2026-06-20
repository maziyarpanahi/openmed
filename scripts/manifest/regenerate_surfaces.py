"""Regenerate manifest-derived repository surfaces from ``models.jsonl``."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models.jsonl"
LABELS_PATH = ROOT / "openmed" / "core" / "labels.py"
MODEL_REGISTRY_PATH = ROOT / "openmed" / "core" / "model_registry.py"

LANGUAGE_ORDER = (
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "te",
    "pt",
    "ar",
    "ja",
    "tr",
)
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
    "hi": "Hindi",
    "te": "Telugu",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ja": "Japanese",
    "tr": "Turkish",
}
DEFAULT_PII_MODEL_IDS = {
    "en": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    "fr": "OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1",
    "de": "OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1",
    "it": "OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1",
    "es": "OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1",
    "nl": "OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1",
    "hi": "OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1",
    "te": "OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1",
    "pt": "OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1",
    "ar": "OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1",
    "ja": "OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1",
    "tr": "OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1",
}
TIER_ORDER = {"Tiny": 0, "Small": 1, "Base": 2, "Medium": 3, "Large": 4, "XLarge": 5}

CATALOG_TABLE_BEGIN = "<!-- BEGIN MANIFEST MODEL TABLE -->"
CATALOG_TABLE_END = "<!-- END MANIFEST MODEL TABLE -->"
BENCHMARK_TABLE_BEGIN = "<!-- BEGIN MANIFEST BENCHMARK TABLE -->"
BENCHMARK_TABLE_END = "<!-- END MANIFEST BENCHMARK TABLE -->"


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load the committed JSONL manifest without doing remote discovery."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _format_count(value: int) -> str:
    return f"{value:,}"


def _language_sort_key(language: str) -> tuple[int, str]:
    try:
        return (LANGUAGE_ORDER.index(language), language)
    except ValueError:
        return (len(LANGUAGE_ORDER), language)


def supported_pii_languages(rows: Iterable[dict[str, Any]]) -> list[str]:
    languages = {
        language
        for row in rows
        if row.get("family") == "PII"
        for language in row.get("languages", [])
    }
    unknown = sorted(language for language in languages if language not in LANGUAGE_NAMES)
    if unknown:
        raise ValueError(
            "PII manifest uses unsupported language code(s): " + ", ".join(unknown)
        )
    return sorted(languages, key=_language_sort_key)


def _is_pii_row(row: dict[str, Any]) -> bool:
    return row.get("family") == "PII" and isinstance(row.get("repo_id"), str)


def _is_model_file_variant(repo_id: str) -> bool:
    name = repo_id.rsplit("/", 1)[-1].lower()
    return name.endswith("-mlx") or name.endswith("-mlx-8bit")


def default_pii_model_id(rows: Sequence[dict[str, Any]], language: str) -> str:
    preferred = DEFAULT_PII_MODEL_IDS.get(language)
    manifest_ids = {row["repo_id"] for row in rows if isinstance(row.get("repo_id"), str)}
    if preferred in manifest_ids:
        return preferred

    candidates = [
        row
        for row in rows
        if _is_pii_row(row)
        and row.get("languages") == [language]
        and not _is_model_file_variant(row["repo_id"])
    ]
    if not candidates:
        raise ValueError(f"No default PII model candidate found for {language!r}")

    def score(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
        repo_id = row["repo_id"]
        expected_language = LANGUAGE_NAMES[language]
        if language == "en":
            language_score = 0
        else:
            language_score = 0 if f"-{expected_language}-" in repo_id else 1
        superclinical_score = 0 if "SuperClinical" in repo_id else 1
        tier_score = TIER_ORDER.get(str(row.get("tier")), 99)
        param_count = row.get("param_count")
        param_score = param_count if isinstance(param_count, int) else 10**12
        return (language_score, superclinical_score, tier_score, param_score, repo_id)

    return sorted(candidates, key=score)[0]["repo_id"]


def default_pii_models(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    return {
        language: default_pii_model_id(rows, language)
        for language in supported_pii_languages(rows)
    }


def _inline_string_set(values: Sequence[str], indent: str = "    ") -> list[str]:
    quoted = [f'"{value}"' for value in values]
    lines: list[str] = []
    for index in range(0, len(quoted), 6):
        lines.append(indent + ", ".join(quoted[index:index + 6]) + ",")
    return lines


def render_pii_constants(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    languages = supported_pii_languages(rows)
    defaults = default_pii_models(rows)

    supported = ["SUPPORTED_LANGUAGES: Set[str] = {"]
    supported.extend(_inline_string_set(languages))
    supported.append("}")

    names = ["LANGUAGE_NAMES: Dict[str, str] = {"]
    names.extend(f'    "{language}": "{LANGUAGE_NAMES[language]}",' for language in languages)
    names.append("}")

    prefixes = ["LANGUAGE_MODEL_PREFIX: Dict[str, str] = {"]
    for language in languages:
        prefix = "" if language == "en" else f"{LANGUAGE_NAMES[language]}-"
        prefixes.append(f'    "{language}": "{prefix}",')
    prefixes.append("}")

    models = ["DEFAULT_PII_MODELS: Dict[str, str] = {"]
    models.extend(f'    "{language}": "{defaults[language]}",' for language in languages)
    models.append("}")

    return {
        "SUPPORTED_LANGUAGES": "\n".join(supported),
        "LANGUAGE_NAMES": "\n".join(names),
        "LANGUAGE_MODEL_PREFIX": "\n".join(prefixes),
        "DEFAULT_PII_MODELS": "\n".join(models),
    }


def _assignment_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def replace_assignments(text: str, replacements: dict[str, str]) -> str:
    tree = ast.parse(text)
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        name = _assignment_name(node)
        if name in replacements and hasattr(node, "end_lineno"):
            ranges[name] = (node.lineno - 1, node.end_lineno or node.lineno)

    missing = sorted(set(replacements) - set(ranges))
    if missing:
        raise ValueError("Could not find assignment(s): " + ", ".join(missing))

    lines = text.splitlines()
    for name, (start, end) in sorted(ranges.items(), key=lambda item: item[1][0], reverse=True):
        lines[start:end] = replacements[name].splitlines()
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def regenerate_pii_i18n(text: str, rows: Sequence[dict[str, Any]]) -> str:
    return replace_assignments(text, render_pii_constants(rows))


def _must_replace(pattern: str, replacement: str, text: str, *, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=flags)
    if count == 0:
        raise ValueError(f"Pattern did not match: {pattern}")
    return updated


def regenerate_readme(text: str, rows: Sequence[dict[str, Any]]) -> str:
    model_count = len(rows)
    languages = supported_pii_languages(rows)
    language_count = len(languages)
    pii_count = sum(1 for row in rows if row.get("family") == "PII")
    model_count_text = _format_count(model_count)
    encoded_model_count = model_count_text.replace(",", "%2C")
    language_codes = ", ".join(f"`{language}`" for language in languages)

    text = _must_replace(
        r"Entity extraction, PII de-identification, and [0-9,]+\+? specialized medical models",
        f"Entity extraction, PII de-identification, and {model_count_text} specialized medical models",
        text,
    )
    text = _must_replace(
        r"(%F0%9F%A4%97%20Models-)[^-]+(-F5E27A)",
        rf"\g<1>{encoded_model_count}\2",
        text,
    )
    text = _must_replace(
        r"<b>[0-9,]+\+? models</b> &nbsp;·&nbsp; <b>[0-9,]+\+? languages</b> &nbsp;·&nbsp; <b>[0-9,]+\+? PII checkpoints</b>",
        (
            f"<b>{model_count_text} models</b> &nbsp;·&nbsp; "
            f"<b>{language_count} languages</b> &nbsp;·&nbsp; "
            f"<b>{_format_count(pii_count)} PII checkpoints</b>"
        ),
        text,
    )
    text = _must_replace(
        r"(\| Specialized medical models\s+\|\s+)[0-9,]+\+?(\s+\|\s+Limited\s+\|)",
        rf"\g<1>{model_count_text}\2",
        text,
    )
    text = _must_replace(
        r"(\| Languages\s+\|\s+)[0-9,]+\+?(\s+\|\s+Varies\s+\|)",
        rf"\g<1>{language_count}\2",
        text,
    )
    text = _must_replace(
        r"\*\*Specialized models\*\* — [0-9,]+\+? curated biomedical & clinical models",
        f"**Specialized models** — {model_count_text} curated biomedical & clinical models",
        text,
    )
    text = _must_replace(
        r"## Multilingual PII \([0-9,]+ languages\)",
        f"## Multilingual PII ({language_count} languages)",
        text,
    )
    text = _must_replace(
        r"Extraction and de-identification across .*? — \*\*[0-9,]+\+? PII checkpoints\*\* total\.",
        (
            f"Extraction and de-identification across {language_codes} — "
            f"**{_format_count(pii_count)} PII checkpoints** total."
        ),
        text,
    )
    return text


def _join_values(values: Iterable[Any]) -> str:
    cleaned = [str(value) for value in values if value not in (None, "")]
    return ", ".join(cleaned) if cleaned else "-"


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_catalog_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Model | Family | Task | Languages | Tier | Formats |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("family") or ""),
            str(item.get("task") or ""),
            str(item.get("repo_id") or ""),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['repo_id']}`",
                    str(row.get("family") or "-"),
                    str(row.get("task") or "-"),
                    _join_values(row.get("languages") or []),
                    str(row.get("tier") or "-"),
                    _join_values(row.get("formats") or []),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _has_benchmark(row: dict[str, Any]) -> bool:
    benchmark = row.get("benchmark") or {}
    return any(benchmark.get(key) is not None for key in ("dataset", "micro_f1", "recall"))


def render_benchmark_table(rows: Sequence[dict[str, Any]]) -> str:
    benchmark_rows = [row for row in rows if _has_benchmark(row)]
    lines = [
        "| Model | Family | Dataset | Micro F1 | Recall | Tier | Formats |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(
        benchmark_rows,
        key=lambda item: (
            str((item.get("benchmark") or {}).get("dataset") or ""),
            str(item.get("repo_id") or ""),
        ),
    ):
        benchmark = row.get("benchmark") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['repo_id']}`",
                    str(row.get("family") or "-"),
                    str(benchmark.get("dataset") or "-"),
                    _format_metric(benchmark.get("micro_f1")),
                    _format_metric(benchmark.get("recall")),
                    str(row.get("tier") or "-"),
                    _join_values(row.get("formats") or []),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _replace_marker_block(text: str, begin: str, end: str, body: str) -> str:
    pattern = re.compile(
        rf"{re.escape(begin)}\n.*?\n{re.escape(end)}",
        flags=re.DOTALL,
    )
    replacement = f"{begin}\n{body}\n{end}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"Expected exactly one generated block for {begin}")
    return updated


def regenerate_model_registry_doc(text: str, rows: Sequence[dict[str, Any]]) -> str:
    model_count = len(rows)
    language_count = len(supported_pii_languages(rows))
    family_counts = Counter(str(row.get("family") or "Unknown") for row in rows)
    summary = (
        f"The committed manifest currently contains {_format_count(model_count)} "
        f"models across {language_count} supported PII languages. Family counts: "
        + ", ".join(
            f"{family}={_format_count(count)}"
            for family, count in sorted(family_counts.items())
        )
        + "."
    )
    text = _must_replace(
        r"^The committed manifest currently contains .*$",
        summary,
        text,
        flags=re.MULTILINE,
    )
    text = _replace_marker_block(
        text,
        CATALOG_TABLE_BEGIN,
        CATALOG_TABLE_END,
        render_catalog_table(rows),
    )
    text = _replace_marker_block(
        text,
        BENCHMARK_TABLE_BEGIN,
        BENCHMARK_TABLE_END,
        render_benchmark_table(rows),
    )
    return text


def validate_canonical_labels(rows: Sequence[dict[str, Any]]) -> None:
    labels_module = _load_module("_openmed_manifest_labels", LABELS_PATH)
    canonical_labels = set(labels_module.CANONICAL_LABELS)
    unknown: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        repo_id = str(row.get("repo_id") or "<unknown>")
        for label in row.get("canonical_labels") or []:
            if label not in canonical_labels:
                unknown[label].append(repo_id)
    if unknown:
        details = "; ".join(
            f"{label}: {', '.join(repo_ids[:3])}"
            for label, repo_ids in sorted(unknown.items())
        )
        raise ValueError(f"Manifest canonical_labels outside CANONICAL_LABELS: {details}")


def validate_registry_derivation(rows: Sequence[dict[str, Any]]) -> None:
    registry_module = _load_module("_openmed_manifest_registry", MODEL_REGISTRY_PATH)
    registry = registry_module._build_registry(rows)
    manifest_ids = {
        row["repo_id"]
        for row in rows
        if isinstance(row.get("repo_id"), str) and row["repo_id"]
    }
    registry_ids = {info.model_id for info in registry.values()}
    missing = sorted(manifest_ids - registry_ids)
    if missing:
        raise ValueError(
            "Manifest rows missing from generated registry: " + ", ".join(missing[:20])
        )


def _write_or_report(path: Path, content: str, *, check: bool) -> bool:
    current = path.read_text(encoding="utf-8")
    if current == content:
        return False
    if not check:
        path.write_text(content, encoding="utf-8")
    return True


def regenerate_surfaces(root: Path = ROOT, *, check: bool = False) -> list[Path]:
    rows = load_manifest(root / "models.jsonl")
    validate_canonical_labels(rows)
    validate_registry_derivation(rows)

    updates = {
        root / "openmed" / "core" / "pii_i18n.py": regenerate_pii_i18n(
            (root / "openmed" / "core" / "pii_i18n.py").read_text(encoding="utf-8"),
            rows,
        ),
        root / "README.md": regenerate_readme(
            (root / "README.md").read_text(encoding="utf-8"),
            rows,
        ),
        root / "docs" / "model-registry.md": regenerate_model_registry_doc(
            (root / "docs" / "model-registry.md").read_text(encoding="utf-8"),
            rows,
        ),
    }

    changed: list[Path] = []
    for path, content in updates.items():
        if _write_or_report(path, content, check=check):
            changed.append(path)
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate manifest-derived surfaces from committed models.jsonl."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without writing files.",
    )
    args = parser.parse_args(argv)

    changed = regenerate_surfaces(check=args.check)
    if args.check and changed:
        for path in changed:
            print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
