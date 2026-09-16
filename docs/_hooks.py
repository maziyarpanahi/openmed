"""MkDocs hook helpers for documentation builds."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from mkdocs.plugins import event_priority

_BLOCK_MARKERS = ("```", "~~~", "!!!", "???", "|", "-", "*", "+", "1.")
_DESCRIPTION_LIMIT = 180
_DEFAULT_BUILD_ARTIFACTS: dict[str, bytes] = {}


def on_config(config: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Populate the current year placeholder in ``config``."""
    year = str(datetime.now(timezone.utc).year)
    copyright_text = config.get("copyright")
    if isinstance(copyright_text, str) and "{year}" in copyright_text:
        config["copyright"] = copyright_text.replace("{year}", year)
    return config


def on_page_markdown(markdown: str, page: Any, **_: Any) -> str:
    """Give every indexable page a useful, deterministic metadata summary."""
    heading_match = re.search(r"^#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    heading = _plain_text(heading_match.group(1)) if heading_match else ""
    title = str(page.meta.get("title") or heading or page.title or "OpenMed")
    page.meta.setdefault("title", title)

    if not page.meta.get("description"):
        summary = _first_prose_paragraph(markdown)
        description = (
            f"{title}. {summary}"
            if summary
            else (f"{title}. Local-first OpenMed healthcare NLP documentation.")
        )
        page.meta["description"] = _truncate_description(description)
    return markdown


@event_priority(-200)
def on_files(files: Any, config: Any, **_: Any) -> Any:
    """Keep the default-language LLM feeds out of localized sub-builds.

    ``mkdocs-static-i18n`` performs one internal MkDocs build per locale. The
    ``llmstxt`` plugin is not locale-aware, so on translated builds it looks
    for the default source URI of each translated page and emits strict-mode
    warnings. The root English build still generates the canonical feeds;
    localized builds skip their feed sections after i18n has selected files.
    """
    plugins = config.plugins
    i18n_plugin = plugins.get("i18n")
    llmstxt_plugin = plugins.get("llmstxt")
    if (
        i18n_plugin is not None
        and llmstxt_plugin is not None
        and not i18n_plugin.is_default_language_build
    ):
        llmstxt_plugin._sections = {}
        llmstxt_plugin._file_uris = set()
    return files


@event_priority(-50)
def on_post_build(config: Any, **_: Any) -> None:
    """Preserve default-language LLM feeds across localized sub-builds.

    ``mkdocs-static-i18n`` runs each locale through the complete plugin event
    sequence. The locale-aware ``on_files`` hook above intentionally clears
    the non-default page selection, but ``llmstxt`` would then overwrite the
    canonical feeds with header-only files in its localized ``on_post_build``.
    Cache the English outputs after ``llmstxt`` runs, then restore them after
    every localized build. The root ``404.html`` is preserved in the same
    lifecycle so the final locale cannot overwrite its language and chrome.
    A locale-only build fails instead of publishing misleading canonical
    artifacts without a captured default-language build.
    """
    plugins = config.plugins
    i18n_plugin = plugins.get("i18n")
    llmstxt_plugin = plugins.get("llmstxt")
    if i18n_plugin is None or llmstxt_plugin is None:
        return

    output_names = ["404.html", "llms.txt"]
    full_output = getattr(llmstxt_plugin.config, "full_output", None)
    if isinstance(full_output, str) and full_output.strip():
        output_names.append(full_output.strip())

    site_dir = Path(config.site_dir)
    if i18n_plugin.is_default_language_build:
        _DEFAULT_BUILD_ARTIFACTS.clear()
        for output_name in output_names:
            output = site_dir / output_name
            if output.is_file():
                _DEFAULT_BUILD_ARTIFACTS[output_name] = output.read_bytes()
        return

    for output_name in output_names:
        output = site_dir / output_name
        payload = _DEFAULT_BUILD_ARTIFACTS.get(output_name)
        if payload is None:
            raise RuntimeError(
                "Localized MkDocs build cannot restore the canonical "
                f"default-language artifact: {output_name}"
            )
        output.write_bytes(payload)


def on_page_context(context: Any, config: Any, **_: Any) -> Any:
    """Prevent duplicate page assets after locale reconfiguration."""
    _deduplicate_extra_assets(config)
    return context


def on_template_context(context: Any, config: Any, **_: Any) -> Any:
    """Prevent duplicate assets on generated templates such as ``404.html``."""
    _deduplicate_extra_assets(config)
    return context


def _deduplicate_extra_assets(config: Any) -> None:
    """Deduplicate configured stylesheets and scripts while preserving order."""
    for key in ("extra_css", "extra_javascript"):
        unique: list[Any] = []
        for value in config.get(key, []):
            if value not in unique:
                unique.append(value)
        config[key] = unique


def _first_prose_paragraph(markdown: str) -> str:
    """Extract the first substantive prose paragraph from Markdown."""
    for block in re.split(r"\n\s*\n", markdown):
        candidate = block.strip()
        if (
            not candidate
            or candidate.startswith("#")
            or candidate.startswith(_BLOCK_MARKERS)
        ):
            continue
        plain = _plain_text(candidate)
        if len(plain) >= 40:
            return plain
    return ""


def _plain_text(value: str) -> str:
    """Remove common inline Markdown while preserving readable link labels."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", "`*_"))
    return re.sub(r"\s+", " ", text).strip()


def _truncate_description(value: str) -> str:
    """Keep descriptions concise without splitting the final word."""
    if len(value) <= _DESCRIPTION_LIMIT:
        return value
    truncated = value[: _DESCRIPTION_LIMIT - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{truncated}…"
