"""MkDocs hook helpers for documentation builds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from mkdocs.plugins import event_priority


def on_config(config: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Populate the current year placeholder in ``config``."""
    year = str(datetime.now(timezone.utc).year)
    copyright_text = config.get("copyright")
    if isinstance(copyright_text, str) and "{year}" in copyright_text:
        config["copyright"] = copyright_text.replace("{year}", year)
    return config


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
