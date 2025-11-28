"""MkDocs hook helpers for documentation builds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def on_config(config: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Populate the current year placeholder in ``config``."""
    year = str(datetime.now(timezone.utc).year)
    copyright_text = config.get("copyright")
    if isinstance(copyright_text, str) and "{year}" in copyright_text:
        config["copyright"] = copyright_text.replace("{year}", year)
    return config
