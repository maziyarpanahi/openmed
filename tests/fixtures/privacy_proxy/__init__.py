"""Load repository-owned synthetic privacy-proxy fixtures."""

import json
from pathlib import Path


def load_fixture(name: str) -> dict:
    """Read one named JSON fixture without modifying its request envelope."""
    return json.loads(
        Path(__file__).with_name(f"{name}.json").read_text(encoding="utf-8")
    )
