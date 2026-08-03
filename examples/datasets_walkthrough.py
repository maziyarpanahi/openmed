#!/usr/bin/env python3
"""Walk through a bundled synthetic golden fixture with the public PII API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from openmed import deidentify, extract_pii

FIXTURE_ID = "golden-multilingual-en-ssn"
ALLOW_DOWNLOAD_ENV = "OPENMED_EXAMPLE_ALLOW_DOWNLOAD"


def fixture_path() -> Path:
    """Return the repository-relative path to the bundled fixture file."""
    return (
        Path(__file__).resolve().parents[1]
        / "openmed"
        / "eval"
        / "golden"
        / "fixtures"
        / "multilingual.json"
    )


def load_fixture(fixture_id: str = FIXTURE_ID) -> dict[str, Any]:
    """Load one synthetic, redistributable fixture without model access."""
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    for fixture in payload["fixtures"]:
        if fixture["id"] == fixture_id:
            if not fixture.get("metadata", {}).get("synthetic", False):
                raise ValueError(f"Fixture is not marked synthetic: {fixture_id}")
            return fixture
    raise KeyError(f"Unknown bundled fixture: {fixture_id}")


@contextmanager
def offline_model_loading() -> Iterator[None]:
    """Keep the walkthrough offline unless downloads are explicitly enabled."""
    if os.getenv(ALLOW_DOWNLOAD_ENV) == "1":
        yield
        return

    previous = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_walkthrough(fixture_id: str = FIXTURE_ID) -> None:
    """Print a compact before/after summary for one bundled fixture."""
    fixture = load_fixture(fixture_id)
    text = fixture["text"]
    expected = fixture["metadata"]["expected_output"]["text"]

    print(f"Fixture: {fixture['id']} ({fixture['language']})")
    print("Synthetic data: yes; no DUA or external dataset required")
    print(f"Before: {text}")

    try:
        with offline_model_loading():
            extracted = extract_pii(text, lang=fixture["language"])
            result = deidentify(
                text,
                method="mask",
                lang=fixture["language"],
            )
    except Exception as exc:
        print(f"After: model unavailable offline ({exc})")
        print("Set OPENMED_EXAMPLE_ALLOW_DOWNLOAD=1 to allow first-run downloads.")
        return

    print(f"Detected entities: {len(extracted.entities)}")
    print(f"After: {result.deidentified_text}")
    print(f"Matches bundled expected output: {result.deidentified_text == expected}")


def main() -> None:
    run_walkthrough()


if __name__ == "__main__":
    main()
