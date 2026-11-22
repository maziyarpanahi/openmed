"""Regression tests for the private-learning stability page."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "training" / "private-learning-stability.md"
MKDOCS = ROOT / "mkdocs.yml"
PUBLICATION = ROOT / "docs" / "brand" / "system" / "publication.yml"

GA_CRITERIA = ("2876", "3209", "3212")
SCHEMA_MODULES = (
    "federated_round",
    "federated_schedule",
    "federated_status",
    "federated_update_metadata",
    "federated_metrics",
)


def _page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_is_navigated_and_published() -> None:
    page = "training/private-learning-stability.md"

    assert page in MKDOCS.read_text(encoding="utf-8")
    assert f"- {page}" in PUBLICATION.read_text(encoding="utf-8")


def test_page_names_every_ga_criterion_and_the_tracking_release() -> None:
    text = _page_text()

    for issue in GA_CRITERIA:
        assert f"issues/{issue})" in text, f"missing GA criterion link: #{issue}"
    assert "**v7.0**" in text
    assert "experimental" in text.lower()


def test_quoted_schema_versions_match_the_modules_they_describe() -> None:
    text = _page_text()

    for module_name in SCHEMA_MODULES:
        module = importlib.import_module(f"openmed.training.{module_name}")
        versions = [
            value
            for name, value in vars(module).items()
            if name.endswith("SCHEMA_VERSION") and isinstance(value, str)
        ]
        assert len(versions) == 1, f"{module_name} must expose one schema version"
        assert versions[0] in text, f"page omits {versions[0]}"


def test_stable_contracts_are_separated_from_the_experimental_surface() -> None:
    text = _page_text()
    stable_section = text.split("## What stays stable", 1)[1].split(
        "## GA criteria", 1
    )[0]

    linked = re.findall(
        r"\[[^\]]+\]\((federated-[a-z-]+\.md|private-training-[a-z-]+\.md)\)",
        stable_section,
    )
    assert set(linked) == {
        "federated-round-lifecycle.md",
        "federated-scheduling.md",
        "federated-round-status.md",
        "federated-update-metadata.md",
        "federated-metrics.md",
        "federated-schema-fingerprints.md",
        "private-training-schemas.md",
    }
    for guide in sorted(set(linked)):
        assert (PAGE.parent / guide).is_file(), f"guide does not exist: {guide}"
    # The stable section must stay metadata-only: the experimental execution
    # package and the governance package are not part of the shipped contracts.
    assert "openmed/training/federated/" not in stable_section
    assert "openmed/compliance/governance" not in stable_section
    assert "do not run training" in stable_section
