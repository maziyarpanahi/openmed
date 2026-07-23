"""Acceptance checks for African developer onboarding documentation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_swahili_readme_is_linked_from_english_readme() -> None:
    source = (ROOT / "README.md").read_text(encoding="utf-8")
    translation = (ROOT / "README.sw.md").read_text(encoding="utf-8")

    assert '<a href="README.sw.md">Kiswahili</a>' in source
    assert '<a href="README.md">English</a>' in translation


def test_africa_guide_covers_required_onboarding_paths() -> None:
    guide = (ROOT / "docs" / "africa-onboarding.md").read_text(encoding="utf-8")

    for required in (
        "OpenMed-NER-AnatomyDetect-ElectraMed-33M",
        "openmed models size",
        "--budget-mb 100",
        "snapshot_download",
        "OPENMED_OFFLINE=1",
        "POPIA",
        "NDPA",
        "strict_no_leak",
        "clinical_minimal_redaction",
        "research_limited_dataset",
        "to_bundle",
        "/openmrs/ws/fhir2/R4",
        "/api/tracker",
    ):
        assert required in guide

    community_links = (
        "https://www.masakhane.io/",
        "https://datasciencenigeria.org/",
        "https://www.zindi.africa/",
        "https://deeplearningindaba.com/",
        "https://www.meetup.com/python-ghana/",
        "https://www.meetup.com/python-nairobi/",
    )
    assert sum(link in guide for link in community_links) >= 4


def test_africa_guide_is_in_nav_and_linked_from_quick_start() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    quick_start = (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")

    assert "African Developer Onboarding: africa-onboarding.md" in nav
    assert "[African developer onboarding guide](africa-onboarding.md)" in quick_start


def test_africa_guide_python_examples_compile() -> None:
    guide = (ROOT / "docs" / "africa-onboarding.md").read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)```", guide, flags=re.DOTALL)

    assert snippets
    for index, snippet in enumerate(snippets, start=1):
        compile(snippet, f"africa-onboarding-python-{index}", "exec")
