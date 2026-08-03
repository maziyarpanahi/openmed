"""Smoke tests for the section 4.2 package scaffold."""

from importlib import import_module
from pathlib import Path


def test_section_4_2_packages_import_cleanly():
    """All new top-level package shells should describe and import cleanly."""
    expected_docstring_content = {
        "openmed.clinical": (
            "sections.py",
            "context.py",
            "grounding.py",
            "relations.py",
            "sdoh.py",
            "FHIR/OMOP",
        ),
        "openmed.eval": (
            "harness.py",
            "metrics.py",
            "suites/",
            "golden/",
            "report.py",
            "calibrate.py",
            "release_gates.py",
        ),
        "openmed.multimodal": (
            "PDF/DOCX/HTML->text+offsets",
            "OCR",
            "image/DICOM",
        ),
        "openmed.structured": (
            "column classification",
            "k-anonymity",
            "l-diversity",
            "t-closeness",
            "differential privacy",
        ),
        "openmed.risk": (
            "quasi-identifier",
            "uniqueness/k-anonymity",
            "adversarial",
        ),
        "openmed.interop": ("canonical spans", "bridges/", "subprocess"),
        "openmed.interop.bridges": ("Permissive-only", "GPL", "invariant I2"),
    }

    for package_name, expected_content in expected_docstring_content.items():
        package = import_module(package_name)
        assert package.__doc__ is not None
        normalized_docstring = " ".join(package.__doc__.split())
        missing_content = [
            content
            for content in expected_content
            if content not in normalized_docstring
        ]
        assert not missing_content, (
            f"{package_name} docstring missing {missing_content}"
        )

    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "openmed" / "evals").exists()
