"""Verified license record for dependencies"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "DependencyLicense",
    "SystemBinaryLicense",
    "MULTIMODAL_EXTRAS",
    "MULTIMODAL_DEPENDENCY_LICENSES",
    "SYSTEM_BINARY_LICENSES",
    "TCIA_COLLECTION_CAVEAT",
    "normalize_distribution",
    "license_for",
    "recorded_distributions",
]

# Optional-dependency groups in pyproject.toml that this record covers.
MULTIMODAL_EXTRAS = ("multimodal", "ocr-paddle")

_NORMALIZE_RE = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class DependencyLicense:
    """A verified license for one distribution pinned by a multimodal extra.

    Attributes:
        distribution: PEP 503 normalized distribution name on PyPI.
        spdx: SPDX license expression as published by the project.
        source_url: Where the license was read from during verification.
        verified_on: ISO-8601 date the license was last confirmed.
        in_process: Whether OpenMed imports the package into its own process.
            Permissive-only packages may; anything else must go through
            :mod:`openmed.interop.bridges` to preserve invariant I2.
        note: Optional clarification, used where the SPDX identifier alone is
            misleading or where policy needs an explicit rationale.
    """

    distribution: str
    spdx: str
    source_url: str
    verified_on: str
    in_process: bool = True
    note: str = ""


@dataclass(frozen=True)
class SystemBinaryLicense:
    """A verified license for a non-PyPI system binary an OCR engine shells to.

    Attributes:
        name: Human-readable binary or project name.
        spdx: SPDX license expression for the binary.
        source_url: Where the license was read from during verification.
        verified_on: ISO-8601 date the license was last confirmed.
        engine: OpenMed OCR engine name that invokes the binary.
        note: Why the binary is outside the Python dependency graph.
    """

    name: str
    spdx: str
    source_url: str
    verified_on: str
    engine: str
    note: str = ""


# Verified against the PyPI project metadata for each distribution on the date
# recorded below. Keep this tuple sorted by distribution name.
MULTIMODAL_DEPENDENCY_LICENSES: tuple[DependencyLicense, ...] = (
    DependencyLicense(
        distribution="easyocr",
        spdx="Apache-2.0",
        source_url="https://pypi.org/project/easyocr/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="markdown-it-py",
        spdx="MIT",
        source_url="https://pypi.org/project/markdown-it-py/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="numpy",
        spdx="BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        source_url="https://pypi.org/project/numpy/",
        verified_on="2026-08-03",
        note=(
            "Multi-license expression covering vendored components; every "
            "term is permissive and none is copyleft."
        ),
    ),
    DependencyLicense(
        distribution="onnx",
        spdx="Apache-2.0",
        source_url="https://pypi.org/project/onnx/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="paddleocr",
        spdx="Apache-2.0",
        source_url="https://pypi.org/project/paddleocr/",
        verified_on="2026-08-03",
        note=(
            "Split into its own extra because paddlepaddle is heavy and "
            "platform-sensitive, not for licensing reasons."
        ),
    ),
    DependencyLicense(
        distribution="pdfplumber",
        spdx="MIT",
        source_url="https://pypi.org/project/pdfplumber/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="piexif",
        spdx="MIT",
        source_url="https://pypi.org/project/piexif/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="pikepdf",
        spdx="MPL-2.0",
        source_url="https://pypi.org/project/pikepdf/",
        verified_on="2026-08-03",
        note=(
            "Weak copyleft at file scope. Permitted in-process by the "
            "repository license policy, which lists MPL-2.0 as allowed: "
            "obligations attach to modified pikepdf source files, not to "
            "OpenMed code that merely imports it. Contributing patches back "
            "to pikepdf itself would trigger MPL disclosure for those files."
        ),
    ),
    DependencyLicense(
        distribution="pillow",
        spdx="MIT-CMU",
        source_url="https://pypi.org/project/pillow/",
        verified_on="2026-08-03",
        note=(
            "Pillow publishes MIT-CMU, the current SPDX identifier for the "
            "HPND-style license it has always used. Permissive either way."
        ),
    ),
    DependencyLicense(
        distribution="pydicom",
        spdx="MIT",
        source_url="https://pypi.org/project/pydicom/",
        verified_on="2026-08-03",
        note=(
            "Supersedes the roadmap's 'license unverified - confirm before "
            "bundling' caveat (sec 2.2c/5.8/4.6). Confirmed MIT; safe to "
            "bundle in-process."
        ),
    ),
    DependencyLicense(
        distribution="pytesseract",
        spdx="Apache-2.0",
        source_url="https://pypi.org/project/pytesseract/",
        verified_on="2026-08-03",
        note=(
            "Wrapper only. The Tesseract binary it drives is licensed "
            "separately; see SYSTEM_BINARY_LICENSES."
        ),
    ),
    DependencyLicense(
        distribution="python-docx",
        spdx="MIT",
        source_url="https://pypi.org/project/python-docx/",
        verified_on="2026-08-03",
    ),
    DependencyLicense(
        distribution="python-doctr",
        spdx="Apache-2.0",
        source_url="https://pypi.org/project/python-doctr/",
        verified_on="2026-08-03",
    ),
)

# OCR engines shell out to binaries the user installs themselves. These are not
# in the Python dependency graph, so the pyproject-driven gate cannot see them.
SYSTEM_BINARY_LICENSES: tuple[SystemBinaryLicense, ...] = (
    SystemBinaryLicense(
        name="Tesseract OCR",
        spdx="Apache-2.0",
        source_url="https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE",
        verified_on="2026-08-03",
        engine="tesseract",
        note=(
            "Installed by the user via a system package manager and invoked "
            "as a subprocess by pytesseract. Not redistributed by OpenMed."
        ),
    ),
)

# DICOM corpora are data, not dependencies, and their terms do not follow the
# pydicom license. Recorded here because the DICOM path is the most likely
# place for a contributor to reach for public imaging data.
TCIA_COLLECTION_CAVEAT = (
    "The Cancer Imaging Archive (TCIA) licenses data per collection, not "
    "archive-wide. Individual collections range from CC BY to restricted "
    "terms requiring explicit permission, so no TCIA collection may be "
    "committed as a test fixture or bundled with OpenMed without checking "
    "that specific collection's license. Use synthetic DICOM fixtures "
    "instead. See https://www.cancerimagingarchive.net/data-usage-policies-"
    "and-restrictions/"
)


def normalize_distribution(name: str) -> str:
    """Normalize a distribution name according to PEP 503.

    Args:
        name: Raw distribution name, e.g. ``"Pillow"`` or ``"python_docx"``.

    Returns:
        The normalized name, e.g. ``"pillow"`` or ``"python-docx"``.
    """

    return _NORMALIZE_RE.sub("-", name).lower()


def recorded_distributions() -> frozenset[str]:
    """Return the normalized distribution names covered by this record.

    Returns:
        Every distribution name in :data:`MULTIMODAL_DEPENDENCY_LICENSES`.
    """

    return frozenset(
        normalize_distribution(entry.distribution)
        for entry in MULTIMODAL_DEPENDENCY_LICENSES
    )


def license_for(name: str) -> DependencyLicense | None:
    """Look up the verified license record for a distribution.

    Args:
        name: Distribution name, normalized or not.

    Returns:
        The matching :class:`DependencyLicense`, or ``None`` when the
        distribution is not pinned by a multimodal extra.
    """

    normalized = normalize_distribution(name)
    for entry in MULTIMODAL_DEPENDENCY_LICENSES:
        if normalize_distribution(entry.distribution) == normalized:
            return entry
    return None
