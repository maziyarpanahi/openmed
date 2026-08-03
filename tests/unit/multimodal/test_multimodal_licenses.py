"""License policy tests for dependencies"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from openmed.multimodal._licenses import (
    MULTIMODAL_DEPENDENCY_LICENSES,
    MULTIMODAL_EXTRAS,
    SYSTEM_BINARY_LICENSES,
    TCIA_COLLECTION_CAVEAT,
    license_for,
    normalize_distribution,
    recorded_distributions,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"
MULTIMODAL_PACKAGE = ROOT / "openmed" / "multimodal"
POLICY_SCRIPT = ROOT / "scripts" / "release" / "check_license_policy.py"

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MODULE_TO_DISTRIBUTION = {
    "PIL": "pillow",
    "docx": "python-docx",
    "doctr": "python-doctr",
    "easyocr": "easyocr",
    "markdown_it": "markdown-it-py",
    "numpy": "numpy",
    "onnx": "onnx",
    "paddleocr": "paddleocr",
    "pdfplumber": "pdfplumber",
    "piexif": "piexif",
    "pikepdf": "pikepdf",
    "pydicom": "pydicom",
    "pytesseract": "pytesseract",
}


def _load_policy():
    """Load the OM-036 license gate as a module, without importing scripts/."""
    spec = importlib.util.spec_from_file_location("check_license_policy", POLICY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = _load_policy()


def _pinned_distributions() -> set[str]:
    """Return the normalized distributions pinned by any multimodal extra."""
    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    optional = pyproject["project"]["optional-dependencies"]
    return {
        policy.dependency_name(str(requirement))
        for extra in MULTIMODAL_EXTRAS
        for requirement in optional[extra]
    }


def _statically_imported_modules() -> set[str]:
    """Return top-level modules imported at module scope across the package."""
    modules: set[str] = set()
    for path in sorted(MULTIMODAL_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_every_pinned_dependency_has_a_license_record():
    """A dependency added to a multimodal extra must be licensed-audited."""
    missing = sorted(_pinned_distributions() - recorded_distributions())

    assert not missing, (
        f"Dependencies pinned by {list(MULTIMODAL_EXTRAS)} with no entry in "
        f"openmed/multimodal/_licenses.py: {missing}"
    )


def test_record_has_no_entries_for_unpinned_dependencies():
    """The record must not drift ahead of pyproject.toml."""
    stale = sorted(recorded_distributions() - _pinned_distributions())

    assert not stale, (
        f"Entries in openmed/multimodal/_licenses.py that no multimodal extra "
        f"pins any more: {stale}"
    )


@pytest.mark.parametrize(
    "entry", MULTIMODAL_DEPENDENCY_LICENSES, ids=lambda e: e.distribution
)
def test_recorded_license_is_on_the_permissive_allow_list(entry):
    allowed, reason = policy.is_allowed_license(entry.distribution, entry.spdx)

    assert allowed, f"{entry.distribution} ({entry.spdx}) is not permissive: {reason}"


@pytest.mark.parametrize(
    "entry", MULTIMODAL_DEPENDENCY_LICENSES, ids=lambda e: e.distribution
)
def test_recorded_license_agrees_with_the_repository_gate(entry):
    """The record and the OM-036 table must not reach opposite verdicts.

    They are allowed to spell a license differently -- Pillow publishes
    MIT-CMU where the gate's table says HPND -- so this compares policy
    outcomes rather than strings.
    """
    distribution = normalize_distribution(entry.distribution)
    gate_license = policy.resolve_license(distribution)
    if not gate_license:
        pytest.skip(f"{distribution} has no reviewed license in the OM-036 table")

    gate_allowed, _ = policy.is_allowed_license(distribution, gate_license)
    record_allowed, _ = policy.is_allowed_license(distribution, entry.spdx)

    assert gate_allowed == record_allowed, (
        f"{distribution}: record says {entry.spdx} (allowed={record_allowed}) "
        f"but the OM-036 gate says {gate_license} (allowed={gate_allowed})"
    )


def test_a_non_permissive_dependency_would_fail_the_gate():
    """Guard against the allow-list silently accepting everything."""
    allowed, reason = policy.is_allowed_license("example-copyleft", "GPL-3.0-only")

    assert allowed is False
    assert "not allowed" in reason


# --- Provenance: every claim is traceable and dated --------------------------


@pytest.mark.parametrize(
    "entry",
    [*MULTIMODAL_DEPENDENCY_LICENSES, *SYSTEM_BINARY_LICENSES],
    ids=lambda e: getattr(e, "distribution", None) or e.name,
)
def test_every_record_carries_spdx_source_and_verification_date(entry):
    label = getattr(entry, "distribution", None) or entry.name

    assert entry.spdx.strip(), f"{label} has no SPDX identifier"
    assert entry.source_url.startswith("https://"), f"{label} has no https source URL"
    assert ISO_DATE_RE.match(entry.verified_on), (
        f"{label} verification date {entry.verified_on!r} is not ISO-8601"
    )


def test_pydicom_is_recorded_as_verified_mit():
    """OM-092: replaces the roadmap's 'license unverified' caveat."""
    entry = license_for("pydicom")

    assert entry is not None, "pydicom must carry an explicit license record"
    assert entry.spdx == "MIT"
    assert entry.source_url.startswith("https://")
    assert ISO_DATE_RE.match(entry.verified_on)


# --- Invariant I2: nothing non-permissive runs in-process --------------------


def test_no_non_permissive_dependency_is_marked_in_process():
    offenders = [
        entry.distribution
        for entry in MULTIMODAL_DEPENDENCY_LICENSES
        if entry.in_process
        and not policy.is_allowed_license(entry.distribution, entry.spdx)[0]
    ]

    assert not offenders, (
        f"Invariant I2: non-permissive dependencies imported in-process by the "
        f"multimodal package must move to openmed/interop/bridges: {offenders}"
    )


def test_multimodal_package_only_imports_recorded_permissive_distributions():
    """Any third-party import in the package must trace to a permissive record."""
    third_party = {
        module
        for module in _statically_imported_modules()
        if module not in sys.stdlib_module_names
        and module != "openmed"
        and not module.startswith("_")
    }

    unrecorded = sorted(third_party - set(MODULE_TO_DISTRIBUTION))
    assert not unrecorded, (
        f"Third-party modules imported by openmed/multimodal with no known "
        f"distribution mapping: {unrecorded}. Add them to the license record."
    )

    for module in sorted(third_party):
        entry = license_for(MODULE_TO_DISTRIBUTION[module])
        assert entry is not None, (
            f"{module} maps to {MODULE_TO_DISTRIBUTION[module]}, which has no "
            f"license record"
        )
        allowed, reason = policy.is_allowed_license(entry.distribution, entry.spdx)
        assert allowed, (
            f"Invariant I2: {module} ({entry.spdx}) is imported in-process by "
            f"the multimodal package but is not permissive: {reason}"
        )


def test_module_to_distribution_map_only_names_recorded_distributions():
    """Keep the test's module map from drifting away from the record."""
    unknown = sorted(
        distribution
        for distribution in MODULE_TO_DISTRIBUTION.values()
        if license_for(distribution) is None
    )

    assert not unknown, f"Module map names unrecorded distributions: {unknown}"


# --- OCR system binaries and dataset caveats ---------------------------------


def test_tesseract_system_binary_license_is_documented():
    tesseract = next(
        (entry for entry in SYSTEM_BINARY_LICENSES if entry.engine == "tesseract"),
        None,
    )

    assert tesseract is not None, "The Tesseract binary license must be recorded"
    assert tesseract.spdx == "Apache-2.0"


def test_tcia_per_collection_caveat_is_recorded():
    assert "per collection" in TCIA_COLLECTION_CAVEAT
    assert "cancerimagingarchive.net" in TCIA_COLLECTION_CAVEAT
