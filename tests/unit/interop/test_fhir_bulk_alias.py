"""Check both import paths from a fresh module cache."""

import subprocess
import sys


def test_legacy_module_import_warns_once_and_preserves_exports():
    script = """
import importlib
import warnings
from openmed.interop.fhir import bulk
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    legacy = importlib.import_module("openmed.interop.fhir_bulk")
    assert importlib.import_module("openmed.interop.fhir_bulk") is legacy
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(bulk, name)
assert len(caught) == 1 and caught[0].category is DeprecationWarning
assert "openmed.interop.fhir.bulk" in str(caught[0].message)
"""
    subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", script], check=True
    )


def test_canonical_path_and_openmrs_import_without_alias():
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import sys, openmed.interop.fhir.bulk, openmed.interop.openmrs; "
            "assert 'openmed.interop.fhir_bulk' not in sys.modules",
        ],
        check=True,
    )
