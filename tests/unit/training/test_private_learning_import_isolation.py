"""Import-isolation tests for the private-learning metadata modules.

The private-training schema catalog is documented as import-light: the
metadata modules under ``openmed/training/`` use only the standard library,
so an air-gapped site can validate metadata without installing a training
framework. These tests hold that line by importing the modules in a clean
subprocess and reporting any heavy optional package that gets loaded.
"""

from __future__ import annotations

import subprocess
import sys

# Heavy optional packages the metadata modules must never pull in.
HEAVY_MODULES = (
    "torch",
    "transformers",
    "numpy",
    "cryptography",
    "flwr",
)

METADATA_MODULES = (
    "openmed.training.federated_round",
    "openmed.training.federated_schedule",
    "openmed.training.federated_status",
    "openmed.training.federated_update_metadata",
    "openmed.training.federated_metrics",
    "openmed.training.federated_schema_fingerprint",
    "openmed.training.private_training_schemas",
)


def _loaded_heavy_modules(script: str) -> set[str]:
    """Run ``script`` in a clean subprocess; return heavy modules it loaded."""
    code = (
        f"{script}\n"
        "import sys\n"
        f"loaded = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return {m for m in result.stdout.strip().split(",") if m}


def test_metadata_modules_do_not_load_heavy_optional_packages():
    imports = "\n".join(f"import {name}" for name in METADATA_MODULES)
    assert _loaded_heavy_modules(imports) == set()


def test_training_package_and_lazy_export_do_not_load_heavy_packages():
    script = (
        "import openmed.training\n"
        "from openmed.training import FederatedRoundLifecycle\n"
    )
    assert _loaded_heavy_modules(script) == set()


def test_preloaded_heavy_module_is_reported_by_name():
    # Negative control: a stub heavy module already present in sys.modules
    # must be reported, proving the check above can actually fail.
    script = (
        "import sys\n"
        "import types\n"
        "sys.modules['torch'] = types.ModuleType('torch')\n"
        "import openmed.training.federated_round\n"
    )
    assert _loaded_heavy_modules(script) == {"torch"}
