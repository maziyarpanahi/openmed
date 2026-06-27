from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
from typing import Any

# Optional extras mapping with their actual import names
OPTIONAL_EXTRAS = {
    "mlx": "mlx",
    "coreml": "coremltools",
    "onnx": "onnxruntime",
    "hf": "transformers",
    "multimodal": "PIL",
}

# Known architecture mappings for validation
SUPPORTED_ARCHS = {
    "x86_64",
    "AMD64",  # Intel/AMD 64-bit
    "arm64",
    "aarch64",  # ARM 64-bit (Apple Silicon, ARM servers)
    "armv7l",  # 32-bit ARM
}


def run_diagnostics() -> list[dict[str, Any]]:
    """Run comprehensive system diagnostics for OpenMed."""
    checks: list[dict[str, Any]] = []

    # 1. Python version check
    python_version = platform.python_version()
    version_parts = tuple(int(x) for x in python_version.split(".")[:2])

    status = "FAIL" if version_parts < (3, 9) else "PASS"

    checks.append(
        {
            "name": "python_version",
            "status": status,
            "details": python_version,
        }
    )

    # 2. Python architecture
    arch = platform.machine()
    is_supported = arch in SUPPORTED_ARCHS
    status = "FAIL" if not is_supported else "PASS"
    checks.append(
        {
            "name": "python_arch",
            "status": "PASS" if is_supported else "WARN",
            "details": arch,
        }
    )

    # 3. OpenMed version - search for __version__ in multiple locations
    try:
        from openmed.__about__ import __version__

        checks.append(
            {
                "name": "openmed_version",
                "status": "PASS",
                "details": __version__,
            }
        )
    except Exception:
        checks.append(
            {
                "name": "openmed_version",
                "status": "WARN",
                "details": "version unavailable",
            }
        )

    # 4. Optional dependencies with proper graceful degradation
    for name, module_name in OPTIONAL_EXTRAS.items():
        try:
            # For multimodal, check PIL/Pillow specifically
            if name == "multimodal":
                try:
                    importlib.import_module("PIL")
                    checks.append(
                        {
                            "name": name,
                            "status": "PASS",
                            "details": "Pillow installed",
                        }
                    )
                    continue
                except ImportError:
                    # Some systems might have PIL installed as Pillow but import as PIL
                    pass

            importlib.import_module(module_name)
            checks.append(
                {
                    "name": name,
                    "status": "PASS",
                    "details": "installed",
                }
            )
        except ImportError:
            # Not installed - just WARN, not FAIL (optional)
            hint_map = {
                "mlx": "Install with: pip install mlx",
                "coreml": "Install with: pip install coremltools",
                "onnx": "Install with: pip install onnxruntime",
                "hf": "Install with: pip install transformers",
                "multimodal": "Install with: pip install pillow",
            }

            checks.append(
                {
                    "name": name,
                    "status": "WARN",
                    "details": f"{module_name} not installed",
                    "hint": hint_map.get(name),
                }
            )

    # 5. HF token - check safely without exposing value
    token_present = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))

    checks.append(
        {
            "name": "hf_token",
            "status": "PASS" if token_present else "WARN",
            "details": f"present={token_present}",
            "hint": (
                None
                if token_present
                else "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable"
            ),
        }
    )

    # 6. OPENMED_OFFLINE environment variable
    offline = os.getenv("OPENMED_OFFLINE", "0")
    checks.append(
        {
            "name": "openmed_offline",
            "status": "PASS",
            "details": offline,
        }
    )

    # 7. Manifest existence and row count
    manifest = Path("models.jsonl")

    if manifest.exists():
        checks.append(
            {
                "name": "manifest_exists",
                "status": "PASS",
                "details": str(manifest),
            }
        )

        try:
            with manifest.open("r", encoding="utf-8") as f:
                rows = sum(1 for _ in f)

            checks.append(
                {
                    "name": "manifest_rows",
                    "status": "PASS",
                    "details": f"{rows} rows",
                }
            )

        except Exception as exc:
            checks.append(
                {
                    "name": "manifest_rows",
                    "status": "WARN",
                    "details": f"error reading manifest: {exc}",
                }
            )

    else:
        checks.append(
            {
                "name": "manifest_exists",
                "status": "WARN",
                "details": "models.jsonl not found",
            }
        )

    return checks
