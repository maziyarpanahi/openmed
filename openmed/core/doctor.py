"""Offline-safe environment diagnostics for OpenMed."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import Any

from .manifest_schema import MANIFEST_PATH

# Optional extras mapping with their actual import names
OPTIONAL_EXTRAS = {
    "mlx": "mlx",
    "coreml": "coremltools",
    "onnx": "onnxruntime",
    "hf": "transformers",
    "multimodal": "PIL",
}

# Known architecture mappings for validation
SUPPORTED_ARCHS = frozenset(
    {
        "x86_64",
        "AMD64",  # Intel/AMD 64-bit
        "arm64",
        "aarch64",  # ARM 64-bit (Apple Silicon, ARM servers)
        "armv7l",  # 32-bit ARM
    }
)

MIN_PYTHON_VERSION = (3, 10)


def _check(
    name: str,
    status: str,
    details: str,
    hint: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "details": details,
    }
    if hint is not None:
        payload["hint"] = hint
    return payload


def run_diagnostics() -> list[dict[str, Any]]:
    """Run local OpenMed diagnostics without network calls or secret exposure.

    Returns:
        A list of check dictionaries with ``name``, ``status``, ``details``,
        and optional remediation ``hint`` fields.
    """
    checks: list[dict[str, Any]] = []

    python_version = platform.python_version()
    python_supported = sys.version_info[:2] >= MIN_PYTHON_VERSION
    checks.append(
        _check(
            "python_version",
            "PASS" if python_supported else "FAIL",
            python_version,
            None if python_supported else "Use Python 3.10 or newer.",
        )
    )

    arch = platform.machine() or "unknown"
    arch_supported = arch in SUPPORTED_ARCHS
    checks.append(
        _check(
            "python_arch",
            "PASS" if arch_supported else "FAIL",
            arch,
            None if arch_supported else "Use a supported 64-bit Python architecture.",
        )
    )

    _check_openmed_version(checks)
    _check_optional_dependencies(checks)
    _check_hf_token(checks)
    _check_offline_mode(checks)
    _check_manifest(checks)

    return checks


def _check_openmed_version(checks: list[dict[str, Any]]) -> None:
    try:
        from ..__about__ import __version__

        checks.append(_check("openmed_version", "PASS", __version__))
    except Exception:
        checks.append(_check("openmed_version", "WARN", "version unavailable"))


def _check_optional_dependencies(checks: list[dict[str, Any]]) -> None:
    hint_map = {
        "mlx": "Install with: pip install mlx",
        "coreml": "Install with: pip install coremltools",
        "onnx": "Install with: pip install onnxruntime",
        "hf": "Install with: pip install transformers",
        "multimodal": "Install with: pip install pillow",
    }

    for name, module_name in OPTIONAL_EXTRAS.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            checks.append(
                _check(
                    name,
                    "WARN",
                    f"{module_name} not installed",
                    hint_map.get(name),
                )
            )
            continue

        details = "Pillow installed" if name == "multimodal" else "installed"
        checks.append(_check(name, "PASS", details))


def _check_hf_token(checks: list[dict[str, Any]]) -> None:
    token_present = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    checks.append(
        _check(
            "hf_token",
            "PASS" if token_present else "WARN",
            f"present={token_present}",
            None
            if token_present
            else "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable",
        )
    )


def _check_offline_mode(checks: list[dict[str, Any]]) -> None:
    checks.append(_check("openmed_offline", "PASS", os.getenv("OPENMED_OFFLINE", "0")))


def _check_manifest(checks: list[dict[str, Any]]) -> None:
    if not MANIFEST_PATH.exists():
        checks.append(
            _check(
                "manifest_exists",
                "WARN",
                f"{MANIFEST_PATH.name} not found",
            )
        )
        checks.append(
            _check(
                "manifest_rows",
                "WARN",
                "not checked because manifest is missing",
            )
        )
        return

    checks.append(_check("manifest_exists", "PASS", str(MANIFEST_PATH)))

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    except OSError as exc:
        checks.append(_check("manifest_rows", "WARN", f"error reading manifest: {exc}"))
        return

    checks.append(_check("manifest_rows", "PASS", f"{rows} rows"))
