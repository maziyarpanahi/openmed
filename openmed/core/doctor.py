"""Offline-safe environment diagnostics for OpenMed."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
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
LOW_RESOURCE_MIN_RAM_BYTES = 4 * 1024**3
LOW_RESOURCE_SUGGEST_RAM_BYTES = 8 * 1024**3


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
    _check_low_resource_envelope(checks)
    _check_optional_dependencies(checks)
    _check_hf_token(checks)
    _check_offline_mode(checks)
    _check_manifest(checks)

    return checks


def _check_low_resource_envelope(checks: list[dict[str, Any]]) -> None:
    total_bytes = _effective_memory_bytes()
    total_gib = total_bytes / 1024**3
    fits = total_bytes >= LOW_RESOURCE_MIN_RAM_BYTES
    suggest = total_bytes < LOW_RESOURCE_SUGGEST_RAM_BYTES

    hint = None
    if not fits:
        hint = (
            "This host has less than a 4 GiB effective memory limit; close other "
            "applications before de-identification."
        )
    elif suggest:
        hint = "Set OPENMED_PROFILE=low_resource for CPU-only ONNX INT8 inference."

    check = _check(
        "low_resource_memory",
        "PASS" if fits else "WARN",
        f"effective_ram={total_gib:.2f} GiB; fits_4gb_envelope={str(fits).lower()}",
        hint,
    )
    check["effective_ram_bytes"] = total_bytes
    check["fits_low_resource"] = fits
    check["profile_suggested"] = suggest
    checks.append(check)


def _effective_memory_bytes() -> int:
    """Return physical RAM capped by the current cgroup limit when present."""
    physical = _physical_memory_bytes()
    limits: list[int] = []
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = open(path, encoding="utf-8").read().strip()  # noqa: PTH123
        except OSError:
            continue
        if raw != "max":
            try:
                value = int(raw)
            except ValueError:
                continue
            if 0 < value < 1 << 60:
                limits.append(value)
    return min([physical, *limits]) if limits else physical


def _physical_memory_bytes() -> int:
    if os.name == "nt":  # pragma: no cover - Windows-only
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)

    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        if platform.system() != "Darwin":
            raise RuntimeError("Unable to determine physical memory")
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
        )
        return int(output.strip())


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
    check = _check(
        "hf_token",
        "PASS" if token_present else "WARN",
        f"present={token_present}",
        None
        if token_present
        else "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable",
    )
    check["present"] = token_present
    checks.append(check)


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
