"""Conservative runtime CPU feature detection for ONNX fast paths."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_X86_ARCHITECTURES = frozenset({"x86", "x86_64"})
_ARM_ARCHITECTURES = frozenset({"arm", "arm64"})


@dataclass(frozen=True)
class CpuFeatures:
    """CPU architecture and instruction sets safe for the current process."""

    architecture: str
    flags: frozenset[str]
    avx2: bool = False
    avx512: bool = False
    neon: bool = False

    @property
    def tier(self) -> str:
        """Return the strongest supported classification-head kernel tier."""

        if self.avx512:
            return "avx512"
        if self.avx2:
            return "avx2"
        if self.neon:
            return "neon"
        return "scalar"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable CPU feature record."""

        return {
            "architecture": self.architecture,
            "flags": sorted(self.flags),
            "avx2": self.avx2,
            "avx512": self.avx512,
            "neon": self.neon,
            "tier": self.tier,
        }


def detect_cpu_features(
    *,
    machine: str | None = None,
    flags: str | Iterable[str] | None = None,
) -> CpuFeatures:
    """Detect AVX2, AVX-512, or NEON without optimistic guessing.

    Args:
        machine: Optional architecture override for deterministic tests.
        flags: Optional CPU flag text or iterable for deterministic tests.

    Returns:
        A conservative feature record. Unknown architectures and missing flag
        sources select the scalar tier.
    """

    architecture = _normalize_architecture(machine or platform.machine())
    normalized_flags = (
        _normalize_flags(flags)
        if flags is not None
        else _runtime_cpu_flags(architecture)
    )

    is_x86 = architecture in _X86_ARCHITECTURES
    is_arm = architecture in _ARM_ARCHITECTURES
    avx2 = is_x86 and "avx2" in normalized_flags
    avx512 = is_x86 and "avx512f" in normalized_flags and "avx512bw" in normalized_flags
    neon = is_arm and bool(normalized_flags & {"asimd", "neon"})
    return CpuFeatures(
        architecture=architecture,
        flags=normalized_flags,
        avx2=avx2,
        avx512=avx512,
        neon=neon,
    )


def select_cpu_kernel(features: CpuFeatures | None = None) -> str:
    """Select the strongest safe kernel name for the current process."""

    return (features or detect_cpu_features()).tier


def _normalize_architecture(machine: str) -> str:
    normalized = machine.strip().lower().replace("-", "_")
    aliases = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "armv7": "arm",
        "armv7l": "arm",
        "i386": "x86",
        "i686": "x86",
        "x64": "x86_64",
    }
    return aliases.get(normalized, normalized or "unknown")


def _normalize_flags(flags: str | Iterable[str]) -> frozenset[str]:
    values = flags.replace(",", " ").split() if isinstance(flags, str) else flags
    return frozenset(
        str(flag).strip().lower().replace(".", "_")
        for flag in values
        if str(flag).strip()
    )


def _runtime_cpu_flags(architecture: str) -> frozenset[str]:
    system = platform.system().lower()
    if system == "linux":
        return _linux_cpu_flags()
    if system == "darwin":
        return _darwin_cpu_flags(architecture)
    if system == "windows":
        return _normalize_flags(platform.processor())
    return frozenset()


def _linux_cpu_flags() -> frozenset[str]:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return frozenset()

    per_processor: list[frozenset[str]] = []
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"features", "flags"}:
            per_processor.append(_normalize_flags(value))
    if not per_processor:
        return frozenset()
    return frozenset.intersection(*per_processor)


def _darwin_cpu_flags(architecture: str) -> frozenset[str]:
    collected: set[str] = set()
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        value = _read_sysctl(key)
        if value:
            collected.update(_normalize_flags(value))

    if architecture in _ARM_ARCHITECTURES:
        for key, flag in (
            ("hw.optional.neon", "neon"),
            ("hw.optional.arm.FEAT_DotProd", "dotprod"),
        ):
            if _read_sysctl(key) == "1":
                collected.add(flag)
    return frozenset(collected)


def _read_sysctl(key: str) -> str:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


__all__ = ["CpuFeatures", "detect_cpu_features", "select_cpu_kernel"]
