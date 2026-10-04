"""Offline-safe environment diagnostics for OpenMed."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .manifest_schema import MANIFEST_PATH
from .offline import env_flag_enabled

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
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


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
    _check_network_environment(checks)
    _check_offline_mode(checks)
    _check_manifest(checks)
    checks.extend(clinical_brief_readiness())

    return checks


def clinical_brief_readiness() -> list[dict[str, Any]]:
    """Inspect local brief prerequisites without importing runtimes or loading models.

    Returns:
        Value-free PASS/WARN checks with controlled codes and remediation hints.
        Cache presence is not inference, calibration, review or clinical approval.
        The raw-note check covers the default English PII artifact only; callers
        selecting another language/model must provision that artifact separately.
    """
    from .model_registry import (
        get_default_nli_model,
        get_default_pii_model,
        resolve_summarizer_model,
    )

    def check(name: str, available: bool, ready: str, missing: str, hint: str):
        code = ready if available else missing
        result = _check(
            name, "PASS" if available else "WARN", code, None if available else hint
        )
        result["code"] = code
        return result

    platform_ready = platform.system() == "Darwin" and platform.machine() in {
        "arm64",
        "aarch64",
    }
    runtime_ready = all(
        _brief_module_present(name) for name in ("mlx", "mlx_lm", "huggingface_hub")
    )
    summarizer, revision = resolve_summarizer_model("mlx")
    pii = get_default_pii_model("en")
    nli = get_default_nli_model()
    return [
        check(
            "brief_extractive",
            True,
            "extractive_available",
            "extractive_unavailable",
            "",
        ),
        check(
            "brief_mlx_platform",
            platform_ready,
            "mlx_platform_supported",
            "mlx_platform_unsupported",
            "Use Apple silicon for the MLX alias, or explicitly choose extractive/local caller-supplied generation.",
        ),
        check(
            "brief_mlx_runtime",
            runtime_ready,
            "mlx_runtime_present",
            "mlx_runtime_missing",
            "Install the mlx extra before provisioning local inference.",
        ),
        check(
            "brief_summarizer_cache",
            _brief_cached_artifact(summarizer, revision),
            "pinned_summarizer_cached",
            "pinned_summarizer_missing",
            "Provision the registered pinned Maple revision before offline use.",
        ),
        check(
            "brief_raw_note_pii_cache",
            pii is not None and _brief_cached_artifact(pii, "main"),
            "default_pii_cached",
            "default_pii_missing",
            "Provision the default English PII artifact, or supply an already de-identified artifact and reviewed context.",
        ),
        check(
            "brief_nli_provider",
            nli is not None,
            "released_nli_registered",
            "caller_nli_provider_required",
            "Supply a calibrated local NLI provider; training a new checkpoint is not required to use the SDK.",
        ),
    ]


def _brief_module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, OSError):
        return False


def _brief_cached_artifact(model_id: str, revision: str) -> bool:
    """Read only cache metadata/stat results; never download or construct a model."""
    try:
        cache, _ = _hf_cache_location()
        repository = cache / ("models--" + model_id.replace("/", "--"))
        if revision == "main":
            with (repository / "refs" / "main").open(encoding="ascii") as handle:
                revision = handle.read(42).strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            return False
        snapshot = repository / "snapshots" / revision

        def present(name: str) -> bool:
            path = snapshot / name
            return path.is_file() and path.stat().st_size > 0

        if not all(present(name) for name in ("config.json", "tokenizer.json")):
            return False
        if any(present(name) for name in ("model.safetensors", "pytorch_model.bin")):
            return True
        index = snapshot / "model.safetensors.index.json"
        if index.stat().st_size > 1_048_576:
            return False
        data = json.loads(index.read_text(encoding="utf-8"))
        mapping = data.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or len(mapping) > 100_000:
            return False
        weights = set(mapping.values())
        return len(weights) <= 256 and all(
            type(name) is str
            and re.fullmatch(r"[A-Za-z0-9_-]+\.safetensors", name) is not None
            and present(name)
            for name in weights
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


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
    token_present = bool(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    check = _check(
        "hf_token",
        "PASS" if token_present else "WARN",
        f"present={token_present}",
        None if token_present else "Set the HF_TOKEN environment variable",
    )
    check["present"] = token_present
    checks.append(check)


def _check_network_environment(checks: list[dict[str, Any]]) -> None:
    endpoint = os.getenv("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    endpoint_is_secure = _url_uses_https(endpoint)
    endpoint_check = _check(
        "hf_endpoint",
        "PASS" if endpoint_is_secure else "WARN",
        _redact_url_credentials(endpoint),
        None
        if endpoint_is_secure
        else "Use an HTTPS Hugging Face endpoint to protect credentials and models.",
    )
    endpoint_check["source"] = "HF_ENDPOINT" if os.getenv("HF_ENDPOINT") else "default"
    checks.append(endpoint_check)

    for env_name in PROXY_ENV_VARS:
        value, source = _environment_value(env_name)
        details = "not set" if value is None else _redact_url_credentials(value)
        proxy_check = _check(env_name.lower(), "PASS", details)
        proxy_check["present"] = value is not None
        if source is not None:
            proxy_check["source"] = source
        checks.append(proxy_check)

    cache_path, cache_source = _hf_cache_location()
    cache_check = _check(
        "hf_cache", "PASS", _escape_control_characters(str(cache_path))
    )
    cache_check["source"] = cache_source
    checks.append(cache_check)


def _environment_value(name: str) -> tuple[str | None, str | None]:
    # Match urllib's proxy selection: lowercase values override uppercase ones.
    for candidate in (name.lower(), name):
        value = os.getenv(candidate)
        if value:
            return value, candidate
    return None, None


def _redact_url_credentials(value: str) -> str:
    """Redact URL secrets while preserving a useful diagnostic address."""
    sanitized = _escape_control_characters(value)
    added_authority_prefix = False
    try:
        parsed = urlsplit(sanitized)
        if parsed.username is None and not parsed.netloc and "@" in sanitized:
            parsed = urlsplit(f"//{sanitized}")
            added_authority_prefix = True
        username = parsed.username
        password = parsed.password
    except ValueError:
        return "configured (value could not be parsed safely)"

    if username is None:
        if "@" in sanitized:
            return "configured (credentials redacted)"
        return urlunsplit(parsed._replace(query="", fragment=""))

    address = parsed.netloc.rsplit("@", 1)[-1]
    redacted_userinfo = "***:***" if password is not None else "***"
    redacted = urlunsplit(
        parsed._replace(
            netloc=f"{redacted_userinfo}@{address}",
            query="",
            fragment="",
        )
    )
    if added_authority_prefix and redacted.startswith("//"):
        return redacted[2:]
    return redacted


def _escape_control_characters(value: str) -> str:
    """Render control characters visibly instead of emitting them to a terminal."""
    return "".join(
        character if character.isprintable() else f"\\u{ord(character):04x}"
        for character in value
    )


def _url_uses_https(value: str) -> bool:
    """Return whether *value* names an HTTPS endpoint without control bytes."""
    if any(not character.isprintable() for character in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


def _hf_cache_location() -> tuple[Path, str]:
    explicit_cache = os.getenv("HF_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser(), "HF_HUB_CACHE"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub", "HF_HOME"

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface" / "hub", "XDG_CACHE_HOME"

    return Path.home() / ".cache" / "huggingface" / "hub", "default"


def _check_offline_mode(checks: list[dict[str, Any]]) -> None:
    raw_value = os.getenv("OPENMED_OFFLINE")
    enabled = env_flag_enabled(raw_value)
    rendered_value = _escape_control_characters(raw_value or "0")
    details = (
        f"{'enabled' if enabled else 'disabled'} (OPENMED_OFFLINE={rendered_value})"
    )
    check = _check("openmed_offline", "PASS", details)
    check["enabled"] = enabled
    checks.append(check)


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
