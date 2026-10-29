"""Offline verification CLI for local clinical SLM packages.

The command inspects package metadata only.  It never imports a model runtime,
constructs a tokenizer or model, reads weights into memory, or opens a socket,
so an air-gapped operator can gate a package before any clinical request
reaches the local inference path.

The report contains counts, fixed reason codes, closed capability names, and
the canonical manifest digest.  It never echoes a package path, model
identifier, component name, or any other free-form manifest value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from openmed.models import clinical_slm_manifest as manifest_module
from openmed.models.clinical_slm_capabilities import (
    ClinicalSLMCapabilityError,
    probe_clinical_slm_capabilities,
)
from openmed.models.clinical_slm_manifest import (
    ClinicalSLMManifestError,
    load_clinical_slm_manifest,
    verify_clinical_slm_package,
)
from openmed.models.clinical_slm_memory import (
    ClinicalSLMMemoryError,
    preflight_clinical_slm_memory,
)

from ._output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, emit

UNSUPPORTED_PLATFORM_CODE = "unsupported_platform"
MANIFEST_INVALID_CODE = "slm_manifest_invalid"
PACKAGE_INVALID_CODE = "slm_package_invalid"
UNKNOWN_TASK_CODE = "slm_unknown_task"
CAPABILITY_METADATA_CODE = "slm_capability_metadata_invalid"
MEMORY_PROFILE_CODE = "slm_memory_profile_invalid"

_ARTIFACT_REASON_CODES = frozenset(
    {
        "artifact_invalid",
        "artifact_missing",
        "artifact_unreadable",
        "component_mutated",
        "component_unreadable",
        "duplicate_field",
        "missing_component",
        "package_invalid",
        "weights_missing",
    }
)


def add_slm_verify_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the offline clinical SLM package verification command."""

    parser = subparsers.add_parser(
        "slm-verify",
        help="Verify and probe a local clinical SLM package without loading it.",
    )
    parser.add_argument(
        "package_dir",
        type=Path,
        metavar="PACKAGE_DIR",
        help="Local package directory holding a clinical SLM manifest.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        required=True,
        metavar="CAPABILITY",
        help=(
            "Required capability or task name, such as bounded_summarization "
            "or nli; repeat to request several."
        ),
    )
    parser.add_argument(
        "--memory-budget",
        type=_positive_int,
        required=True,
        metavar="BYTES",
        help="Explicit device memory budget in bytes for the loading preflight.",
    )
    parser.add_argument(
        "--headroom-bytes",
        type=_non_negative_int,
        default=0,
        metavar="BYTES",
        help="Memory that must stay free after loading (default: 0).",
    )
    parser.set_defaults(handler=handle_slm_verify)


def handle_slm_verify(args: argparse.Namespace) -> int:
    """Report an offline verification verdict for one clinical SLM package."""

    package_dir = Path(args.package_dir)
    if not _secure_local_reads_available():
        raise CliError(
            "Clinical SLM package verification needs secure local reads, which "
            "this platform does not provide",
            code=UNSUPPORTED_PLATFORM_CODE,
            exit_code=EXIT_ERROR,
        )

    try:
        manifest = load_clinical_slm_manifest(package_dir)
    except ClinicalSLMManifestError as exc:
        raise CliError(
            f"Clinical SLM manifest rejected; reason code {exc.code}",
            code=MANIFEST_INVALID_CODE,
            exit_code=EXIT_ERROR,
        ) from exc
    except (OSError, ValueError) as exc:
        raise CliError(
            "Clinical SLM manifest could not be read",
            code=MANIFEST_INVALID_CODE,
            exit_code=EXIT_ERROR,
        ) from exc

    verification = _verification_report(package_dir, manifest)
    capabilities = _capability_report(package_dir, args.tasks)
    memory = _memory_report(
        package_dir,
        memory_budget=args.memory_budget,
        headroom_bytes=args.headroom_bytes,
    )

    reason_codes = sorted(
        {
            *verification["reason_codes"],
            *capabilities["reason_codes"],
            *memory["reason_codes"],
        }
    )
    passed = (
        bool(verification["verified"])
        and bool(capabilities["supported"])
        and bool(memory["ready"])
    )
    payload: dict[str, Any] = {
        "verdict": "pass" if passed else "fail",
        "manifest": {
            "component_count": manifest.component_count,
            "manifest_digest": manifest.manifest_digest,
        },
        "verification": verification,
        "capabilities": capabilities,
        "memory": memory,
        "reason_codes": reason_codes,
        "network": {"mandatory": False},
    }
    emit(args, payload, human=_render_summary(payload))
    return EXIT_OK if passed else EXIT_ERROR


def _secure_local_reads_available() -> bool:
    """Report whether the platform can hash package components securely."""

    return bool(getattr(manifest_module, "_HAS_SECURE_LOCAL_READ", False))


def _verification_report(
    package_dir: Path,
    manifest: Any,
) -> dict[str, Any]:
    try:
        result = verify_clinical_slm_package(package_dir, manifest)
    except ClinicalSLMManifestError as exc:
        return {"verified": False, "reason_codes": [exc.code]}
    except (OSError, ValueError):
        return {"verified": False, "reason_codes": ["package_invalid"]}
    return dict(result.to_dict())


def _capability_report(
    package_dir: Path,
    tasks: list[str],
) -> dict[str, Any]:
    try:
        report = probe_clinical_slm_capabilities(
            package_dir,
            required_capabilities=tuple(tasks),
        )
    except ClinicalSLMCapabilityError as exc:
        if exc.code == "unknown_capability":
            raise CliError(
                "Unknown capability requested; name a declared capability or task",
                code=UNKNOWN_TASK_CODE,
                exit_code=EXIT_USAGE,
            ) from exc
        raise CliError(
            f"Clinical SLM capability metadata rejected; reason code {exc.code}",
            code=CAPABILITY_METADATA_CODE,
            exit_code=EXIT_ERROR,
        ) from exc
    except (OSError, ValueError) as exc:
        raise CliError(
            "Clinical SLM capability metadata could not be read",
            code=CAPABILITY_METADATA_CODE,
            exit_code=EXIT_ERROR,
        ) from exc
    return dict(report.to_dict())


def _memory_report(
    package_dir: Path,
    *,
    memory_budget: int,
    headroom_bytes: int,
) -> dict[str, Any]:
    try:
        report = preflight_clinical_slm_memory(
            package_dir,
            memory_budget_bytes=memory_budget,
            headroom_bytes=headroom_bytes,
        )
    except ClinicalSLMMemoryError as exc:
        package_fault = exc.code in _ARTIFACT_REASON_CODES
        raise CliError(
            f"Clinical SLM memory preflight rejected its inputs; "
            f"reason code {exc.code}",
            code=PACKAGE_INVALID_CODE if package_fault else MEMORY_PROFILE_CODE,
            exit_code=EXIT_ERROR if package_fault else EXIT_USAGE,
        ) from exc
    except (OSError, ValueError) as exc:
        raise CliError(
            "Clinical SLM memory metadata could not be read",
            code=PACKAGE_INVALID_CODE,
            exit_code=EXIT_ERROR,
        ) from exc
    return dict(report.to_dict())


def _render_summary(payload: dict[str, Any]) -> str:
    verification = payload["verification"]
    reasons = ",".join(payload["reason_codes"]) or "none"
    return (
        f"slm-verify: {payload['verdict']}; "
        f"components={payload['manifest']['component_count']}; "
        f"verified={str(bool(verification['verified'])).lower()}; "
        f"capabilities={payload['capabilities']['decision']}; "
        f"memory={payload['memory']['decision']}; "
        f"reason_codes={reasons}"
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a whole number of bytes") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number of bytes")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a whole number of bytes") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative number of bytes")
    return parsed


__all__ = [
    "add_slm_verify_command",
    "handle_slm_verify",
]
