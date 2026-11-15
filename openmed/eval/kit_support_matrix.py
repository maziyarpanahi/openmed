"""Frozen OpenMedKit support matrix: loader, validator and Markdown renderer.

The matrix lives in ``openmed/eval/data/kit_support_matrix.json`` beside the
canonical device-tier budgets in :mod:`openmed.eval.tiers`. For every portable
kit surface it declares which combinations of platform, coarse device class,
ABI, runtime pin, execution provider, artifact format, model family, precision
and tier are ``supported``, ``experimental`` or ``unsupported``.

Two rules carry the contract:

* A cell is ``supported`` only when it names at least one evidence record.
  ``"evidence": "pending"`` is allowed and forces the cell to ``experimental``.
* The matrix names coarse device classes only. Device identifiers, serials,
  user data and local paths are rejected by :func:`validate_kit_support_matrix`.

This module only reads JSON and renders text. Coherence with the repository's
package manifests is checked by ``scripts/release/check_kit_support_matrix.py``
because those manifests are not shipped in the wheel.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from openmed.eval.tiers import TIERS

SCHEMA_VERSION: Final = 1
MATRIX_NAME: Final = "openmedkit-support-matrix"
DEFAULT_KIT_SUPPORT_MATRIX: Final = (
    Path(__file__).with_name("data") / "kit_support_matrix.json"
)

SUPPORTED: Final = "supported"
EXPERIMENTAL: Final = "experimental"
UNSUPPORTED: Final = "unsupported"
SUPPORT_STATUSES: Final = (SUPPORTED, EXPERIMENTAL, UNSUPPORTED)
PENDING_EVIDENCE: Final = "pending"
ANY: Final = "*"

DOC_BEGIN_MARKER: Final = "<!-- kit-support-matrix:begin -->"
DOC_END_MARKER: Final = "<!-- kit-support-matrix:end -->"

# Surfaces named by the v4.0 portable-kit epic (#3604, which still says v3.5).
EPIC_SURFACES: Final = (
    "python",
    "swift",
    "android",
    "react-native",
    "web",
    "document-intake",
    "key-store",
    "flutter",
    "electron",
    "tauri",
)
EPIC_SCOPES: Final = ("reference", "P0", "P1", "P2")
SUPPORT_CLAIMS: Final = ("supported", "experimental-only")

CELL_PLATFORMS: Final = frozenset(
    {
        "android",
        "chromium",
        "gecko",
        "ios",
        "linux",
        "mac-catalyst",
        "macos",
        "visionos",
        "watchos",
        "webkit",
        "windows",
    }
)
FLOOR_PLATFORMS: Final = CELL_PLATFORMS | frozenset(
    {"electron", "node", "python", "react-native", "swift-tools", "tauri"}
)
FLOOR_BOUNDS: Final = ("min", "compile", "target")
DEVICE_CLASSES: Final = frozenset(
    {
        "android-device",
        "android-emulator",
        "apple-silicon-mac",
        "apple-vision",
        "apple-watch",
        "desktop-app",
        "desktop-browser",
        "intel-mac",
        "ios-simulator",
        "iphone-ipad",
        "mac-catalyst",
        "mobile-browser",
        "server-or-desktop",
        "visionos-simulator",
        "watchos-simulator",
    }
)
ABIS: Final = frozenset(
    {"any", "arm64", "arm64-v8a", "armeabi-v7a", "wasm32", "x86", "x86_64"}
)
PROVIDERS: Final = frozenset(
    {
        "coreml",
        "cpu",
        "keychain",
        "metal",
        "nnapi",
        "on-device",
        "qnn",
        "secure-enclave",
        "strongbox",
        "tee",
        "wasm-basic",
        "wasm-simd-threads",
        "webgpu",
    }
)
ARTIFACTS: Final = frozenset(
    {
        "coreml",
        "gguf",
        "mlx",
        "onnx",
        "onnx-android",
        "onnx-webgpu",
        "pytorch",
        "transformersjs",
    }
)
PRECISIONS: Final = frozenset({"fp16", "fp32", "int4", "int8"})
RUNTIME_KINDS: Final = ("inference", "ocr", "keystore")
PINNING_MODES: Final = ("exact", "minimum", "major", "unpinned", "system")
EVIDENCE_KINDS: Final = ("ci-lane", "device-run", "release-packet")
EVIDENCE_COVERAGE: Final = (
    "build",
    "contract-parity",
    "runtime-synthetic",
    "runtime-model",
    "memory-budget",
    "device-qualification",
    "signed-release",
)
UNSUPPORTED_REASONS: Final = (
    "below-platform-floor",
    "budget-exceeded",
    "family-not-exported",
    "hardware-unavailable",
    "not-implemented",
    "not-qualified",
    "platform-deprecated",
    "policy-forbidden",
    "runtime-not-bundled",
    "runtime-not-linked",
    "runtime-unavailable",
    "script-unsupported",
)

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "matrix",
        "release",
        "state",
        "frozen_on",
        "sources",
        "surfaces",
        "families",
        "platform_floors",
        "runtime_pins",
        "runtime_skew",
        "evidence",
        "cells",
        "unsupported",
    }
)
_SOURCES_FIELDS: Final = frozenset({"families", "tiers"})
_SURFACE_REQUIRED: Final = frozenset({"id", "name", "epic_scope", "support_claim"})
_SURFACE_OPTIONAL: Final = frozenset({"builds_on"})
_FAMILY_FIELDS: Final = frozenset({"id", "export_matrix_row"})
_FLOOR_FIELDS: Final = frozenset(
    {"id", "surface", "bound", "platform", "version", "sources"}
)
_PIN_FIELDS: Final = frozenset(
    {"id", "runtime", "kind", "surface", "version", "pinning", "sources"}
)
_SKEW_FIELDS: Final = frozenset({"id", "runtime", "pins", "related", "note"})
_EVIDENCE_FIELDS: Final = frozenset({"id", "kind", "covers", "refs", "note"})
_PLATFORM_REQUIRED: Final = frozenset({"name"})
_PLATFORM_OPTIONAL: Final = frozenset({"min", "max"})
_CELL_REQUIRED: Final = frozenset(
    {
        "id",
        "surface",
        "platform",
        "device_class",
        "abi",
        "runtime",
        "provider",
        "artifact",
        "families",
        "precisions",
        "tier",
        "status",
        "evidence",
    }
)
_CELL_OPTIONAL: Final = frozenset({"condition", "related", "note"})
_UNSUPPORTED_REQUIRED: Final = frozenset(
    {
        "id",
        "surface",
        "platform",
        "device_class",
        "abi",
        "runtime",
        "provider",
        "families",
        "precisions",
        "status",
        "reason",
    }
)
_UNSUPPORTED_OPTIONAL: Final = frozenset(
    {"artifact", "tier", "condition", "related", "note"}
)

_ID: Final = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_VERSION: Final = re.compile(r"\d+(?:\.\d+)*\Z")
_EXACT_VERSION: Final = re.compile(r"\d+(?:\.\d+)+\Z")
_MINIMUM_VERSION: Final = re.compile(r">=\d+(?:\.\d+)*\Z")
_MAJOR_VERSION: Final = re.compile(r"\d+\Z")
_RELEASE: Final = re.compile(r"v\d+\.\d+\Z")
_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_ISSUE: Final = re.compile(r"#\d+\Z")
_CONDITION: Final = re.compile(r"[a-z0-9-]+:[a-z0-9.-]+\Z")
_REPO_PATH: Final = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*\Z")
_NOTE_MAX_CHARS: Final = 240

# Value shapes that would identify a physical device, a person or a local
# machine. The matrix may only carry coarse classes, so any match is refused.
_IDENTIFIER_PATTERNS: Final = (
    re.compile(r"\b(?:iPhone|iPad|iPod|Watch|Mac|MacBook\w*|RealityDevice)\d+,\d+\b"),
    re.compile(
        r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{12}\b"
    ),
    re.compile(r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{16}\b"),
    re.compile(r"\b[0-9A-Fa-f]{16,}\b"),
    re.compile(r"\b\d{14,16}\b"),
    re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:^|\s)(?:/Users/|/home/|/private/|/var/|~/|[A-Za-z]:\\)"),
    re.compile(r"\b(?:SM-[A-Z]\d{3,}|GT-[A-Z]\d{3,})\b"),
    re.compile(r"(?i)\b(?:serial|udid|imei|android_id|advertising[_ -]id)\b"),
)


class KitSupportMatrixError(ValueError):
    """Raised when the support matrix cannot be read or fails validation.

    Attributes:
        problems: Value-free problem codes. They name fields and controlled
            matrix identifiers only, never field values.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems) or "kit_support_matrix_invalid")


def load_kit_support_matrix(
    path: Path | str | None = None, *, validate: bool = True
) -> dict[str, Any]:
    """Load the kit support matrix JSON document.

    Args:
        path: Matrix location. Defaults to the packaged
            ``openmed/eval/data/kit_support_matrix.json``.
        validate: When true, run :func:`validate_kit_support_matrix` and raise
            on any problem.

    Returns:
        The decoded matrix document.

    Raises:
        KitSupportMatrixError: If the file cannot be decoded as a JSON object or,
            with ``validate``, if validation reports problems.
    """
    source = Path(path) if path is not None else DEFAULT_KIT_SUPPORT_MATRIX
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise KitSupportMatrixError(("matrix_unreadable",)) from None
    if not isinstance(document, dict):
        raise KitSupportMatrixError(("matrix_not_object",))
    if validate:
        problems = validate_kit_support_matrix(document)
        if problems:
            raise KitSupportMatrixError(problems)
    return document


def tier_names(tiers: Mapping[str, Mapping[str, Any]] = TIERS) -> frozenset[str]:
    """Return canonical tier names plus their named sub-tiers.

    Args:
        tiers: Tier table shaped like :data:`openmed.eval.tiers.TIERS`.

    Returns:
        Every tier and sub-tier name a matrix cell may reference.
    """
    names = set(tiers)
    for row in tiers.values():
        sub_tiers = row.get("sub_tiers")
        if isinstance(sub_tiers, Mapping):
            names.update(sub_tiers)
    return frozenset(names)


def tier_ram_ceiling_mb(
    tier: str, tiers: Mapping[str, Mapping[str, Any]] = TIERS
) -> int | None:
    """Return the resident-RAM ceiling for a tier or sub-tier name.

    Args:
        tier: Tier name such as ``"Tiny"`` or the ``"Nano"`` sub-tier.
        tiers: Tier table shaped like :data:`openmed.eval.tiers.TIERS`.

    Returns:
        The ``ram_mb_max`` budget, or ``None`` for an unknown name.
    """
    row = tiers.get(tier)
    if isinstance(row, Mapping):
        value = row.get("ram_mb_max")
        return value if isinstance(value, int) else None
    for parent in tiers.values():
        sub_tiers = parent.get("sub_tiers")
        if isinstance(sub_tiers, Mapping) and isinstance(sub_tiers.get(tier), Mapping):
            value = sub_tiers[tier].get("ram_mb_max")
            return value if isinstance(value, int) else None
    return None


def version_key(version: str) -> tuple[int, ...]:
    """Return a comparable tuple for a dotted numeric version.

    Trailing zero components are dropped so ``"2"`` and ``"2.0.0"`` compare
    equal. A leading ``>=``, ``^`` or ``~`` is ignored.

    Args:
        version: Dotted numeric version, optionally prefixed by a constraint.

    Returns:
        Integer components without trailing zeros.

    Raises:
        ValueError: If the version is not dotted numeric.
    """
    text = version.strip().lstrip(">=^~").strip()
    if not _VERSION.fullmatch(text):
        raise ValueError("version_not_numeric")
    parts = [int(part) for part in text.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def iter_entries(matrix: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield every supported, experimental and unsupported entry in order.

    Args:
        matrix: A decoded matrix document.

    Yields:
        Cells first, then unsupported combinations.
    """
    for key in ("cells", "unsupported"):
        rows = matrix.get(key)
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, Mapping))


def entries_with_status(
    matrix: Mapping[str, Any], status: str, *, surface: str | None = None
) -> list[Mapping[str, Any]]:
    """Return entries with one support status, optionally for one surface.

    Args:
        matrix: A decoded matrix document.
        status: One of :data:`SUPPORT_STATUSES`.
        surface: Optional surface identifier filter.

    Returns:
        Matching entries in document order.
    """
    return [
        entry
        for entry in iter_entries(matrix)
        if entry.get("status") == status
        and (surface is None or entry.get("surface") == surface)
    ]


def validate_kit_support_matrix(
    matrix: Mapping[str, Any],
    *,
    tiers: Mapping[str, Mapping[str, Any]] = TIERS,
) -> tuple[str, ...]:
    """Validate the matrix structure, evidence rule and privacy boundary.

    The check is self-contained: it needs no repository checkout. Problem codes
    are ``code`` or ``code:<matrix id>`` and never repeat field values.

    Args:
        matrix: A decoded matrix document.
        tiers: Tier table used to resolve tier names.

    Returns:
        Sorted, de-duplicated problem codes. Empty means valid.
    """
    problems: set[str] = set()
    if not isinstance(matrix, Mapping):
        return ("matrix_not_object",)

    _check_fields(problems, matrix, "matrix", _TOP_LEVEL_FIELDS, frozenset())
    _scan_identifiers(problems, matrix, "matrix")
    if matrix.get("schema_version") != SCHEMA_VERSION:
        problems.add("schema_version_unsupported")
    if matrix.get("matrix") != MATRIX_NAME:
        problems.add("matrix_name_invalid")
    if not _matches(_RELEASE, matrix.get("release")):
        problems.add("release_invalid")
    if matrix.get("state") not in {"frozen", "draft"}:
        problems.add("state_invalid")
    if not _matches(_DATE, matrix.get("frozen_on")):
        problems.add("frozen_on_invalid")
    sources = matrix.get("sources")
    if isinstance(sources, Mapping):
        _check_fields(problems, sources, "sources", _SOURCES_FIELDS, frozenset())
    else:
        problems.add("sources_invalid")

    surfaces = _rows(problems, matrix, "surfaces")
    families = _rows(problems, matrix, "families")
    floors = _rows(problems, matrix, "platform_floors")
    pins = _rows(problems, matrix, "runtime_pins")
    skews = _rows(problems, matrix, "runtime_skew")
    evidence = _rows(problems, matrix, "evidence")
    cells = _rows(problems, matrix, "cells")
    unsupported = _rows(problems, matrix, "unsupported")

    surface_ids = _ids(problems, surfaces, "surface")
    family_ids = _ids(problems, families, "family")
    _ids(problems, floors, "floor")
    pin_ids = _ids(problems, pins, "pin")
    _ids(problems, skews, "skew")
    evidence_ids = _ids(problems, evidence, "evidence")
    _ids(problems, [*cells, *unsupported], "entry")
    pins_by_id = {row["id"]: row for row in pins if isinstance(row.get("id"), str)}
    known_tiers = tier_names(tiers)

    if set(surface_ids) != set(EPIC_SURFACES):
        problems.add("surfaces_mismatch_epic")
    for row in surfaces:
        _check_surface(problems, row, surface_ids)
    for row in families:
        _check_fields(problems, row, "family", _FAMILY_FIELDS, frozenset())
        if not _nonempty_str(row.get("export_matrix_row")):
            problems.add(f"family_row_invalid:{_label(row)}")
    for row in floors:
        _check_floor(problems, row, surface_ids)
    for row in pins:
        _check_pin(problems, row, surface_ids)
    for row in skews:
        _check_skew(problems, row, pins_by_id)
    for row in evidence:
        _check_evidence(problems, row)

    used_evidence: set[str] = set()
    for row in cells:
        _check_entry(
            problems,
            row,
            unsupported=False,
            surface_ids=surface_ids,
            family_ids=family_ids,
            pins_by_id=pins_by_id,
            known_tiers=known_tiers,
        )
        refs = row.get("evidence")
        status = row.get("status")
        if refs == PENDING_EVIDENCE:
            if status != EXPERIMENTAL:
                problems.add(f"cell_supported_without_evidence:{_label(row)}")
        elif isinstance(refs, list) and refs and _all_str(refs):
            if len(set(refs)) != len(refs):
                problems.add(f"cell_evidence_duplicate:{_label(row)}")
            for ref in refs:
                if ref in evidence_ids:
                    used_evidence.add(ref)
                else:
                    problems.add(f"cell_evidence_unknown:{_label(row)}")
        else:
            problems.add(f"cell_evidence_invalid:{_label(row)}")
            if status == SUPPORTED:
                problems.add(f"cell_supported_without_evidence:{_label(row)}")
    for row in unsupported:
        _check_entry(
            problems,
            row,
            unsupported=True,
            surface_ids=surface_ids,
            family_ids=family_ids,
            pins_by_id=pins_by_id,
            known_tiers=known_tiers,
        )
    for evidence_id in evidence_ids:
        if evidence_id not in used_evidence:
            problems.add(f"evidence_unused:{evidence_id}")

    for row in surfaces:
        surface = row.get("id")
        if not isinstance(surface, str):
            continue
        supported = entries_with_status(matrix, SUPPORTED, surface=surface)
        experimental = entries_with_status(matrix, EXPERIMENTAL, surface=surface)
        if not supported and not experimental:
            problems.add(f"surface_without_cells:{surface}")
        if not entries_with_status(matrix, UNSUPPORTED, surface=surface):
            problems.add(f"surface_without_unsupported:{surface}")
        claim = row.get("support_claim")
        if (claim == "supported") != bool(supported):
            problems.add(f"surface_claim_mismatch:{surface}")
    return tuple(sorted(problems))


def render_kit_support_matrix_markdown(
    matrix: Mapping[str, Any],
    *,
    tiers: Mapping[str, Mapping[str, Any]] = TIERS,
) -> str:
    """Render the generated documentation block for a valid matrix.

    The output is deterministic and contains no begin/end markers; the
    repository script places it between :data:`DOC_BEGIN_MARKER` and
    :data:`DOC_END_MARKER` in ``docs/openmedkit-support-matrix.md``.

    Args:
        matrix: A validated matrix document.
        tiers: Tier table used for RAM ceilings.

    Returns:
        Markdown text ending with a newline.
    """
    pins = {row["id"]: row for row in matrix["runtime_pins"]}
    lines: list[str] = [
        "<!-- Generated from openmed/eval/data/kit_support_matrix.json by",
        "     scripts/release/check_kit_support_matrix.py --write-docs.",
        "     Edit the JSON, then regenerate; do not edit this block by hand. -->",
        "",
        f"Matrix `{matrix['matrix']}` schema {matrix['schema_version']}, release "
        f"{matrix['release']}, state **{matrix['state']}** on {matrix['frozen_on']}.",
        "",
        "## Surfaces",
        "",
        "| Surface | Epic scope | Claim | Supported | Experimental | Unsupported |",
        "|---|---|---|---:|---:|---:|",
    ]
    for surface in matrix["surfaces"]:
        counts = [
            len(entries_with_status(matrix, status, surface=surface["id"]))
            for status in SUPPORT_STATUSES
        ]
        lines.append(
            f"| {_cell(surface['name'])} (`{surface['id']}`) | "
            f"{surface['epic_scope']} | {surface['support_claim']} | "
            + " | ".join(str(count) for count in counts)
            + " |"
        )

    lines += [
        "",
        "## Platform floors and SDK bounds",
        "",
        "| Floor | Surface | Bound | Platform | Version | Sources |",
        "|---|---|---|---|---|---|",
    ]
    for floor in matrix["platform_floors"]:
        lines.append(
            f"| `{floor['id']}` | `{floor['surface']}` | {floor['bound']} | "
            f"{floor['platform']} | {floor['version']} | "
            f"{_paths(floor['sources'])} |"
        )

    lines += [
        "",
        "## Runtime pins",
        "",
        "| Pin | Runtime | Kind | Surface | Version | Pinning | Sources |",
        "|---|---|---|---|---|---|---|",
    ]
    for pin in matrix["runtime_pins"]:
        lines.append(
            f"| `{pin['id']}` | {pin['runtime']} | {pin['kind']} | "
            f"`{pin['surface']}` | {_cell(pin['version'])} | {pin['pinning']} | "
            f"{_paths(pin['sources']) or 'operating system'} |"
        )

    lines += [
        "",
        "## Runtime-version skew",
        "",
        "Pins of the same runtime that differ between kits are listed here "
        "explicitly. They are not aligned by this matrix.",
        "",
        "| Skew | Runtime | Pins | Related | Note |",
        "|---|---|---|---|---|",
    ]
    for skew in matrix["runtime_skew"]:
        members = "; ".join(
            f"`{pin_id}` {_cell(pins[pin_id]['version'])}" for pin_id in skew["pins"]
        )
        lines.append(
            f"| `{skew['id']}` | {skew['runtime']} | {members} | "
            f"{_related(skew['related'])} | {_cell(skew['note'])} |"
        )

    for surface in matrix["surfaces"]:
        surface_id = surface["id"]
        lines += ["", f"## {surface['name']}", ""]
        for status, heading in (
            (SUPPORTED, "Supported"),
            (EXPERIMENTAL, "Experimental"),
            (UNSUPPORTED, "Unsupported combinations"),
        ):
            rows = entries_with_status(matrix, status, surface=surface_id)
            lines += [f"### {heading}", ""]
            if not rows:
                lines += [_empty_status_sentence(status), ""]
                continue
            if status == UNSUPPORTED:
                lines += [
                    "| Entry | Platform | Device class | ABI | Runtime | Provider "
                    "| Families | Precision | Condition | Reason | Related | Note |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|",
                ]
                for row in rows:
                    lines.append(
                        f"| `{row['id']}` | {_platform(row['platform'])} | "
                        f"{_word(row['device_class'])} | {_word(row['abi'])} | "
                        f"{_runtime(row['runtime'], pins)} | "
                        f"{_word(row['provider'])} | {_list(row['families'])} | "
                        f"{_list(row['precisions'], sep='/')} | "
                        f"{_word(row.get('condition'))} | {row['reason']} | "
                        f"{_related(row.get('related', []))} | "
                        f"{_cell(row.get('note', ''))} |"
                    )
            else:
                lines += [
                    "| Cell | Platform | Device class | ABI | Runtime | Provider "
                    "| Artifact | Families | Precision | Tier (RAM ceiling) "
                    "| Condition | Evidence | Related | Note |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
                ]
                for row in rows:
                    lines.append(
                        f"| `{row['id']}` | {_platform(row['platform'])} | "
                        f"{_word(row['device_class'])} | {_word(row['abi'])} | "
                        f"{_runtime(row['runtime'], pins)} | "
                        f"{_word(row['provider'])} | {_word(row['artifact'])} | "
                        f"{_list(row['families'])} | "
                        f"{_list(row['precisions'], sep='/')} | "
                        f"{_tier(row['tier'], tiers)} | "
                        f"{_word(row.get('condition'))} | "
                        f"{_evidence(row['evidence'])} | "
                        f"{_related(row.get('related', []))} | "
                        f"{_cell(row.get('note', ''))} |"
                    )
            lines.append("")
        if lines[-1] == "":
            lines.pop()

    lines += [
        "",
        "## Evidence records",
        "",
        "| Evidence | Kind | Covers | References | Note |",
        "|---|---|---|---|---|",
    ]
    for record in matrix["evidence"]:
        lines.append(
            f"| `{record['id']}` | {record['kind']} | "
            f"{', '.join(record['covers'])} | {_paths(record['refs'])} | "
            f"{_cell(record['note'])} |"
        )

    lines += [
        "",
        "## Model families",
        "",
        "| Family | Export matrix row |",
        "|---|---|",
    ]
    for family in matrix["families"]:
        lines.append(f"| `{family['id']}` | {_cell(family['export_matrix_row'])} |")
    return "\n".join(lines) + "\n"


def _empty_status_sentence(status: str) -> str:
    if status == SUPPORTED:
        return (
            "None. No cell on this surface has evidence and an in-scope "
            "declaration at freeze."
        )
    if status == EXPERIMENTAL:
        return "None."
    return "None declared."


def _platform(platform: Mapping[str, Any]) -> str:
    name = platform.get("name")
    if name == ANY:
        return "any"
    low = platform.get("min")
    high = platform.get("max")
    if low and high:
        return f"{name} {low}" if low == high else f"{name} {low}–{high}"
    if low:
        return f"{name} {low}+"
    if high:
        return f"{name} ≤ {high}"
    return str(name)


def _runtime(pin_id: str, pins: Mapping[str, Mapping[str, Any]]) -> str:
    if pin_id == ANY:
        return "any"
    pin = pins[pin_id]
    return f"{pin['runtime']} {_cell(pin['version'])}"


def _tier(tier: str | None, tiers: Mapping[str, Mapping[str, Any]]) -> str:
    if tier is None:
        return "—"
    if tier == ANY:
        return "any"
    ceiling = tier_ram_ceiling_mb(tier, tiers)
    return f"{tier} (≤ {ceiling} MB)" if ceiling is not None else tier


def _evidence(refs: Any) -> str:
    if refs == PENDING_EVIDENCE:
        return "pending"
    return ", ".join(f"`{ref}`" for ref in refs)


def _list(values: Sequence[str], *, sep: str = ", ") -> str:
    if not values:
        return "—"
    if list(values) == [ANY]:
        return "any"
    return sep.join(values)


def _word(value: str | None) -> str:
    if value is None:
        return "—"
    if value == ANY:
        return "any"
    return value


def _related(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "—"


def _paths(values: Sequence[str]) -> str:
    return "<br>".join(f"`{value}`" for value in values)


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _all_str(values: Sequence[Any]) -> bool:
    return all(isinstance(value, str) for value in values)


def _label(row: Mapping[str, Any]) -> str:
    value = row.get("id")
    return value if _matches(_ID, value) else "invalid-id"


def _rows(
    problems: set[str], matrix: Mapping[str, Any], key: str
) -> list[Mapping[str, Any]]:
    rows = matrix.get(key)
    if not isinstance(rows, list):
        problems.add(f"{key}_invalid")
        return []
    valid = [row for row in rows if isinstance(row, Mapping)]
    if len(valid) != len(rows):
        problems.add(f"{key}_entry_not_object")
    return valid


def _ids(problems: set[str], rows: Sequence[Mapping[str, Any]], kind: str) -> list[str]:
    seen: list[str] = []
    for row in rows:
        value = row.get("id")
        if not _matches(_ID, value):
            problems.add(f"{kind}_id_invalid")
            continue
        if value in seen:
            problems.add(f"{kind}_id_duplicate:{value}")
            continue
        seen.append(value)
    return seen


def _check_fields(
    problems: set[str],
    row: Mapping[str, Any],
    context: str,
    required: frozenset[str],
    optional: frozenset[str],
) -> None:
    keys = set(row)
    label = _label(row) if "id" in required else context
    for key in sorted(required - keys):
        problems.add(f"{context}_field_missing:{label}:{key}")
    if keys - required - optional:
        problems.add(f"{context}_field_unknown:{label}")


def _scan_identifiers(problems: set[str], value: Any, location: str) -> None:
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _IDENTIFIER_PATTERNS):
            problems.add(f"matrix_identifier_value:{location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = key if isinstance(key, str) and _ID.fullmatch(key) else "key"
            if not isinstance(key, str) or any(
                pattern.search(key) for pattern in _IDENTIFIER_PATTERNS
            ):
                problems.add(f"matrix_identifier_value:{location}")
            _scan_identifiers(problems, item, f"{location}.{name}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_identifiers(problems, item, f"{location}[{index}]")


def _check_surface(
    problems: set[str], row: Mapping[str, Any], surface_ids: Sequence[str]
) -> None:
    _check_fields(problems, row, "surface", _SURFACE_REQUIRED, _SURFACE_OPTIONAL)
    label = _label(row)
    if not _nonempty_str(row.get("name")):
        problems.add(f"surface_name_invalid:{label}")
    if row.get("epic_scope") not in EPIC_SCOPES:
        problems.add(f"surface_scope_invalid:{label}")
    if row.get("support_claim") not in SUPPORT_CLAIMS:
        problems.add(f"surface_claim_invalid:{label}")
    builds_on = row.get("builds_on", [])
    if not isinstance(builds_on, list) or any(
        item not in surface_ids or item == row.get("id") for item in builds_on
    ):
        problems.add(f"surface_builds_on_invalid:{label}")


def _check_floor(
    problems: set[str], row: Mapping[str, Any], surface_ids: Sequence[str]
) -> None:
    _check_fields(problems, row, "floor", _FLOOR_FIELDS, frozenset())
    label = _label(row)
    if row.get("surface") not in surface_ids:
        problems.add(f"floor_surface_unknown:{label}")
    if row.get("bound") not in FLOOR_BOUNDS:
        problems.add(f"floor_bound_invalid:{label}")
    if row.get("platform") not in FLOOR_PLATFORMS:
        problems.add(f"floor_platform_invalid:{label}")
    if not _matches(_VERSION, row.get("version")):
        problems.add(f"floor_version_invalid:{label}")
    if not _valid_paths(row.get("sources"), allow_empty=False):
        problems.add(f"floor_sources_invalid:{label}")


def _check_pin(
    problems: set[str], row: Mapping[str, Any], surface_ids: Sequence[str]
) -> None:
    _check_fields(problems, row, "pin", _PIN_FIELDS, frozenset())
    label = _label(row)
    if not _matches(_ID, row.get("runtime")):
        problems.add(f"pin_runtime_invalid:{label}")
    if row.get("kind") not in RUNTIME_KINDS:
        problems.add(f"pin_kind_invalid:{label}")
    if row.get("surface") not in surface_ids:
        problems.add(f"pin_surface_unknown:{label}")
    pinning = row.get("pinning")
    version = row.get("version")
    shapes = {
        "exact": lambda value: _matches(_EXACT_VERSION, value),
        "minimum": lambda value: _matches(_MINIMUM_VERSION, value),
        "major": lambda value: _matches(_MAJOR_VERSION, value),
        "unpinned": lambda value: value == "unpinned",
        "system": lambda value: value == "system",
    }
    check = shapes.get(pinning) if isinstance(pinning, str) else None
    if check is None:
        problems.add(f"pin_pinning_invalid:{label}")
    elif not check(version):
        problems.add(f"pin_version_invalid:{label}")
    if not _valid_paths(row.get("sources"), allow_empty=pinning == "system"):
        problems.add(f"pin_sources_invalid:{label}")
    elif pinning == "system" and row.get("sources"):
        problems.add(f"pin_sources_invalid:{label}")


def _check_skew(
    problems: set[str],
    row: Mapping[str, Any],
    pins_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    _check_fields(problems, row, "skew", _SKEW_FIELDS, frozenset())
    label = _label(row)
    members = row.get("pins")
    if (
        not isinstance(members, list)
        or len(members) < 2
        or len(set(members)) != len(members)
        or any(
            member not in pins_by_id
            or pins_by_id[member].get("runtime") != row.get("runtime")
            for member in members
        )
    ):
        problems.add(f"skew_pins_invalid:{label}")
    if not _valid_related(row.get("related")):
        problems.add(f"skew_related_invalid:{label}")
    if not _valid_note(row.get("note"), required=True):
        problems.add(f"skew_note_invalid:{label}")


def _check_evidence(problems: set[str], row: Mapping[str, Any]) -> None:
    _check_fields(problems, row, "evidence", _EVIDENCE_FIELDS, frozenset())
    label = _label(row)
    if row.get("kind") not in EVIDENCE_KINDS:
        problems.add(f"evidence_kind_invalid:{label}")
    covers = row.get("covers")
    if (
        not isinstance(covers, list)
        or not covers
        or any(item not in EVIDENCE_COVERAGE for item in covers)
    ):
        problems.add(f"evidence_covers_invalid:{label}")
    if not _valid_paths(row.get("refs"), allow_empty=False):
        problems.add(f"evidence_refs_invalid:{label}")
    if not _valid_note(row.get("note"), required=True):
        problems.add(f"evidence_note_invalid:{label}")


def _check_entry(
    problems: set[str],
    row: Mapping[str, Any],
    *,
    unsupported: bool,
    surface_ids: Sequence[str],
    family_ids: Sequence[str],
    pins_by_id: Mapping[str, Mapping[str, Any]],
    known_tiers: frozenset[str],
) -> None:
    kind = "unsupported" if unsupported else "cell"
    if unsupported:
        _check_fields(problems, row, kind, _UNSUPPORTED_REQUIRED, _UNSUPPORTED_OPTIONAL)
    else:
        _check_fields(problems, row, kind, _CELL_REQUIRED, _CELL_OPTIONAL)
    label = _label(row)
    allow_any = unsupported

    if row.get("surface") not in surface_ids:
        problems.add(f"{kind}_surface_unknown:{label}")
    status = row.get("status")
    if unsupported and status != UNSUPPORTED:
        problems.add(f"{kind}_status_invalid:{label}")
    if not unsupported and status not in {SUPPORTED, EXPERIMENTAL}:
        problems.add(f"{kind}_status_invalid:{label}")

    platform = row.get("platform")
    if not isinstance(platform, Mapping):
        problems.add(f"{kind}_platform_invalid:{label}")
    else:
        _check_fields(
            problems,
            platform,
            f"{kind}_platform",
            _PLATFORM_REQUIRED,
            _PLATFORM_OPTIONAL,
        )
        name = platform.get("name")
        if name not in CELL_PLATFORMS and not (allow_any and name == ANY):
            problems.add(f"{kind}_platform_invalid:{label}")
        bounds = [platform.get("min"), platform.get("max")]
        if any(value is not None and not _matches(_VERSION, value) for value in bounds):
            problems.add(f"{kind}_platform_version_invalid:{label}")
        elif all(value is not None for value in bounds) and version_key(
            bounds[0]
        ) > version_key(bounds[1]):
            problems.add(f"{kind}_platform_version_invalid:{label}")

    device_class = row.get("device_class")
    if device_class not in DEVICE_CLASSES and not (allow_any and device_class == ANY):
        problems.add(f"{kind}_device_class_invalid:{label}")
    abi = row.get("abi")
    if abi not in ABIS and not (allow_any and abi == ANY):
        problems.add(f"{kind}_abi_invalid:{label}")
    provider = row.get("provider")
    if provider not in PROVIDERS and not (allow_any and provider == ANY):
        problems.add(f"{kind}_provider_invalid:{label}")

    runtime = row.get("runtime")
    pin = pins_by_id.get(runtime) if isinstance(runtime, str) else None
    if pin is None and not (allow_any and runtime == ANY):
        problems.add(f"{kind}_runtime_unknown:{label}")
    runtime_kind = pin.get("kind") if pin is not None else None
    inference = runtime_kind == "inference" or runtime == ANY

    families = row.get("families")
    precisions = row.get("precisions")
    artifact = row.get("artifact")
    tier = row.get("tier")
    # A wildcard runtime may describe a non-model combination (for example a
    # key store that does not exist on a surface), so empty lists are allowed.
    allow_empty = runtime == ANY
    if inference:
        if not (allow_empty and families == []) and not _valid_members(
            families, family_ids, allow_any=allow_any
        ):
            problems.add(f"{kind}_families_invalid:{label}")
        if not (allow_empty and precisions == []) and not _valid_members(
            precisions, PRECISIONS, allow_any=allow_any
        ):
            problems.add(f"{kind}_precisions_invalid:{label}")
        if artifact is None and not unsupported:
            problems.add(f"{kind}_artifact_invalid:{label}")
        if artifact is not None and artifact not in ARTIFACTS:
            problems.add(f"{kind}_artifact_invalid:{label}")
        if tier is None and not unsupported:
            problems.add(f"{kind}_tier_unknown:{label}")
        if tier is not None and tier not in known_tiers and tier != ANY:
            problems.add(f"{kind}_tier_unknown:{label}")
        if tier == ANY and not unsupported:
            problems.add(f"{kind}_tier_unknown:{label}")
    elif runtime_kind is not None:
        if families != [] or precisions != []:
            problems.add(f"{kind}_families_invalid:{label}")
        if artifact is not None or tier is not None:
            problems.add(f"{kind}_artifact_invalid:{label}")

    condition = row.get("condition")
    if condition is not None and not _matches(_CONDITION, condition):
        problems.add(f"{kind}_condition_invalid:{label}")
    if not _valid_related(row.get("related", [])):
        problems.add(f"{kind}_related_invalid:{label}")
    if not _valid_note(row.get("note", ""), required=False):
        problems.add(f"{kind}_note_invalid:{label}")
    if unsupported and row.get("reason") not in UNSUPPORTED_REASONS:
        problems.add(f"{kind}_reason_invalid:{label}")


def _valid_members(values: Any, allowed: Iterable[str], *, allow_any: bool) -> bool:
    if not isinstance(values, list) or not values or not _all_str(values):
        return False
    if allow_any and values == [ANY]:
        return True
    allowed_set = set(allowed)
    return len(set(values)) == len(values) and all(
        value in allowed_set for value in values
    )


def _valid_paths(values: Any, *, allow_empty: bool) -> bool:
    if not isinstance(values, list) or (not values and not allow_empty):
        return False
    return all(
        isinstance(value, str)
        and _REPO_PATH.fullmatch(value) is not None
        and ".." not in value.split("/")
        and not value.startswith("/")
        for value in values
    )


def _valid_related(values: Any) -> bool:
    return isinstance(values, list) and all(_matches(_ISSUE, value) for value in values)


def _valid_note(value: Any, *, required: bool) -> bool:
    if not isinstance(value, str):
        return False
    if required and not value.strip():
        return False
    return len(value) <= _NOTE_MAX_CHARS and "\n" not in value


__all__ = [
    "ABIS",
    "ANY",
    "ARTIFACTS",
    "CELL_PLATFORMS",
    "DEFAULT_KIT_SUPPORT_MATRIX",
    "DEVICE_CLASSES",
    "DOC_BEGIN_MARKER",
    "DOC_END_MARKER",
    "EPIC_SCOPES",
    "EPIC_SURFACES",
    "EVIDENCE_COVERAGE",
    "EVIDENCE_KINDS",
    "EXPERIMENTAL",
    "FLOOR_BOUNDS",
    "FLOOR_PLATFORMS",
    "KitSupportMatrixError",
    "MATRIX_NAME",
    "PENDING_EVIDENCE",
    "PINNING_MODES",
    "PRECISIONS",
    "PROVIDERS",
    "RUNTIME_KINDS",
    "SCHEMA_VERSION",
    "SUPPORTED",
    "SUPPORT_CLAIMS",
    "SUPPORT_STATUSES",
    "UNSUPPORTED",
    "UNSUPPORTED_REASONS",
    "entries_with_status",
    "iter_entries",
    "load_kit_support_matrix",
    "render_kit_support_matrix_markdown",
    "tier_names",
    "tier_ram_ceiling_mb",
    "validate_kit_support_matrix",
    "version_key",
]
