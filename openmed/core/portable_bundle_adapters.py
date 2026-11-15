"""Build portable-bundle manifests from existing converter and catalog metadata.

The Core ML, ONNX, and MLX converters each write their own manifest
(``openmed-coreml.json``, ``openmed-onnx.json``, ``openmed-mlx.json``) and the
catalog keeps one row per model in ``models.jsonl``. These adapters project
that metadata into the portable schema in :mod:`openmed.core.portable_bundle`
without silently discarding anything:

* every field that has a portable home is mapped, and conflicting sources are
  refused;
* every field that does not is listed as an :class:`AdapterLoss`; converter
  manifests stay in the bundle as signed ``evidence`` files, so their
  non-portable details (latency, residency, parity, optimization) remain
  covered by the content identity;
* unsupported formats, ambiguous precisions, unmapped licenses, and
  non-portable source identities raise :class:`PortableBundleAdapterError`
  with a controlled reason code instead of guessing.

Adapters read only local files, never contact a network, and never place
local filesystem paths in the manifest. Export-provenance records (path, size,
and digest rows) may be passed in and are cross-checked against the measured
tree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .portable_bundle import (
    PortableBundleError,
    encode_portable_manifest,
    inventory_files,
)

ADAPTER_REASONS = (
    "source_manifest_missing",
    "source_manifest_malformed",
    "source_format_unsupported",
    "source_identity_not_portable",
    "precision_unmapped",
    "precision_ambiguous",
    "license_unmapped",
    "metadata_missing",
    "metadata_conflict",
    "artifact_missing",
    "label_map_malformed",
    "segmenter_invalid",
    "inventory_invalid",
)
"""Controlled reason codes raised by :class:`PortableBundleAdapterError`."""

LOSS_PRESERVED = "preserved_in_converter_manifest"
LOSS_CATALOG_ONLY = "catalog_only"
LOSS_DEFAULTED = "defaulted"
LOSS_UNKNOWN = "unknown"
LOSS_ARTIFACT_SKIPPED = "artifact_format_not_portable"

DEFAULT_APPLE_PLATFORMS = (("ios", "17.0"), ("macos", "14.0"))
"""Minimum Apple OS versions declared by OpenMedKit's Swift package."""

DEFAULT_ANDROID_MIN_SDK = "26"
"""Minimum Android API level declared by the OpenMedKit Android library."""

DEFAULT_SDK_RANGE = ("4.0.0", "5.0.0")
"""Default compatible SDK range for adapter-built manifests."""

_SPDX_ALIASES = {
    "apache-2.0": "apache-2.0",
    "bsd-2-clause": "bsd-2-clause",
    "bsd-3-clause": "bsd-3-clause",
    "cc-by-3.0": "cc-by-3.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc0-1.0": "cc0-1.0",
    "icu": "icu",
    "isc": "isc",
    "mit": "mit",
    "unicode-3.0": "unicode-3.0",
    "unlicense": "unlicense",
}
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_REPO_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
)
_LOCAL_ID = re.compile(r"local:sha256:[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

_PRECISIONS = {
    "float32": "fp32",
    "fp32": "fp32",
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "int8": "int8",
    "int4": "int4",
    "q4": "int4",
}

_ONNX_FORMATS: Mapping[str, tuple[str, tuple[tuple[str, str | None], ...]]] = {
    "onnx": (
        "onnx",
        (
            ("android", DEFAULT_ANDROID_MIN_SDK),
            ("ios", "17.0"),
            ("macos", "14.0"),
            ("linux", None),
            ("windows", None),
            ("web", None),
        ),
    ),
    "onnx-int8": (
        "onnx",
        (
            ("android", DEFAULT_ANDROID_MIN_SDK),
            ("linux", None),
            ("macos", "14.0"),
            ("windows", None),
            ("web", None),
        ),
    ),
    "onnx-android": ("onnx", (("android", DEFAULT_ANDROID_MIN_SDK),)),
    "ort-android": ("ort", (("android", DEFAULT_ANDROID_MIN_SDK),)),
    "webgpu": ("onnx", (("web", None),)),
    "transformersjs": ("transformersjs", (("web", None),)),
}
"""Converter artifact formats with a portable runtime and default platforms."""

_MLX_PRECISIONS = {"mlx-8bit": "int8", "mlx-4bit": "int4", "mlx-2bit": "int2"}

_CATALOG_PORTABLE = frozenset(
    {"repo_id", "family", "task", "languages", "architecture", "license"}
)


class PortableBundleAdapterError(ValueError):
    """Raised when source metadata cannot be mapped without guessing.

    Attributes:
        reason: One of :data:`ADAPTER_REASONS`; the message is the code only.
    """

    def __init__(self, reason: str) -> None:
        """Store a controlled reason code."""

        self.reason = reason if reason in ADAPTER_REASONS else "inventory_invalid"
        super().__init__(self.reason)


@dataclass(frozen=True)
class AdapterLoss:
    """One source field that has no portable home, and why.

    Attributes:
        source_field: Dotted source field name (never a value).
        reason: ``preserved_in_converter_manifest``, ``catalog_only``,
            ``defaulted``, ``unknown``, or ``artifact_format_not_portable``.
    """

    source_field: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible record."""

        return {"reason": self.reason, "source_field": self.source_field}


@dataclass(frozen=True)
class BundleRelease:
    """Publisher-chosen identity and validity window for a bundle.

    Attributes:
        bundle_id: Stable bundle identifier.
        version: ``MAJOR.MINOR.PATCH`` bundle version.
        sequence: Monotonic anti-rollback sequence number.
        issued_at: ``YYYY-MM-DDTHH:MM:SSZ`` start of validity.
        expires_at: ``YYYY-MM-DDTHH:MM:SSZ`` end of validity.
    """

    bundle_id: str
    version: str
    sequence: int
    issued_at: str
    expires_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest ``bundle`` section."""

        return {
            "expires_at": self.expires_at,
            "id": self.bundle_id,
            "issued_at": self.issued_at,
            "sequence": self.sequence,
            "version": self.version,
        }


@dataclass(frozen=True)
class PortableManifestDraft:
    """A mapped manifest plus every explicit loss.

    Attributes:
        manifest: Portable manifest mapping.
        losses: Source fields that were not projected, with reasons.
    """

    manifest: Mapping[str, Any]
    losses: tuple[AdapterLoss, ...] = field(default_factory=tuple)

    def encode(self) -> bytes:
        """Return validated canonical manifest bytes ready to sign.

        Raises:
            PortableBundleError: If the mapped manifest is not valid.
        """

        return encode_portable_manifest(self.manifest)


def model_section_from_catalog(
    entry: Mapping[str, Any],
    *,
    revision: str | None = None,
    license_id: str | None = None,
) -> tuple[dict[str, Any], str, tuple[AdapterLoss, ...]]:
    """Map one ``models.jsonl`` row into the portable ``model`` section.

    Args:
        entry: Catalog row.
        revision: Immutable source commit, if known.
        license_id: SPDX identifier to use when the catalog records a license
            the portable schema cannot express, such as ``other``.

    Returns:
        ``(model_section, license_id, losses)``.

    Raises:
        PortableBundleAdapterError: For missing fields, an unmapped license, or
            a non-portable repository identity.
    """

    if not isinstance(entry, Mapping):
        raise PortableBundleAdapterError("source_manifest_malformed")
    repo_id = entry.get("repo_id")
    if not isinstance(repo_id, str) or _REPO_ID.fullmatch(repo_id) is None:
        raise PortableBundleAdapterError("source_identity_not_portable")
    languages = entry.get("languages")
    if not isinstance(languages, list) or not languages:
        raise PortableBundleAdapterError("metadata_missing")
    model = {
        "architecture": _token(entry.get("architecture"), "metadata_missing"),
        "family": _token(entry.get("family"), "metadata_missing"),
        "id": repo_id,
        "languages": [str(item) for item in languages],
        "revision": _revision(revision),
        "task": _token(entry.get("task"), "metadata_missing"),
    }
    losses = [
        AdapterLoss(f"catalog.{name}", LOSS_CATALOG_ONLY)
        for name in sorted(entry)
        if name not in _CATALOG_PORTABLE
    ]
    if license_id is None:
        mapped_license = _license(entry.get("license"))
    else:
        mapped_license = _license(license_id)
        losses.append(AdapterLoss("catalog.license", LOSS_DEFAULTED))
    if revision is None:
        losses.append(AdapterLoss("model.revision", LOSS_UNKNOWN))
    return model, mapped_license, tuple(losses)


def manifest_from_onnx(
    root: str | Path,
    *,
    release: BundleRelease,
    catalog_entry: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    license_id: str | None = None,
    export_provenance: Mapping[str, Any] | None = None,
    sdks: Sequence[str] = ("openmed-python", "openmedkit-android", "openmedkit-web"),
    skip_unsupported: bool = False,
    decoder: tuple[str, int] | None = None,
    policy_range: tuple[int, int] = (1, 2),
) -> PortableManifestDraft:
    """Map an ``openmed-onnx.json`` export directory into a portable manifest.

    Args:
        root: ONNX export directory.
        release: Bundle identity and validity window.
        catalog_entry: Optional ``models.jsonl`` row for model identity.
        model: Explicit ``model`` section, used when no catalog row exists.
        license_id: SPDX identifier overriding the catalog license.
        export_provenance: Optional export-provenance record whose ``files``
            rows are cross-checked and whose source digest is recorded.
        sdks: SDK names declared compatible with :data:`DEFAULT_SDK_RANGE`.
        skip_unsupported: Record unsupported artifacts as losses instead of
            refusing the export.
        decoder: ``(kind, contract_version)``; defaults to BIO version 1.
        policy_range: Supported privacy-policy versions ``[min, max)``.

    Returns:
        The mapped draft.

    Raises:
        PortableBundleAdapterError: When the export cannot be mapped exactly.
    """

    from openmed.onnx.convert import MANIFEST_FILENAME, validate_onnx_bundle

    base = Path(root)
    source = _load_source(base, MANIFEST_FILENAME, validate_onnx_bundle)
    if source.get("format") != "openmed-onnx":
        raise PortableBundleAdapterError("source_manifest_malformed")
    files = _inventory(base, export_provenance)
    paths = {item["path"] for item in files}
    losses: list[AdapterLoss] = []
    runtimes = []
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PortableBundleAdapterError("source_manifest_malformed")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise PortableBundleAdapterError("source_manifest_malformed")
        source_format = artifact.get("format")
        if source_format not in _ONNX_FORMATS:
            if skip_unsupported:
                losses.append(
                    AdapterLoss(f"artifacts.{index}.format", LOSS_ARTIFACT_SKIPPED)
                )
                continue
            raise PortableBundleAdapterError("source_format_unsupported")
        runtime_format, platforms = _ONNX_FORMATS[str(source_format)]
        entrypoint = _artifact_entry(artifact.get("path"), paths, (".onnx", ".ort"))
        precision = _precision(artifact.get("precision"))
        companions = [
            path
            for path in sorted(paths)
            if path.startswith((f"{entrypoint}_data", f"{entrypoint}.data"))
        ]
        runtime_files = sorted({entrypoint, *companions} | ({"config.json"} & paths))
        runtimes.append(
            _runtime(
                f"{source_format}-{precision}",
                runtime_format,
                precision,
                entrypoint,
                runtime_files,
                platforms,
                files,
            )
        )
        if artifact.get("metadata"):
            losses.append(AdapterLoss(f"artifacts.{index}.metadata", LOSS_PRESERVED))
    if not runtimes:
        raise PortableBundleAdapterError("source_format_unsupported")
    runtimes = _dedupe_runtime_ids(runtimes)
    for name in (
        "dynamic_shapes",
        "max_sequence_length",
        "minimum_opset",
        "operator_fallbacks",
        "optimization",
        "validation",
    ):
        if name in source:
            losses.append(AdapterLoss(name, LOSS_PRESERVED))
    return _assemble(
        base,
        source,
        files=files,
        release=release,
        runtimes=runtimes,
        catalog_entry=catalog_entry,
        model=model,
        license_id=license_id,
        export_provenance=export_provenance,
        sdks=sdks,
        decoder=decoder,
        policy_range=policy_range,
        converter_manifest=MANIFEST_FILENAME,
        losses=losses,
    )


def manifest_from_mlx(
    root: str | Path,
    *,
    release: BundleRelease,
    catalog_entry: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    license_id: str | None = None,
    export_provenance: Mapping[str, Any] | None = None,
    precision: str | None = None,
    sdks: Sequence[str] = ("openmed-python", "openmedkit-swift"),
    decoder: tuple[str, int] | None = None,
    policy_range: tuple[int, int] = (1, 2),
) -> PortableManifestDraft:
    """Map an ``openmed-mlx.json`` artifact directory into a portable manifest.

    Args:
        root: MLX artifact directory.
        release: Bundle identity and validity window.
        catalog_entry: Optional ``models.jsonl`` row for model identity.
        model: Explicit ``model`` section, used when no catalog row exists.
        license_id: SPDX identifier overriding the catalog license.
        export_provenance: Optional export-provenance record.
        precision: Required for ``mlx-fp`` artifacts, whose dtype the MLX
            manifest does not record.
        sdks: SDK names declared compatible with :data:`DEFAULT_SDK_RANGE`.
        decoder: ``(kind, contract_version)``; defaults to BIO version 1.
        policy_range: Supported privacy-policy versions ``[min, max)``.

    Returns:
        The mapped draft.

    Raises:
        PortableBundleAdapterError: When the artifact cannot be mapped exactly.
    """

    from openmed.mlx.artifact import MANIFEST_FILENAME, validate_mlx_bundle

    base = Path(root)
    source = _load_source(base, MANIFEST_FILENAME, validate_mlx_bundle)
    if source.get("format") != "openmed-mlx":
        raise PortableBundleAdapterError("source_manifest_malformed")
    files = _inventory(base, export_provenance)
    paths = {item["path"] for item in files}
    formats = source.get("formats")
    if not isinstance(formats, list) or len(formats) != 1:
        raise PortableBundleAdapterError("source_manifest_malformed")
    source_format = formats[0]
    if source_format in _MLX_PRECISIONS:
        mapped = _MLX_PRECISIONS[source_format]
        if precision is not None and _precision(precision) != mapped:
            raise PortableBundleAdapterError("metadata_conflict")
    elif source_format == "mlx-fp":
        if precision is None:
            raise PortableBundleAdapterError("precision_ambiguous")
        mapped = _precision(precision)
    else:
        raise PortableBundleAdapterError("source_format_unsupported")
    available = source.get("available_weights")
    if not isinstance(available, list) or not available:
        raise PortableBundleAdapterError("artifact_missing")
    weights = [
        _artifact_entry(name, paths, (".safetensors", ".npz")) for name in available
    ]
    preferred = source.get("preferred_weights")
    entrypoint = preferred if preferred in weights else weights[0]
    runtime_files = sorted(set(weights) | ({"config.json"} & paths))
    runtimes = [
        _runtime(
            f"mlx-{mapped}",
            "mlx",
            mapped,
            entrypoint,
            runtime_files,
            DEFAULT_APPLE_PLATFORMS,
            files,
        )
    ]
    losses = [
        AdapterLoss(name, LOSS_PRESERVED)
        for name in (
            "certification",
            "certified",
            "max_sequence_length",
            "prompt_spec",
            "quant_recall_delta",
            "quantization",
            "recall_delta_path",
            "runtime",
        )
        if source.get(name) is not None
    ]
    losses.append(AdapterLoss("platforms", LOSS_DEFAULTED))
    return _assemble(
        base,
        source,
        files=files,
        release=release,
        runtimes=runtimes,
        catalog_entry=catalog_entry,
        model=model,
        license_id=license_id,
        export_provenance=export_provenance,
        sdks=sdks,
        decoder=decoder,
        policy_range=policy_range,
        converter_manifest=MANIFEST_FILENAME,
        losses=losses,
    )


def manifest_from_coreml(
    root: str | Path,
    *,
    release: BundleRelease,
    catalog_entry: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    license_id: str | None = None,
    export_provenance: Mapping[str, Any] | None = None,
    sdks: Sequence[str] = ("openmedkit-swift",),
    decoder: tuple[str, int] | None = None,
    policy_range: tuple[int, int] = (1, 2),
) -> PortableManifestDraft:
    """Map an ``openmed-coreml.json`` conversion directory into a manifest.

    Args:
        root: Core ML conversion output directory.
        release: Bundle identity and validity window.
        catalog_entry: Optional ``models.jsonl`` row for model identity.
        model: Explicit ``model`` section, used when no catalog row exists.
        license_id: SPDX identifier overriding the catalog license.
        export_provenance: Optional export-provenance record.
        sdks: SDK names declared compatible with :data:`DEFAULT_SDK_RANGE`.
        decoder: ``(kind, contract_version)``; defaults to BIO version 1.
        policy_range: Supported privacy-policy versions ``[min, max)``.

    Returns:
        The mapped draft.

    Raises:
        PortableBundleAdapterError: When the conversion cannot be mapped exactly.
    """

    from openmed.coreml.convert import COREML_MANIFEST_FILENAME

    base = Path(root)
    source = _load_source(base, COREML_MANIFEST_FILENAME, None)
    if source.get("format") != "openmed-coreml":
        raise PortableBundleAdapterError("source_manifest_malformed")
    if "family" not in source and isinstance(source.get("source_model_type"), str):
        source = {**source, "family": source["source_model_type"]}
    files = _inventory(base, export_provenance)
    paths = {item["path"] for item in files}
    variants = source.get("variants")
    if not isinstance(variants, list) or not variants:
        raise PortableBundleAdapterError("source_manifest_malformed")
    runtimes = []
    losses: list[AdapterLoss] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping):
            raise PortableBundleAdapterError("source_manifest_malformed")
        package = variant.get("path")
        if not isinstance(package, str) or not package:
            raise PortableBundleAdapterError("source_manifest_malformed")
        prefix = package.rstrip("/") + "/"
        package_files = sorted(path for path in paths if path.startswith(prefix))
        if package in paths:
            package_files = [package]
        if not package_files:
            raise PortableBundleAdapterError("artifact_missing")
        quantization = variant.get("quantization")
        precision = (
            _precision(quantization)
            if quantization not in (None, "none", "")
            else _precision(variant.get("precision"))
        )
        entrypoint = next(
            (
                path
                for candidate in ("Manifest.json", "coremldata.bin")
                for path in package_files
                if path == prefix + candidate
            ),
            package_files[0],
        )
        runtime_files = sorted(set(package_files) | ({"config.json"} & paths))
        runtimes.append(
            _runtime(
                f"coreml-{precision}",
                "coreml",
                precision,
                entrypoint,
                runtime_files,
                DEFAULT_APPLE_PLATFORMS,
                files,
            )
        )
        for name in ("latency_ms", "parity", "residency"):
            if variant.get(name):
                losses.append(AdapterLoss(f"variants.{index}.{name}", LOSS_PRESERVED))
    runtimes = _dedupe_runtime_ids(runtimes)
    for name in (
        "ane_optimization_profile",
        "compute_units",
        "generated_at",
        "max_sequence_length",
    ):
        if name in source:
            losses.append(AdapterLoss(name, LOSS_PRESERVED))
    losses.append(AdapterLoss("platforms", LOSS_DEFAULTED))
    return _assemble(
        base,
        source,
        files=files,
        release=release,
        runtimes=runtimes,
        catalog_entry=catalog_entry,
        model=model,
        license_id=license_id,
        export_provenance=export_provenance,
        sdks=sdks,
        decoder=decoder,
        policy_range=policy_range,
        converter_manifest=COREML_MANIFEST_FILENAME,
        losses=losses,
    )


# --------------------------------------------------------------------------
# Shared assembly
# --------------------------------------------------------------------------


def _load_source(base: Path, name: str, validator: Any) -> Mapping[str, Any]:
    path = base / name
    if path.is_symlink() or not path.is_file():
        raise PortableBundleAdapterError("source_manifest_missing")
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise PortableBundleAdapterError("source_manifest_malformed") from None
    if not isinstance(source, dict):
        raise PortableBundleAdapterError("source_manifest_malformed")
    if validator is not None:
        try:
            validator(base)
        except (OSError, ValueError):
            raise PortableBundleAdapterError("segmenter_invalid") from None
    return source


def _inventory(
    base: Path, export_provenance: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    records = None
    if export_provenance is not None:
        rows = export_provenance.get("files")
        if not isinstance(rows, list):
            raise PortableBundleAdapterError("inventory_invalid")
        records = rows
    try:
        return inventory_files(base, records=records)
    except PortableBundleError:
        raise PortableBundleAdapterError("inventory_invalid") from None


def _assemble(
    base: Path,
    source: Mapping[str, Any],
    *,
    files: list[dict[str, Any]],
    release: BundleRelease,
    runtimes: list[dict[str, Any]],
    catalog_entry: Mapping[str, Any] | None,
    model: Mapping[str, Any] | None,
    license_id: str | None,
    export_provenance: Mapping[str, Any] | None,
    sdks: Sequence[str],
    decoder: tuple[str, int] | None,
    policy_range: tuple[int, int],
    converter_manifest: str,
    losses: list[AdapterLoss],
) -> PortableManifestDraft:
    paths = {item["path"] for item in files}
    revision, source_digest = _provenance(export_provenance)
    model_section, chosen_license, catalog_losses = _model(
        source, catalog_entry, model, revision, license_id
    )
    losses.extend(catalog_losses)
    if export_provenance is None:
        losses.append(AdapterLoss("provenance.source_digest", LOSS_UNKNOWN))

    by_path = {item["path"]: item for item in files}
    tokenizer = None
    tokenizer_source = source.get("tokenizer")
    if not isinstance(tokenizer_source, Mapping) or not tokenizer_source.get("files"):
        losses.append(AdapterLoss("tokenizer", LOSS_UNKNOWN))
    else:
        names = tokenizer_source.get("files") or []
        prefix = str(tokenizer_source.get("path") or ".").strip("/")
        tokenizer_files = sorted(
            {
                name if prefix in {"", "."} else f"{prefix}/{name}"
                for name in names
                if isinstance(name, str)
            }
        )
        if not tokenizer_files or not set(tokenizer_files) <= paths:
            raise PortableBundleAdapterError("artifact_missing")
        for path in tokenizer_files:
            by_path[path]["role"] = "tokenizer"
        tokenizer = {
            "contract_version": 1,
            "files": tokenizer_files,
            "kind": _tokenizer_kind(base, tokenizer_files, losses),
        }

    labels = None
    label_path = source.get("label_map_path")
    if label_path is not None:
        if not isinstance(label_path, str) or label_path not in paths:
            raise PortableBundleAdapterError("artifact_missing")
        by_path[label_path]["role"] = "labels"
        count, scheme = _label_summary(base / label_path)
        labels = {"count": count, "path": label_path, "scheme": scheme}

    licenses: dict[str, list[str]] = {}
    license_files = sorted(path for path in paths if by_path[path]["role"] == "license")
    licenses[chosen_license] = license_files

    segmenter = None
    descriptor = source.get("segmenter")
    if isinstance(descriptor, Mapping):
        resources = descriptor.get("resource_files")
        segmenter_id = _token(
            str(descriptor.get("id") or "").lower(), "segmenter_invalid"
        )
        if not isinstance(resources, list) or not resources:
            raise PortableBundleAdapterError("segmenter_invalid")
        segmenter_files = []
        for resource in resources:
            if not isinstance(resource, Mapping):
                raise PortableBundleAdapterError("segmenter_invalid")
            resource_path = resource.get("path")
            if not isinstance(resource_path, str) or resource_path not in by_path:
                raise PortableBundleAdapterError("artifact_missing")
            if (
                resource.get("sha256") is not None
                and resource.get("sha256") != by_path[resource_path]["sha256"]
            ):
                raise PortableBundleAdapterError("segmenter_invalid")
            by_path[resource_path]["role"] = "segmenter"
            segmenter_files.append(resource_path)
            resource_license = _license(resource.get("license"))
            licenses.setdefault(resource_license, [])
            licenses[resource_license] = sorted(
                {*licenses[resource_license], resource_path}
            )
        segmenter = {"files": sorted(segmenter_files), "id": segmenter_id}

    evidence = []
    if converter_manifest in paths:
        by_path[converter_manifest]["role"] = "evidence"
        evidence.append({"kind": "converter-manifest", "path": converter_manifest})

    if decoder is None:
        decoder = ("bio", 1)
        losses.append(AdapterLoss("decoder", LOSS_DEFAULTED))
    losses.append(AdapterLoss("runtimes.min_memory_bytes", LOSS_UNKNOWN))

    manifest = {
        "bundle": release.to_dict(),
        "calibration": None,
        "compatibility": {
            "sdks": [
                {
                    "max_exclusive": DEFAULT_SDK_RANGE[1],
                    "min": DEFAULT_SDK_RANGE[0],
                    "name": name,
                }
                for name in sdks
            ]
        },
        "critical_extensions": [],
        "decoder": {"contract_version": decoder[1], "kind": decoder[0]},
        "evidence": evidence,
        "extensions": {},
        "files": files,
        "labels": labels,
        "licenses": [
            {"files": items, "id": name} for name, items in sorted(licenses.items())
        ],
        "model": model_section,
        "offsets": {"contract_version": 1, "unit": "unicode_scalar"},
        "policy": {
            "max_version_exclusive": policy_range[1],
            "min_version": policy_range[0],
            "path": None,
        },
        "provenance": {"export_record": None, "source_digest": source_digest},
        "runtimes": runtimes,
        "schema": "openmed.portable_bundle",
        "schema_version": 1,
        "segmenter": segmenter,
        "tokenizer": tokenizer,
    }
    draft = PortableManifestDraft(
        manifest=manifest,
        losses=tuple(sorted(set(losses), key=lambda item: item.source_field)),
    )
    try:
        draft.encode()
    except PortableBundleError as error:
        raise PortableBundleAdapterError(_adapter_reason(error)) from None
    return draft


def _adapter_reason(error: PortableBundleError) -> str:
    if error.reason.value == "runtime_combination_unsupported":
        return "source_format_unsupported"
    if error.reason.value in {"path_unsafe", "duplicate_path", "path_conflict"}:
        return "inventory_invalid"
    return "metadata_missing"


def _model(
    source: Mapping[str, Any],
    catalog_entry: Mapping[str, Any] | None,
    model: Mapping[str, Any] | None,
    revision: str | None,
    license_id: str | None,
) -> tuple[dict[str, Any], str, list[AdapterLoss]]:
    losses: list[AdapterLoss] = []
    known_ids: set[str] = set()
    if catalog_entry is not None:
        section, mapped_license, catalog_losses = model_section_from_catalog(
            catalog_entry, revision=revision, license_id=license_id
        )
        losses.extend(catalog_losses)
        if isinstance(catalog_entry.get("base_model"), str):
            known_ids.add(str(catalog_entry["base_model"]))
    elif model is not None:
        section = dict(model)
        if revision is not None and section.get("revision") is None:
            section["revision"] = revision
        if license_id is None:
            raise PortableBundleAdapterError("license_unmapped")
        mapped_license = _license(license_id)
    else:
        raise PortableBundleAdapterError("metadata_missing")
    known_ids.add(str(section.get("id")))
    source_id = source.get("source_model_id")
    if isinstance(source_id, str) and source_id:
        if (
            _REPO_ID.fullmatch(source_id) is None
            and _LOCAL_ID.fullmatch(source_id) is None
        ):
            # Local filesystem paths are never copied into a manifest.
            losses.append(AdapterLoss("source_model_id", LOSS_UNKNOWN))
        elif source_id not in known_ids:
            raise PortableBundleAdapterError("metadata_conflict")
    source_task = source.get("task")
    if isinstance(source_task, str) and source_task != section.get("task"):
        raise PortableBundleAdapterError("metadata_conflict")
    source_family = source.get("family")
    if (
        isinstance(source_family, str)
        and source_family not in {"", "unknown"}
        and source_family.lower() != str(section.get("architecture", "")).lower()
    ):
        raise PortableBundleAdapterError("metadata_conflict")
    return section, mapped_license, losses


def _provenance(
    export_provenance: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    if export_provenance is None:
        return None, None
    source = export_provenance.get("source")
    if not isinstance(source, Mapping):
        raise PortableBundleAdapterError("inventory_invalid")
    revision = source.get("revision")
    digest = source.get("content_digest")
    if revision is not None and (
        not isinstance(revision, str) or _REVISION.fullmatch(revision) is None
    ):
        raise PortableBundleAdapterError("inventory_invalid")
    if digest is not None and (
        not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
    ):
        raise PortableBundleAdapterError("inventory_invalid")
    return revision, digest


def _runtime(
    runtime_id: str,
    runtime_format: str,
    precision: str,
    entrypoint: str,
    runtime_files: list[str],
    platforms: Sequence[tuple[str, str | None]],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    sizes = {item["path"]: item["size_bytes"] for item in files}
    return {
        "entrypoint": entrypoint,
        "fallback": None,
        "files": runtime_files,
        "format": runtime_format,
        "id": runtime_id,
        "min_memory_bytes": 0,
        "min_storage_bytes": sum(sizes[path] for path in runtime_files),
        "platforms": [
            {"max_os_exclusive": None, "min_os": minimum, "platform": name}
            for name, minimum in platforms
        ],
        "precision": precision,
    }


def _dedupe_runtime_ids(runtimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    for runtime in runtimes:
        base = runtime["id"]
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            runtime["id"] = f"{base}-{count + 1}"
    return runtimes


def _artifact_entry(value: object, paths: set[str], suffixes: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value:
        raise PortableBundleAdapterError("source_manifest_malformed")
    if value in paths:
        return value
    prefix = value.rstrip("/") + "/"
    candidates = sorted(
        path for path in paths if path.startswith(prefix) and path.endswith(suffixes)
    )
    if not candidates:
        raise PortableBundleAdapterError("artifact_missing")
    return candidates[0]


def _precision(value: object) -> str:
    if not isinstance(value, str) or value.lower() not in _PRECISIONS:
        raise PortableBundleAdapterError("precision_unmapped")
    return _PRECISIONS[value.lower()]


def _license(value: object) -> str:
    if not isinstance(value, str):
        raise PortableBundleAdapterError("license_unmapped")
    mapped = _SPDX_ALIASES.get(value.strip().lower())
    if mapped is None:
        raise PortableBundleAdapterError("license_unmapped")
    return mapped


def _token(value: object, reason: str) -> str:
    if not isinstance(value, str):
        raise PortableBundleAdapterError(reason)
    lowered = value.strip().lower().replace(" ", "-")
    if _TOKEN.fullmatch(lowered) is None:
        raise PortableBundleAdapterError(reason)
    return lowered


def _revision(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise PortableBundleAdapterError("source_identity_not_portable")
    return value


def _tokenizer_kind(
    base: Path, tokenizer_files: Sequence[str], losses: list[AdapterLoss]
) -> str:
    if "tokenizer.json" in tokenizer_files:
        try:
            payload = json.loads((base / "tokenizer.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            payload = None
        kind = (
            payload.get("model", {}).get("type") if isinstance(payload, dict) else None
        )
        if isinstance(kind, str) and _TOKEN.fullmatch(kind.lower()):
            return kind.lower()
    if "vocab.txt" in tokenizer_files:
        return "wordpiece"
    losses.append(AdapterLoss("tokenizer.kind", LOSS_UNKNOWN))
    return "unknown"


def _label_summary(path: Path) -> tuple[int, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise PortableBundleAdapterError("label_map_malformed") from None
    if isinstance(payload, Mapping) and isinstance(payload.get("id2label"), Mapping):
        payload = payload["id2label"]
    if not isinstance(payload, Mapping) or not payload:
        raise PortableBundleAdapterError("label_map_malformed")
    labels = [str(item) for item in payload.values()]
    prefixes = {label.split("-", 1)[0] for label in labels if "-" in label}
    if prefixes & {"S", "E"}:
        scheme = "bioes"
    elif prefixes & {"B", "I"}:
        scheme = "bio"
    else:
        scheme = "span"
    return len(labels), scheme


__all__ = [
    "ADAPTER_REASONS",
    "DEFAULT_ANDROID_MIN_SDK",
    "DEFAULT_APPLE_PLATFORMS",
    "DEFAULT_SDK_RANGE",
    "LOSS_ARTIFACT_SKIPPED",
    "LOSS_CATALOG_ONLY",
    "LOSS_DEFAULTED",
    "LOSS_PRESERVED",
    "LOSS_UNKNOWN",
    "AdapterLoss",
    "BundleRelease",
    "PortableBundleAdapterError",
    "PortableManifestDraft",
    "manifest_from_coreml",
    "manifest_from_mlx",
    "manifest_from_onnx",
    "model_section_from_catalog",
]
