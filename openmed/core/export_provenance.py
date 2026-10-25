"""Portable, deterministic provenance for converted model artifacts.

Verification is explicit and offline. These manifests describe input identity
and output integrity; they do not establish publisher trust or sign artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .model_integrity import sha256_file

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REPO_PART = re.compile(r"\w(?:[\w.-]*\w)?\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .+_()\-]{0,127}\Z")
# Keep remote snapshots limited to the conventional PyTorch BERT-family inputs.
# Root-specific names avoid pulling published ONNX/CoreML exports or datasets.
_SOURCE_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "spm.model",
    "sentencepiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "tokenizer.vocab",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "COPYING",
    "COPYING.txt",
)
_COMMON_DISTRIBUTIONS = (
    "openmed",
    "transformers",
    "huggingface-hub",
    "torch",
    "safetensors",
)
_BACKEND_DISTRIBUTIONS = {
    "mlx": ("mlx", "numpy"),
    "onnx": ("onnx", "onnxruntime", "onnxconverter-common"),
    "coreml": ("coremltools",),
}


@dataclass(frozen=True)
class SourceContext:
    """Source identity and the private reference used by all converter loaders.

    ``load_reference`` is internal and is never serialized into provenance.
    Remote resolved contexts carry a commit revision; local contexts carry a
    content digest independent of the input directory's name or location.

    Attributes:
        source_model_id: Public repository ID or opaque local content identity.
        load_reference: Private local snapshot or input directory for loaders.
        revision: Resolved remote commit, or null when unavailable.
        source_digest: Local source tree digest, or null for remote snapshots.
        license: Available scalar license identifier, otherwise null.
    """

    source_model_id: str
    load_reference: str
    revision: str | None = None
    source_digest: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_model_id, str):
            raise ValueError("source model ID must be a string")
        if self.source_model_id.startswith("local:"):
            if (
                not isinstance(self.source_digest, str)
                or not _SHA256.fullmatch(self.source_digest)
                or self.source_model_id != f"local:{self.source_digest}"
                or self.revision is not None
            ):
                raise ValueError("local source identity needs its content digest")
        else:
            _validate_model_id(self.source_model_id)
            if self.source_digest is not None:
                raise ValueError("remote source identity must use a revision")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision)
        ):
            raise ValueError("source revision must be an immutable commit SHA")
        if self.license is not None and _license_value(self.license) != self.license:
            raise ValueError("source license must be a scalar license identifier")


def resolve_source_context(
    model_id: str,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> SourceContext:
    """Resolve one source before loading config, model, or tokenizer.

    Local directories are hashed without network access. A remote repository is
    materialized once by Hugging Face and every loader must use the resulting
    ``load_reference`` so that a mutable branch cannot change between loads.

    Args:
        model_id: Local checkpoint directory or plain Hugging Face repository ID.
        cache_dir: Optional Hugging Face cache location.
        revision: Requested remote revision to resolve.

    Returns:
        One source context shared by all converter loaders.

    Raises:
        ValueError: The input identity or resolved revision is unsafe or invalid.
    """
    local = Path(model_id).expanduser()
    if local.is_symlink():
        raise ValueError("source symlinks are not supported")
    if local.is_dir():
        digest = _source_content_digest(local)
        return SourceContext(
            source_model_id=f"local:{digest}",
            load_reference=str(local.resolve()),
            source_digest=digest,
            license=_available_license(local),
        )
    _validate_model_id(model_id)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=list(_SOURCE_ALLOW_PATTERNS),
        )
    )
    if not snapshot.is_dir() or not _REVISION.fullmatch(snapshot.name):
        raise ValueError("resolved source revision must be an immutable commit SHA")
    # HF snapshots normally contain symlinks to cache blobs. They are valid
    # inputs; the output inventory's symlink prohibition does not apply here.
    return SourceContext(
        source_model_id=model_id,
        load_reference=str(snapshot),
        revision=snapshot.name,
        license=_available_license(snapshot),
    )


def assert_source_unchanged(source: SourceContext) -> None:
    """Reject local source changes between resolution and final manifest writing.

    Args:
        source: The original resolved context used by converter loaders.

    Raises:
        ValueError: Local source files changed or can no longer be inventoried.
    """
    if source.source_digest is None:
        return
    try:
        if not source.load_reference:
            raise ValueError("local source reference is unavailable")
        actual = _source_content_digest(Path(source.load_reference))
    except (OSError, ValueError):
        raise ValueError("local source changed during conversion") from None
    if actual != source.source_digest:
        raise ValueError("local source changed during conversion")


def context_from_metadata(
    model_id: str, config: Mapping[str, Any] | None = None
) -> SourceContext:
    """Build writer context from available metadata without remote resolution.

    Direct writer callers may not have a resolved revision. Their unavailable
    metadata is represented by null; converter entry points should instead pass
    the context from :func:`resolve_source_context`.

    Args:
        model_id: Local source directory or plain repository ID.
        config: Already available source config, without any remote fetch.

    Returns:
        A safe source identity with unavailable metadata represented by null.
    """
    local = Path(model_id).expanduser()
    if local.is_symlink() or local.is_dir():
        return resolve_source_context(model_id)
    _validate_model_id(model_id)
    config = config or {}
    commit = config.get("_commit_hash")
    return SourceContext(
        source_model_id=model_id,
        load_reference=model_id,
        revision=commit
        if isinstance(commit, str) and _REVISION.fullmatch(commit)
        else None,
        license=_license_value(config.get("license")),
    )


def build_provenance(
    root: str | Path,
    *,
    source: SourceContext,
    manifest_filename: str,
    backend: str,
) -> dict[str, Any]:
    """Inventory all output files except the named root manifest.

    Output symlinks and non-regular files are rejected. Toolchain versions come
    from an explicit distribution allowlist without importing backend engines.

    Args:
        root: Converted artifact directory.
        source: Input identity shared by the converter loaders.
        manifest_filename: Basename of the root manifest excluded from inventory.
        backend: One of mlx, onnx, or coreml.

    Returns:
        A versioned source, toolchain, and sorted output file inventory.
    """
    _manifest_name(manifest_filename)
    if backend not in _BACKEND_DISTRIBUTIONS:
        raise ValueError("unsupported provenance backend")
    versions: dict[str, str | None] = {
        "python": ".".join(str(part) for part in sys.version_info[:3])
    }
    for name in (*_COMMON_DISTRIBUTIONS, *_BACKEND_DISTRIBUTIONS[backend]):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1,
        "source": {
            "model_id": source.source_model_id,
            "revision": source.revision,
            "content_digest": source.source_digest,
            "license": source.license,
        },
        "toolchain": versions,
        "files": _inventory(Path(root), exclude=manifest_filename),
    }


def verify_export_provenance(
    root: str | Path, manifest_filename: str
) -> dict[str, Any]:
    """Verify portable provenance and the exact output file set, fully offline.

    Returns the verified provenance object. Raises ``ValueError`` for missing or
    invalid provenance, unsafe paths, unexpected files, or size/digest mismatch.
    Legacy manifests remain readable by their existing readers, but cannot pass
    this explicit verification without provenance.

    Args:
        root: Converted artifact directory.
        manifest_filename: Basename of its producer manifest.

    Returns:
        The verified provenance object.

    Raises:
        ValueError: Provenance is absent, malformed, unsafe, or inconsistent.
    """
    name = _manifest_name(manifest_filename)
    root = Path(root)
    if root.is_symlink() or (root / name).is_symlink():
        raise ValueError("output symlinks are not supported")
    if not root.is_dir() or not (root / name).is_file():
        raise ValueError("export manifest must be a regular file")
    try:
        payload = json.loads(
            (root / name).read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("export manifest must be readable UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("export manifest must be an object")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema_version",
        "source",
        "toolchain",
        "files",
    }:
        raise ValueError("export manifest needs valid provenance")
    if (
        type(provenance["schema_version"]) is not int
        or provenance["schema_version"] != 1
    ):
        raise ValueError("unsupported export provenance schema")
    source = provenance["source"]
    if not isinstance(source, dict) or set(source) != {
        "model_id",
        "revision",
        "content_digest",
        "license",
    }:
        raise ValueError("invalid provenance source")
    if not isinstance(source["model_id"], str):
        raise ValueError("invalid provenance source identity")
    SourceContext(
        source["model_id"],
        "",
        source["revision"],
        source["content_digest"],
        source["license"],
    )
    _validate_toolchain(provenance["toolchain"])
    records = provenance["files"]
    if not isinstance(records, list):
        raise ValueError("provenance files must be a list")
    declared: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("invalid provenance file record")
        relative = _safe_path(record["path"])
        if relative == name or relative in declared:
            raise ValueError("duplicate or self-referencing provenance file")
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ValueError("provenance file size must be a non-negative integer")
        if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(
            record["sha256"]
        ):
            raise ValueError("invalid provenance file digest")
        declared[relative] = record
    actual = {record["path"]: record for record in _inventory(root, exclude=name)}
    if set(actual) != set(declared):
        raise ValueError("export files do not exactly match provenance inventory")
    if actual != declared:
        raise ValueError("export file size or digest does not match provenance")
    return provenance


def write_canonical_manifest(
    root: str | Path,
    manifest_filename: str,
    payload: Mapping[str, Any],
    *,
    source: SourceContext,
    backend: str,
) -> Path:
    """Write a canonical manifest with fresh provenance for the final output tree.

    Args:
        root: Converted artifact directory, created if missing.
        manifest_filename: Basename of the manifest to write.
        payload: Existing backend-specific manifest fields to preserve.
        source: Input identity shared by all converter loaders.
        backend: One of mlx, onnx, or coreml.

    Returns:
        Path to the UTF-8 manifest with a trailing newline.
    """
    name = _manifest_name(manifest_filename)
    root = Path(root)
    if root.is_symlink():
        raise ValueError("output symlinks are not supported")
    root.mkdir(parents=True, exist_ok=True)
    result = dict(payload)
    result["provenance"] = build_provenance(
        root, source=source, manifest_filename=name, backend=backend
    )
    manifest = root / name
    manifest.write_bytes(_canonical_bytes(result))
    return manifest


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _validate_model_id(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("source model ID must be a plain Hugging Face repository ID")
    parts = value.split("/")
    if (
        len(parts) not in {1, 2}
        or any(not _REPO_PART.fullmatch(part) for part in parts)
        or len(parts[-1]) > 96
        or ".." in value
        or "--" in value
        or value.endswith(".git")
    ):
        raise ValueError("source model ID must be a plain Hugging Face repository ID")


def _safe_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("provenance paths must be safe relative POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("provenance paths must be safe relative POSIX paths")
    return value


def _manifest_name(value: str) -> str:
    name = _safe_path(value)
    if "/" in name:
        raise ValueError("manifest filename must be a basename")
    return name


def _source_content_digest(root: Path) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(_inventory(root))).hexdigest()


def _inventory(root: Path, *, exclude: str | None = None) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise ValueError("symlinks are not supported in artifact trees")
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    records: list[dict[str, Any]] = []

    def traversal_error(error: OSError) -> None:
        raise ValueError("artifact tree could not be inventoried") from error

    for directory, directories, files in os.walk(
        root, followlinks=False, onerror=traversal_error
    ):
        parent = Path(directory)
        for name in sorted([*directories, *files]):
            path = parent / name
            if path.is_symlink():
                raise ValueError("symlinks are not supported in artifact trees")
            relative = _safe_path(path.relative_to(root).as_posix())
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("artifact trees must contain only regular files")
            if relative != exclude:
                records.append(
                    {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return sorted(records, key=lambda item: item["path"])


def _license_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _LICENSE.fullmatch(value) else None


def _available_license(root: Path) -> str | None:
    try:
        with (root / "config.json").open(encoding="utf-8") as handle:
            config = json.loads(handle.read(1024**2))
        if isinstance(config, dict):
            value = _license_value(config.get("license"))
            if value is not None:
                return value
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    try:
        with (root / "README.md").open(encoding="utf-8") as handle:
            lines = handle.read(64 * 1024).splitlines()
    except (OSError, UnicodeError):
        return None
    if not lines or lines[0].strip() != "---":
        return None
    end = next(
        (
            index
            for index, line in enumerate(lines[1:], 1)
            if line.strip() in {"---", "..."}
        ),
        None,
    )
    if end is None:
        return None
    for line in lines[1:end]:
        if line.startswith("license:"):
            value = line.partition(":")[2].strip()
            if value.startswith(("'", '"')):
                closing = value.find(value[0], 1)
                if closing < 0:
                    return None
                remainder = value[closing + 1 :].strip()
                if remainder and not remainder.startswith("#"):
                    return None
                value = value[1:closing]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0]
            return _license_value(value)
    return None


def _validate_toolchain(value: Any) -> None:
    allowed = {"python", *_COMMON_DISTRIBUTIONS}
    for distributions in _BACKEND_DISTRIBUTIONS.values():
        allowed.update(distributions)
    if (
        not isinstance(value, dict)
        or not {"python", *_COMMON_DISTRIBUTIONS}.issubset(value)
        or not set(value).issubset(allowed)
        or any(
            version is not None
            and (not isinstance(version, str) or not version or len(version) > 128)
            for version in value.values()
        )
    ):
        raise ValueError("invalid provenance toolchain versions")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate export manifest key")
        result[name] = value
    return result
