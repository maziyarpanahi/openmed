"""Build and validate portable Maple ONNX Runtime bundles.

The published Maple MLX checkpoint is not an ONNX model. This module packages
separately exported ONNX or ORT decoder graphs with the tokenizer and immutable
provenance required by the Android and browser demos. It deliberately performs
no model conversion and never reads or stores clinical input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

MAPLE_BUNDLE_FILENAME = "maple-bundle.json"
MAPLE_BUNDLE_SCHEMA_VERSION = 1
MAPLE_SOURCE_MODEL = "deepgrove/maple-preview"
MAPLE_SOURCE_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
MAPLE_ARCHITECTURE = "MapleForCausalLM"
MAPLE_VOCAB_SIZE = 151_936
MAPLE_MAX_CONTEXT_TOKENS = 131_072
MAPLE_DEFAULT_EOS_TOKEN_IDS = (151_645,)
MAPLE_BUNDLE_RUNTIMES = frozenset({"onnxruntime-mobile", "onnxruntime-web"})

_MAX_BUNDLE_FILES = 512
_MAX_BUNDLE_BYTES = 12 * 1024**3
_MAX_MANIFEST_BYTES = 1024**2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_GRAPH_SUFFIXES = frozenset({".onnx", ".ort"})
_COPY_BUFFER_BYTES = 1024**2


class MapleBundleError(ValueError):
    """Raised when a Maple bundle violates the portable runtime contract."""


@dataclass(frozen=True)
class MapleBundleFile:
    """One integrity-bound payload entry in a Maple model bundle."""

    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized manifest representation."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MapleBundleBuild:
    """Result and immutable metadata for a packaged Maple runtime bundle."""

    bundle_path: Path
    manifest: Mapping[str, Any]

    @property
    def total_size_bytes(self) -> int:
        """Return the declared uncompressed model payload size."""

        return sum(int(item["size_bytes"]) for item in self.manifest["files"])


def build_maple_onnx_bundle(
    source_directory: str | Path,
    output_path: str | Path,
    *,
    prefill_path: str = "decoder_model.ort",
    decode_path: str | None = "decoder_with_past_model.ort",
    tokenizer_path: str = "tokenizer.json",
    extra_files: Iterable[str] = ("tokenizer_config.json", "config.json"),
    runtime: str = "onnxruntime-mobile",
    quantization: str = "qmoe-4bit-blockwise-128",
    source_revision: str = MAPLE_SOURCE_REVISION,
    max_context_tokens: int = 4096,
    max_input_tokens: int = 3072,
    eos_token_ids: Iterable[int] = MAPLE_DEFAULT_EOS_TOKEN_IDS,
) -> MapleBundleBuild:
    """Package exported Maple decoder graphs in a deterministic ZIP container.

    The output uses stored ZIP entries because already-quantized model graphs
    generally compress poorly and mobile import should not need extra scratch
    space. Existing output is rejected to avoid silently replacing a validated
    artifact.
    """

    root = Path(source_directory).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    if not root.is_dir():
        raise FileNotFoundError(f"Maple bundle source directory does not exist: {root}")

    requested_paths = _deduplicate_paths(
        path
        for path in (prefill_path, decode_path, tokenizer_path, *extra_files)
        if path is not None
    )
    file_records = tuple(_record_source_file(root, path) for path in requested_paths)
    manifest: dict[str, Any] = {
        "schema_version": MAPLE_BUNDLE_SCHEMA_VERSION,
        "source_model": MAPLE_SOURCE_MODEL,
        "source_revision": source_revision,
        "architecture": MAPLE_ARCHITECTURE,
        "quantization": quantization,
        "runtime": runtime,
        "tokenizer_path": tokenizer_path,
        "graphs": {
            "prefill_path": prefill_path,
            "decode_path": decode_path,
            "input_ids_name": "input_ids",
            "attention_mask_name": "attention_mask",
            "position_ids_name": "position_ids",
            "logits_name": "logits",
        },
        "cache": (
            {
                "past_input_prefix": "past_key_values.",
                "present_output_prefix": "present.",
            }
            if decode_path is not None
            else None
        ),
        "generation": {
            "eos_token_ids": list(dict.fromkeys(int(item) for item in eos_token_ids)),
            "max_context_tokens": max_context_tokens,
            "max_input_tokens": max_input_tokens,
        },
        "files": [record.to_dict() for record in file_records],
    }
    _validate_manifest(manifest)
    manifest_bytes = _manifest_bytes(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                _reproducible_zip_info(MAPLE_BUNDLE_FILENAME), manifest_bytes
            )
            for record in file_records:
                source_path = _resolve_source_file(root, record.path)
                with source_path.open("rb") as source:
                    with archive.open(
                        _reproducible_zip_info(record.path),
                        mode="w",
                        force_zip64=True,
                    ) as target:
                        while chunk := source.read(_COPY_BUFFER_BYTES):
                            target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return validate_maple_onnx_bundle(destination)


def validate_maple_onnx_bundle(bundle_path: str | Path) -> MapleBundleBuild:
    """Validate manifest shape, entry set, sizes, and SHA-256 checksums."""

    path = Path(bundle_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Maple bundle does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or entries[0].filename != MAPLE_BUNDLE_FILENAME:
                raise MapleBundleError(
                    f"{MAPLE_BUNDLE_FILENAME} must be the first ZIP entry"
                )
            if entries[0].file_size > _MAX_MANIFEST_BYTES:
                raise MapleBundleError(f"{MAPLE_BUNDLE_FILENAME} is too large")
            try:
                manifest = json.loads(archive.read(entries[0]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MapleBundleError(
                    f"{MAPLE_BUNDLE_FILENAME} must be valid UTF-8 JSON"
                ) from exc
            _validate_manifest(manifest)

            declared = {item["path"]: item for item in manifest["files"]}
            payload_entries = entries[1:]
            names = [entry.filename for entry in payload_entries]
            if len(names) != len(set(names)):
                raise MapleBundleError("bundle contains duplicate ZIP entries")
            if set(names) != set(declared):
                raise MapleBundleError(
                    "bundle payload entries do not exactly match the manifest"
                )
            for entry in payload_entries:
                if entry.is_dir():
                    raise MapleBundleError(
                        "bundle payload must not contain directories"
                    )
                expected = declared[entry.filename]
                if entry.file_size != expected["size_bytes"]:
                    raise MapleBundleError(
                        f"bundle file size mismatch: {entry.filename}"
                    )
                digest = hashlib.sha256()
                with archive.open(entry) as source:
                    while chunk := source.read(_COPY_BUFFER_BYTES):
                        digest.update(chunk)
                if digest.hexdigest() != expected["sha256"]:
                    raise MapleBundleError(
                        f"bundle checksum mismatch: {entry.filename}"
                    )
    except zipfile.BadZipFile as exc:
        raise MapleBundleError("Maple bundle must be a valid ZIP archive") from exc
    return MapleBundleBuild(bundle_path=path, manifest=manifest)


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise MapleBundleError("Maple bundle manifest must be a JSON object")
    required_keys = {
        "schema_version",
        "source_model",
        "source_revision",
        "architecture",
        "quantization",
        "runtime",
        "tokenizer_path",
        "graphs",
        "cache",
        "generation",
        "files",
    }
    if set(manifest) != required_keys:
        raise MapleBundleError(
            "Maple bundle manifest keys do not match schema version 1"
        )
    if manifest["schema_version"] != MAPLE_BUNDLE_SCHEMA_VERSION:
        raise MapleBundleError("unsupported Maple bundle schema")
    if manifest["source_model"] != MAPLE_SOURCE_MODEL:
        raise MapleBundleError(f"source_model must be {MAPLE_SOURCE_MODEL}")
    revision = manifest["source_revision"]
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise MapleBundleError(
            "source_revision must be an immutable lowercase commit SHA"
        )
    if manifest["architecture"] != MAPLE_ARCHITECTURE:
        raise MapleBundleError(f"architecture must be {MAPLE_ARCHITECTURE}")
    if manifest["runtime"] not in MAPLE_BUNDLE_RUNTIMES:
        raise MapleBundleError("runtime must be onnxruntime-mobile or onnxruntime-web")
    if (
        not isinstance(manifest["quantization"], str)
        or not manifest["quantization"].strip()
    ):
        raise MapleBundleError("quantization must be a non-empty string")

    graphs = _require_object(manifest, "graphs")
    graph_keys = {
        "prefill_path",
        "decode_path",
        "input_ids_name",
        "attention_mask_name",
        "position_ids_name",
        "logits_name",
    }
    if set(graphs) != graph_keys:
        raise MapleBundleError("graphs keys do not match the Maple decoder contract")
    prefill_path = _validate_graph_path(graphs["prefill_path"], "prefill_path")
    decode_value = graphs["decode_path"]
    decode_path = (
        _validate_graph_path(decode_value, "decode_path")
        if decode_value is not None
        else None
    )
    for name in (
        "input_ids_name",
        "attention_mask_name",
        "position_ids_name",
        "logits_name",
    ):
        if not isinstance(graphs[name], str) or not graphs[name].strip():
            raise MapleBundleError(f"graphs.{name} must be a non-empty string")

    cache = manifest["cache"]
    if decode_path is None:
        if cache is not None:
            raise MapleBundleError("a cache contract requires a decode graph")
    else:
        if not isinstance(cache, dict) or set(cache) != {
            "past_input_prefix",
            "present_output_prefix",
        }:
            raise MapleBundleError("a cached decode graph requires its cache contract")
        if any(not isinstance(value, str) or not value for value in cache.values()):
            raise MapleBundleError("cache tensor prefixes must be non-empty strings")

    tokenizer_path = _validate_relative_path(manifest["tokenizer_path"])
    generation = _require_object(manifest, "generation")
    if set(generation) != {
        "eos_token_ids",
        "max_context_tokens",
        "max_input_tokens",
    }:
        raise MapleBundleError("generation keys do not match the Maple contract")
    eos_ids = generation["eos_token_ids"]
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(not isinstance(item, int) or isinstance(item, bool) for item in eos_ids)
    ):
        raise MapleBundleError("generation.eos_token_ids must contain integers")
    max_context = generation["max_context_tokens"]
    max_input = generation["max_input_tokens"]
    if (
        not isinstance(max_context, int)
        or not 64 <= max_context <= MAPLE_MAX_CONTEXT_TOKENS
    ):
        raise MapleBundleError("max_context_tokens is outside Maple's supported range")
    if not isinstance(max_input, int) or not 32 <= max_input < max_context:
        raise MapleBundleError("max_input_tokens must leave room for generation")

    files = manifest["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= _MAX_BUNDLE_FILES:
        raise MapleBundleError("files must contain between 1 and 512 entries")
    seen: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise MapleBundleError(
                "each files entry must contain path, size_bytes, sha256"
            )
        item_path = _validate_relative_path(item["path"])
        if item_path in seen:
            raise MapleBundleError(f"duplicate bundle path: {item_path}")
        seen.add(item_path)
        size = item["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise MapleBundleError("bundle file sizes must be positive integers")
        checksum = item["sha256"]
        if (
            not isinstance(checksum, str)
            or not _SHA256_PATTERN.fullmatch(checksum)
            or set(checksum) == {"0"}
        ):
            raise MapleBundleError(f"bundle file {item_path} needs a valid SHA-256")
        total_bytes += size
    if total_bytes > _MAX_BUNDLE_BYTES:
        raise MapleBundleError("bundle exceeds the 12 GiB import limit")
    for required_path in (prefill_path, decode_path, tokenizer_path):
        if required_path is not None and required_path not in seen:
            raise MapleBundleError(f"required file is not declared: {required_path}")


def _require_object(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent[key]
    if not isinstance(value, dict):
        raise MapleBundleError(f"{key} must be a JSON object")
    return value


def _validate_graph_path(value: Any, name: str) -> str:
    path = _validate_relative_path(value)
    if PurePosixPath(path).suffix not in _GRAPH_SUFFIXES:
        raise MapleBundleError(f"{name} must point to an ONNX or ORT graph")
    return path


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "\\" in value:
        raise MapleBundleError("bundle paths must be non-empty POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MapleBundleError(f"unsafe bundle path: {value!r}")
    if str(path) != value:
        raise MapleBundleError(f"bundle path is not normalized: {value!r}")
    return value


def _deduplicate_paths(paths: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = _validate_relative_path(path)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def _resolve_source_file(root: Path, relative_path: str) -> Path:
    normalized = _validate_relative_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    if candidate.is_symlink():
        raise MapleBundleError(
            f"bundle source files must not be symlinks: {normalized}"
        )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise MapleBundleError(
            f"bundle source path escapes or is missing: {normalized}"
        ) from exc
    if not resolved.is_file():
        raise MapleBundleError(f"bundle source path is not a file: {normalized}")
    return resolved


def _record_source_file(root: Path, relative_path: str) -> MapleBundleFile:
    source = _resolve_source_file(root, relative_path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise MapleBundleError(f"bundle source file is empty: {relative_path}")
    return MapleBundleFile(relative_path, size, digest.hexdigest())


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _reproducible_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--runtime",
        choices=sorted(MAPLE_BUNDLE_RUNTIMES),
        default="onnxruntime-mobile",
    )
    parser.add_argument("--prefill", default="decoder_model.ort")
    parser.add_argument("--decode", default="decoder_with_past_model.ort")
    parser.add_argument("--quantization", default="qmoe-4bit-blockwise-128")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one Maple bundle from command-line arguments."""

    arguments = _build_parser().parse_args(argv)
    result = build_maple_onnx_bundle(
        arguments.source_directory,
        arguments.output_path,
        prefill_path=arguments.prefill,
        decode_path=arguments.decode,
        runtime=arguments.runtime,
        quantization=arguments.quantization,
    )
    print(json.dumps({"bundle": str(result.bundle_path), **result.manifest}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` tests
    raise SystemExit(main())


__all__ = [
    "MAPLE_ARCHITECTURE",
    "MAPLE_BUNDLE_FILENAME",
    "MAPLE_BUNDLE_RUNTIMES",
    "MAPLE_SOURCE_MODEL",
    "MAPLE_SOURCE_REVISION",
    "MAPLE_VOCAB_SIZE",
    "MapleBundleBuild",
    "MapleBundleError",
    "MapleBundleFile",
    "build_maple_onnx_bundle",
    "validate_maple_onnx_bundle",
]
