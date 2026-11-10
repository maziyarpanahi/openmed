"""Export local Hugging Face embedding backbones to GGUF.

The converter delegates tensor serialization to llama.cpp's
``convert_hf_to_gguf.py`` contract. It deliberately rejects token-classification
heads: llama.cpp can run BERT-family encoder embeddings, but it does not expose
the classifier head required for token-label predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GGUF_FORMAT = "gguf"
MANIFEST_FORMAT = "openmed-gguf"
MANIFEST_VERSION = 1
MANIFEST_FILENAME = "openmed-gguf.json"
DEFAULT_TIMEOUT_SECONDS = 3600.0

_CONVERTER_FILENAMES = (
    "convert_hf_to_gguf.py",
    "convert-hf-to-gguf.py",
)
_VARIANTS = (
    ("f16", "F16", "float16", "model-f16.gguf"),
    ("q8_0", "Q8_0", "q8_0", "model-q8_0.gguf"),
)
_EMBEDDING_TASKS = frozenset(
    {
        "embedding",
        "embeddings",
        "feature-extraction",
        "sentence-embedding",
        "sentence-similarity",
    }
)
_TOKEN_CLASSIFICATION_TASKS = frozenset(
    {
        "ner",
        "token-classification",
        "token-classifier",
        "tokenclassification",
    }
)


class GgufExportError(RuntimeError):
    """Raised when an embedding backbone cannot be exported to GGUF."""


class UnsupportedGgufModelError(GgufExportError):
    """Raised when a checkpoint is not an embedding backbone."""


@dataclass(frozen=True)
class GgufArtifact:
    """One GGUF precision or quantization variant."""

    path: Path
    quantization: str
    precision: str

    def to_manifest(self, root: Path) -> dict[str, str]:
        """Return manifest metadata relative to *root*."""

        return {
            "format": GGUF_FORMAT,
            "path": self.path.relative_to(root).as_posix(),
            "precision": self.precision,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class GgufConversionResult:
    """Paths and metadata produced by :func:`convert`."""

    output_dir: Path
    manifest_path: Path
    artifacts: tuple[GgufArtifact, ...]

    @property
    def formats(self) -> list[str]:
        """Return canonical formats for models.jsonl publication."""

        return [GGUF_FORMAT]


def convert(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    converter_path: str | Path | None = None,
    llama_cpp_dir: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    source_model_id: str | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = False,
) -> GgufConversionResult:
    """Export an embedding checkpoint as F16 and Q8_0 GGUF artifacts.

    Args:
        model_path: Local Hugging Face checkpoint directory. It must include a
            ``config.json`` and the weights/tokenizer files expected by
            llama.cpp.
        output_dir: Directory that receives both GGUF files, the source config,
            and ``openmed-gguf.json``.
        converter_path: Direct path to llama.cpp's ``convert_hf_to_gguf.py``.
        llama_cpp_dir: llama.cpp checkout containing the converter. This is
            optional when ``converter_path`` or ``LLAMA_CPP_DIR`` is supplied.
        python_executable: Python interpreter used to run the converter.
        source_model_id: Stable source repository ID for manifest provenance.
            Defaults to the local checkpoint directory name.
        timeout_seconds: Per-variant subprocess timeout, or ``None`` to disable.
        overwrite: Replace OpenMed GGUF output files that already exist.

    Returns:
        Metadata and paths for the F16 and Q8_0 artifacts.

    Raises:
        FileNotFoundError: If the checkpoint, config, or converter is missing.
        UnsupportedGgufModelError: If the config describes a task-specific
            head instead of an embedding/feature-extraction backbone.
        GgufExportError: If conversion fails or does not create a GGUF file.
    """

    source_dir = Path(model_path).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"model checkpoint directory not found: {source_dir}")

    config_path = source_dir / "config.json"
    config = _read_config(config_path)
    task = _validate_embedding_backbone(config)
    converter = _resolve_converter_path(
        converter_path=converter_path,
        llama_cpp_dir=llama_cpp_dir,
    )
    _validate_timeout(timeout_seconds)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_names = [variant[3] for variant in _VARIANTS]
    target_names.extend(("config.json", MANIFEST_FILENAME))
    _check_output_conflicts(destination, target_names, overwrite=overwrite)

    resolved_source_id = source_model_id or source_dir.name
    artifacts: list[GgufArtifact] = []
    with tempfile.TemporaryDirectory(
        prefix=".openmed-gguf-",
        dir=destination,
    ) as staging_value:
        staging_dir = Path(staging_value)
        for outtype, quantization, precision, filename in _VARIANTS:
            staged_path = staging_dir / filename
            _run_converter(
                converter=converter,
                python_executable=Path(python_executable),
                model_path=source_dir,
                output_path=staged_path,
                outtype=outtype,
                timeout_seconds=timeout_seconds,
            )
            artifacts.append(
                GgufArtifact(
                    path=destination / filename,
                    quantization=quantization,
                    precision=precision,
                )
            )

        shutil.copy2(config_path, staging_dir / "config.json")
        manifest_path = staging_dir / MANIFEST_FILENAME
        _write_manifest(
            manifest_path,
            root=destination,
            source_model_id=resolved_source_id,
            config=config,
            task=task,
            artifacts=artifacts,
        )

        for filename in target_names:
            os.replace(staging_dir / filename, destination / filename)

    return GgufConversionResult(
        output_dir=destination,
        manifest_path=destination / MANIFEST_FILENAME,
        artifacts=tuple(artifacts),
    )


def export_gguf(
    model_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> GgufConversionResult:
    """Alias for :func:`convert` with an explicit export-oriented name."""

    return convert(model_path, output_dir, **kwargs)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GgufExportError(f"model config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise GgufExportError(f"model config must contain a JSON object: {path}")
    return payload


def _validate_embedding_backbone(config: Mapping[str, Any]) -> str:
    tasks = {
        _normalize_task(config[key])
        for key in ("task", "pipeline_tag", "_mlx_task")
        if config.get(key)
    }
    architectures = _string_sequence(config.get("architectures"))
    auto_map = config.get("auto_map")
    auto_map_values: list[str] = []
    if isinstance(auto_map, Mapping):
        auto_map_values.extend(str(key) for key in auto_map)
        auto_map_values.extend(str(value) for value in auto_map.values())

    architecture_hints = [*architectures, *auto_map_values]
    token_head = any(
        "tokenclassification" in hint.replace("_", "").replace("-", "").lower()
        for hint in architecture_hints
    )
    if tasks & _TOKEN_CLASSIFICATION_TASKS or token_head:
        raise UnsupportedGgufModelError(
            "GGUF embedding export does not support token-classification heads: "
            "llama.cpp cannot run the token classifier head. Export the encoder "
            "backbone (for example BertModel/AutoModel) instead."
        )

    unsupported_tasks = tasks - _EMBEDDING_TASKS
    if unsupported_tasks:
        task_list = ", ".join(sorted(unsupported_tasks))
        raise UnsupportedGgufModelError(
            "GGUF export is limited to embedding/feature-extraction backbones; "
            f"checkpoint task is {task_list}."
        )

    task_specific_architectures = [
        architecture
        for architecture in architectures
        if "For" in architecture and "FeatureExtraction" not in architecture
    ]
    if task_specific_architectures:
        names = ", ".join(task_specific_architectures)
        raise UnsupportedGgufModelError(
            "GGUF export is limited to encoder backbones without task-specific "
            f"heads; checkpoint architecture is {names}."
        )

    if not architectures and not tasks and not config.get("model_type"):
        raise UnsupportedGgufModelError(
            "GGUF export needs config evidence for an embedding backbone: set "
            "architectures, pipeline_tag, task, or model_type."
        )

    return "feature-extraction"


def _normalize_task(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item]


def _resolve_converter_path(
    *,
    converter_path: str | Path | None,
    llama_cpp_dir: str | Path | None,
) -> Path:
    if converter_path is not None and llama_cpp_dir is not None:
        raise ValueError("provide converter_path or llama_cpp_dir, not both")

    if converter_path is not None:
        converter = Path(converter_path).expanduser().resolve()
        if not converter.is_file():
            raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
        return converter

    checkout_value = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if checkout_value is None:
        raise FileNotFoundError(
            "llama.cpp converter not configured; pass converter_path or "
            "llama_cpp_dir, or set LLAMA_CPP_DIR"
        )

    checkout = Path(checkout_value).expanduser().resolve()
    for filename in _CONVERTER_FILENAMES:
        candidate = checkout / filename
        if candidate.is_file():
            return candidate
    names = " or ".join(_CONVERTER_FILENAMES)
    raise FileNotFoundError(f"{names} not found in llama.cpp checkout: {checkout}")


def _validate_timeout(timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")


def _check_output_conflicts(
    output_dir: Path,
    filenames: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    conflicts = [name for name in filenames if (output_dir / name).exists()]
    if conflicts:
        names = ", ".join(conflicts)
        raise FileExistsError(
            f"GGUF output already exists ({names}); pass overwrite=True to replace it"
        )


def _run_converter(
    *,
    converter: Path,
    python_executable: Path,
    model_path: Path,
    output_path: Path,
    outtype: str,
    timeout_seconds: float | None,
) -> None:
    command = [
        str(python_executable),
        str(converter),
        str(model_path),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise GgufExportError(
            f"llama.cpp GGUF {outtype} conversion exceeded {timeout_seconds} seconds"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = _subprocess_error_detail(exc)
        raise GgufExportError(
            f"llama.cpp GGUF {outtype} conversion failed: {detail}"
        ) from exc
    except OSError as exc:
        raise GgufExportError(
            f"could not start llama.cpp GGUF {outtype} conversion: {exc}"
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise GgufExportError(
            f"llama.cpp GGUF {outtype} conversion did not write {output_path.name}"
        )


def _subprocess_error_detail(exc: subprocess.CalledProcessError) -> str:
    value = exc.stderr or exc.stdout or f"exit status {exc.returncode}"
    detail = str(value).strip()
    if len(detail) > 1000:
        detail = detail[-1000:]
    return detail


def _write_manifest(
    path: Path,
    *,
    root: Path,
    source_model_id: str,
    config: Mapping[str, Any],
    task: str,
    artifacts: Sequence[GgufArtifact],
) -> None:
    architectures = _string_sequence(config.get("architectures"))
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "formats": [GGUF_FORMAT],
        "task": task,
        "family": str(config.get("model_type") or "unknown"),
        "architecture": architectures[0] if architectures else None,
        "source_model_id": source_model_id,
        "source_revision": str(config.get("_commit_hash") or "local"),
        "config_path": "config.json",
        "artifacts": [artifact.to_manifest(root) for artifact in artifacts],
        "converter": {
            "contract": "llama.cpp/convert_hf_to_gguf.py",
            "outtypes": [variant[0] for variant in _VARIANTS],
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GGUF embedding export command-line interface."""

    parser = argparse.ArgumentParser(
        description="Export a local embedding backbone as F16 and Q8_0 GGUF",
    )
    parser.add_argument("--model", required=True, help="Local Hugging Face model")
    parser.add_argument("--output", required=True, help="Output directory")
    converter_group = parser.add_mutually_exclusive_group()
    converter_group.add_argument(
        "--converter",
        help="Path to llama.cpp convert_hf_to_gguf.py",
    )
    converter_group.add_argument(
        "--llama-cpp",
        help="Path to a llama.cpp checkout (or set LLAMA_CPP_DIR)",
    )
    parser.add_argument(
        "--source-model-id",
        help="Stable source repository ID recorded in the export manifest",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-variant timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing OpenMed GGUF output files",
    )
    args = parser.parse_args(argv)
    result = convert(
        args.model,
        args.output,
        converter_path=args.converter,
        llama_cpp_dir=args.llama_cpp,
        source_model_id=args.source_model_id,
        timeout_seconds=args.timeout,
        overwrite=args.overwrite,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
