#!/usr/bin/env python3
"""Run a PHI-free post-publish smoke test in a fresh virtual environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[2]

# The entire probe is synthetic. Child output is captured and never relayed so
# neither model predictions nor de-identified text can enter workflow logs.
_SYNTHETIC_NOTE = "Synthetic patient contact number is 202-555-0100."
_SYNTHETIC_PHONE = "202-555-0100"
_SAFE_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_ONNX_FORMATS = frozenset({"onnx", "webgpu", "int8"})
_MLX_FORMATS = frozenset({"mlx", "mlx-fp", "mlx-8bit", "mlx-4bit"})


class SmokeTestError(RuntimeError):
    """Raised when the fresh-environment artifact probe fails closed."""


class SmokeResult(NamedTuple):
    """PHI-free result carrying only counts and an offsets hash."""

    span_count: int
    span_offsets_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_count": self.span_count,
            "span_offsets_hash": self.span_offsets_hash,
        }


def probe_artifact(artifact_dir: str | Path, *, format_name: str) -> SmokeResult:
    """Run ``extract_pii`` and ``deidentify`` against a downloaded artifact."""

    from openmed.core.config import OpenMedConfig
    from openmed.core.models import ModelLoader
    from openmed.core.pii import deidentify, extract_pii

    artifact = Path(artifact_dir)
    if not artifact.is_dir():
        raise SmokeTestError("downloaded artifact directory is missing")
    normalized_format = format_name.lower().replace("_", "-")
    if normalized_format in _ONNX_FORMATS:
        backend = "onnx"
        variant = "int8" if normalized_format == "int8" else "auto"
    elif normalized_format in _MLX_FORMATS:
        backend = "mlx"
        variant = "auto"
    else:
        backend = "hf"
        variant = "auto"

    config = OpenMedConfig(
        backend=backend,
        device="cpu",
        onnx_variant=variant,
        use_medical_tokenizer=False,
    )
    loader = ModelLoader(config)
    extracted = extract_pii(
        _SYNTHETIC_NOTE,
        model_name=str(artifact),
        config=config,
        loader=loader,
        use_smart_merging=True,
    )
    expected_start = _SYNTHETIC_NOTE.index(_SYNTHETIC_PHONE)
    expected_end = expected_start + len(_SYNTHETIC_PHONE)
    spans = sorted(
        (int(entity.start), int(entity.end), str(entity.label))
        for entity in extracted.entities
    )
    if not any(
        start <= expected_start and end >= expected_end for start, end, _ in spans
    ):
        raise SmokeTestError(
            "extract_pii did not return the expected synthetic span offsets"
        )

    deidentified = deidentify(
        _SYNTHETIC_NOTE,
        model_name=str(artifact),
        config=config,
        loader=loader,
        use_smart_merging=True,
    )
    if _SYNTHETIC_PHONE in deidentified.deidentified_text:
        raise SmokeTestError("deidentify retained the expected synthetic span")

    encoded = json.dumps(spans, ensure_ascii=True, separators=(",", ":")).encode()
    return SmokeResult(
        span_count=len(spans),
        span_offsets_hash=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
    )


def download_and_probe(
    repo_id: str,
    *,
    format_name: str,
    download_dir: str | Path,
) -> SmokeResult:
    """Download the just-published repository, then probe its local snapshot."""

    if not _SAFE_REPO_ID_RE.fullmatch(repo_id):
        raise SmokeTestError("published repository id is malformed")
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(download_dir),
        token=os.environ.get("HF_WRITE_TOKEN"),
    )
    return probe_artifact(local_path, format_name=format_name)


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def _run_captured(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    cwd: Path,
) -> None:
    completed = runner(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SmokeTestError("fresh-environment smoke command failed")


def run_fresh_venv_smoke(
    repo_id: str,
    *,
    format_name: str,
    repository_root: str | Path = ROOT,
    python_executable: str = sys.executable,
    runner: CommandRunner = subprocess.run,
) -> None:
    """Create a venv, install the narrow runtime, download, and probe."""

    if not _SAFE_REPO_ID_RE.fullmatch(repo_id):
        raise SmokeTestError("published repository id is malformed")
    root = Path(repository_root).resolve()
    normalized_format = format_name.lower().replace("_", "-")
    with tempfile.TemporaryDirectory(prefix="openmed-release-smoke-") as temporary:
        temporary_path = Path(temporary)
        venv_dir = temporary_path / "venv"
        venv_python = (
            venv_dir / "Scripts" / "python.exe"
            if os.name == "nt"
            else venv_dir / "bin" / "python"
        )
        _run_captured(
            [python_executable, "-m", "venv", str(venv_dir)],
            runner=runner,
            cwd=root,
        )

        if normalized_format in _ONNX_FORMATS:
            install_args = [f"{root}[onnx-runtime]"]
        elif normalized_format in _MLX_FORMATS:
            install_args = [f"{root}[mlx]"]
        else:
            install_args = [f"{root}[hf]", "torch>=2.0"]
        _run_captured(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *install_args,
            ],
            runner=runner,
            cwd=root,
        )
        _run_captured(
            [
                str(venv_python),
                str(Path(__file__).resolve()),
                "--probe",
                "--repo-id",
                repo_id,
                "--format",
                normalized_format,
                "--download-dir",
                str(temporary_path / "artifact"),
            ],
            runner=runner,
            cwd=root,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--format", dest="format_name", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--download-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.probe:
            if args.download_dir is None:
                raise SmokeTestError("probe mode requires a download directory")
            result = download_and_probe(
                args.repo_id,
                format_name=args.format_name,
                download_dir=args.download_dir,
            )
            print(json.dumps(result.to_dict(), ensure_ascii=True, sort_keys=True))
        else:
            run_fresh_venv_smoke(
                args.repo_id,
                format_name=args.format_name,
            )
    except (OSError, ValueError, SmokeTestError):
        print("post-publish smoke test failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
