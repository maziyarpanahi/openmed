"""Build the pinned, browser-only Transformers.js tokenizer dependency."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

TRANSFORMERS_JS_VERSION = "3.8.1"
TRANSFORMERS_JS_FILE = "transformers.web.min.js"


class TokenizerRuntimeBuildError(RuntimeError):
    """Raised when the pinned tokenizer runtime cannot be built."""


def build_tokenizer_runtime(output_directory: str | Path) -> Path:
    """Install the pinned npm package temporarily and collect its browser module."""

    if shutil.which("npm") is None:
        raise TokenizerRuntimeBuildError("required build command is unavailable: npm")
    output = Path(output_directory).expanduser().resolve()
    destination = output / TRANSFORMERS_JS_FILE
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer runtime: {destination}")

    with tempfile.TemporaryDirectory(prefix="openmed-maple-tokenizer-") as temporary:
        root = Path(temporary)
        subprocess.run(
            [
                "npm",
                "install",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                "--prefix",
                str(root),
                f"@huggingface/transformers@{TRANSFORMERS_JS_VERSION}",
            ],
            check=True,
        )
        source = (
            root
            / "node_modules"
            / "@huggingface"
            / "transformers"
            / "dist"
            / TRANSFORMERS_JS_FILE
        )
        if not source.is_file():
            raise TokenizerRuntimeBuildError(
                f"Transformers.js did not produce {TRANSFORMERS_JS_FILE}"
            )
        payload = source.read_text(encoding="utf-8")
        for module_name in ("onnxruntime-common", "onnxruntime-web"):
            specifier = f'from"{module_name}"'
            if payload.count(specifier) != 1:
                raise TokenizerRuntimeBuildError(
                    f"unexpected Transformers.js import for {module_name}"
                )
            payload = payload.replace(
                specifier,
                'from"./ort.webgpu.min.mjs"',
            )
        output.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the tokenizer runtime from the command line."""

    arguments = _build_parser().parse_args(argv)
    print(build_tokenizer_runtime(arguments.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
