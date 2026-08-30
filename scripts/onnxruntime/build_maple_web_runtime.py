"""Build the pinned ONNX Runtime WebGPU EP fork that enables Maple QMoE2.

The upstream WebGPU MatMulNBits implementation already supports packed two-bit
weights. OpenMed's narrow patch only allows the QMoE wrapper to select that
kernel and passes the correct four-values-per-byte packing ratio.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

ONNXRUNTIME_REVISION = "8c546c37b43caaca1fa25db430dab94b901cf277"
PATCH_MARKER = "openmed-qmoe2-webgpu-v1"
PATCH_PATH = Path(__file__).with_name("patches") / "maple-qmoe-2bit-webgpu-v1.patch"


class RuntimeBuildError(RuntimeError):
    """Raised when the pinned runtime source or build output is invalid."""


def prepare_source(source_directory: str | Path) -> Path:
    """Validate the pinned checkout and apply the reviewed patch exactly once."""

    source = Path(source_directory).expanduser().resolve()
    if not (source / ".git").exists():
        raise RuntimeBuildError(f"not an ONNX Runtime git checkout: {source}")
    revision = _capture(["git", "rev-parse", "HEAD"], cwd=source).strip()
    if revision != ONNXRUNTIME_REVISION:
        raise RuntimeBuildError(
            f"ONNX Runtime must be at {ONNXRUNTIME_REVISION}; found {revision}"
        )
    if _succeeds(["git", "apply", "--reverse", "--check", str(PATCH_PATH)], cwd=source):
        return source
    _run(["git", "apply", "--check", str(PATCH_PATH)], cwd=source)
    _run(["git", "apply", str(PATCH_PATH)], cwd=source)
    return source


def build_runtime(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    install_emsdk: bool = True,
) -> tuple[Path, ...]:
    """Build and collect the three browser files consumed by the Maple demo."""

    source = prepare_source(source_directory)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime output: {output}")
    for command in ("cmake", "node", "npm"):
        if shutil.which(command) is None:
            raise RuntimeBuildError(f"required build command is unavailable: {command}")

    _run(["git", "submodule", "sync", "--recursive"], cwd=source)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
    emsdk = source / "cmake" / "external" / "emsdk"
    if install_emsdk:
        _run([str(emsdk / "emsdk"), "install", "latest"], cwd=emsdk)
        _run([str(emsdk / "emsdk"), "activate", "latest"], cwd=emsdk)

    build_command = " ".join(
        (
            f"source '{emsdk / 'emsdk_env.sh'}' >/dev/null",
            "&&",
            "./build.sh --config Release --build_wasm --skip_tests",
            "--disable_rtti --use_webgpu",
            "--enable_wasm_simd --enable_wasm_threads --parallel",
            "--target onnxruntime_webassembly",
        )
    )
    _run(["bash", "-lc", build_command], cwd=source)

    for directory in (source / "js", source / "js" / "common", source / "js" / "web"):
        _run(["npm", "ci"], cwd=directory)
    web = source / "js" / "web"
    _run(["npm", "run", "pull:wasm"], cwd=web)
    wasm_root = _find_wasm_output(source)
    web_dist = web / "dist"
    web_dist.mkdir(parents=True, exist_ok=True)
    for name in (
        "ort-wasm-simd-threaded.asyncify.mjs",
        "ort-wasm-simd-threaded.asyncify.wasm",
    ):
        shutil.copy2(wasm_root / name, web_dist / name)
    _run(["npm", "run", "build"], cwd=web)

    candidates = {
        "ort.webgpu.min.mjs": web / "dist" / "ort.webgpu.min.mjs",
        "ort-wasm-simd-threaded.asyncify.mjs": (
            wasm_root / "ort-wasm-simd-threaded.asyncify.mjs"
        ),
        "ort-wasm-simd-threaded.asyncify.wasm": (
            wasm_root / "ort-wasm-simd-threaded.asyncify.wasm"
        ),
    }
    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        raise RuntimeBuildError("runtime build did not produce: " + ", ".join(missing))
    output.mkdir(parents=True)
    built: list[Path] = []
    for name, source_path in candidates.items():
        destination = output / name
        shutil.copy2(source_path, destination)
        built.append(destination)
    (output / "OPENMED_QMOE2_RUNTIME").write_text(
        f"{PATCH_MARKER}\nonnxruntime={ONNXRUNTIME_REVISION}\n",
        encoding="utf-8",
    )
    return tuple(built)


def _find_wasm_output(source: Path) -> Path:
    candidates = (
        source / "build" / "WebAssembly" / "Release",
        source / "build" / "MacOS" / "Release",
    )
    for candidate in candidates:
        if all(
            (candidate / name).is_file()
            for name in (
                "ort-wasm-simd-threaded.asyncify.mjs",
                "ort-wasm-simd-threaded.asyncify.wasm",
            )
        ):
            return candidate
    raise RuntimeBuildError("unable to locate the built WebGPU WebAssembly artifacts")


def _run(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _capture(command: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _succeeds(command: Sequence[str], *, cwd: Path) -> bool:
    return (
        subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path, nargs="?")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-emsdk-install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pinned runtime preparation or complete WebGPU build."""

    arguments = _build_parser().parse_args(argv)
    if arguments.prepare_only:
        prepare_source(arguments.source_directory)
        return 0
    if arguments.output_directory is None:
        raise SystemExit("output_directory is required unless --prepare-only is used")
    build_runtime(
        arguments.source_directory,
        arguments.output_directory,
        install_emsdk=not arguments.skip_emsdk_install,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
