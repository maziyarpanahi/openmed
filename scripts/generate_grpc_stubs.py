"""Generate and verify committed gRPC Python stubs for OpenMed."""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = REPO_ROOT / "openmed" / "service" / "proto"
PROTO_FILES = (PROTO_DIR / "openmed.proto", PROTO_DIR / "kserve_v2.proto")
GENERATED_DIR = PROTO_DIR / "generated"
GENERATED_FILES = (
    "openmed_pb2.py",
    "openmed_pb2_grpc.py",
    "kserve_v2_pb2.py",
    "__init__.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated stubs differ from the proto contract",
    )
    args = parser.parse_args()

    if args.check:
        return _check_generated_stubs()

    _generate_into(GENERATED_DIR)
    return 0


def _check_generated_stubs() -> int:
    with tempfile.TemporaryDirectory(prefix="openmed-grpc-stubs-") as tmp:
        expected_dir = Path(tmp) / "generated"
        _generate_into(expected_dir)
        mismatches = _diff_generated_files(expected_dir, GENERATED_DIR)
        if mismatches:
            joined = ", ".join(mismatches)
            print(
                "gRPC generated stubs are out of date; run "
                "`make grpc-proto` and commit the result. Drift: "
                f"{joined}",
                file=sys.stderr,
            )
            return 1
    return 0


def _generate_into(output_dir: Path) -> None:
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:
        raise SystemExit(
            "grpcio-tools is required to generate gRPC stubs. "
            "Install dev dependencies with `uv sync --extra dev`."
        ) from exc

    grpc_tools_include = Path(grpc_tools.__file__).resolve().parent / "_proto"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"-I{grpc_tools_include}",
    ]
    python_args = [
        *common_args,
        f"--python_out={output_dir}",
        *(str(path) for path in PROTO_FILES),
    ]
    result = protoc.main(python_args)
    if result != 0:
        raise SystemExit(result)

    grpc_args = [
        *common_args,
        f"--grpc_python_out={output_dir}",
        str(PROTO_DIR / "openmed.proto"),
    ]
    result = protoc.main(grpc_args)
    if result != 0:
        raise SystemExit(result)

    _patch_relative_import(output_dir / "openmed_pb2_grpc.py")
    (output_dir / "__init__.py").write_text(
        '"""Generated protobuf modules for the OpenMed gRPC service."""\n\n'
        "from . import (\n"
        "    kserve_v2_pb2,\n"
        "    openmed_pb2,\n"
        "    openmed_pb2_grpc,\n"
        ")\n\n"
        "__all__ = [\n"
        '    "kserve_v2_pb2",\n'
        '    "openmed_pb2",\n'
        '    "openmed_pb2_grpc",\n'
        "]\n",
        encoding="utf-8",
    )


def _patch_relative_import(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "import openmed_pb2 as openmed__pb2",
        "from . import openmed_pb2 as openmed__pb2",
    )
    text = text.replace(
        "import kserve_v2_pb2 as kserve__v2__pb2",
        "from . import kserve_v2_pb2 as kserve__v2__pb2",
    )
    path.write_text(text, encoding="utf-8")


def _diff_generated_files(expected_dir: Path, actual_dir: Path) -> list[str]:
    mismatches: list[str] = []
    for name in GENERATED_FILES:
        expected = expected_dir / name
        actual = actual_dir / name
        if not actual.exists() or not _generated_files_match(expected, actual, name):
            mismatches.append(name)
    return mismatches


def _generated_files_match(expected: Path, actual: Path, name: str) -> bool:
    if name.endswith("_pb2.py"):
        return _normalized_pb2_text(expected) == _normalized_pb2_text(actual)
    return filecmp.cmp(expected, actual, shallow=False)


def _normalized_pb2_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"# Protobuf Python Version: \d+\.\d+\.\d+",
        "# Protobuf Python Version: <normalized>",
        text,
    )
    return re.sub(
        r"(_runtime_version\.ValidateProtobufRuntimeVersion\(\n"
        r"\s+_runtime_version\.Domain\.PUBLIC,\n)"
        r"\s+\d+,\n\s+\d+,\n\s+\d+,",
        r"\g<1>    0,\n    0,\n    0,",
        text,
    )


if __name__ == "__main__":
    raise SystemExit(main())
