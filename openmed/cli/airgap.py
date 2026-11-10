"""Build and verify portable OpenMed air-gapped installation kits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

from ..__about__ import __version__
from ..core.hf_hub import prefetch_model, resolve_repo_id

MANIFEST_NAME = "bundle-manifest.json"
DEFAULT_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
DEFAULT_EXTRAS = ("cli",)
SUPPORTED_EXTRAS = ("cli", "onnx")
DEFAULT_MODELS = (DEFAULT_MODEL,)
DEFAULT_BUNDLE_LIMIT_BYTES = 900_000_000
_HASH_CHUNK_SIZE = 1024 * 1024
_PYTHON_VERSION_RE = re.compile(r"^(?:cp|py)?(\d)(?:\.?)(\d{1,2})$")

_BUNDLED_OFFLINE_GUIDE = """\
# OpenMed offline installation

This kit contains the Python wheels, model cache, integrity manifest, and
installation instructions needed to install OpenMed without internet access.

1. On the connected build machine, run `openmed airgap verify <bundle>`.
2. Copy the complete bundle to removable media.
3. On the offline machine, copy it to local storage and run `./install.sh`.
4. Set `OPENMED_OFFLINE=1` before processing clinical text.
5. Run `openmed doctor` and a synthetic de-identification check before using
   the installation with clinical data.

Never put raw PHI in build logs, bundle manifests, or transfer records.
"""


def add_airgap_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed airgap`` command group."""
    parser = subparsers.add_parser(
        "airgap",
        help="Build or verify an offline installation kit.",
    )
    commands = parser.add_subparsers(dest="airgap_command")

    bundle = commands.add_parser(
        "bundle",
        help="Download wheels and models into a portable offline kit.",
    )
    bundle.add_argument(
        "output",
        type=Path,
        help="New output directory, .tar.gz, or .tgz path.",
    )
    bundle.add_argument(
        "--archive",
        action="store_true",
        help="Write a gzip-compressed tar archive instead of a directory.",
    )
    bundle.add_argument(
        "--extra",
        dest="extras",
        action="append",
        choices=SUPPORTED_EXTRAS,
        help="OpenMed extra to include; repeat for multiple extras (default: cli).",
    )
    bundle.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Model alias or Hub repository to cache; repeat for multiple models "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    bundle.add_argument(
        "--platform",
        "--target-platform",
        dest="target_platform",
        help=(
            "pip target platform, for example manylinux2014_aarch64 "
            "(default: current platform)."
        ),
    )
    bundle.add_argument(
        "--python-version",
        "--target-python",
        dest="python_version",
        type=_python_version,
        help="Target Python version, for example 3.11 or cp311 (default: current).",
    )
    bundle.add_argument(
        "--implementation",
        choices=("cp", "pp", "py"),
        help="Target Python implementation tag (default: current implementation).",
    )
    bundle.add_argument(
        "--abi",
        help="Target Python ABI tag (default: derived from the target Python).",
    )
    bundle.set_defaults(handler=_handle_bundle)

    verify = commands.add_parser(
        "verify",
        help="Verify every bundled artifact against its SHA-256 digest.",
    )
    verify.add_argument("bundle", type=Path, help="Bundle directory or tar archive.")
    verify.set_defaults(handler=_handle_verify)


def _handle_bundle(args: argparse.Namespace) -> int:
    extras = _unique(args.extras or DEFAULT_EXTRAS)
    models = _unique(args.models or DEFAULT_MODELS)
    archive = bool(
        args.archive
        or str(args.output).endswith(".tar.gz")
        or args.output.suffix == ".tgz"
    )
    try:
        result = build_bundle(
            args.output,
            archive=archive,
            extras=extras,
            models=models,
            target_platform=args.target_platform,
            python_version=args.python_version,
            implementation=args.implementation,
            abi=args.abi,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"Failed to build air-gap bundle: {exc}\n")
        return 1

    sys.stdout.write(f"Created air-gap bundle: {result.path}\n")
    sys.stdout.write(
        f"Payload: {result.total_size_bytes} bytes "
        f"across {result.artifact_count} artifacts\n"
    )
    sys.stdout.write(f"Verify before transfer: openmed airgap verify {result.path}\n")
    return 0


def _handle_verify(args: argparse.Namespace) -> int:
    try:
        errors = verify_bundle(args.bundle)
    except (OSError, ValueError, tarfile.TarError) as exc:
        sys.stderr.write(f"Air-gap bundle verification failed: {exc}\n")
        return 1

    if errors:
        for error in errors:
            sys.stderr.write(f"FAIL {error}\n")
        return 1

    sys.stdout.write(f"Verified air-gap bundle: {args.bundle}\n")
    return 0


class BundleResult:
    """Summary of a completed bundle build."""

    def __init__(
        self,
        *,
        path: Path,
        total_size_bytes: int,
        artifact_count: int,
    ) -> None:
        self.path = path
        self.total_size_bytes = total_size_bytes
        self.artifact_count = artifact_count


def build_bundle(
    output: Path,
    *,
    archive: bool = False,
    extras: Sequence[str] = DEFAULT_EXTRAS,
    models: Sequence[str] = DEFAULT_MODELS,
    target_platform: str | None = None,
    python_version: str | None = None,
    implementation: str | None = None,
    abi: str | None = None,
) -> BundleResult:
    """Create a complete offline kit at a new path.

    Args:
        output: New directory or archive path to create.
        archive: Package the completed directory as a gzip-compressed tarball.
        extras: Optional OpenMed extras selected for the wheelhouse.
        models: Model aliases or repository ids to pre-fetch.
        target_platform: Optional pip target platform tag.
        python_version: Optional target Python version.
        implementation: Optional pip implementation tag.
        abi: Optional pip ABI tag.

    Returns:
        Summary containing the output path and exact payload size.

    Raises:
        FileExistsError: If *output* already exists.
        RuntimeError: If a download fails or the default bundle exceeds 900 MB.
        ValueError: If target values or downloaded paths are unsafe.
    """
    output = output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {output}")

    normalized_extras = _validate_extras(extras)
    normalized_models = _unique(models)
    if not normalized_models:
        raise ValueError("at least one model must be included")

    normalized_platform = _normalize_platform(target_platform)
    normalized_python = python_version or _current_python_version()
    normalized_implementation = implementation or _current_implementation()
    normalized_abi = abi or _default_abi(
        normalized_python,
        normalized_implementation,
    )
    package_spec = _package_spec(normalized_extras)
    target = {
        "platform": normalized_platform or _current_platform(),
        "python_version": normalized_python,
        "implementation": normalized_implementation,
        "abi": normalized_abi,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".openmed-airgap-",
        dir=output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        bundle_root = temporary_root / "openmed-airgap"
        wheels_dir = bundle_root / "wheels"
        model_cache = bundle_root / "models" / "hub"
        docs_dir = bundle_root / "docs"
        wheels_dir.mkdir(parents=True)
        model_cache.mkdir(parents=True)
        docs_dir.mkdir(parents=True)

        _download_wheels(
            wheels_dir,
            package_spec=package_spec,
            target_platform=normalized_platform,
            python_version=(
                normalized_python
                if target_platform is not None or python_version is not None
                else None
            ),
            implementation=(
                normalized_implementation
                if target_platform is not None or python_version is not None
                else None
            ),
            abi=(
                normalized_abi
                if target_platform is not None or python_version is not None
                else None
            ),
        )
        if not any(wheels_dir.glob("*.whl")):
            raise RuntimeError("pip download completed without producing any wheels")

        model_records = _download_models(normalized_models, model_cache)
        shutil.rmtree(model_cache / ".locks", ignore_errors=True)

        (docs_dir / "offline-install.md").write_text(
            _BUNDLED_OFFLINE_GUIDE,
            encoding="utf-8",
        )
        (bundle_root / "INSTALL.md").write_text(
            _render_install_readme(target, package_spec, model_records),
            encoding="utf-8",
        )
        install_script = bundle_root / "install.sh"
        install_script.write_text(
            _render_install_script(package_spec),
            encoding="utf-8",
        )
        install_script.chmod(0o755)

        artifacts = _collect_artifacts(bundle_root)
        total_size = sum(item["size_bytes"] for item in artifacts)
        is_default_bundle = (
            tuple(normalized_extras) == DEFAULT_EXTRAS
            and tuple(normalized_models) == DEFAULT_MODELS
        )
        if is_default_bundle and total_size > DEFAULT_BUNDLE_LIMIT_BYTES:
            raise RuntimeError(
                f"default bundle is larger than the 900 MB limit: {total_size} bytes"
            )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "openmed_version": __version__,
            "package_spec": package_spec,
            "target": target,
            "extras": normalized_extras,
            "models": model_records,
            "artifact_count": len(artifacts),
            "total_size_bytes": total_size,
            "bundle_size_bytes": total_size,
            "default_bundle_limit_bytes": DEFAULT_BUNDLE_LIMIT_BYTES,
            "artifacts": artifacts,
        }
        (bundle_root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if archive:
            temporary_archive = temporary_root / "openmed-airgap.tar.gz"
            with tarfile.open(temporary_archive, "w:gz") as tar:
                tar.add(
                    bundle_root,
                    arcname="openmed-airgap",
                    filter=_portable_tar_info,
                )
            os.replace(temporary_archive, output)
        else:
            shutil.move(str(bundle_root), output)

    return BundleResult(
        path=output,
        total_size_bytes=total_size,
        artifact_count=len(artifacts),
    )


def verify_bundle(bundle: Path) -> list[str]:
    """Return integrity errors for a directory or tar bundle."""
    bundle = bundle.expanduser().resolve(strict=True)
    with _verified_bundle_root(bundle) as root:
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            return [f"missing {MANIFEST_NAME}"]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [f"invalid {MANIFEST_NAME}: {exc}"]
        return _verify_manifest(root, manifest)


def _download_wheels(
    wheels_dir: Path,
    *,
    package_spec: str,
    target_platform: str | None,
    python_version: str | None,
    implementation: str | None,
    abi: str | None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--dest",
        str(wheels_dir),
    ]
    if target_platform is not None:
        command.extend(("--platform", target_platform))
    if python_version is not None:
        command.extend(("--python-version", python_version))
    if implementation is not None:
        command.extend(("--implementation", implementation))
    if abi is not None:
        command.extend(("--abi", abi))
    command.append(package_spec)
    try:
        subprocess.run(command, check=True)  # noqa: S603 - fixed executable/argv
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pip download failed with exit code {exc.returncode}"
        ) from exc


def _download_models(models: Sequence[str], model_cache: Path) -> list[dict[str, str]]:
    cache_root = model_cache.resolve()
    records: list[dict[str, str]] = []
    for model in models:
        repo_id = resolve_repo_id(model)
        snapshot = Path(
            prefetch_model(
                model,
                cache_dir=str(model_cache),
            )
        )
        try:
            snapshot = snapshot.resolve(strict=True)
            relative_snapshot = snapshot.relative_to(cache_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"model snapshot escaped the bundle cache: {snapshot}"
            ) from exc
        if not snapshot.is_dir():
            raise ValueError(f"model snapshot is not a directory: {snapshot}")
        records.append(
            {
                "repo_id": repo_id,
                "snapshot_path": (
                    Path("models") / "hub" / relative_snapshot
                ).as_posix(),
            }
        )
    return records


def _collect_artifacts(root: Path) -> list[dict[str, Any]]:
    root_resolved = root.resolve()
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.name == MANIFEST_NAME or not (path.is_file() or path.is_symlink()):
            continue
        _ensure_artifact_within(root_resolved, path)
        relative = path.relative_to(root).as_posix()
        artifacts.append(
            {
                "path": relative,
                "kind": relative.split("/", 1)[0],
                "type": "symlink" if path.is_symlink() else "file",
                "size_bytes": _artifact_size(path),
                "sha256": _artifact_sha256(path),
            }
        )
        if path.is_symlink():
            artifacts[-1]["link_target"] = _portable_link_target(path)
    return artifacts


def _verify_manifest(root: Path, manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest root must be a JSON object"]
    if manifest.get("schema_version") != 1:
        return ["unsupported manifest schema_version"]
    declared = manifest.get("artifacts")
    if not isinstance(declared, list):
        return ["manifest artifacts must be a list"]

    root_resolved = root.resolve()
    errors: list[str] = []
    expected_paths: set[str] = set()
    declared_total = 0
    for entry in declared:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object artifact")
            continue
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        expected_type = entry.get("type", "file")
        if not isinstance(relative, str) or not _safe_relative_path(relative):
            errors.append(f"unsafe artifact path: {relative!r}")
            continue
        if relative in expected_paths:
            errors.append(f"duplicate artifact path: {relative}")
            continue
        expected_paths.add(relative)
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"invalid size for {relative}")
            continue
        declared_total += expected_size
        path = root / relative
        if not (path.is_file() or path.is_symlink()):
            errors.append(f"missing artifact: {relative}")
            continue
        try:
            _ensure_artifact_within(root_resolved, path)
            actual_type = "symlink" if path.is_symlink() else "file"
            actual_size = _artifact_size(path)
            actual_hash = _artifact_sha256(path)
        except OSError as exc:
            errors.append(f"unreadable artifact {relative}: {exc}")
            continue
        if expected_type not in {"file", "symlink"}:
            errors.append(f"invalid artifact type for {relative}")
        elif actual_type != expected_type:
            errors.append(
                f"type mismatch for {relative}: expected {expected_type}, "
                f"found {actual_type}"
            )
        if actual_type == "symlink" and entry.get(
            "link_target"
        ) != _portable_link_target(path):
            errors.append(f"link target mismatch for {relative}")
        if actual_size != expected_size:
            errors.append(
                f"size mismatch for {relative}: expected {expected_size}, "
                f"found {actual_size}"
            )
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            errors.append(f"sha256 mismatch for {relative}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name != MANIFEST_NAME and (path.is_file() or path.is_symlink())
    }
    for unexpected in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected artifact: {unexpected}")

    if manifest.get("artifact_count") != len(declared):
        errors.append("artifact_count does not match the manifest inventory")
    if manifest.get("total_size_bytes") != declared_total:
        errors.append("total_size_bytes does not match artifact sizes")
    if manifest.get("bundle_size_bytes") != declared_total:
        errors.append("bundle_size_bytes does not match artifact sizes")
    return errors


@contextmanager
def _verified_bundle_root(bundle: Path) -> Iterator[Path]:
    if bundle.is_dir():
        yield bundle
        return
    if not bundle.is_file():
        raise ValueError(f"bundle does not exist: {bundle}")

    with tempfile.TemporaryDirectory(prefix="openmed-airgap-verify-") as temporary:
        destination = Path(temporary)
        with tarfile.open(bundle, "r:*") as tar:
            _extract_regular_archive(tar, destination)
        manifests = list(destination.rglob(MANIFEST_NAME))
        if len(manifests) != 1:
            raise ValueError(
                f"archive must contain exactly one {MANIFEST_NAME}; "
                f"found {len(manifests)}"
            )
        yield manifests[0].parent


def _extract_regular_archive(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract regular entries and safe in-tree file symlinks."""
    destination_resolved = destination.resolve()
    members = tar.getmembers()
    member_types: dict[str, str] = {}
    symlinks: list[tarfile.TarInfo] = []
    for member in members:
        if not _safe_relative_path(member.name):
            raise ValueError(f"unsafe archive member: {member.name!r}")
        if member.name in member_types:
            raise ValueError(f"duplicate archive member: {member.name}")
        if member.isdir():
            member_types[member.name] = "directory"
        elif member.isfile():
            member_types[member.name] = "file"
        elif member.issym():
            member_types[member.name] = "symlink"
            symlinks.append(member)
        else:
            raise ValueError(f"unsupported archive member type: {member.name}")

    for member in members:
        if member.issym():
            continue
        target = destination / PurePosixPath(member.name)
        _ensure_path_within(destination_resolved, target, strict=False)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member.name}")
        with source, target.open("xb") as handle:
            shutil.copyfileobj(source, handle)
        target.chmod(member.mode & 0o777)

    for member in symlinks:
        if not member.linkname or "\\" in member.linkname:
            raise ValueError(f"unsafe archive link: {member.name}")
        normalized_target = posixpath.normpath(
            posixpath.join(posixpath.dirname(member.name), member.linkname)
        )
        if (
            not _safe_relative_path(normalized_target)
            or member_types.get(normalized_target) != "file"
        ):
            raise ValueError(
                f"archive link escapes or targets a non-file: {member.name}"
            )
        target = destination / PurePosixPath(member.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(*PurePosixPath(member.linkname).parts))
        _ensure_artifact_within(destination_resolved, target)


def _render_install_script(package_spec: str) -> str:
    return f"""\
#!/bin/sh
set -eu

BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${{PYTHON:-python3}}
WHEELS_DIR="$BUNDLE_ROOT/wheels"
MODEL_SOURCE="$BUNDLE_ROOT/models/hub"

"$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELS_DIR" \\
  '{package_spec}'

if [ -n "${{HF_HUB_CACHE:-}}" ]; then
  MODEL_TARGET=$HF_HUB_CACHE
elif [ -n "${{HF_HOME:-}}" ]; then
  MODEL_TARGET="$HF_HOME/hub"
else
  MODEL_TARGET="${{HOME:?HOME must be set}}/.cache/huggingface/hub"
fi

mkdir -p "$MODEL_TARGET"
cp -R "$MODEL_SOURCE/." "$MODEL_TARGET/"

# OpenMed 1.x uses ~/.cache/openmed by default while Transformers uses the
# standard Hub cache above. Bridge each repository without duplicating blobs.
OPENMED_CACHE=${{OPENMED_AIRGAP_CACHE:-"${{HOME:?HOME must be set}}/.cache/openmed"}}
if [ "$OPENMED_CACHE" != "$MODEL_TARGET" ]; then
  mkdir -p "$OPENMED_CACHE"
  for MODEL_REPO in "$MODEL_TARGET"/models--*; do
    [ -e "$MODEL_REPO" ] || continue
    OPENMED_REPO="$OPENMED_CACHE/$(basename -- "$MODEL_REPO")"
    if [ ! -e "$OPENMED_REPO" ]; then
      ln -s "$MODEL_REPO" "$OPENMED_REPO"
    elif [ ! -L "$OPENMED_REPO" ]; then
      cp -R "$MODEL_REPO/." "$OPENMED_REPO/"
    fi
  done
fi

cat <<'EOF'
OpenMed and its model cache are installed.

Before offline use:
  export OPENMED_OFFLINE=1
  openmed doctor

Keep OPENMED_OFFLINE=1 set whenever clinical data is processed without a
network connection. See INSTALL.md for a synthetic de-identification check.
EOF
"""


def _render_install_readme(
    target: dict[str, str],
    package_spec: str,
    models: Sequence[dict[str, str]],
) -> str:
    model_lines = "\n".join(f"- `{item['repo_id']}`" for item in models)
    return f"""\
# Install OpenMed without internet access

This bundle targets `{target["platform"]}`, Python `{target["python_version"]}`,
implementation `{target["implementation"]}`, and ABI `{target["abi"]}`.

Included package: `{package_spec}`

Included model snapshots:

{model_lines}

## On the connected machine

Verify the copied kit before disconnecting:

```console
openmed airgap verify /path/to/openmed-airgap
```

Copy the whole directory or archive to removable media. Do not edit files after
verification.

## On the offline machine

Copy the kit to local storage, verify it if OpenMed is already available, then
run:

```console
./install.sh
export OPENMED_OFFLINE=1
openmed doctor
```

The installer uses only `pip install --no-index --find-links wheels/`, copies the
bundled snapshots into `HF_HUB_CACHE`, `$HF_HOME/hub`, or the default
`$HOME/.cache/huggingface/hub` cache, and links those repositories into
OpenMed's default model cache without duplicating the model blobs. Set
`OPENMED_AIRGAP_CACHE` during installation if OpenMed is configured with a
different `cache_dir`.

## Synthetic safety check

Never use real patient data for the first check:

```console
printf '%s\n' 'Patient Jane Example called 555-0100 on 2025-01-20.' | \\
  OPENMED_OFFLINE=1 openmed deid --model {models[0]["repo_id"]}
```

The command must finish from the local cache. OpenMed's offline guard blocks
outbound socket connections while local-only inference runs.
"""


def _package_spec(extras: Sequence[str]) -> str:
    # Model inference needs the HF runtime even though it is not a selectable
    # bundle profile. The user-selected profiles remain visible in the manifest.
    runtime_extras = sorted(set(extras) | {"hf"})
    return f"openmed[{','.join(runtime_extras)}]=={__version__}"


def _validate_extras(extras: Sequence[str]) -> list[str]:
    unique = _unique(extras)
    invalid = sorted(set(unique) - set(SUPPORTED_EXTRAS))
    if invalid:
        raise ValueError(f"unsupported extras: {', '.join(invalid)}")
    return unique


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _python_version(value: str) -> str:
    match = _PYTHON_VERSION_RE.fullmatch(value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError(
            "Python version must look like 3.11, 311, or cp311"
        )
    return f"{match.group(1)}.{match.group(2)}"


def _normalize_platform(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    aliases = {
        "linux/aarch64": "manylinux2014_aarch64",
        "linux/arm64": "manylinux2014_aarch64",
        "linux/x86_64": "manylinux2014_x86_64",
        "linux/amd64": "manylinux2014_x86_64",
    }
    return aliases.get(normalized, normalized)


def _current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _current_implementation() -> str:
    return {"cpython": "cp", "pypy": "pp"}.get(sys.implementation.name, "py")


def _current_platform() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_") or (
        f"{platform.system().lower()}_{platform.machine().lower()}"
    )


def _default_abi(python_version: str, implementation: str) -> str:
    digits = python_version.replace(".", "")
    return f"{implementation}{digits}" if implementation in {"cp", "pp"} else "none"


def _artifact_sha256(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(_portable_link_target(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_size(path: Path) -> int:
    if path.is_symlink():
        return len(_portable_link_target(path).encode("utf-8"))
    return path.stat().st_size


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return False
    normalized = posixpath.normpath(value)
    return normalized == value and not normalized.startswith("../")


def _portable_link_target(path: Path) -> str:
    """Return a relative symlink target using archive-style separators."""
    return os.readlink(path).replace("\\", "/")


def _portable_tar_info(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize relative symlink targets for portable tar extraction."""
    if member.issym():
        member.linkname = member.linkname.replace("\\", "/")
    return member


def _ensure_path_within(
    root: Path,
    path: Path,
    *,
    strict: bool = True,
) -> None:
    try:
        resolved = path.resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"path escapes bundle root: {path}") from exc


def _ensure_artifact_within(root: Path, path: Path) -> None:
    """Reject artifacts or symlink targets that leave *root*."""
    if not path.is_symlink():
        _ensure_path_within(root, path)
        return

    _ensure_path_within(root, path.parent)
    raw_target = os.readlink(path)
    if Path(raw_target).is_absolute():
        raise ValueError(f"path escapes bundle root: {path}")
    portable_target = _portable_link_target(path)
    link_target = PurePosixPath(portable_target)
    if link_target.is_absolute():
        raise ValueError(f"path escapes bundle root: {path}")
    target = path.parent.resolve(strict=True) / Path(*link_target.parts)
    _ensure_path_within(root, target)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"path escapes bundle root: {path}")


__all__ = [
    "DEFAULT_BUNDLE_LIMIT_BYTES",
    "DEFAULT_MODEL",
    "MANIFEST_NAME",
    "BundleResult",
    "add_airgap_command",
    "build_bundle",
    "verify_bundle",
]
