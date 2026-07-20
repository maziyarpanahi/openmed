#!/usr/bin/env python3
"""Batch Android ONNX conversion and private Hub publishing.

The runner consumes the canonical ``models.jsonl`` manifest, selects PyTorch
token-classification source checkpoints, converts each with the Android ONNX
profile, and optionally uploads the resulting artifact repositories as private
Hugging Face model repos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

# Hugging Face reads this setting at import time. Direct HTTP transfers were
# more reliable across the full Android rollout while remaining overridable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from openmed.core.hf_publish import publish_artifact, target_repo_id  # noqa: E402
from openmed.onnx.convert import convert  # noqa: E402

DEFAULT_MANIFEST = Path("models.jsonl")
DEFAULT_OUTPUT_ROOT = Path("dist/onnx-android")
DEFAULT_STATUS_LOG = DEFAULT_OUTPUT_ROOT / "status.jsonl"
ANDROID_PROFILE = "android"
ANDROID_FORMAT = "onnx-android"
FINAL_STATUSES = {
    "converted",
    "published_private",
    "skipped_existing_private",
}
PUBLISH_FINAL_STATUSES = {
    "published_private",
    "skipped_existing_private",
}
DERIVED_FORMAT_PREFIXES = ("mlx", "coreml", "onnx", "tflite")
ANDROID_FORMAT_PREFIXES = ("onnx", "tflite")
UNSUPPORTED_ANDROID_ARCHITECTURES = {"gliner", "privacy-filter"}
UNSUPPORTED_ANDROID_REPO_PREFIXES = ("OpenMed/privacy-filter-",)
DERIVED_REPO_SUFFIXES = (
    "-mlx",
    "-mlx-8bit",
    "-mlx-4bit",
    "-coreml",
    "-onnx",
    "-onnx-android",
    "-onnx-int8",
)


@dataclass(frozen=True)
class Candidate:
    """One manifest row selected for Android ONNX conversion."""

    repo_id: str
    row: Mapping[str, Any]

    @property
    def architecture(self) -> str:
        return normalize_token(self.row.get("architecture") or "unknown")

    @property
    def family(self) -> str:
        return normalize_token(self.row.get("family") or "unknown")


@dataclass(frozen=True)
class ExistingArtifactResult:
    """A converted artifact directory that can be published without reconversion."""

    output_dir: Path
    manifest_path: Path
    formats: list[str]


def normalize_token(value: object) -> str:
    """Normalize a manifest token for filtering."""

    return str(value).strip().lower().replace("_", "-")


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load manifest JSONL rows from *path*."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"{path}:{line_number}: invalid JSON: {exc.msg}"
                raise ValueError(message) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def is_android_source_candidate(
    row: Mapping[str, Any],
    *,
    include_derived: bool = False,
    include_existing_android: bool = False,
) -> bool:
    """Return whether *row* should enter the Android ONNX batch."""

    repo_id = str(row.get("repo_id") or "").strip()
    if not repo_id:
        return False
    if repo_id.startswith(UNSUPPORTED_ANDROID_REPO_PREFIXES):
        return False
    if normalize_token(row.get("task")) != "token-classification":
        return False
    if normalize_token(row.get("architecture")) in UNSUPPORTED_ANDROID_ARCHITECTURES:
        return False

    formats = [normalize_token(item) for item in row.get("formats") or []]
    if "pytorch" not in formats:
        return False
    if not include_existing_android and any(
        item.startswith(ANDROID_FORMAT_PREFIXES) for item in formats
    ):
        return False
    if include_derived:
        return True

    has_derived_format = any(
        item != "pytorch" and item.startswith(DERIVED_FORMAT_PREFIXES)
        for item in formats
    )
    has_derived_suffix = repo_id.endswith(DERIVED_REPO_SUFFIXES)
    return not has_derived_format and not has_derived_suffix


def select_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_derived: bool = False,
    include_existing_android: bool = False,
    repo_filters: Sequence[str] = (),
    architecture_filters: Sequence[str] = (),
    family_filters: Sequence[str] = (),
    start_after: str | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    """Return ordered conversion candidates from manifest *rows*."""

    repo_filter_set = {item.strip() for item in repo_filters if item.strip()}
    architecture_filter_set = {
        normalize_token(item) for item in architecture_filters if item.strip()
    }
    family_filter_set = {normalize_token(item) for item in family_filters if item}
    found_start = start_after is None
    candidates: list[Candidate] = []

    for row in rows:
        if not is_android_source_candidate(
            row,
            include_derived=include_derived,
            include_existing_android=include_existing_android,
        ):
            continue
        repo_id = str(row["repo_id"])
        if not found_start:
            found_start = repo_id == start_after
            continue
        candidate = Candidate(repo_id=repo_id, row=row)
        if repo_filter_set and not _repo_matches(repo_id, repo_filter_set):
            continue
        if (
            architecture_filter_set
            and candidate.architecture not in architecture_filter_set
        ):
            continue
        if family_filter_set and candidate.family not in family_filter_set:
            continue
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break

    return candidates


def load_completed_statuses(
    path: Path,
    *,
    final_statuses: set[str] = FINAL_STATUSES,
) -> set[str]:
    """Return source repo ids with final statuses in an existing status log."""

    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if event.get("status") in final_statuses and event.get("source_model_id"):
                completed.add(str(event["source_model_id"]))
    return completed


def append_status(path: Path, event: Mapping[str, Any]) -> None:
    """Append one compact status event to *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def output_dir_for(output_root: Path, repo_id: str) -> Path:
    """Return a deterministic artifact directory for *repo_id*."""

    owner, _, name = repo_id.partition("/")
    if not name:
        owner, name = "local", owner
    return output_root / owner / name


def run(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
) -> int:
    """Run the Android ONNX batch and return a process exit code."""

    stdout = stdout or sys.stdout
    rows = load_manifest_rows(args.manifest)
    candidates = select_candidates(
        rows,
        include_derived=args.include_derived,
        include_existing_android=args.include_existing_android,
        repo_filters=args.model,
        architecture_filters=args.architecture,
        family_filters=args.family,
        start_after=args.start_after,
        limit=args.limit,
    )

    final_statuses = PUBLISH_FINAL_STATUSES if args.publish_to_hub else FINAL_STATUSES
    completed = (
        set()
        if args.no_resume
        else load_completed_statuses(args.status_log, final_statuses=final_statuses)
    )
    pending = [
        candidate for candidate in candidates if candidate.repo_id not in completed
    ]

    if args.publish_to_hub and not os.environ.get(args.publish_token_env):
        stdout.write(
            f"{args.publish_token_env} is required for private Hub publishing.\n"
        )
        return 2

    stdout.write(f"Selected {len(candidates)} candidate(s); {len(pending)} pending.\n")
    if args.dry_run:
        for candidate in pending[: args.dry_run_list_limit]:
            target = target_repo_id(
                candidate.repo_id,
                ANDROID_FORMAT,
                org=args.publish_org,
                version=args.publish_version,
            )
            stdout.write(f"{candidate.repo_id}\t{target}\n")
        return 0

    failures = 0
    for index, candidate in enumerate(pending, start=1):
        target = target_repo_id(
            candidate.repo_id,
            ANDROID_FORMAT,
            org=args.publish_org,
            version=args.publish_version,
        )
        artifact_dir = output_dir_for(args.output_root, candidate.repo_id)
        base_event = {
            "source_model_id": candidate.repo_id,
            "target_repo_id": target,
            "artifact_dir": artifact_dir.as_posix(),
            "private": bool(args.publish_to_hub),
            "timestamp": _utc_now(),
        }
        append_status(args.status_log, {**base_event, "status": "started"})
        stdout.write(f"[{index}/{len(pending)}] {candidate.repo_id}\n")
        try:
            result = None
            if args.reuse_existing_artifacts:
                result = _load_existing_artifact_result(
                    artifact_dir,
                    include_int8=not args.no_int8,
                )
                if result is not None:
                    stdout.write(f"Reusing existing artifact at {artifact_dir}\n")
            if result is None:
                result = convert(
                    candidate.repo_id,
                    artifact_dir,
                    profile=ANDROID_PROFILE,
                    include_int8=not args.no_int8,
                    cache_dir=args.cache_dir,
                    max_seq_length=args.max_seq_length,
                    eval_suite_path=args.eval_suite,
                    publish_to_hub=False,
                )
            event = {
                **base_event,
                "formats": result.formats,
                "manifest_path": result.manifest_path.as_posix(),
            }
            if args.publish_to_hub:
                publish_result = publish_artifact(
                    artifact_dir=result.output_dir,
                    source_model_id=candidate.repo_id,
                    format_name=result.formats[0],
                    formats=result.formats,
                    org=args.publish_org,
                    version=args.publish_version,
                    token_env=args.publish_token_env,
                    manifest_path=args.publish_manifest,
                    private=True,
                    skip_existing=not args.publish_overwrite_existing,
                )
                status = (
                    "skipped_existing_private"
                    if publish_result.skipped
                    else "published_private"
                )
                cleanup_event = (
                    _delete_artifact_dir(result.output_dir, args.output_root)
                    if args.delete_successful_artifacts
                    else {}
                )
                append_status(
                    args.status_log,
                    {
                        **event,
                        **cleanup_event,
                        "status": status,
                        "target_repo_id": publish_result.repo_id,
                    },
                )
            else:
                append_status(args.status_log, {**event, "status": "converted"})
        except Exception as exc:  # pragma: no cover - branch details are logged
            failures += 1
            append_status(
                args.status_log,
                {
                    **base_event,
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error": _redact(str(exc), args.publish_token_env),
                    "traceback": _redact(
                        traceback.format_exc(limit=8),
                        args.publish_token_env,
                    ),
                },
            )
            stdout.write(f"FAILED {candidate.repo_id}: {exc.__class__.__name__}\n")
            if _is_hub_repo_creation_limit(exc):
                stdout.write(
                    "Stopping because Hugging Face repo creation is rate-limited; "
                    "retry after the limit resets.\n"
                )
                return 1
            if args.stop_on_failure:
                return 1

    stdout.write(
        f"Finished {len(pending)} pending candidate(s) with {failures} failure(s).\n"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert PyTorch token-classification source models to Android "
            "ONNX artifacts and optionally publish them as private Hub repos."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-log", type=Path, default=DEFAULT_STATUS_LOG)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--eval-suite", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--start-after", default=None)
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--include-existing-android", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help=(
            "Publish a complete artifact directory already present under "
            "--output-root instead of reconverting it."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-list-limit", type=int, default=25)
    parser.add_argument("--publish-to-hub", action="store_true")
    parser.add_argument("--publish-org", default="OpenMed")
    parser.add_argument("--publish-version", type=int, default=1)
    parser.add_argument("--publish-token-env", default="HF_WRITE_TOKEN")
    parser.add_argument("--publish-manifest", type=Path, default=None)
    parser.add_argument("--publish-overwrite-existing", action="store_true")
    parser.add_argument(
        "--delete-successful-artifacts",
        action="store_true",
        help=(
            "Delete each local per-model artifact directory after a successful "
            "private publish or confirmed existing private target repo."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    return run(build_parser().parse_args(argv))


def _repo_matches(repo_id: str, filters: set[str]) -> bool:
    return repo_id in filters or repo_id.rsplit("/", 1)[-1] in filters


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(text: str, token_env: str) -> str:
    token = os.environ.get(token_env)
    if token:
        text = text.replace(token, "<redacted>")
    return text


def _load_existing_artifact_result(
    artifact_dir: Path,
    *,
    include_int8: bool,
) -> ExistingArtifactResult | None:
    """Load a complete existing conversion artifact, if one is present."""

    manifest_path = artifact_dir / "openmed-onnx.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    formats = manifest.get("formats")
    if not isinstance(formats, list) or not all(
        isinstance(item, str) for item in formats
    ):
        return None
    if ANDROID_FORMAT not in formats:
        return None
    if include_int8 and "onnx-int8" not in formats:
        return None

    required_files = {
        "config.json",
        "id2label.json",
        "model.onnx",
        "model_fp16.onnx",
        "tokenizer.json",
    }
    if "ort-android" in formats:
        required_files.add("model.ort")
    if "onnx-int8" in formats:
        required_files.add("model_int8.onnx")
    if any(not (artifact_dir / name).is_file() for name in required_files):
        return None

    onnx_files = [artifact_dir / "model.onnx", artifact_dir / "model_fp16.onnx"]
    if "onnx-int8" in formats:
        onnx_files.append(artifact_dir / "model_int8.onnx")
    if any(not _onnx_external_data_files_present(path) for path in onnx_files):
        return None

    return ExistingArtifactResult(
        output_dir=artifact_dir,
        manifest_path=manifest_path,
        formats=formats,
    )


def _onnx_external_data_files_present(model_path: Path) -> bool:
    """Validate that ONNX external-data references stay local and are complete."""

    try:
        import onnx
        from google.protobuf.message import DecodeError
    except ImportError:
        return False

    try:
        model = onnx.load(str(model_path), load_external_data=False)
        model_dir = model_path.parent.resolve(strict=False)
        for tensor in model.graph.initializer:
            metadata = {item.key: item.value for item in tensor.external_data}
            location = metadata.get("location")
            if not location:
                if (
                    tensor.external_data
                    or tensor.data_location == onnx.TensorProto.EXTERNAL
                ):
                    return False
                continue
            data_path = (model_path.parent / location).resolve(strict=False)
            if not data_path.is_relative_to(model_dir) or not data_path.is_file():
                return False
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", "0"))
            if offset < 0 or length < 0:
                return False
            data_size = data_path.stat().st_size
            if offset > data_size or (length and data_size < offset + length):
                return False
    except (DecodeError, OSError, TypeError, ValueError):
        return False
    return True


def _is_hub_repo_creation_limit(exc: Exception) -> bool:
    """Return whether *exc* is the Hub daily repository creation limit."""

    text = str(exc).lower()
    return (
        exc.__class__.__name__ == "HfHubHTTPError"
        and "429" in text
        and ("repository creation" in text or "/api/repos/create" in text)
    )


def _delete_artifact_dir(path: Path, output_root: Path) -> dict[str, Any]:
    """Delete one generated artifact directory if it is inside *output_root*."""

    resolved_path = path.resolve(strict=False)
    resolved_root = output_root.resolve(strict=False)
    if resolved_path == resolved_root or not resolved_path.is_relative_to(
        resolved_root
    ):
        return {
            "artifact_deleted": False,
            "cleanup_error": (
                f"refusing to delete artifact path outside output root: {path}"
            ),
        }

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return {
            "artifact_deleted": False,
            "cleanup_error": f"{exc.__class__.__name__}: {exc}",
        }
    return {"artifact_deleted": True}


if __name__ == "__main__":
    raise SystemExit(main())
