"""Publish converted OpenMed model artifacts to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.core.baseline import load_baseline_store, update_baseline_entry
from openmed.core.model_card import DEFAULT_ARXIV, render_model_card, write_model_card
from openmed.core.model_registry import load_manifest_rows
from openmed.core.repro_hash import compute_reproducibility_hash, resolve_git_sha
from openmed.eval.report import read_reports, write_benchmark_cards, write_leaderboard

logger = logging.getLogger(__name__)

DEFAULT_ORG = "OpenMed"
DEFAULT_TOKEN_ENV = "HF_WRITE_TOKEN"
DEFAULT_MANIFEST_PATH = Path("models.jsonl")
DEFAULT_MODEL_CARD_COMMIT_MESSAGE = "Update generated model card"

_VERSION_SUFFIX_RE = re.compile(r"-v\d+(?=$|-)")
_SAFE_REPO_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ONNX_FORMAT_ALIASES = {
    "onnx",
    "onnx-fp32",
    "onnx-float32",
    "webgpu",
    "onnx-webgpu",
    "webgpu-onnx",
}
_QUANTIZED_FORMAT_ALIASES = {
    "int8": "int8",
    "8bit": "int8",
    "8-bit": "int8",
    "onnx-int8": "int8",
    "int4": "int4",
    "4bit": "int4",
    "4-bit": "int4",
    "onnx-int4": "int4",
    "awq": "awq",
    "openmed-awq": "awq",
    "gptq": "gptq",
    "openmed-gptq": "gptq",
}


class HfPublishError(RuntimeError):
    """Raised when a model artifact cannot be published safely."""


@dataclass(frozen=True)
class PublishResult:
    """Result for one artifact publish attempt."""

    repo_id: str
    artifact_dir: Path
    manifest_row: dict[str, Any]
    skipped: bool = False


def read_hf_token(env_var: str = DEFAULT_TOKEN_ENV) -> str:
    """Read the write token from *env_var* without logging or exposing it."""

    token = os.environ.get(env_var)
    if not token:
        raise HfPublishError(
            f"{env_var} is required to publish model artifacts. "
            "Configure it as a protected CI secret before enabling publish."
        )
    return token


def target_repo_id(
    source_model_id: str,
    format_name: str,
    *,
    org: str = DEFAULT_ORG,
    version: int = 1,
) -> str:
    """Return the OpenMed target repo id for a source model and artifact format."""

    repo_name = _source_repo_name(source_model_id)
    if not _VERSION_SUFFIX_RE.search(repo_name):
        repo_name = f"{repo_name}-v{version}"

    suffix = _format_repo_suffix(format_name)
    if suffix and not repo_name.endswith(f"-{suffix}"):
        repo_name = f"{repo_name}-{suffix}"

    return f"{org}/{repo_name}"


def build_manifest_row(
    *,
    repo_id: str,
    source_model_id: str,
    artifact_dir: str | Path,
    format_name: str,
    formats: list[str] | tuple[str, ...] | None = None,
    released: str | None = None,
    recipe: Any | None = None,
    data_manifest: Any | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Build a models.jsonl-compatible row for a published artifact."""

    artifact_dir = Path(artifact_dir)
    config = _read_optional_json(artifact_dir / "config.json")
    labels = _canonical_labels(repo_id, config)
    released = released or datetime.now(timezone.utc).date().isoformat()
    artifact_hash = artifact_sha256(artifact_dir)
    manifest_format = _manifest_format_name(format_name)
    manifest_formats = _manifest_format_names(formats or [format_name])
    reproducibility_hash = compute_reproducibility_hash(
        recipe=recipe
        or _default_repro_recipe(
            repo_id=repo_id,
            format_name=manifest_format,
            artifact_hash=artifact_hash,
        ),
        data_manifest=data_manifest
        if data_manifest is not None
        else _default_data_manifest(artifact_dir, artifact_hash),
        base_model=source_model_id,
        git_sha=git_sha,
    )

    if not _SHA256_RE.fullmatch(reproducibility_hash):
        raise HfPublishError("invalid reproducibility hash generated for publish row")

    return {
        "repo_id": repo_id,
        "family": _family(repo_id),
        "task": str(config.get("_mlx_task") or "token-classification"),
        "languages": _languages(repo_id),
        "tier": _tier(repo_id),
        "param_count": _param_count(repo_id),
        "architecture": _architecture(repo_id, config),
        "base_model": source_model_id,
        "formats": manifest_formats,
        "canonical_labels": labels,
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": DEFAULT_ARXIV,
        "license": "apache-2.0",
        "reproducibility_hash": reproducibility_hash,
        "released": released,
    }


def append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append or replace *row* in a JSONL manifest keyed by repo_id."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    replaced = False

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s: %s",
                        path,
                        line.strip()[:120],
                    )
                    continue
                if existing.get("repo_id") == row["repo_id"]:
                    rows.append(_merge_manifest_row(existing, row))
                    replaced = True
                else:
                    rows.append(existing)

    if not replaced:
        rows.append(row)

    with path.open("w", encoding="utf-8") as handle:
        for manifest_row in rows:
            handle.write(
                json.dumps(manifest_row, sort_keys=False, separators=(",", ":"))
            )
            handle.write("\n")


def publish_model_card(
    row: dict[str, Any],
    *,
    token: str | None = None,
    api: Any | None = None,
    commit_message: str | None = None,
) -> Any:
    """Render and upload a manifest row as ``README.md`` for its model repo."""

    api = api or _load_hf_api()
    return api.upload_file(
        path_or_fileobj=render_model_card(row).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=row["repo_id"],
        repo_type="model",
        token=token,
        commit_message=commit_message or DEFAULT_MODEL_CARD_COMMIT_MESSAGE,
    )


def publish_artifact(
    *,
    artifact_dir: str | Path,
    source_model_id: str,
    format_name: str,
    formats: list[str] | tuple[str, ...] | None = None,
    repo_id: str | None = None,
    org: str = DEFAULT_ORG,
    version: int = 1,
    token: str | None = None,
    token_env: str = DEFAULT_TOKEN_ENV,
    manifest_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    benchmark_report_paths: list[str | Path] | None = None,
    benchmarks_dir: str | Path | None = None,
    leaderboard_dir: str | Path | None = None,
    status_output_path: str | Path | None = None,
    smoke_status: str = "green",
    smoke_failure_reason: str | None = None,
    api: Any | None = None,
    private: bool = False,
    skip_existing: bool = True,
    released: str | None = None,
    recipe: Any | None = None,
    data_manifest: Any | None = None,
    git_sha: str | None = None,
) -> PublishResult:
    """Create a target repo, upload *artifact_dir*, and emit a manifest row."""

    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise HfPublishError(f"artifact directory does not exist: {artifact_dir}")

    token = token or read_hf_token(token_env)
    repo_id = repo_id or target_repo_id(
        source_model_id,
        format_name,
        org=org,
        version=version,
    )
    api = api or _load_hf_api()
    resolved_git_sha = git_sha or resolve_git_sha()
    row = build_manifest_row(
        repo_id=repo_id,
        source_model_id=source_model_id,
        artifact_dir=artifact_dir,
        format_name=format_name,
        formats=formats,
        released=released,
        recipe=recipe,
        data_manifest=data_manifest,
        git_sha=resolved_git_sha,
    )
    write_model_card(artifact_dir / "README.md", row)

    skipped = False
    if skip_existing and _repo_exists(api, repo_id=repo_id, token=token):
        skipped = True
    else:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(artifact_dir),
            token=token,
            commit_message=f"Publish {format_name} artifact",
        )

    if manifest_path is not None:
        append_manifest_row(manifest_path, row)
    if baseline_path is not None:
        update_baseline_entry(
            baseline_path,
            family=str(row["family"]),
            tier=row.get("tier"),
            format_name=str(row["formats"][0]),
            metrics=baseline_metrics
            if baseline_metrics is not None
            else row["benchmark"],
            reproducibility_hash=str(row["reproducibility_hash"]),
            repo_id=str(row["repo_id"]),
            source_model_id=source_model_id,
            released=str(row["released"]) if row.get("released") else None,
            git_sha=resolved_git_sha,
            metadata={
                "architecture": row.get("architecture"),
                "base_model": row.get("base_model"),
                "task": row.get("task"),
            },
        )
    if any((benchmarks_dir, leaderboard_dir, status_output_path)):
        if manifest_path is None or baseline_path is None:
            raise HfPublishError(
                "manifest and baseline paths are required to refresh status outputs"
            )
        _refresh_status_outputs(
            manifest_path=manifest_path,
            baseline_path=baseline_path,
            benchmark_report_paths=benchmark_report_paths or [],
            benchmarks_dir=benchmarks_dir,
            leaderboard_dir=leaderboard_dir,
            status_output_path=status_output_path,
            smoke_status=smoke_status,
            smoke_failure_reason=smoke_failure_reason,
        )

    return PublishResult(
        repo_id=repo_id,
        artifact_dir=artifact_dir,
        manifest_row=row,
        skipped=skipped,
    )


def artifact_sha256(path: str | Path) -> str:
    """Return a deterministic sha256 digest for a file or directory tree."""

    path = Path(path)
    if not path.exists():
        raise HfPublishError(f"artifact path does not exist: {path}")

    digest = hashlib.sha256()
    root = path if path.is_dir() else path.parent
    paths = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )

    for file_path in paths:
        relative = file_path.relative_to(root).as_posix()
        if relative == "README.md":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish one converted OpenMed model artifact to the Hub.",
    )
    parser.add_argument("--model", required=True, help="Source model id")
    parser.add_argument(
        "--artifact-dir", required=True, help="Converted artifact directory"
    )
    parser.add_argument(
        "--format",
        required=True,
        dest="format_name",
        help="Published artifact format, such as mlx-fp, mlx-8bit, or coreml",
    )
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated manifest formats when one repo carries multiple artifacts",
    )
    parser.add_argument("--repo-id", default=None, help="Explicit target repo id")
    parser.add_argument("--org", default=DEFAULT_ORG, help="Target organization")
    parser.add_argument(
        "--version", type=int, default=1, help="Version suffix for new repos"
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="JSONL manifest path to append or update",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional last-green baseline JSON path to update after publish",
    )
    parser.add_argument(
        "--baseline-metrics",
        default=None,
        help="Optional JSON object, or @file, with metrics for the baseline entry",
    )
    parser.add_argument(
        "--benchmark-report",
        action="append",
        default=[],
        help="BenchmarkReport JSON file used to refresh status artifacts",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default=None,
        help="Optional docs/benchmarks output directory to refresh",
    )
    parser.add_argument(
        "--leaderboard-dir",
        default=None,
        help="Optional docs/leaderboard output directory to refresh",
    )
    parser.add_argument(
        "--status-output",
        default=None,
        help="Optional status Markdown output path to refresh",
    )
    parser.add_argument(
        "--smoke-status",
        choices=["green", "red"],
        default="green",
        help="Smoke-test status rendered into the status page",
    )
    parser.add_argument(
        "--smoke-failure-reason",
        default=None,
        help="Optional smoke-test failure reason rendered for red status",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Git SHA to include in the reproducibility hash",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help="Environment variable containing the write token",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repo as private if it does not already exist",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Upload into an existing target repo instead of skipping it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = publish_artifact(
        artifact_dir=args.artifact_dir,
        source_model_id=args.model,
        format_name=args.format_name,
        formats=_parse_formats_arg(args.formats),
        repo_id=args.repo_id,
        org=args.org,
        version=args.version,
        token_env=args.token_env,
        manifest_path=args.manifest,
        baseline_path=args.baseline,
        baseline_metrics=_parse_json_object_arg(args.baseline_metrics)
        if args.baseline_metrics
        else None,
        benchmark_report_paths=args.benchmark_report,
        benchmarks_dir=args.benchmarks_dir,
        leaderboard_dir=args.leaderboard_dir,
        status_output_path=args.status_output,
        smoke_status=args.smoke_status,
        smoke_failure_reason=args.smoke_failure_reason,
        private=args.private,
        skip_existing=not args.overwrite_existing,
        git_sha=args.git_sha,
    )
    action = "Skipped existing" if result.skipped else "Published"
    print(f"{action} model artifact: {result.repo_id}")


def _load_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HfPublishError(
            "huggingface-hub is required to publish model artifacts. "
            "Install the hf extra before enabling publish."
        ) from exc
    return HfApi()


def _repo_exists(api: Any, *, repo_id: str, token: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        return True
    except Exception as exc:
        if _is_not_found(exc):
            return False
        raise


def _is_not_found(exc: Exception) -> bool:
    if exc.__class__.__name__ == "RepositoryNotFoundError":
        return True
    if getattr(exc, "response", None) is not None:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            return True
    return getattr(exc, "status_code", None) == 404


def _source_repo_name(source_model_id: str) -> str:
    name = source_model_id.rstrip("/").rsplit("/", 1)[-1]
    name = _SAFE_REPO_RE.sub("-", name).strip(".-_")
    if not name:
        raise HfPublishError(
            f"could not derive target repo name from {source_model_id!r}"
        )
    return name


def _format_repo_suffix(format_name: str) -> str:
    normalized = format_name.lower().replace("_", "-")
    if normalized in {"mlx", "mlx-fp", "mlx-float", "mlx-float16"}:
        return "mlx"
    if normalized in {"mlx-8bit", "mlx-int8"}:
        return "mlx-8bit"
    if normalized in {"mlx-4bit", "mlx-int4"}:
        return "mlx-4bit"
    if normalized == "coreml":
        return "coreml"
    if normalized in _ONNX_FORMAT_ALIASES:
        return "onnx"
    quantized = _QUANTIZED_FORMAT_ALIASES.get(normalized)
    if quantized is not None:
        return f"onnx-{quantized}"
    return normalized


def _manifest_format_name(format_name: str) -> str:
    normalized = format_name.lower().replace("_", "-")
    if normalized in {"mlx", "mlx-fp", "mlx-float", "mlx-float16"}:
        return "mlx-fp"
    if normalized in {"mlx-8bit", "mlx-int8"}:
        return "mlx-8bit"
    if normalized in {"mlx-4bit", "mlx-int4"}:
        return "mlx-4bit"
    if normalized in {"onnx-fp32", "onnx-float32"}:
        return "onnx"
    if normalized in {"onnx-webgpu", "webgpu-onnx"}:
        return "webgpu"
    quantized = _QUANTIZED_FORMAT_ALIASES.get(normalized)
    if quantized is not None:
        return quantized
    return normalized


def _manifest_format_names(format_names: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for format_name in format_names:
        normalized = _manifest_format_name(format_name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _merge_manifest_row(
    existing: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(replacement)
    existing_formats = existing.get("formats")
    replacement_formats = replacement.get("formats")
    if isinstance(existing_formats, list) and isinstance(replacement_formats, list):
        merged["formats"] = _manifest_format_names(
            [
                *[str(item) for item in existing_formats],
                *[str(item) for item in replacement_formats],
            ]
        )
    return merged


def _parse_json_object_arg(value: str) -> dict[str, Any]:
    source = value
    if value.startswith("@"):
        source = Path(value[1:]).read_text(encoding="utf-8")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise HfPublishError(
            f"--baseline-metrics is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HfPublishError("--baseline-metrics must be a JSON object")
    return payload


def _parse_formats_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    formats = [item.strip() for item in value.split(",") if item.strip()]
    return formats or None


def _refresh_status_outputs(
    *,
    manifest_path: str | Path,
    baseline_path: str | Path,
    benchmark_report_paths: list[str | Path],
    benchmarks_dir: str | Path | None,
    leaderboard_dir: str | Path | None,
    status_output_path: str | Path | None,
    smoke_status: str,
    smoke_failure_reason: str | None,
) -> None:
    manifest_rows = load_manifest_rows(Path(manifest_path))
    baseline_store = load_baseline_store(Path(baseline_path))
    reports = read_reports(benchmark_report_paths)
    if benchmarks_dir is not None:
        write_benchmark_cards(
            reports,
            benchmarks_dir,
            manifest_rows=manifest_rows,
        )
    if leaderboard_dir is not None:
        write_leaderboard(
            leaderboard_dir,
            manifest_rows=manifest_rows,
            reports=reports,
            baseline_store=baseline_store,
        )
    if status_output_path is not None:
        from scripts.status.generate_status import write_status_page

        write_status_page(
            status_output_path,
            manifest_rows=manifest_rows,
            baseline_store=baseline_store,
            reports=reports,
            smoke_status=smoke_status,
            smoke_failure_reason=smoke_failure_reason,
        )


def _default_repro_recipe(
    *,
    repo_id: str,
    format_name: str,
    artifact_hash: str,
) -> dict[str, Any]:
    return {
        "artifact_hash": artifact_hash,
        "format": format_name,
        "repo_id": repo_id,
    }


def _default_data_manifest(artifact_dir: Path, artifact_hash: str) -> dict[str, Any]:
    artifact_manifest = _read_optional_json(artifact_dir / "openmed-mlx.json")
    return {
        "artifact_hash": artifact_hash,
        "artifact_manifest": artifact_manifest or None,
    }


def _family(repo_id: str) -> str:
    lowered = repo_id.lower()
    if "pii" in lowered or "privacy-filter" in lowered:
        return "PII"
    if "zero-shot" in lowered or "zeroshot" in lowered:
        return "ZeroShot"
    if "ner" in lowered:
        return "NER"
    return "General"


def _languages(repo_id: str) -> list[str]:
    lowered = repo_id.lower()
    language_names = {
        "arabic": "ar",
        "dutch": "nl",
        "french": "fr",
        "german": "de",
        "hindi": "hi",
        "italian": "it",
        "japanese": "ja",
        "portuguese": "pt",
        "spanish": "es",
        "telugu": "te",
        "turkish": "tr",
    }
    for name, code in language_names.items():
        if name in lowered:
            return [code]
    if "pii" in lowered or "ner" in lowered or "privacy-filter" in lowered:
        return ["en"]
    return []


def _tier(repo_id: str) -> str | None:
    match = re.search(
        r"(?<![A-Za-z])(TinyMed|Tiny|Small|Base|Medium|Large|XLarge)(?![A-Za-z])",
        repo_id,
    )
    if not match:
        return None
    value = match.group(1)
    return "Tiny" if value == "TinyMed" else value


def _param_count(repo_id: str) -> int | None:
    matches = list(re.finditer(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[mMbB])", repo_id))
    if not matches:
        return None
    match = matches[-1]
    multiplier = 1_000_000_000 if match.group("unit").lower() == "b" else 1_000_000
    return int(float(match.group("value")) * multiplier)


def _architecture(repo_id: str, config: dict[str, Any]) -> str | None:
    model_type = config.get("_mlx_model_type") or config.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type

    lowered = repo_id.lower()
    for name in (
        "deberta-v2",
        "xlm-roberta",
        "modernbert",
        "distilbert",
        "eurobert",
        "roberta",
        "bert",
        "gliner",
        "t5",
        "qwen",
        "bge",
        "e5",
    ):
        if name.replace("-", "") in lowered.replace("-", ""):
            return name
    return None


def _canonical_labels(repo_id: str, config: dict[str, Any]) -> list[str]:
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        return []

    labels: list[str] = []
    for value in id2label.values():
        if not isinstance(value, str):
            continue
        label = value
        if label.startswith(("B-", "I-")):
            label = label[2:]
        if label == "O" or not label:
            continue
        if label not in labels:
            labels.append(label)
    return labels


if __name__ == "__main__":
    main()
