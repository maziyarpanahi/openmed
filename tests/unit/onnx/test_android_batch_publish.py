"""Tests for Android ONNX batch conversion and private publishing."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

from scripts.onnx import batch_android_convert_publish as batch


def test_batch_disables_xet_before_hugging_face_import() -> None:
    env = os.environ.copy()
    env.pop("HF_HUB_DISABLE_XET", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, types; "
                "openmed = types.ModuleType('openmed'); "
                "openmed.__path__ = []; "
                "core = types.ModuleType('openmed.core'); "
                "core.__path__ = []; "
                "onnx = types.ModuleType('openmed.onnx'); "
                "onnx.__path__ = []; "
                "hf_publish = types.ModuleType('openmed.core.hf_publish'); "
                "hf_publish.publish_artifact = object(); "
                "hf_publish.target_repo_id = object(); "
                "convert = types.ModuleType('openmed.onnx.convert'); "
                "convert.convert = object(); "
                "sys.modules['openmed'] = openmed; "
                "sys.modules['openmed.core'] = core; "
                "sys.modules['openmed.onnx'] = onnx; "
                "sys.modules['openmed.core.hf_publish'] = hf_publish; "
                "sys.modules['openmed.onnx.convert'] = convert; "
                "import scripts.onnx.batch_android_convert_publish; "
                "from huggingface_hub.constants import HF_HUB_DISABLE_XET; "
                "print(HF_HUB_DISABLE_XET)"
            ),
        ],
        cwd=Path(__file__).parents[3],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "True"


def test_candidate_selection_defaults_to_pytorch_source_rows() -> None:
    rows = [
        {
            "repo_id": "OpenMed/source",
            "task": "token-classification",
            "formats": ["pytorch"],
            "architecture": "bert",
            "family": "NER",
        },
        {
            "repo_id": "OpenMed/source-mlx",
            "task": "token-classification",
            "formats": ["mlx-fp", "pytorch"],
            "architecture": "bert",
            "family": "NER",
        },
        {
            "repo_id": "OpenMed/source-onnx-android",
            "task": "token-classification",
            "formats": ["onnx-android", "onnx-int8"],
            "architecture": "bert",
            "family": "NER",
        },
        {
            "repo_id": "OpenMed/generator",
            "task": "text-generation",
            "formats": ["pytorch"],
            "architecture": "llama",
            "family": "General",
        },
        {
            "repo_id": "OpenMed/zero-shot-span-model",
            "task": "token-classification",
            "formats": ["pytorch"],
            "architecture": "gliner",
            "family": "ZeroShot",
        },
        {
            "repo_id": "OpenMed/privacy-filter-example",
            "task": "token-classification",
            "formats": ["pytorch"],
            "architecture": "unknown",
            "family": "PII",
        },
    ]

    candidates = batch.select_candidates(rows)

    assert [candidate.repo_id for candidate in candidates] == ["OpenMed/source"]


def test_private_publish_is_forced_for_batch_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
                "architecture": "bert",
                "family": "NER",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    convert_calls = []
    publish_calls = []

    def fake_convert(model_id, output_dir, **kwargs):
        convert_calls.append((model_id, Path(output_dir), kwargs))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        manifest_path = output_dir / "openmed-onnx.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return types.SimpleNamespace(
            output_dir=output_dir,
            manifest_path=manifest_path,
            formats=["onnx-android", "onnx-int8"],
        )

    def fake_publish_artifact(**kwargs):
        publish_calls.append(kwargs)
        return types.SimpleNamespace(
            repo_id="OpenMed/source-v1-onnx-android",
            skipped=False,
        )

    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(batch, "convert", fake_convert)
    monkeypatch.setattr(batch, "publish_artifact", fake_publish_artifact)
    stdout = io.StringIO()

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(tmp_path / "out"),
                "--status-log",
                str(tmp_path / "status.jsonl"),
                "--publish-to-hub",
            ]
        ),
        stdout=stdout,
    )

    assert exit_code == 0
    assert "secret-token" not in stdout.getvalue()
    assert convert_calls[0][2]["profile"] == "android"
    assert publish_calls[0]["private"] is True
    assert publish_calls[0]["format_name"] == "onnx-android"
    assert publish_calls[0]["formats"] == ["onnx-android", "onnx-int8"]

    events = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["status"] == "published_private"
    assert events[-1]["private"] is True


def test_successful_private_publish_can_delete_local_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    artifact_dir = output_root / "OpenMed" / "source"

    def fake_convert(model_id, output_dir, **kwargs):
        del model_id, kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        manifest_path = output_dir / "openmed-onnx.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return types.SimpleNamespace(
            output_dir=output_dir,
            manifest_path=manifest_path,
            formats=["onnx-android", "onnx-int8"],
        )

    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(batch, "convert", fake_convert)
    monkeypatch.setattr(
        batch,
        "publish_artifact",
        lambda **kwargs: types.SimpleNamespace(
            repo_id="OpenMed/source-v1-onnx-android",
            skipped=False,
        ),
    )

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--status-log",
                str(tmp_path / "status.jsonl"),
                "--publish-to-hub",
                "--delete-successful-artifacts",
            ]
        ),
        stdout=io.StringIO(),
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert not artifact_dir.exists()
    assert events[-1]["status"] == "published_private"
    assert events[-1]["artifact_deleted"] is True


def test_missing_token_stops_before_conversion(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    monkeypatch.setattr(batch, "convert", lambda *args, **kwargs: None)
    stdout = io.StringIO()

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--status-log",
                str(tmp_path / "status.jsonl"),
                "--publish-to-hub",
            ]
        ),
        stdout=stdout,
    )

    assert exit_code == 2
    assert "HF_WRITE_TOKEN is required" in stdout.getvalue()
    assert not (tmp_path / "status.jsonl").exists()


def test_publish_mode_does_not_resume_from_convert_only_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status_log = tmp_path / "status.jsonl"
    status_log.write_text(
        json.dumps(
            {
                "source_model_id": "OpenMed/source",
                "status": "converted",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    convert_calls = []

    def fake_convert(model_id, output_dir, **kwargs):
        convert_calls.append(model_id)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "openmed-onnx.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return types.SimpleNamespace(
            output_dir=output_dir,
            manifest_path=manifest_path,
            formats=["onnx-android"],
        )

    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(batch, "convert", fake_convert)
    monkeypatch.setattr(
        batch,
        "publish_artifact",
        lambda **kwargs: types.SimpleNamespace(
            repo_id="OpenMed/source-v1-onnx-android",
            skipped=True,
        ),
    )

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(tmp_path / "out"),
                "--status-log",
                str(status_log),
                "--publish-to-hub",
            ]
        ),
        stdout=io.StringIO(),
    )

    assert exit_code == 0
    assert convert_calls == ["OpenMed/source"]


def test_reuse_existing_artifact_skips_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    artifact_dir = output_root / "OpenMed" / "source"
    artifact_dir.mkdir(parents=True)
    for name in ["config.json", "model.onnx", "model.ort", "model_int8.onnx"]:
        (artifact_dir / name).write_text("artifact", encoding="utf-8")
    (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest_path = artifact_dir / "openmed-onnx.json"
    manifest_path.write_text(
        json.dumps({"formats": ["onnx-android", "onnx-int8", "ort-android"]}),
        encoding="utf-8",
    )
    publish_calls = []

    def fail_convert(*args, **kwargs):
        del args, kwargs
        raise AssertionError("existing artifact should have been reused")

    def fake_publish_artifact(**kwargs):
        publish_calls.append(kwargs)
        return types.SimpleNamespace(
            repo_id="OpenMed/source-v1-onnx-android",
            skipped=False,
        )

    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(batch, "convert", fail_convert)
    monkeypatch.setattr(batch, "publish_artifact", fake_publish_artifact)
    stdout = io.StringIO()

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(output_root),
                "--status-log",
                str(tmp_path / "status.jsonl"),
                "--publish-to-hub",
                "--reuse-existing-artifacts",
            ]
        ),
        stdout=stdout,
    )

    assert exit_code == 0
    assert "Reusing existing artifact" in stdout.getvalue()
    assert publish_calls[0]["artifact_dir"] == artifact_dir
    assert publish_calls[0]["manifest_path"] is None
    assert publish_calls[0]["formats"] == [
        "onnx-android",
        "onnx-int8",
        "ort-android",
    ]


def test_hub_repo_creation_limit_stops_batch(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "task": "token-classification",
                    "formats": ["pytorch"],
                }
            )
            for repo_id in ["OpenMed/one", "OpenMed/two"]
        )
        + "\n",
        encoding="utf-8",
    )
    convert_calls = []

    class HfHubHTTPError(RuntimeError):
        pass

    def fake_convert(model_id, output_dir, **kwargs):
        del kwargs
        convert_calls.append(model_id)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        manifest_path = output_dir / "openmed-onnx.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return types.SimpleNamespace(
            output_dir=output_dir,
            manifest_path=manifest_path,
            formats=["onnx-android"],
        )

    def fake_publish_artifact(**kwargs):
        del kwargs
        raise HfHubHTTPError(
            "429 Too Many Requests: You have exceeded the rate limit for "
            "repository creation. Url: https://huggingface.co/api/repos/create."
        )

    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")
    monkeypatch.setattr(batch, "convert", fake_convert)
    monkeypatch.setattr(batch, "publish_artifact", fake_publish_artifact)
    stdout = io.StringIO()

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(tmp_path / "out"),
                "--status-log",
                str(tmp_path / "status.jsonl"),
                "--publish-to-hub",
            ]
        ),
        stdout=stdout,
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 1
    assert convert_calls == ["OpenMed/one"]
    assert events[-1]["status"] == "failed"
    assert "repo creation is rate-limited" in stdout.getvalue()


def test_dry_run_reports_private_target_repo(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "OpenMed/source",
                "task": "token-classification",
                "formats": ["pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stdout = io.StringIO()

    exit_code = batch.run(
        batch.build_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--dry-run",
            ]
        ),
        stdout=stdout,
    )

    assert exit_code == 0
    assert "OpenMed/source-v1-onnx-android" in stdout.getvalue()
