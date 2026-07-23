"""Tests for multi-architecture container build metadata."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DOCKERFILE = ROOT / "deploy" / "docker" / "Dockerfile"
ROOT_DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github" / "workflows" / "container-multiarch.yml"
ARM_LATENCY_DOCS = ROOT / "docs" / "benchmarks" / "arm-latency.md"
DOCS = ROOT / "docs" / "deploy" / "multi-arch.md"
MKDOCS = ROOT / "mkdocs.yml"

PINNED_PYTHON_BASE_RE = re.compile(
    r"^FROM(?: --platform=\$TARGETPLATFORM)? "
    r"python:3\.11-slim@sha256:[0-9a-f]{64}$",
    re.MULTILINE,
)


def test_deployment_dockerfile_uses_digest_pinned_python_base():
    content = DEPLOY_DOCKERFILE.read_text(encoding="utf-8")

    assert PINNED_PYTHON_BASE_RE.search(content)


def test_root_dockerfile_keeps_compose_base_digest_pinned():
    content = ROOT_DOCKERFILE.read_text(encoding="utf-8")

    assert PINNED_PYTHON_BASE_RE.search(content)


def test_multiarch_workflow_builds_manifest_and_smokes_each_platform():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "docker/build-push-action@v6" in content
    assert "platforms: linux/amd64,linux/arm64" in content
    assert "docker buildx imagetools inspect" in content
    assert "grep -Eq 'Platform:[[:space:]]+linux/amd64'" in content
    assert "grep -Eq 'Platform:[[:space:]]+linux/arm64'" in content
    assert "type=gha,scope=openmed-${{ matrix.arch }}" in content
    assert "openmed.deidentify" in content
    assert "linux/amd64" in content
    assert "linux/arm64" in content


def test_multiarch_workflow_gates_native_arm_sms_latency_offline():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04-arm" in content
    assert 'test "$(uname -m)" = "aarch64"' in content
    assert "test_intentionally_slowed_fixture_trips_gate" in content
    assert 'OPENMED_OFFLINE: "1"' in content
    assert "openmed benchmark latency" in content
    assert "--output arm-latency-report.json" in content
    assert "name: arm-latency-report" in content
    assert "- arm-latency" in content


def test_arm_latency_docs_record_budget_and_reproduction_commands():
    content = ARM_LATENCY_DOCS.read_text(encoding="utf-8")

    assert "Raspberry Pi 5 with 8 GB RAM" in content
    assert "1,500 ms" in content
    assert "1,800 ms" in content
    assert "OPENMED_OFFLINE=1 openmed benchmark latency" in content
    assert "model_int8.onnx" in content
    assert "48a0b2e9269933bef0cf8913239d07996fa2afb107cd223ced95c8decd24ae6b" in content
    assert "SHA-256" in content
    assert "test_intentionally_slowed_fixture_trips_gate" in content
    assert "test_latency_command_emits_offline_int8_json_and_blocks_sockets" in content


def test_multiarch_docs_cover_supported_platforms_and_pull_commands():
    content = DOCS.read_text(encoding="utf-8")

    assert "ghcr.io/maziyarpanahi/openmed:latest" in content
    assert "docker pull --platform linux/arm64" in content
    assert "docker pull --platform linux/amd64" in content
    assert "docker buildx imagetools inspect" in content
    assert "`linux/amd64`" in content
    assert "`linux/arm64`" in content


def test_multiarch_docs_are_in_mkdocs_nav():
    content = MKDOCS.read_text(encoding="utf-8")

    assert "deploy/multi-arch.md" in content
