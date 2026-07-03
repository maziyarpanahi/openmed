"""Tests for multi-architecture container build metadata."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DOCKERFILE = ROOT / "deploy" / "docker" / "Dockerfile"
ROOT_DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github" / "workflows" / "container-multiarch.yml"
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
