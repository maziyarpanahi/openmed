"""Conversion workflow credential policy tests."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONVERT_WORKFLOW = ROOT / ".github" / "workflows" / "convert-models.yml"
HF_TOKEN_POLICY = ROOT / "docs" / "security" / "hf-token-policy.md"


def test_convert_workflow_uses_protected_hf_publish_environment():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "publish_to_hub:" in workflow
    assert "  publish-hf:" in workflow
    assert "name: hf-publish" in workflow
    assert "HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}" in workflow


def test_convert_workflow_fails_publish_job_when_hf_token_is_missing():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "Require HF write token before publish" in workflow
    assert 'if [ -z "${HF_WRITE_TOKEN:-}" ]; then' in workflow
    assert "::error title=Missing HF_WRITE_TOKEN::" in workflow
    assert "exit 1" in workflow
    assert re.search(r"hf_[A-Za-z0-9]{20,}", workflow) is None
    assert 'echo "$HF_WRITE_TOKEN' not in workflow


def test_hf_token_policy_documents_scope_storage_rotation_and_revocation():
    policy = HF_TOKEN_POLICY.read_text(encoding="utf-8")

    assert "org-write access" in policy
    assert "`HF_WRITE_TOKEN` secret" in policy
    assert "`hf-publish` GitHub Actions protected environment" in policy
    assert "Rotate the token every 90 days" in policy
    assert "Revoke the token" in policy
    assert "org-wide write access" in policy
