"""Contract tests for the native aarch64 edge benchmark workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "edge-benchmark.yml"


def test_edge_workflow_runs_both_profiles_on_native_aarch64() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04-arm" in content
    assert 'test "$(uname -m)" = "aarch64"' in content
    assert "- jetson-nano" in content
    assert "- raspberry-pi-5" in content
    assert "--require-aarch64" in content


def test_edge_workflow_uses_frozen_minimal_install_and_offline_gate() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "uv sync --frozen --extra edge-sbc --no-dev --no-editable" in content
    assert 'OPENMED_OFFLINE: "1"' in content
    assert "-m openmed.eval.edge_benchmark" in content
    assert "-m openmed.eval.footprint_gate" in content
    assert "--install-path" in content
    assert "CPUExecutionProvider" in content
    assert "upload-artifact@v7" in content
