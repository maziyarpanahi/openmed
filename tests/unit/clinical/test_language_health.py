"""Focused tests for the offline language-route health matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.language_health import (
    LanguageHealthError,
    build_language_health_matrix,
    check_language_health,
    require_language_health,
)
from openmed.core import LanguagePack, LanguagePackRegistry


def _registry(*, code: str = "en", model: str = "OpenMed/synthetic-pii"):
    registry = LanguagePackRegistry()
    registry.register(
        LanguagePack(
            code=code,
            scripts=("Latin",),
            default_model=model,
            segmenter_id="unicode-sentence",
            recognizers=("regex", "model"),
            surrogate_locale="en_US",
            policy_overrides={"profile": "balanced"},
        )
    )
    return registry


def _manifest(model: str = "OpenMed/synthetic-pii") -> list[dict[str, object]]:
    return [
        {
            "repo_id": model,
            "family": "PII",
            "languages": ["en"],
        }
    ]


def _fixture_root(tmp_path: Path, payload: dict[str, object]) -> Path:
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "synthetic.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return root


def test_matrix_is_deterministic_json_ready_and_text_free(tmp_path: Path) -> None:
    root = _fixture_root(
        tmp_path,
        {
            "language": "en-US",
            "text": "Synthetic Patient 0001",
            "metadata": {"synthetic": True},
        },
    )
    kwargs = {
        "registry": _registry(),
        "manifest_rows": _manifest(),
        "fixture_roots": (root,),
        "languages": ("en-US",),
        "policy_names": ("clinical_minimal_redaction",),
    }

    first = build_language_health_matrix(**kwargs)
    second = build_language_health_matrix(**kwargs)

    assert first == second
    assert first["summary"]["issue_count"] == 0
    row = first["languages"][0]
    assert row["language"] == "en"
    assert row["status"] == "healthy"
    assert all(
        row[component]["status"] == "filled" for component in first["components"]
    )
    assert row["fixture"]["includes_text"] is False
    assert first["sources"]["includes_fixture_text"] is False
    serialized = json.dumps(first, sort_keys=True)
    assert "Synthetic Patient 0001" not in serialized


def test_matrix_reports_missing_route_model_fixture_and_policy_entries(
    tmp_path: Path,
) -> None:
    root = _fixture_root(
        tmp_path,
        {
            "language": "yy",
            "text": "Synthetic fixture only",
            "metadata": {"synthetic": True},
        },
    )
    report = build_language_health_matrix(
        registry=_registry(code="xx", model="OpenMed/missing-pii"),
        manifest_rows=_manifest("OpenMed/other-pii"),
        fixture_roots=(root,),
        languages=("xx", "yy"),
        policy_names=(),
    )

    rows = {row["language"]: row for row in report["languages"]}
    assert rows["xx"]["status"] == "missing"
    assert rows["xx"]["route"]["status"] == "filled"
    assert rows["xx"]["model"]["status"] == "missing"
    assert rows["xx"]["fixture"]["status"] == "missing"
    assert rows["xx"]["policy"]["status"] == "missing"
    assert rows["yy"]["route"]["status"] == "missing"
    assert rows["yy"]["fixture"]["status"] == "filled"
    assert any(issue["component"] == "route" for issue in report["issues"])
    assert (
        check_language_health(
            registry=_registry(code="xx", model="OpenMed/missing-pii"),
            manifest_rows=_manifest("OpenMed/other-pii"),
            fixture_roots=(root,),
            languages=("xx",),
            policy_names=(),
        )
        > 0
    )
    with pytest.raises(LanguageHealthError, match="issue"):
        require_language_health(
            registry=_registry(code="xx", model="OpenMed/missing-pii"),
            manifest_rows=_manifest("OpenMed/other-pii"),
            fixture_roots=(root,),
            languages=("xx",),
            policy_names=(),
        )


def test_fixture_safety_findings_do_not_echo_fixture_values(tmp_path: Path) -> None:
    root = _fixture_root(
        tmp_path,
        {
            "language": "en",
            "text": "Sensitive-looking value must stay out of reports",
            "metadata": {"synthetic": False, "contains_real_phi": True},
        },
    )
    report = build_language_health_matrix(
        registry=_registry(),
        manifest_rows=_manifest(),
        fixture_roots=(root,),
        languages=("en",),
        policy_names=("clinical_minimal_redaction",),
    )

    row = report["languages"][0]
    assert row["fixture"]["status"] == "contradictory"
    assert report["sources"]["includes_fixture_text"] is False
    assert "Sensitive-looking value" not in json.dumps(report)
