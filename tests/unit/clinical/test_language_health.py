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


def test_fixture_source_paths_are_hashed_without_exposing_values(
    tmp_path: Path,
) -> None:
    sensitive_value = "Synthetic Patient Alice 123-45-6789"
    root = tmp_path / sensitive_value
    root.mkdir()
    (root / f"{sensitive_value}.json").write_text(
        json.dumps(
            {
                "language": "en",
                "text": sensitive_value,
                "metadata": {"synthetic": True},
            }
        ),
        encoding="utf-8",
    )

    report = build_language_health_matrix(
        registry=_registry(),
        manifest_rows=_manifest(),
        fixture_roots=(root,),
        languages=("en",),
        policy_names=("clinical_minimal_redaction",),
    )

    serialized = json.dumps(report, sort_keys=True)
    assert sensitive_value not in serialized
    assert report["languages"][0]["fixture"]["source_ids"][0].startswith(
        "external:sha256:"
    )
    assert report["sources"]["fixture_roots"][0].startswith("external:sha256:")


def test_fixture_language_metadata_does_not_echo_arbitrary_values(
    tmp_path: Path,
) -> None:
    marker = "SyntheticCaseSecret"
    root = _fixture_root(
        tmp_path,
        {"language": marker, "metadata": {"synthetic": True}},
    )

    report = build_language_health_matrix(
        registry=_registry(),
        manifest_rows=_manifest(),
        fixture_roots=(root,),
        policy_names=("clinical_minimal_redaction",),
    )

    assert marker.casefold() not in json.dumps(report).casefold()


@pytest.mark.parametrize("marker_value", ["false", 0, None])
def test_nonboolean_fixture_safety_flags_never_certify_data(tmp_path, marker_value):
    root = _fixture_root(
        tmp_path,
        {
            "language": "en",
            "metadata": {"synthetic": True, "contains_phi": marker_value},
        },
    )
    report = build_language_health_matrix(
        registry=_registry(),
        manifest_rows=_manifest(),
        fixture_roots=(root,),
        languages=("en",),
    )
    assert report["languages"][0]["fixture"]["status"] == "contradictory"


def test_nested_span_marker_cannot_certify_an_entire_record(tmp_path):
    root = _fixture_root(tmp_path, {"language": "en", "spans": [{"synthetic": True}]})
    report = build_language_health_matrix(
        registry=_registry(),
        manifest_rows=_manifest(),
        fixture_roots=(root,),
        languages=("en",),
    )
    assert report["languages"][0]["fixture"]["status"] != "filled"


def test_model_and_policy_metadata_are_not_copied(tmp_path):
    marker = "SyntheticPatientSecret"
    root = _fixture_root(tmp_path, {"language": "en", "synthetic": True})
    report = build_language_health_matrix(
        registry=_registry(model=marker),
        manifest_rows=_manifest(marker),
        fixture_roots=(root,),
        languages=("en",),
        policy_names=(marker,),
    )
    assert marker not in json.dumps(report)


def test_script_verdict_cannot_copy_untrusted_values(tmp_path):
    marker = "SyntheticPatientSecret"
    rows = _manifest()
    rows[0]["languages"] = ["hi"]
    rows[0]["script_coverage"] = {"devanagari": {"verdict": marker}}
    report = build_language_health_matrix(
        registry=_registry(code="hi"),
        manifest_rows=rows,
        fixture_roots=(),
        languages=("hi",),
    )
    assert marker not in json.dumps(report)


def test_manifest_iterator_failure_has_no_sensitive_error_context():
    def broken():
        raise RuntimeError("SyntheticPatientSecret")
        yield {}

    with pytest.raises(ValueError) as caught:
        build_language_health_matrix(manifest_rows=broken(), fixture_roots=())
    assert "SyntheticPatientSecret" not in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_language_health_bounds_manifest_collection():
    with pytest.raises(ValueError):
        build_language_health_matrix(
            manifest_rows=({} for _ in range(8193)), fixture_roots=()
        )


def test_fixture_source_order_does_not_change_matrix(tmp_path):
    roots = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
    kwargs = dict(registry=_registry(), manifest_rows=_manifest(), languages=("en",))
    assert build_language_health_matrix(
        fixture_roots=roots, **kwargs
    ) == build_language_health_matrix(fixture_roots=reversed(roots), **kwargs)


def test_route_extension_values_are_hashed(tmp_path):
    marker = "SyntheticPatientSecret"
    registry = LanguagePackRegistry()
    registry.register(
        LanguagePack(
            code="en",
            scripts=(marker,),
            default_model=marker,
            segmenter_id=marker,
            recognizers=(marker,),
            surrogate_locale=marker,
            policy_overrides={"profile": marker},
        )
    )
    report = build_language_health_matrix(
        registry=registry, manifest_rows=[], fixture_roots=(), languages=("en",)
    )
    assert marker not in json.dumps(report)


def test_cyclic_manifest_metadata_fails_without_context():
    row = {}
    row["cycle"] = row
    with pytest.raises(ValueError) as caught:
        build_language_health_matrix(manifest_rows=[row], fixture_roots=())
    assert caught.value.__context__ is None


def test_default_matrix_is_offline_and_serializable():
    report = build_language_health_matrix()
    assert report["summary"]["language_count"] >= 35
    assert report == build_language_health_matrix()
    json.dumps(report)
