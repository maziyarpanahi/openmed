"""Tests for the offline ``openmed i18n check`` locale-pack conformance CLI."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any, cast

import pytest

from openmed.cli.i18n_check import (
    CHECK_COMPONENTS,
    CHECK_STATUSES,
    I18N_CHECK_SCHEMA_VERSION,
    REASON_CODES,
    ConformanceFinding,
    ConformanceReport,
    _display_path,
    _relative_reference,
    format_conformance_report,
    run_locale_pack_conformance,
)
from openmed.cli.main import build_parser, main
from openmed.core.language_pack import LANGUAGE_PACK_REGISTRY, LanguagePack
from openmed.core.language_pack_catalog import LANG_TO_LOCALE
from openmed.core.locale_tag import LocaleTagError, normalize_locale_tag

_GOLDEN_TEXT = """\
locale pack conformance: zh [openmed.i18n.locale_pack_conformance.v1]
metadata pass pack_declared: pack 'zh' declares scripts Han
registry pass registry_wired: pack 'zh' resolves segmenter 'jieba'
validator pass national_id_providers_resolved: resolved 1 national-ID validator(s)
surrogate pass surrogate_locale_resolved: surrogate locale 'zh_CN' resolves
fixtures pass fixtures_verified: 1 verified synthetic fixture record(s) for 'zh'
span_integrity pass spans_verified: 1 span payload(s) keep offsets aligned
evidence pass evidence_verified: 1 evidence reference(s) resolve
summary pass=7 fail=0 skipped=0 ok=true"""

_GOLDEN_JSON_SHA256 = "32eb2c8b6556f489219d80778dd0eb7f3ef7e19794f6789c0e6699c509e59aad"

_SPAN_TEXT = "Patient Zhang Wei reports fever."


def _canonical_json(report: ConformanceReport) -> str:
    return json.dumps(
        report.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write(path: Path, payload: object, *, raw: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        path.write_text(str(payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _synthetic_pack(**overrides: Any) -> LanguagePack:
    payload: dict[str, Any] = {
        "code": "zz",
        "scripts": ("Latin",),
        "default_model": "OpenMed/custom-pack",
        "segmenter_id": "pysbd",
        "recognizers": ("builtin-patterns",),
        "surrogate_locale": "en_US",
        "national_id_providers": {"chinese_resident_id": "zh_CN"},
    }
    payload.update(overrides)
    return LanguagePack(**cast(Any, payload))


def _complete_inputs(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    fixture_root = root / "fixtures"
    _write(
        fixture_root / "zh_synthetic.jsonl",
        {"language": "zh", "safety": "verified_synthetic", "count": 3},
    )
    fixtures = [fixture_root]
    spans = [
        _write(
            root / "spans.json",
            {
                "text": _SPAN_TEXT,
                "spans": [
                    {"start": 8, "end": 17, "label": "PERSON", "text": "Zhang Wei"},
                    {"start": 26, "end": 31, "label": "SYMPTOM", "text": "fever"},
                ],
            },
        )
    ]
    evidence = [
        _write(root / "evidence" / "review.json", {"reviewer": "fixture"}),
    ]
    return fixtures, spans, evidence


def _reasons(report: ConformanceReport, component: str) -> list[str]:
    return [
        finding.reason for finding in report.findings if finding.component == component
    ]


def _status(report: ConformanceReport, component: str) -> str:
    statuses = {
        finding.status for finding in report.findings if finding.component == component
    }
    return statuses.pop()


def _complete_report(root: Path) -> ConformanceReport:
    fixtures, spans, evidence = _complete_inputs(root)
    return run_locale_pack_conformance(
        "zh",
        fixture_roots=fixtures,
        span_payloads=spans,
        evidence_paths=evidence,
        repository_root=root,
    )


def test_check_constants_are_closed_sets() -> None:
    assert CHECK_COMPONENTS == (
        "metadata",
        "registry",
        "validator",
        "surrogate",
        "fixtures",
        "span_integrity",
        "evidence",
    )
    assert CHECK_STATUSES == ("pass", "fail", "skipped")
    assert I18N_CHECK_SCHEMA_VERSION == "openmed.i18n.locale_pack_conformance.v1"
    assert REASON_CODES


@pytest.mark.parametrize(
    "kwargs",
    [
        {"component": "nope", "status": "pass", "reason": "pack_declared"},
        {"component": "metadata", "status": "maybe", "reason": "pack_declared"},
        {"component": "metadata", "status": "pass", "reason": "made_up_reason"},
    ],
)
def test_finding_rejects_unknown_component_status_and_reason(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        ConformanceFinding(detail="detail", **kwargs)


def test_report_rejects_empty_schema_version() -> None:
    with pytest.raises(ValueError):
        ConformanceReport(language="zh", findings=(), schema_version="  ")


def test_builtin_pack_passes_declared_components_and_skips_opt_in_ones() -> None:
    report = run_locale_pack_conformance("zh", repository_root=Path.cwd())

    assert report.ok is True
    assert report.counts() == {"pass": 4, "fail": 0, "skipped": 3}
    assert _status(report, "metadata") == "pass"
    assert _status(report, "registry") == "pass"
    assert _status(report, "validator") == "pass"
    assert _status(report, "surrogate") == "pass"
    assert _reasons(report, "fixtures") == ["no_fixture_roots"]
    assert _reasons(report, "span_integrity") == ["no_span_payloads"]
    assert _reasons(report, "evidence") == ["no_evidence_references"]


def test_complete_synthetic_pack_passes_every_component(tmp_path: Path) -> None:
    report = _complete_report(tmp_path)

    assert report.ok is True
    assert report.counts() == {"pass": 7, "fail": 0, "skipped": 0}
    assert [finding.component for finding in report.findings] == list(CHECK_COMPONENTS)


def test_report_text_is_byte_stable(tmp_path: Path) -> None:
    report = _complete_report(tmp_path)

    assert format_conformance_report(report) == _GOLDEN_TEXT


def test_report_json_is_byte_stable(tmp_path: Path) -> None:
    report = _complete_report(tmp_path)
    canonical = _canonical_json(report)

    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == _GOLDEN_JSON_SHA256


def test_repeated_runs_are_deterministic(tmp_path: Path) -> None:
    first = _complete_report(tmp_path)
    second = _complete_report(tmp_path)

    assert first.to_dict() == second.to_dict()
    assert _canonical_json(first) == _canonical_json(second)


def test_check_runs_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("conformance checks must stay offline")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    assert _complete_report(tmp_path).ok is True


def test_registry_and_segmenter_failures_are_independent() -> None:
    unwired = run_locale_pack_conformance("zz", pack=_synthetic_pack())
    bad_segmenter = run_locale_pack_conformance(
        "zz",
        pack=_synthetic_pack(segmenter_id="not-a-segmenter"),
    )

    assert _reasons(unwired, "registry") == ["registry_not_wired"]
    assert _reasons(bad_segmenter, "registry") == [
        "registry_not_wired",
        "segmenter_not_registered",
    ]
    assert _status(unwired, "metadata") == "pass"
    assert _status(unwired, "surrogate") == "pass"


def test_validator_failure_is_isolated() -> None:
    report = run_locale_pack_conformance(
        "zz",
        pack=_synthetic_pack(
            national_id_providers={"does_not_exist": "en_US"},
        ),
    )

    assert _reasons(report, "validator") == ["national_id_validator_missing"]
    assert _status(report, "metadata") == "pass"


def test_validator_component_skips_without_providers() -> None:
    report = run_locale_pack_conformance(
        "zz",
        pack=_synthetic_pack(national_id_providers={}),
    )

    assert _status(report, "validator") == "skipped"
    assert _reasons(report, "validator") == ["no_national_id_providers"]


def test_surrogate_failure_is_isolated() -> None:
    report = run_locale_pack_conformance(
        "zz",
        pack=_synthetic_pack(surrogate_locale="xx_YY"),
    )

    assert _reasons(report, "surrogate") == ["surrogate_locale_invalid"]
    assert _status(report, "metadata") == "pass"


def test_missing_fixture_root_fails(tmp_path: Path) -> None:
    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[tmp_path / "absent"],
        repository_root=tmp_path,
    )

    assert _reasons(report, "fixtures") == ["fixture_root_missing"]
    assert report.ok is False


def test_fixture_without_language_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    _write(root / "fr_only.json", {"language": "fr", "safety": "verified_synthetic"})

    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[root],
        repository_root=tmp_path,
    )

    assert _reasons(report, "fixtures") == ["fixture_language_missing"]


def test_fixture_with_text_keys_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    _write(
        root / "leaky.json",
        {"language": "zh", "safety": "verified_synthetic", "text": "Zhang Wei"},
    )

    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[root],
        repository_root=tmp_path,
    )

    assert _reasons(report, "fixtures") == ["fixture_contains_text"]


def test_fixture_without_verified_safety_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    _write(root / "unverified.json", {"language": "zh", "safety": "unverified"})

    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[root],
        repository_root=tmp_path,
    )

    assert _reasons(report, "fixtures") == ["fixture_not_synthetic"]


def test_unreadable_fixture_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    _write(root / "broken.json", "{not json", raw=True)

    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[root],
        repository_root=tmp_path,
    )

    assert _reasons(report, "fixtures") == ["fixture_unreadable"]


def test_span_mismatch_fails_integrity(tmp_path: Path) -> None:
    payload = _write(
        tmp_path / "bad_spans.json",
        {
            "text": _SPAN_TEXT,
            "spans": [{"start": 8, "end": 17, "label": "PERSON", "text": "Zhang Wey"}],
        },
    )

    report = run_locale_pack_conformance(
        "zh",
        span_payloads=[payload],
        repository_root=tmp_path,
    )

    assert _reasons(report, "span_integrity") == ["span_integrity_failed"]
    assert report.ok is False


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "spans": [{"start": 0, "end": 1, "label": "X", "text": "a"}]},
        {"text": "abc", "spans": "not-a-list"},
        {"text": "abc", "spans": []},
        {"text": "abc", "spans": [{"start": "0", "end": 1, "label": "X", "text": "a"}]},
        {"text": "abc", "spans": [{"start": 0, "end": 1, "label": "", "text": "a"}]},
        {"text": "abc", "spans": [{"start": 0, "end": 1, "label": "X", "text": 7}]},
    ],
)
def test_malformed_span_payloads_fail(tmp_path: Path, payload: object) -> None:
    path = _write(tmp_path / "payload.json", payload)

    report = run_locale_pack_conformance(
        "zh",
        span_payloads=[path],
        repository_root=tmp_path,
    )

    assert _reasons(report, "span_integrity") == ["span_payload_invalid"]


def test_unreadable_span_payload_fails(tmp_path: Path) -> None:
    path = _write(tmp_path / "broken.json", "{not json", raw=True)

    report = run_locale_pack_conformance(
        "zh",
        span_payloads=[path],
        repository_root=tmp_path,
    )

    assert _reasons(report, "span_integrity") == ["span_payload_invalid"]


def test_missing_evidence_fails(tmp_path: Path) -> None:
    report = run_locale_pack_conformance(
        "zh",
        evidence_paths=[tmp_path / "absent.json"],
        repository_root=tmp_path,
    )

    assert _reasons(report, "evidence") == ["evidence_reference_missing"]


def test_duplicate_evidence_fails(tmp_path: Path) -> None:
    path = _write(tmp_path / "review.json", {"reviewer": "fixture"})

    report = run_locale_pack_conformance(
        "zh",
        evidence_paths=[path, path],
        repository_root=tmp_path,
    )

    assert _reasons(report, "evidence") == ["evidence_reference_duplicate"]


def test_evidence_references_follow_repository_and_external_scopes(
    tmp_path: Path,
) -> None:
    inside = _write(tmp_path / "nested" / "review.json", {"reviewer": "fixture"})
    outside = Path.cwd() / "pyproject.toml"

    relative, scope = _relative_reference(inside, tmp_path)
    external_relative, external_scope = _relative_reference(outside, tmp_path)

    assert (relative, scope) == ("nested/review.json", "repository")
    assert relative in inside.as_posix()
    assert external_scope == "external"
    assert external_relative == outside.resolve().as_posix()


def test_display_path_does_not_leak_absolute_paths(tmp_path: Path) -> None:
    outside = Path.cwd() / "pyproject.toml"

    assert _display_path(tmp_path / "review.json", tmp_path) == "review.json"
    assert _display_path(outside, tmp_path) == "pyproject.toml"


def test_finding_details_stay_free_of_absolute_paths(tmp_path: Path) -> None:
    report = run_locale_pack_conformance(
        "zh",
        fixture_roots=[tmp_path / "absent"],
        evidence_paths=[tmp_path / "missing.json"],
        repository_root=tmp_path,
    )

    for finding in report.findings:
        assert str(tmp_path) not in finding.detail


@pytest.mark.parametrize("language", ["", "ZH", "zho", "zh-Hans", "1x"])
def test_invalid_language_raises_value_error(language: str) -> None:
    with pytest.raises(ValueError, match="ISO 639-1"):
        run_locale_pack_conformance(language)


def test_national_id_validators_back_builtin_providers() -> None:
    from openmed.core import pii_i18n

    pack = LANGUAGE_PACK_REGISTRY.find("zh")
    assert pack is not None
    assert callable(getattr(pii_i18n, "validate_chinese_resident_id"))
    assert set(pack.national_id_providers) <= {
        name.removeprefix("validate_")
        for name in dir(pii_i18n)
        if name.startswith("validate_")
    }
    assert "zh_CN" in set(LANG_TO_LOCALE.values())
    assert normalize_locale_tag("en-US") == "en-US"
    with pytest.raises(LocaleTagError) as error:
        normalize_locale_tag("zh_CN")
    assert "locale_tag_invalid" in str(error.value)


def test_every_reported_reason_is_documented(tmp_path: Path) -> None:
    reports = [
        run_locale_pack_conformance("zh", repository_root=Path.cwd()),
        _complete_report(tmp_path),
        run_locale_pack_conformance(
            "zz",
            pack=_synthetic_pack(segmenter_id="nope", surrogate_locale="xx_YY"),
            fixture_roots=[tmp_path / "absent"],
            span_payloads=[tmp_path / "absent.json"],
            evidence_paths=[tmp_path / "absent.json"],
            repository_root=tmp_path,
        ),
    ]

    reasons = {finding.reason for report in reports for finding in report.findings}
    assert reasons
    assert reasons <= REASON_CODES


def test_cli_help_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(["i18n", "check", "zh"])

    assert args.command_path == "i18n check"

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["i18n", "check", "--help"])

    assert excinfo.value.code == 0


def test_cli_help_output_starts_with_command_path(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["i18n", "check", "--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: openmed i18n check")


def test_cli_json_envelope_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fixtures, spans, evidence = _complete_inputs(tmp_path)
    code = main(
        [
            "i18n",
            "check",
            "zh",
            "--fixture-root",
            str(fixtures[0]),
            "--spans",
            str(spans[0]),
            "--evidence",
            str(evidence[0]),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 0
    assert envelope["ok"] is True
    assert envelope["command"] == "i18n check"
    assert envelope["data"]["schema_version"] == I18N_CHECK_SCHEMA_VERSION
    assert envelope["data"]["summary"] == {
        "components": 7,
        "fail": 0,
        "pass": 7,
        "skipped": 0,
    }


def test_cli_json_envelope_fails_on_missing_component(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(
        [
            "i18n",
            "check",
            "zh",
            "--fixture-root",
            str(tmp_path / "absent"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)

    assert code == 1
    # The command itself ran, so the envelope is ok; the negative gate result
    # lives in the payload and the exit status, per openmed/cli/_output.py.
    assert envelope["ok"] is True
    assert envelope["data"]["ok"] is False
    assert _reasons(
        ConformanceReport(
            language=envelope["data"]["language"],
            findings=tuple(
                ConformanceFinding(**finding)
                for finding in envelope["data"]["findings"]
            ),
        ),
        "fixtures",
    ) == ["fixture_root_missing"]


def test_cli_unknown_language_uses_usage_exit_code(
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["i18n", "check", "ZH"])
    captured = capsys.readouterr()

    assert code == 2
    assert "ISO 639-1" in captured.err


def test_cli_group_without_subcommand_prints_help(
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["i18n"])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err.startswith("usage: openmed i18n")
