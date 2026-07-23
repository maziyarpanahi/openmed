from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed
from openmed.eval import redteam
from openmed.eval.redteam import (
    DEFAULT_REDTEAM_CORPUS,
    RedTeamCorpusError,
    load_redteam_corpus,
    run_redteam,
)


def _write_corpus(path: Path, *cases: dict[str, object]) -> Path:
    path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "unit-zero-width-ssn",
        "abuse_case_id": "AC-01",
        "attack_type": "zero_width_split",
        "language": "en",
        "synthetic": True,
        "text": "Synthetic SSN 123\u200d-45-6789.",
        "expected_protected": [
            {"label": "ssn", "value": "123\u200d-45-6789", "match": "alnum"}
        ],
    }
    case.update(overrides)
    return case


def _protect_every_assertion(case: redteam.RedTeamCase) -> str:
    output = case.text
    for assertion in case.expected_protected:
        output = output.replace(assertion.value, f"[{assertion.label.upper()}]")
    return output


def test_default_corpus_is_synthetic_mapped_and_covers_required_attacks() -> None:
    cases = load_redteam_corpus()
    attack_types = {case.attack_type for case in cases}
    threat_model = Path("docs/security/threat-model.md").read_text(encoding="utf-8")

    assert DEFAULT_REDTEAM_CORPUS.is_file()
    assert {
        "obfuscated_identifier",
        "homoglyph",
        "zero_width_split",
        "role_played_leakage_request",
        "format_edge_identifier",
    } <= attack_types
    assert all(case.synthetic for case in cases)
    assert all(case.expected_protected for case in cases)
    assert all(case.abuse_case_id in threat_model for case in cases)


def test_wheel_force_includes_default_corpus() -> None:
    configuration = Path("pyproject.toml").read_text(encoding="utf-8")

    assert (
        '"eval/redteam/corpus/adversarial_phi.jsonl" = '
        '"openmed/eval/data/adversarial_phi.jsonl"' in configuration
    )


def test_clean_run_emits_case_and_per_attack_bypass_rates() -> None:
    report = run_redteam(
        deidentifier=_protect_every_assertion,
        max_bypass_rate=0.0,
    )

    assert report.case_count == len(load_redteam_corpus())
    assert report.bypassed_cases == 0
    assert report.bypass_rate == 0.0
    assert report.gate_passed is True
    assert report.decision == "PASSED"
    assert all(item.bypass_rate == 0.0 for item in report.attack_reports)
    assert {item.attack_type for item in report.attack_reports} == {
        case.attack_type for case in load_redteam_corpus()
    }
    assert {item.abuse_case_id for item in report.abuse_case_reports} == {
        case.abuse_case_id for case in load_redteam_corpus()
    }


def test_leaky_output_exceeds_threshold_without_putting_phi_in_report(
    tmp_path: Path,
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())
    loaded_case = load_redteam_corpus(corpus)[0]

    report = run_redteam(
        corpus,
        deidentifier=lambda case: case.text,
        max_bypass_rate=0.0,
    )
    rendered = json.dumps(report.to_dict(), sort_keys=True)

    assert report.bypassed_cases == 1
    assert report.bypass_rate == 1.0
    assert report.gate_passed is False
    assert report.decision == "FAILED"
    assert loaded_case.text not in rendered
    for assertion in loaded_case.expected_protected:
        assert assertion.value not in rendered
        assert assertion.value_hash in rendered


def test_processing_errors_are_scored_as_bypasses(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())

    def broken_runner(case: redteam.RedTeamCase) -> str:
        raise RuntimeError("synthetic source must not appear in the report")

    report = run_redteam(
        corpus,
        deidentifier=broken_runner,
        max_bypass_rate=0.0,
    )

    assert report.bypass_rate == 1.0
    assert report.gate_passed is False
    assert report.case_results[0].error_type == "RuntimeError"
    assert "synthetic source" not in json.dumps(report.to_dict())


def test_threshold_is_inclusive_and_optional(tmp_path: Path) -> None:
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(id="leaky"),
        _case(id="protected"),
    )

    def one_bypass(case: redteam.RedTeamCase) -> str:
        return case.text if case.case_id == "leaky" else _protect_every_assertion(case)

    measured = run_redteam(corpus, deidentifier=one_bypass)
    gated = run_redteam(
        corpus,
        deidentifier=one_bypass,
        max_bypass_rate=0.5,
    )

    assert measured.decision == "MEASURED"
    assert measured.gate_configured is False
    assert measured.gate_passed is True
    assert gated.bypass_rate == 0.5
    assert gated.gate_passed is True


def test_threshold_cannot_hide_bypassed_attack_or_abuse_bucket(
    tmp_path: Path,
) -> None:
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(id="critical", abuse_case_id="AC-01", attack_type="critical"),
        _case(id="easy-one", abuse_case_id="AC-02", attack_type="easy"),
        _case(id="easy-two", abuse_case_id="AC-02", attack_type="easy"),
    )

    def one_bypass(case: redteam.RedTeamCase) -> str:
        return (
            case.text if case.case_id == "critical" else _protect_every_assertion(case)
        )

    report = run_redteam(
        corpus,
        deidentifier=one_bypass,
        max_bypass_rate=0.5,
    )

    assert report.bypass_rate == pytest.approx(1 / 3)
    assert report.gate_passed is False
    assert (
        next(
            item for item in report.attack_reports if item.attack_type == "critical"
        ).bypass_rate
        == 1.0
    )
    assert (
        next(
            item for item in report.abuse_case_reports if item.abuse_case_id == "AC-01"
        ).bypass_rate
        == 1.0
    )


@pytest.mark.parametrize(
    "invalid_update",
    (
        {"synthetic": False},
        {"expected_protected": []},
        {"abuse_case_id": "unknown"},
        {"id": "unsafe id with spaces"},
        {"language": "../unsafe"},
        {"unexpected_field": "not allowed"},
    ),
)
def test_corpus_validation_fails_closed(
    tmp_path: Path,
    invalid_update: dict[str, object],
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case(**invalid_update))

    with pytest.raises(RedTeamCorpusError):
        load_redteam_corpus(corpus)


def test_corpus_validation_rejects_unknown_assertion_fields(tmp_path: Path) -> None:
    case = _case()
    assertions = case["expected_protected"]
    assert isinstance(assertions, list)
    assert isinstance(assertions[0], dict)
    assertions[0]["unexpected_field"] = "not allowed"
    corpus = _write_corpus(tmp_path / "corpus.jsonl", case)

    with pytest.raises(RedTeamCorpusError, match="unsupported fields"):
        load_redteam_corpus(corpus)


def test_corpus_validation_rejects_duplicate_keys(tmp_path: Path) -> None:
    rendered = json.dumps(_case())
    rendered = rendered.replace(
        '"id": "unit-zero-width-ssn"',
        '"id": "first", "id": "second"',
        1,
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(rendered + "\n", encoding="utf-8")

    with pytest.raises(RedTeamCorpusError, match="must not repeat keys"):
        load_redteam_corpus(corpus)


def test_corpus_validation_rejects_invalid_utf8(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b"\xff\xfe")

    with pytest.raises(RedTeamCorpusError, match="valid UTF-8"):
        load_redteam_corpus(corpus)


def test_corpus_validation_rejects_non_finite_json_numbers(tmp_path: Path) -> None:
    rendered = json.dumps(_case()).replace('"language": "en"', '"language": NaN')
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(rendered + "\n", encoding="utf-8")

    with pytest.raises(RedTeamCorpusError, match="non-finite number"):
        load_redteam_corpus(corpus)


@pytest.mark.parametrize(
    ("value", "match"),
    (("\u200d", "normalized"), ("---", "alnum")),
)
def test_corpus_validation_rejects_empty_comparison_surfaces(
    tmp_path: Path,
    value: str,
    match: str,
) -> None:
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(
            text=f"Synthetic protected value: {value}",
            expected_protected=[
                {"label": "identifier", "value": value, "match": match}
            ],
        ),
    )

    with pytest.raises(RedTeamCorpusError, match="becomes empty"):
        load_redteam_corpus(corpus)


def test_corpus_validation_rejects_repeated_assertions(tmp_path: Path) -> None:
    assertion = {"label": "ssn", "value": "123\u200d-45-6789", "match": "alnum"}
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(expected_protected=[assertion, assertion]),
    )

    with pytest.raises(RedTeamCorpusError, match="repeats"):
        load_redteam_corpus(corpus)


def test_oversized_output_is_scored_as_a_bypass(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())

    report = run_redteam(
        corpus,
        deidentifier=lambda case: "x" * (redteam.MAX_REDTEAM_OUTPUT_CHARS + 1),
        max_bypass_rate=0.0,
    )

    assert report.gate_passed is False
    assert report.case_results[0].error_type == "ValueError"


def test_normalized_output_is_computed_once_per_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(
            text="Synthetic values 123-45-6789 and jane@example.com.",
            expected_protected=[
                {"label": "ssn", "value": "123-45-6789", "match": "normalized"},
                {
                    "label": "email",
                    "value": "jane@example.com",
                    "match": "normalized",
                },
            ],
        ),
    )
    original_normalizer = redteam._normalized_surface
    calls: list[str] = []

    def recording_normalizer(value: str) -> str:
        calls.append(value)
        return original_normalizer(value)

    monkeypatch.setattr(redteam, "_normalized_surface", recording_normalizer)
    report = run_redteam(corpus, deidentifier=lambda case: case.text)

    assert report.bypassed_cases == 1
    assert calls.count("Synthetic values 123-45-6789 and jane@example.com.") == 1


def test_report_hashes_the_exact_evaluated_corpus_bytes(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())
    evaluated_bytes = corpus.read_bytes()

    def mutate_after_load(case: redteam.RedTeamCase) -> str:
        _write_corpus(corpus, _case(id="replacement"))
        return _protect_every_assertion(case)

    report = run_redteam(corpus, deidentifier=mutate_after_load)

    assert report.corpus_sha256 == (
        f"sha256:{hashlib.sha256(evaluated_bytes).hexdigest()}"
    )


def test_default_runner_forces_local_only_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(
        tmp_path / "corpus.jsonl",
        _case(id="first"),
        _case(id="second"),
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_deidentify(text: str, **kwargs: object) -> SimpleNamespace:
        calls.append((text, kwargs))
        return SimpleNamespace(deidentified_text="Synthetic SSN [SSN].")

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    report = run_redteam(
        corpus,
        model_name="local-fixture-model",
        max_bypass_rate=0.0,
    )

    assert report.gate_passed is True
    assert len(calls) == 2
    assert calls[0][1]["model_name"] == "local-fixture-model"
    assert calls[0][1]["config"].local_only is True
    assert calls[0][1]["use_safety_sweep"] is True
    assert calls[0][1]["loader"] is calls[1][1]["loader"]


def test_cli_returns_nonzero_and_writes_safe_report_when_threshold_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        redteam,
        "_pipeline_deidentify",
        lambda case, **kwargs: case.text,
    )

    exit_code = redteam.main(
        [
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--max-bypass-rate",
            "0",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["decision"] == "FAILED"
    assert payload["bypass_rate"] == 1.0
    assert "123\u200d-45-6789" not in json.dumps(payload)


def test_report_write_rejects_symbolic_link(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())
    report = run_redteam(corpus, deidentifier=_protect_every_assertion)
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "report.json"
    try:
        output.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        report.write_json(output)

    assert target.read_text(encoding="utf-8") == "unchanged"


def test_cli_uses_environment_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", _case())
    output = tmp_path / "report.json"
    monkeypatch.setenv(redteam.REDTEAM_THRESHOLD_ENV_VAR, "0")
    monkeypatch.setattr(
        redteam,
        "_pipeline_deidentify",
        lambda case, **kwargs: case.text,
    )

    exit_code = redteam.main(["--corpus", str(corpus), "--output", str(output)])

    assert exit_code == 1


@pytest.mark.parametrize("threshold", (-0.01, 1.01))
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_redteam(
            deidentifier=_protect_every_assertion,
            max_bypass_rate=threshold,
        )


def test_boolean_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        run_redteam(
            deidentifier=lambda case: case.text,
            max_bypass_rate=True,
        )
