"""Offline decision-backend qualification with wholly synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path

import pytest

import openmed.eval.decision_qualification as qualification
from openmed.eval.decision_qualification import (
    DecisionQualificationError,
    DecisionQualificationPolicy,
    fit_decision_thresholds,
    qualify_decision_backend,
)
from openmed.structured import decision as decision_module
from openmed.structured.decision import (
    DecisionCalibrationProfile,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    DeterministicDecisionBackend,
    EncoderDecisionBackend,
    decide,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "structured" / "decision_qualification"
DEVELOPMENT = FIXTURES / "development.jsonl"
EVALUATION = FIXTURES / "evaluation.jsonl"
LENIENT = DecisionQualificationPolicy(min_examples_per_option=2)
SCORED: list[str] = []


def synthetic_scorer(text: str, options: tuple[str, ...], mode: DecisionMode):
    """Grade options by synthetic cue words; a stand-in for a local encoder."""
    tokens = set(re.findall(r"\w+", text.casefold()))
    strength = 0.92 if "clearly" in tokens else 0.74 if "likely" in tokens else 0.56
    hits = [option for option in options if option.casefold() in tokens]
    if mode is DecisionMode.MULTI_LABEL:
        return [strength if option in hits else 1 - strength for option in options]
    if len(hits) != 1:
        return [1 / len(options)] * len(options)
    rest = (1 - strength) / (len(options) - 1)
    return [strength if option in hits else rest for option in options]


def synthetic_encoder_backend() -> EncoderDecisionBackend:
    """Return a mocked local encoder backend; no model or network is used."""
    return EncoderDecisionBackend(
        backend_id="synthetic.encoder.v1",
        model_id="synthetic/encoder",
        revision="0.0.0",
        license_id="Apache-2.0",
        scorer=synthetic_scorer,
    )


class SpyBackend:
    """Record the digest of every scored input in call order."""

    def __init__(self, inner) -> None:
        self.identity = inner.identity
        self._inner = inner

    def score(self, request: DecisionRequest):
        SCORED.append(text_digest(request.input_text))
        return self._inner.score(request)


def spy_encoder_backend() -> SpyBackend:
    """Factory used by the command tests to observe scoring order."""
    return SpyBackend(synthetic_encoder_backend())


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    return path


def fictional_evaluation(tmp_path: Path) -> Path:
    rows = []
    for row in records(EVALUATION):
        rows.append(
            {
                **row,
                "id": row["id"].replace("synthetic", "fictional"),
                "input_text": row["input_text"].replace("Synthetic", "Fictional"),
            }
        )
    return write_jsonl(tmp_path / "evaluation-alternate.jsonl", rows)


def run(backend, development=DEVELOPMENT, evaluation=EVALUATION, **options):
    options.setdefault("calibration_id", "site.triage")
    options.setdefault("policy", LENIENT)
    return qualify_decision_backend(
        backend, development=development, evaluation=evaluation, **options
    )


@pytest.fixture(autouse=True)
def _reset_spy():
    SCORED.clear()
    yield
    SCORED.clear()


def test_deterministic_backend_emits_per_mode_profiles_and_file_digests():
    report = run(DeterministicDecisionBackend())
    payload = report.to_dict()

    assert report.qualified and report.status == "qualified"
    assert payload["reasons"] == []
    assert payload["human_review_required"] is True
    assert payload["autonomous_action"] is False
    assert payload["inputs"]["development"] == {
        "digest": "sha256:" + hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
        "records": 26,
    }
    assert payload["inputs"]["evaluation"]["records"] == 20
    assert payload["fit"]["split"] == "development"
    assert payload["fit"]["input_digest"] == payload["inputs"]["development"]["digest"]
    assert payload["fit"]["modes"]["fixed_choice"]["thresholds"] == {
        "minimum_confidence": 1.0,
        "minimum_margin": 0.95,
    }
    assert payload["fit"]["modes"]["multi_label"]["thresholds"] == {
        "multi_label_threshold": 1.0
    }
    fixed = payload["modes"]["fixed_choice"]["evaluation"]
    assert fixed["total"] == 8 and fixed["covered"] == 6
    assert fixed["accuracy"] == 1.0
    assert fixed["coverage"] == 0.75
    assert fixed["abstention_rate"] == 0.25
    assert fixed["states"] == {"abstained": 2, "success": 6}
    assert "expected_calibration_error" in fixed
    recalls = {
        item["evaluation_examples"]: item["evaluation_recall"]
        for item in payload["modes"]["fixed_choice"]["options"]
    }
    assert recalls == {3: 1.0, 5: 0.6}

    profiles = report.profiles()
    assert set(profiles) == {
        "site.triage.fixed_choice",
        "site.triage.boolean_choice",
        "site.triage.multi_label",
    }
    profile = profiles["site.triage.fixed_choice"]
    assert isinstance(profile, DecisionCalibrationProfile)
    assert profile.in_domain_digest == payload["inputs"]["development"]["digest"]
    assert profile.out_of_domain_digest == payload["inputs"]["evaluation"]["digest"]
    # Thresholds for other modes fail closed if a profile is reused.
    assert profile.multi_label_threshold == 1.0
    assert profiles["site.triage.multi_label"].minimum_confidence == 1.0
    assert report.digest == payload["report_digest"]


def test_mocked_encoder_backend_fits_graded_thresholds():
    payload = run(synthetic_encoder_backend()).to_dict()

    assert payload["qualified"] is True
    assert payload["backend"]["kind"] == "encoder"
    assert payload["backend"]["model_id"] == "synthetic/encoder"
    fit = payload["fit"]["modes"]
    assert fit["fixed_choice"]["thresholds"] == {
        "minimum_confidence": 0.74,
        "minimum_margin": pytest.approx(0.48),
    }
    assert fit["boolean_choice"]["thresholds"]["minimum_confidence"] == 0.74
    assert fit["multi_label"]["thresholds"] == {"multi_label_threshold": 0.56}
    evaluation = payload["modes"]["fixed_choice"]["evaluation"]
    assert evaluation["coverage"] == 0.75
    assert evaluation["expected_calibration_error"] == pytest.approx(0.14)


def test_emitted_profiles_reproduce_evaluation_through_decide():
    backend = synthetic_encoder_backend()
    report = run(backend)
    profiles = report.profiles()
    rows = [row for row in records(EVALUATION) if row["mode"] == "fixed_choice"]

    results = [
        decide(
            DecisionRequest(
                mode=row["mode"],
                input_text=row["input_text"],
                options=tuple(row["options"]),
                calibration_id="site.triage.fixed_choice",
            ),
            backend=backend,
            calibration_profiles=profiles,
        )
        for row in rows
    ]

    covered = sum(result.state is DecisionState.SUCCESS for result in results)
    expected = report.to_dict()["modes"]["fixed_choice"]["evaluation"]["covered"]
    assert covered == expected


def test_thresholds_do_not_change_with_disjoint_evaluation_sets(tmp_path):
    alternate = fictional_evaluation(tmp_path)
    dev_inputs = {text_digest(row["input_text"]) for row in records(DEVELOPMENT)}
    first_inputs = {text_digest(row["input_text"]) for row in records(EVALUATION)}
    second_inputs = {text_digest(row["input_text"]) for row in records(alternate)}
    assert dev_inputs.isdisjoint(first_inputs | second_inputs)
    assert first_inputs.isdisjoint(second_inputs)

    first = run(synthetic_encoder_backend()).to_dict()
    second = run(synthetic_encoder_backend(), evaluation=alternate).to_dict()

    assert first["inputs"]["evaluation"] != second["inputs"]["evaluation"]
    assert first["fit"] == second["fit"]
    for mode in first["modes"]:
        assert (
            first["modes"][mode]["development"] == second["modes"][mode]["development"]
        )


def test_fit_sees_only_development_records_before_evaluation_is_scored(
    monkeypatch,
):
    dev_inputs = {text_digest(row["input_text"]) for row in records(DEVELOPMENT)}
    eval_inputs = {text_digest(row["input_text"]) for row in records(EVALUATION)}
    observed = {}

    def spy_fit(examples, **kwargs):
        observed["examples"] = {
            text_digest(example.request.input_text) for example in examples
        }
        fitted = fit_decision_thresholds(examples, **kwargs)
        observed["scored_at_fit"] = list(SCORED)
        return fitted

    monkeypatch.setattr(qualification, "fit_decision_thresholds", spy_fit)
    run(spy_encoder_backend())

    assert observed["examples"] == dev_inputs
    assert set(observed["scored_at_fit"]) == dev_inputs
    after_fit = SCORED[len(observed["scored_at_fit"]) :]
    assert set(after_fit) == eval_inputs
    assert all(digest in dev_inputs for digest in observed["scored_at_fit"])


def test_changing_development_changes_thresholds(tmp_path):
    rows = records(DEVELOPMENT)
    for row in rows:
        if row["mode"] == "fixed_choice" and "likely" in row["input_text"]:
            row["expected_choices"] = [
                next(
                    option
                    for option in row["options"]
                    if option not in row["expected_choices"]
                )
            ]
    harder = write_jsonl(tmp_path / "development-harder.jsonl", rows)

    baseline = run(synthetic_encoder_backend()).to_dict()
    changed = run(synthetic_encoder_backend(), development=harder).to_dict()

    assert baseline["fit"]["modes"]["fixed_choice"]["thresholds"] == {
        "minimum_confidence": 0.74,
        "minimum_margin": pytest.approx(0.48),
    }
    assert changed["fit"]["modes"]["fixed_choice"]["thresholds"] == {
        "minimum_confidence": 0.92,
        "minimum_margin": pytest.approx(0.84),
    }
    assert baseline["inputs"]["evaluation"] == changed["inputs"]["evaluation"]


def test_insufficient_evaluation_support_withholds_profiles():
    report = run(DeterministicDecisionBackend(), policy=DecisionQualificationPolicy())
    payload = report.to_dict()

    assert report.status == "insufficient"
    assert not report.qualified
    assert payload["profiles"] == {}
    assert report.profiles() == {}
    assert payload["reasons"] == [
        "evaluation_insufficient_option_support_fixed_choice",
        "evaluation_insufficient_option_support_boolean_choice",
        "evaluation_insufficient_option_support_multi_label",
    ]
    options = payload["modes"]["multi_label"]["options"]
    assert {item["evaluation_examples"] for item in options} == {2, 3}
    assert not any(item["sufficient"] for item in options)
    # Metrics remain available for review even though no profile is emitted.
    assert payload["modes"]["multi_label"]["evaluation"]["covered"] == 5


def test_one_short_option_withholds_every_profile():
    # Only one multi-label option has two evaluation examples; three are needed.
    policy = DecisionQualificationPolicy(min_examples_per_option=3)

    payload = run(DeterministicDecisionBackend(), policy=policy).to_dict()

    assert payload["reasons"] == ["evaluation_insufficient_option_support_multi_label"]
    short = [
        item
        for item in payload["modes"]["multi_label"]["options"]
        if not item["sufficient"]
    ]
    assert [item["evaluation_examples"] for item in short] == [2]
    assert payload["modes"]["fixed_choice"]["evaluation"]["accuracy"] == 1.0
    assert payload["profiles"] == {}


def test_report_contains_no_text_options_identifiers_or_paths(tmp_path):
    folder = tmp_path / "private-path-marker"
    folder.mkdir()
    options = ["zebra-marker-alpha", "zebra-marker-beta"]

    def row(index: int, choice: str) -> dict:
        return {
            "id": f"private-id-marker-{index}",
            "mode": "fixed_choice",
            "input_text": f"SECRET-MRN-555-01{index:02d} says {choice} with care.",
            "options": options,
            "expected_choices": [choice],
        }

    development = write_jsonl(
        folder / "dev-name-marker.jsonl",
        [row(index, options[index % 2]) for index in range(6)],
    )
    evaluation = write_jsonl(
        folder / "eval-name-marker.jsonl",
        [row(index, options[index % 2]) for index in range(6, 12)],
    )

    rendered = run(
        DeterministicDecisionBackend(),
        development=development,
        evaluation=evaluation,
    ).to_json()

    assert json.loads(rendered)["contains_source_text"] is False
    for marker in (
        "SECRET",
        "555-01",
        "zebra",
        "marker",
        "private",
        "with care",
        str(tmp_path),
    ):
        assert marker not in rendered


def test_split_overlap_and_missing_modes_are_insufficient(tmp_path):
    rows = [row for row in records(EVALUATION) if row["mode"] != "multi_label"]
    rows.append({**records(DEVELOPMENT)[0], "id": "synthetic-eval-overlap"})
    evaluation = write_jsonl(tmp_path / "evaluation.jsonl", rows)

    payload = run(DeterministicDecisionBackend(), evaluation=evaluation).to_dict()

    assert payload["status"] == "insufficient"
    assert payload["reasons"][:2] == ["split_overlap", "evaluation_missing_multi_label"]
    assert payload["modes"]["multi_label"]["evaluation"] is None
    assert payload["profiles"] == {}


def test_unmet_coverage_floor_is_not_qualified():
    policy = DecisionQualificationPolicy(min_examples_per_option=2, coverage_floor=0.8)
    payload = run(DeterministicDecisionBackend(), policy=policy).to_dict()

    assert payload["status"] == "not_qualified"
    assert payload["reasons"] == [
        "evaluation_coverage_floor_unmet_fixed_choice",
        "evaluation_coverage_floor_unmet_boolean_choice",
    ]
    assert payload["profiles"] == {}


def test_unreachable_development_floor_skips_evaluation_scoring(tmp_path):
    rows = [row for row in records(DEVELOPMENT) if row["mode"] == "boolean_choice"]
    for row in rows:
        row["expected_choices"] = [
            next(
                option
                for option in row["options"]
                if option not in row["expected_choices"]
            )
        ]
    development = write_jsonl(tmp_path / "development.jsonl", rows)
    evaluation = write_jsonl(
        tmp_path / "evaluation.jsonl",
        [row for row in records(EVALUATION) if row["mode"] == "boolean_choice"],
    )

    payload = run(
        spy_encoder_backend(), development=development, evaluation=evaluation
    ).to_dict()

    assert payload["fit"]["modes"]["boolean_choice"]["thresholds"] is None
    assert payload["reasons"] == ["development_accuracy_floor_unmet_boolean_choice"]
    assert payload["modes"]["boolean_choice"]["evaluation"] is None
    eval_inputs = {text_digest(row["input_text"]) for row in records(evaluation)}
    assert eval_inputs.isdisjoint(SCORED)


def test_backend_failures_and_timeouts_are_counted_without_messages(monkeypatch):
    def failing(text, options, mode):
        if "likely" in text:
            raise RuntimeError(f"SECRET backend echo {text}")
        return synthetic_scorer(text, options, mode)

    backend = EncoderDecisionBackend(
        backend_id="synthetic.failing.v1",
        model_id="synthetic/failing",
        revision="0.0.0",
        license_id="MIT",
        scorer=failing,
    )
    report = run(backend)
    rendered = report.to_json()
    payload = report.to_dict()

    assert "SECRET" not in rendered
    assert "development_backend_errors_fixed_choice" in payload["reasons"]
    assert "evaluation_backend_errors_fixed_choice" in payload["reasons"]
    assert payload["modes"]["fixed_choice"]["evaluation"]["states"]["failure"] == 2
    assert payload["profiles"] == {}

    ticks = iter(range(0, 10_000_000, 10))
    monkeypatch.setattr(qualification.time, "monotonic", lambda: next(ticks))
    slow = run(synthetic_encoder_backend()).to_dict()
    assert slow["fit"]["modes"]["fixed_choice"]["thresholds"] is None
    assert slow["fit"]["modes"]["fixed_choice"]["backend_errors"] == 10


def test_ordered_preference_mode_is_supported(tmp_path):
    options = ["first", "second", "third"]

    def row(text: str, choice: str) -> dict:
        return {
            "mode": "ordered_preference",
            "input_text": text,
            "options": options,
            "expected_choices": [choice],
        }

    development = write_jsonl(
        tmp_path / "development.jsonl",
        [
            row(f"Synthetic rank {index} puts {choice} ahead.", choice)
            for index, choice in enumerate(options * 2)
        ],
    )
    evaluation = write_jsonl(
        tmp_path / "evaluation.jsonl",
        [
            row(f"Synthetic order {index} keeps {choice} on top.", choice)
            for index, choice in enumerate(options * 2)
        ],
    )

    report = run(
        DeterministicDecisionBackend(), development=development, evaluation=evaluation
    )

    assert report.qualified
    assert set(report.profiles()) == {"site.triage.ordered_preference"}


def _line(**changes) -> str:
    row = {
        "mode": "fixed_choice",
        "input_text": "SECRET-555-0101 synthetic text",
        "options": ["alpha", "beta"],
        "expected_choices": ["alpha"],
    }
    row.update(changes)
    return json.dumps({key: value for key, value in row.items() if value is not None})


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("SECRET-555-0101 not json\n", "invalid_record"),
        ("[]\n", "invalid_record"),
        (_line(note="SECRET") + "\n", "invalid_record"),
        (_line(options=None) + "\n", "invalid_record"),
        (_line(expected_choices=["gamma"]) + "\n", "invalid_record"),
        (_line(expected_choices=["alpha", "beta"]) + "\n", "invalid_record"),
        (
            _line(expected_choices=["alpha", "alpha"], mode="multi_label") + "\n",
            "invalid_record",
        ),
        (_line(options=["alpha", "Alpha"]) + "\n", "invalid_record"),
        (_line(mode="free_text") + "\n", "invalid_record"),
        (_line(id="") + "\n", "invalid_record"),
        (
            _line(mode="scalar_score", options=[], expected_choices=[]) + "\n",
            "unsupported_mode",
        ),
        ((_line(id="synthetic-1") + "\n") * 2, "duplicate_id"),
        ("\n\n", "empty_split"),
        (b"\xff\xfeSECRET", "invalid_record"),
    ],
)
def test_invalid_inputs_raise_value_free_codes(tmp_path, content, code):
    path = tmp_path / "split.jsonl"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, "utf-8")

    with pytest.raises(DecisionQualificationError) as raised:
        run(DeterministicDecisionBackend(), development=path)

    assert raised.value.code == code
    assert str(raised.value) == code
    assert raised.value.__cause__ is None
    assert "SECRET" not in repr(raised.value)


def test_missing_input_calibration_and_policy_errors_are_controlled(tmp_path):
    backend = DeterministicDecisionBackend()
    cases = [
        (dict(development=tmp_path / "absent.jsonl"), "input_unavailable"),
        (
            dict(calibration_id="openmed.synthetic.fixed_option.v1"),
            "reserved_calibration_id",
        ),
        (dict(calibration_id="openmed"), "reserved_calibration_id"),
        (dict(calibration_id="Site Triage"), "invalid_calibration_id"),
        (dict(calibration_id="a" * 109), "invalid_calibration_id"),
        (dict(calibration_version="v1"), "invalid_calibration_version"),
        (dict(policy=object()), "invalid_policy"),
    ]
    for options, code in cases:
        with pytest.raises(DecisionQualificationError) as raised:
            run(backend, **options)
        assert raised.value.code == code
    for values in (
        dict(min_examples_per_option=0),
        dict(min_examples_per_option=True),
        dict(accuracy_floor=1.5),
        dict(coverage_floor=float("nan")),
        dict(accuracy_floor=True),
    ):
        with pytest.raises(DecisionQualificationError, match="invalid_policy"):
            DecisionQualificationPolicy(**values)


def test_backend_is_explicit_and_defaults_are_never_registered():
    default_backend = decision_module.DETERMINISTIC_DECISION_BACKEND
    default_profile = decision_module.DEFAULT_CALIBRATION_PROFILE

    with pytest.raises(DecisionQualificationError, match="backend_invalid"):
        run(None)
    report = run(synthetic_encoder_backend())

    assert report.qualified
    assert decision_module.DETERMINISTIC_DECISION_BACKEND is default_backend
    assert decision_module.DEFAULT_CALIBRATION_PROFILE is default_profile
    assert default_profile == DecisionCalibrationProfile()
    result = decide(
        DecisionRequest(
            mode="fixed_choice",
            input_text="Synthetic review priority is clearly urgent.",
            options=("urgent", "routine"),
        )
    )
    assert result.backend["kind"] == "deterministic"
    assert dict(result.calibration) == DecisionCalibrationProfile().to_dict()
    # A fitted profile ID is unknown to the default decision path.
    unknown = decide(
        DecisionRequest(
            mode="fixed_choice",
            input_text="Synthetic review priority is clearly urgent.",
            options=("urgent", "routine"),
            calibration_id="site.triage.fixed_choice",
        )
    )
    assert unknown.code == "calibration_unavailable"


def test_candidate_grid_is_bounded_and_keeps_extremes():
    values = [index / 1000 for index in range(1001)]
    candidates = qualification._candidates(values)

    assert len(candidates) <= 256
    assert candidates[0] == 0.0 and candidates[-1] == 1.0
    assert candidates == sorted(set(candidates))


def test_qualification_opens_no_network_connections(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert run(DeterministicDecisionBackend()).qualified
    assert run(synthetic_encoder_backend()).qualified
