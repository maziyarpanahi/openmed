"""Android emulator lane: synthetic fixture, instrumentation wiring and CI job."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.android.generate_emulator_lane_fixture import (
    FIXTURE_FILENAME,
    LANE_ROOT,
    MODEL_DIRNAME,
    MODEL_FILES,
    LaneFixtureError,
    aggregate_token_predictions,
    check,
    expected_facade_spans,
    split_token_label,
    validate_fixture,
    validate_lane_text,
)

ROOT = Path(__file__).parents[3]
ANDROID = ROOT / "android"
LIBRARY_BUILD = ANDROID / "openmedkit" / "build.gradle.kts"
CATALOG = ANDROID / "gradle" / "libs.versions.toml"
ANDROID_TEST = ANDROID / "openmedkit" / "src" / "androidTest"
LANE_SOURCES = ANDROID_TEST / "kotlin" / "com" / "openmed" / "openmedkit" / "emulator"
ANDROID_CI = ROOT / ".github" / "workflows" / "android-ci.yml"
LANE_SCRIPT = ROOT / "scripts" / "android" / "run_emulator_lane.sh"
GENERATOR = "scripts/android/generate_emulator_lane_fixture.py"
EMULATOR_RUNNER = "reactivecircus/android-emulator-runner"


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads((LANE_ROOT / FIXTURE_FILENAME).read_text(encoding="utf-8"))


def _token(label: str, start: int, end: int, score: float = 0.9) -> dict[str, Any]:
    return {
        "label": label,
        "start": start,
        "end": end,
        "score": score,
        "special": False,
    }


def test_committed_fixture_is_synthetic_value_free_and_matches_its_model(
    fixture: dict[str, Any],
) -> None:
    validate_fixture(fixture, LANE_ROOT)

    assert fixture["model"]["external_data"] is False
    assert sorted(path.name for path in (LANE_ROOT / MODEL_DIRNAME).iterdir()) == (
        sorted(MODEL_FILES)
    )
    # A tiny committed asset, not a checkpoint.
    assert (LANE_ROOT / MODEL_DIRNAME / "model.onnx").stat().st_size < 1024 * 1024


def test_lane_text_separates_unicode_scalar_and_utf16_offsets(
    fixture: dict[str, Any],
) -> None:
    texts = [case["text"] for case in fixture["cases"]]

    assert any(len(text.encode("utf-16-le")) // 2 != len(text) for text in texts)


def test_model_embeds_every_initializer_inline() -> None:
    onnx = pytest.importorskip("onnx")

    model = onnx.load(
        str(LANE_ROOT / MODEL_DIRNAME / "model.onnx"), load_external_data=False
    )

    assert model.graph.initializer
    assert all(
        tensor.data_location != onnx.TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    )


def test_python_onnx_runtime_reproduces_the_committed_reference() -> None:
    for module in ("numpy", "onnxruntime", "tokenizers"):
        pytest.importorskip(module)

    check(LANE_ROOT)


def test_fixture_validation_rejects_unsafe_or_inconsistent_fixtures(
    fixture: dict[str, Any],
    tmp_path: Path,
    subtests: Any,
) -> None:
    lane_root = tmp_path / "lane"
    shutil.copytree(LANE_ROOT, lane_root)

    def mutated(change: Any) -> dict[str, Any]:
        payload = copy.deepcopy(fixture)
        change(payload)
        return payload

    def span(payload: dict[str, Any]) -> dict[str, Any]:
        return payload["cases"][0]["spans"][0]

    def first_label_token(payload: dict[str, Any]) -> dict[str, Any]:
        return next(t for t in payload["cases"][0]["tokens"] if not t["special"])

    payload_cases = {
        "span surface text": (
            lambda p: span(p).update(text="SYNTH"),
            "value-free",
        ),
        "shifted span": (
            lambda p: span(p).update(end=span(p)["end"] + 1),
            "do not follow",
        ),
        "text without placeholders": (
            lambda p: p["cases"][0].update(text="Seen at the clinic."),
            "SYNTH_",
        ),
        "identifier-shaped text": (
            lambda p: p["cases"][0].update(text="SYNTH_NOTE call 555-123-4567"),
            "identifier-shaped",
        ),
        "not marked synthetic": (
            lambda p: p["cases"][0].update(synthetic=False),
            "synthetic",
        ),
        "inconsistent token label": (
            lambda p: first_label_token(p).update(
                label_id=(first_label_token(p)["label_id"] + 1) % 7
            ),
            "label is inconsistent",
        ),
        "ambiguous token decision": (
            lambda p: p["cases"][0].update(min_top2_logit_margin=0.001),
            "ambiguous",
        ),
        "external data": (
            lambda p: p["model"].update(external_data=True),
            "inline",
        ),
    }
    for name, (change, message) in payload_cases.items():
        with subtests.test(case=name):
            with pytest.raises(LaneFixtureError, match=message):
                validate_fixture(mutated(change), lane_root)

    with subtests.test(case="model file digest"):
        target = lane_root / MODEL_DIRNAME / "tokenizer_config.json"
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        try:
            with pytest.raises(LaneFixtureError, match="digest"):
                validate_fixture(fixture, lane_root)
        finally:
            target.write_bytes(original)

    with subtests.test(case="external-data sidecar"):
        sidecar = lane_root / MODEL_DIRNAME / "model.onnx.data"
        sidecar.write_bytes(b"\0")
        try:
            with pytest.raises(LaneFixtureError, match="only the listed files"):
                validate_fixture(fixture, lane_root)
        finally:
            sidecar.unlink()


def test_lane_text_rules(subtests: Any) -> None:
    rejected = {
        "bad id": ("case-1", "SYNTH_NOTE"),
        "no placeholder": ("synth-lane-001", "Seen on 2099-01-02."),
        "email": ("synth-lane-001", "SYNTH_NOTE a@b.example"),
        "combining mark": ("synth-lane-001", "SYNTH_NOTE é"),
        "zero-width joiner": ("synth-lane-001", "SYNTH_NOTE ‍"),
    }
    for name, (case_id, text) in rejected.items():
        with subtests.test(case=name):
            with pytest.raises(LaneFixtureError):
                validate_lane_text(case_id, text)

    validate_lane_text("synth-lane-001", "SYNTH_NOTE \U0001f600 2099-01-02.")


def test_split_token_label_mirrors_kotlin(subtests: Any) -> None:
    expected = {
        "B-NAME": ("B", "NAME"),
        "i_date": ("I", "date"),
        "S-ID": ("S", "ID"),
        "O": ("", "O"),
        "X-NAME": ("", "X-NAME"),
        "NAME": ("", "NAME"),
        " E-ID ": ("E", "ID"),
    }
    for raw, split in expected.items():
        with subtests.test(label=raw):
            assert split_token_label(raw) == split


def test_aggregation_mirrors_kotlin_runtime_rules(subtests: Any) -> None:
    def spans(text_length: int, *tokens: dict[str, Any]) -> list[tuple[str, int, int]]:
        return [
            (span["label"], span["start"], span["end"])
            for span in aggregate_token_predictions(text_length, tokens)
        ]

    cases = {
        # OpenMedKitApiTest.aggregatesBioTokensAcrossWhitespace.
        "B then E joins across whitespace": (
            spans(12, _token("B-PERSON", 0, 5), _token("E-PERSON", 6, 12)),
            [("PERSON", 0, 12)],
        ),
        "I continues the same label": (
            spans(9, _token("B-NAME", 0, 4), _token("I-NAME", 5, 9)),
            [("NAME", 0, 9)],
        ),
        "I of another label starts a span": (
            spans(9, _token("B-NAME", 0, 4), _token("I-DATE", 5, 9)),
            [("NAME", 0, 4), ("DATE", 5, 9)],
        ),
        "B always starts a span": (
            spans(9, _token("B-NAME", 0, 4), _token("B-NAME", 5, 9)),
            [("NAME", 0, 4), ("NAME", 5, 9)],
        ),
        "lowercase o closes the span": (
            spans(9, _token("B-NAME", 0, 4), _token("o", 4, 5), _token("I-NAME", 5, 9)),
            [("NAME", 0, 4), ("NAME", 5, 9)],
        ),
        "E closes the span": (
            spans(9, _token("E-NAME", 0, 4), _token("I-NAME", 5, 9)),
            [("NAME", 0, 4), ("NAME", 5, 9)],
        ),
        "unprefixed label joins adjacent tokens only": (
            spans(9, _token("NAME", 0, 2), _token("NAME", 2, 4), _token("NAME", 5, 9)),
            [("NAME", 0, 4), ("NAME", 5, 9)],
        ),
        "zero-width token closes the span": (
            spans(
                9,
                _token("B-NAME", 0, 4),
                _token("I-NAME", 4, 4),
                _token("I-NAME", 5, 9),
            ),
            [("NAME", 0, 4), ("NAME", 5, 9)],
        ),
        "special tokens are skipped": (
            spans(
                4,
                {"special": True, "start": 0, "end": 0},
                _token("B-ID", 0, 4),
                {"special": True, "start": 0, "end": 0},
            ),
            [("ID", 0, 4)],
        ),
        "spans past the text end are dropped": (
            spans(3, _token("B-ID", 0, 4)),
            [],
        ),
    }
    for name, (actual, expected) in cases.items():
        with subtests.test(case=name):
            assert actual == expected


def test_facade_spans_average_scores_filter_and_sort() -> None:
    tokens = [
        _token("B-DATE", 6, 8, 0.4),
        _token("I-DATE", 8, 10, 0.5),
        _token("B-NAME", 0, 2, 0.8),
        _token("I-NAME", 2, 4, 0.6),
    ]

    spans = expected_facade_spans("x" * 10, tokens, confidence_threshold=0.5)

    assert [(span["label"], span["start"], span["end"]) for span in spans] == [
        ("NAME", 0, 4)
    ]
    assert spans[0]["confidence"] == pytest.approx(0.7, abs=1e-6)


def test_library_build_wires_the_instrumentation_runner_and_fault_switch() -> None:
    build = LIBRARY_BUILD.read_text(encoding="utf-8")
    catalog = CATALOG.read_text(encoding="utf-8")

    assert (
        'testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"' in build
    )
    assert "androidTestImplementation(libs.androidx.test.runner)" in build
    assert 'module = "androidx.test:runner"' in catalog
    assert 'providers.gradleProperty("openmedNativePackagingFault")' in build
    for library in (
        "libonnxruntime.so",
        "libonnxruntime4j_jni.so",
        "libdjl_tokenizer.so",
    ):
        assert f'"**/{library}"' in build
    assert "variant.androidTest?.packaging?.jniLibs?.excludes" in build


def test_instrumented_test_apk_is_offline_and_every_lane_test_is_guarded() -> None:
    manifest = (ANDROID_TEST / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert re.search(
        r'android:name="android\.permission\.INTERNET"\s+tools:node="remove"', manifest
    )

    guarded = {
        "NativeRuntimePackagingTest.kt",
        "OnnxRuntimeParityTest.kt",
        "TokenizerParityTest.kt",
        "OpenMedKitFromDirectoryTest.kt",
    }
    for name in guarded:
        source = (LANE_SOURCES / name).read_text(encoding="utf-8")
        assert "val noNetwork = NoNetworkSocketRule()" in source, name
    facade = (LANE_SOURCES / "OpenMedKitFromDirectoryTest.kt").read_text(
        encoding="utf-8"
    )
    assert 'OpenMedKit.fromDirectory(modelDirectory, variant = "fp32")' in facade


def test_lane_script_negative_control_matches_the_packaging_test() -> None:
    script = LANE_SCRIPT.read_text(encoding="utf-8")
    packaging = (LANE_SOURCES / "NativeRuntimePackagingTest.kt").read_text(
        encoding="utf-8"
    )
    environment = (LANE_SOURCES / "LaneEnvironment.kt").read_text(encoding="utf-8")

    assert os.access(LANE_SCRIPT, os.X_OK)
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "-PopenmedNativePackagingFault=missing-native-libraries" in script
    assert "com.openmed.openmedkit.emulator.NativeRuntimePackagingTest" in script
    assert '"native_library_unloadable:$library"' in packaging
    assert "native_library_unloadable:$library" in script
    for library in ("onnxruntime", "djl_tokenizer"):
        assert f'"{library}"' in environment
    assert "for library in onnxruntime djl_tokenizer; do" in script


def test_android_ci_runs_x86_64_and_16_kb_emulator_lanes() -> None:
    workflow = yaml.safe_load(ANDROID_CI.read_text(encoding="utf-8"))
    triggers = workflow[True] if True in workflow else workflow["on"]
    job = workflow["jobs"]["emulator-lane"]
    steps = {step.get("name"): step for step in job["steps"]}
    lanes = {entry["lane"]: entry for entry in job["strategy"]["matrix"]["include"]}

    for event in ("push", "pull_request"):
        assert GENERATOR in triggers[event]["paths"]
        assert "scripts/android/run_emulator_lane.sh" in triggers[event]["paths"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["fail-fast"] is False
    assert 'KERNEL=="kvm"' in steps["Enable KVM"]["run"]
    assert steps["Set up Java"]["with"]["java-version"] == "17"

    runner = steps["Run emulator lane"]
    action, _, ref = runner["uses"].partition("@")
    assert action == EMULATOR_RUNNER
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    assert runner["with"]["arch"] == "x86_64"
    assert "scripts/android/run_emulator_lane.sh" in runner["with"]["script"]

    standard = lanes["x86_64-api35"]
    assert (standard["target"], standard["page-size-kb"]) == ("google_apis", 4)
    assert "--negative-control" in standard["lane-args"]
    sixteen = lanes["x86_64-api35-16kb"]
    assert (sixteen["target"], sixteen["page-size-kb"]) == ("google_apis_ps16k", 16)
    assert 26 <= standard["api-level"] <= 35
    assert steps["Upload emulator lane reports"]["if"] == "always()"
