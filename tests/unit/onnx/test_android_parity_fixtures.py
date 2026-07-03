"""Tests for Android span parity fixture generation."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/android/generate_parity_fixtures.py"
FIXTURE = ROOT / "android/openmedkit/src/test/resources/parity/android_span_parity.json"

SPEC = importlib.util.spec_from_file_location("generate_parity_fixtures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)

ID2LABEL = {
    "0": "O",
    "1": "B-PERSON",
    "2": "I-PERSON",
    "3": "B-ORGANIZATION",
    "4": "I-ORGANIZATION",
    "5": "B-DATE",
    "6": "I-DATE",
    "7": "B-PHONE",
    "8": "I-PHONE",
}
LABEL_BY_TOKEN_ID = {
    2100: 1,
    2102: 3,
    2104: 5,
    2203: 1,
    2205: 7,
}


def test_committed_fixture_regenerates_from_android_profile_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_dir = _write_fake_android_export(tmp_path)
    _install_fake_runtime(monkeypatch)
    monkeypatch.setattr(parity, "validate_android_profile", lambda path: None)

    regenerated = parity.build_fixture_payload(export_dir)
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert regenerated == committed
    assert committed["contract"]["token_ids"]["mode"] == "exact"
    assert committed["contract"]["char_offsets"]["mode"] == "exact"
    assert committed["contract"]["span_boundaries"] == {
        "json_path": "cases[].spans[].start/end",
        "mode": "exact",
        "tolerance_chars": 0,
    }
    assert all(case["synthetic"] and case["phi_free"] for case in committed["cases"])
    assert all("SYNTH_" in case["text"] for case in committed["cases"])
    assert all(
        not ({"text", "word", "surface"} & set(span))
        for case in committed["cases"]
        for span in case["spans"]
    )


def test_fixture_validation_rejects_phi_shaped_text() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["cases"][0]["text"] = "SYNTH_PATIENT John Doe"

    with pytest.raises(ValueError, match="PHI-shaped"):
        parity.validate_fixture_payload(payload)


def _write_fake_android_export(tmp_path: Path) -> Path:
    export_dir = tmp_path / "android-export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"onnx")
    (export_dir / "id2label.json").write_text(
        json.dumps(ID2LABEL, indent=2),
        encoding="utf-8",
    )
    (export_dir / "openmed-onnx.json").write_text(
        json.dumps(
            {
                "format": "openmed-onnx",
                "format_version": 1,
                "formats": ["onnx-android"],
                "task": "token-classification",
                "family": "fixture",
                "source_model_id": "OpenMed/android-parity-fixture",
                "config_path": "config.json",
                "label_map_path": "id2label.json",
                "max_sequence_length": 64,
                "tokenizer": {"path": ".", "files": ["tokenizer.json"]},
                "artifacts": [
                    {
                        "format": "onnx-android",
                        "path": "model.onnx",
                        "precision": "float32",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return export_dir


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )

    runtime_mod = types.ModuleType("onnxruntime")
    runtime_mod.InferenceSession = FakeSession

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)


class FakeTokenizer:
    def __call__(self, text: str, **kwargs):
        if "SYNTH_PATIENT_ALPHA" in text:
            ids = [101, 2100, 2101, 2102, 2103, 2104, 102]
            offsets = [
                [0, 0],
                [0, 19],
                [20, 27],
                [28, 45],
                [46, 48],
                [49, 69],
                [0, 0],
            ]
        else:
            ids = [101, 2200, 2201, 2202, 2203, 2204, 2205, 102]
            offsets = [
                [0, 0],
                [0, 5],
                [6, 13],
                [14, 16],
                [17, 35],
                [36, 43],
                [44, 64],
                [0, 0],
            ]
        return {
            "input_ids": [ids],
            "attention_mask": [[1] * len(ids)],
            "offset_mapping": [offsets],
        }


class FakeSession:
    def __init__(self, model_path: str, providers: list[str]) -> None:
        self.model_path = model_path
        self.providers = providers

    def get_inputs(self):
        return [
            types.SimpleNamespace(name="input_ids"),
            types.SimpleNamespace(name="attention_mask"),
        ]

    def run(self, output_names, inputs):
        rows = [
            _logit_row(LABEL_BY_TOKEN_ID.get(int(token_id), 0))
            for token_id in inputs["input_ids"][0]
        ]
        return [[rows]]


def _logit_row(label_id: int) -> list[float]:
    row = [-100.0] * len(ID2LABEL)
    row[label_id] = 100.0
    return row
