"""Tests for compact manifest-declared on-device segmenter resources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openmed.processing.tokenization import (
    DEFAULT_SEGMENTER_ID,
    SEGMENTER_RESOURCE_SIZE_BUDGET_BYTES,
    ResourceSegmenter,
    package_segmenter_resources,
    validate_segmenter_resources,
)

ROOT = Path(__file__).resolve().parents[3]
PARITY_FIXTURE = ROOT / "tests" / "fixtures" / "segmenter" / "zh_hi_parity.json"


def test_packaged_resources_are_license_tagged_and_within_budget(
    tmp_path: Path,
) -> None:
    descriptor = package_segmenter_resources(tmp_path, DEFAULT_SEGMENTER_ID)

    assert descriptor["scripts"] == ["Han", "Devanagari"]
    assert descriptor["license"] == "MIT AND ICU-1.8.1"
    assert {item["license"] for item in descriptor["resource_files"]} == {
        "MIT",
        "ICU-1.8.1",
    }
    assert descriptor["total_size_bytes"] <= descriptor["size_budget_bytes"]
    assert descriptor["size_budget_bytes"] == SEGMENTER_RESOURCE_SIZE_BUDGET_BYTES
    assert validate_segmenter_resources(tmp_path, descriptor)["files"] == [
        "segmenter/han_words.txt",
        "segmenter/indic_rules.json",
    ]


def test_stdlib_segmenter_does_not_import_optional_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = package_segmenter_resources(tmp_path, DEFAULT_SEGMENTER_ID)
    segmenter = ResourceSegmenter(tmp_path, descriptor)

    def fail_import(name: str):
        raise AssertionError(f"unexpected optional runtime import: {name}")

    monkeypatch.setattr(
        "openmed.processing.tokenization.importlib.import_module", fail_import
    )

    spans = segmenter.segment("患者 राहुल")

    assert [(span.text, span.start, span.end) for span in spans] == [
        ("患者", 0, 6),
        ("रा", 7, 13),
        ("हु", 13, 19),
        ("ल", 19, 22),
    ]


def test_python_matches_shared_zh_hi_utf8_offset_fixture(tmp_path: Path) -> None:
    fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
    descriptor = package_segmenter_resources(tmp_path, DEFAULT_SEGMENTER_ID)
    manifest = {"segmenter": descriptor}
    segmenter = ResourceSegmenter.from_manifest(tmp_path, manifest)

    assert segmenter is not None
    actual = [span.__dict__ for span in segmenter.segment(fixture["text"])]
    assert actual == fixture["segments"]


def test_optional_jieba_accelerator_is_lazy_and_offset_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = package_segmenter_resources(tmp_path, "openmed-han-v1")
    segmenter = ResourceSegmenter(tmp_path, descriptor)
    loaded_dictionaries: list[str] = []

    class FakeTokenizer:
        def __init__(self, *, dictionary: str):
            loaded_dictionaries.append(dictionary)

        def tokenize(self, text: str):
            assert text == "患者张三"
            return [("患者", 0, 2), ("张三", 2, 4)]

    class FakeJieba:
        Tokenizer = FakeTokenizer

    monkeypatch.setattr(
        "openmed.processing.tokenization.importlib.import_module",
        lambda name: FakeJieba if name == "jieba" else None,
    )

    assert [
        span.text for span in segmenter.segment("患者张三", use_accelerated=True)
    ] == [
        "患者",
        "张三",
    ]
    assert loaded_dictionaries == [str(tmp_path / "segmenter" / "han_words.txt")]


def test_manifest_han_dictionary_uses_the_bounded_entry_validator(
    tmp_path: Path,
) -> None:
    descriptor = package_segmenter_resources(tmp_path, "openmed-han-v1")
    resource = descriptor["resource_files"][0]
    resource_path = tmp_path / resource["path"]
    payload = ("患" * 257 + "\n").encode()
    resource_path.write_bytes(payload)
    resource["size_bytes"] = len(payload)
    resource["sha256"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    descriptor["total_size_bytes"] = len(payload)

    with pytest.raises(ValueError, match="term_length"):
        ResourceSegmenter(tmp_path, descriptor)


def test_validator_rejects_missing_declared_resource(tmp_path: Path) -> None:
    descriptor = package_segmenter_resources(tmp_path, DEFAULT_SEGMENTER_ID)
    (tmp_path / descriptor["resource_files"][0]["path"]).unlink()

    with pytest.raises(ValueError, match="segmenter resource is missing"):
        validate_segmenter_resources(tmp_path, descriptor)
