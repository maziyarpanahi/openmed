"""Regenerate and verify the synthetic photo question-answer fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.fixtures.multimodal.photo_qa import generate_fixtures as fixtures

Image = pytest.importorskip("PIL.Image")

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "fixtures" / "multimodal" / "photo_qa"
)


def _load_truth() -> dict:
    return json.loads((FIXTURE_DIR / "truth.json").read_text(encoding="utf-8"))


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def test_manifest_has_both_tasks_and_at_least_twenty_closed_items():
    truth = _load_truth()

    assert truth["schema_version"] == fixtures.SCHEMA_VERSION
    assert truth["box_coordinate_system"] == fixtures.BOX_COORDINATE_SYSTEM
    assert len(truth["items"]) >= 20
    assert {item["kind"] for item in truth["items"]} == {
        "label_reading",
        "counting",
    }
    assert len({item["id"] for item in truth["items"]}) == len(truth["items"])
    assert len({item["image"] for item in truth["items"]}) == len(truth["items"])


def test_committed_images_and_boxes_match_the_generator():
    truth = _load_truth()

    for item in truth["items"]:
        generated, box, kind = fixtures.render_case(fixtures.case_by_id(item["id"]))
        committed = _load_rgb(FIXTURE_DIR / item["image"])

        assert generated.mode == committed.mode == "RGB"
        assert generated.size == committed.size == fixtures.IMAGE_SIZE
        assert generated.tobytes() == committed.tobytes()
        assert hashlib.sha256(generated.tobytes()).hexdigest() == item["sha256"]
        assert box == tuple(item["box"])
        assert kind == item["kind"]


def test_every_box_is_inside_its_image_without_path_traversal():
    truth = _load_truth()

    for item in truth["items"]:
        assert Path(item["image"]).name == item["image"]
        image = _load_rgb(FIXTURE_DIR / item["image"])
        x0, y0, x1, y1 = item["box"]
        assert 0 <= x0 < x1 <= image.width
        assert 0 <= y0 < y1 <= image.height


def test_regeneration_reproduces_the_manifest_and_pixels(tmp_path):
    expected = _load_truth()
    regenerated = fixtures.build_fixtures(tmp_path)

    assert regenerated == expected
    for item in expected["items"]:
        committed = _load_rgb(FIXTURE_DIR / item["image"])
        rebuilt = _load_rgb(tmp_path / item["image"])
        assert rebuilt.tobytes() == committed.tobytes()
