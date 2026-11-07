"""Offline acceptance and drift controls for the fixed vision catalog."""

import hashlib
import io
import json
import socket
import unicodedata
from dataclasses import FrozenInstanceError
from importlib.resources import files

import pytest

from openmed.core import labels as L
from openmed.eval import openphi_vision_fixtures as fixtures

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageFont = pytest.importorskip("PIL.ImageFont")
ROOT = files("openmed.eval").joinpath("fixtures", "openphi_vision_v1")


@pytest.fixture
def catalog():
    return json.loads(ROOT.joinpath("catalog.json").read_bytes())


@pytest.fixture
def cases():
    return tuple(fixtures.load_openphi_vision_fixtures())


def test_catalog_is_fixed_synthetic_and_offline(monkeypatch, catalog):
    def denied(*args, **kwargs):
        pytest.fail("fixture loader attempted network access")

    monkeypatch.setattr(socket, "socket", denied)
    assert len(tuple(fixtures.load_openphi_vision_fixtures())) == 6
    assert catalog["version"] == fixtures.CATALOG_VERSION
    assert catalog["synthetic"] is True
    assert catalog["contains_real_phi"] is False
    assert catalog["font_license"] == "OFL-1.1"
    # No seed, custom layout, random generation or training export surface.
    with pytest.raises(TypeError):
        fixtures.load_openphi_vision_fixtures(seed=1)
    for case in fixtures.load_openphi_vision_fixtures():
        assert not hasattr(case, "to_training_item")


def test_every_case_and_page_regenerates_byte_identically(cases, catalog):
    font_bytes = ROOT.joinpath("IBMPlexMono-Regular.ttf").read_bytes()
    second_load = tuple(fixtures.load_openphi_vision_fixtures())
    assert cases == second_load
    for case, declaration in zip(cases, catalog["cases"], strict=True):
        assert case.sha256 == fixtures._case_digest(declaration)
        for page, source in zip(case.pages, declaration["pages"], strict=True):
            payload, boxes, text = fixtures._render_page(source, font_bytes)
            assert payload == page.pgm_bytes
            assert hashlib.sha256(payload).hexdigest() == page.sha256
            assert boxes == page.gold_boxes
            assert text == page.text


def test_gold_bounds_all_ink_and_offsets_are_exact(cases, catalog):
    font = ImageFont.truetype(
        io.BytesIO(ROOT.joinpath("IBMPlexMono-Regular.ttf").read_bytes()),
        18,
        layout_engine=ImageFont.Layout.BASIC,
    )
    for case, declaration in zip(cases, catalog["cases"], strict=True):
        for page, source in zip(case.pages, declaration["pages"], strict=True):
            coverage = Image.new("L", page.size, 0)
            for box, item in zip(page.gold_boxes, source["items"], strict=True):
                assert page.text[box.start : box.end] == box.text
                assert box.label in (L.PERSON, L.ID_NUM, L.DATE)
                x0, y0, x1, y1 = box.bbox
                assert 0 <= x0 < x1 <= page.size[0]
                assert 0 <= y0 < y1 <= page.size[1]
                # Independent large probe: verify ink dimensions and no clipping.
                probe = Image.new("L", (1024, 128), 0)
                ImageDraw.Draw(probe).text((32, 32), box.text, font=font, fill=255)
                ink = probe.getbbox()
                assert (x1 - x0, y1 - y0) == (ink[2] - ink[0], ink[3] - ink[1])
                assert (x0, y0) == tuple(item["origin"])
                crop = page.to_image().crop(box.bbox)
                assert crop.tobytes() == probe.crop(ink).tobytes()
                coverage.paste(crop, (x0, y0))
            assert coverage.tobytes() == page.to_image().tobytes()


def test_tracks_have_critical_ids_and_specific_layouts(cases):
    assert {case.track for case in cases} == {
        "documents",
        "slide_labels",
        "mixed_script",
        "packets",
    }
    for case in cases:
        for page in case.pages:
            assert any(
                box.critical and box.label == L.ID_NUM for box in page.gold_boxes
            )
    regions = {
        box.region for case in cases for page in case.pages for box in page.gold_boxes
    }
    assert {"header", "footer", "form", "accession_code", "specimen_code"} <= regions
    mixed = next(case for case in cases if case.track == "mixed_script")
    names = [unicodedata.name(char, "") for char in mixed.pages[0].text]
    assert any("LATIN" in name for name in names)
    assert any("CYRILLIC" in name for name in names)
    # Font must render Cyrillic letters distinctly, not missing-glyph boxes.
    font = ImageFont.truetype(
        io.BytesIO(ROOT.joinpath("IBMPlexMono-Regular.ttf").read_bytes()), 18
    )
    assert bytes(font.getmask("С")) != bytes(font.getmask("\ufffd"))
    assert bytes(font.getmask("И")) != bytes(font.getmask("\ufffd"))


def test_packet_groups_cover_pages_without_cross_patient_values(cases):
    for case in cases:
        indices = [index for group in case.expected_packet_splits for index in group]
        assert sorted(indices) == list(range(len(case.pages)))
    packet = next(case for case in cases if case.track == "packets")
    assert packet.expected_packet_splits == ((0, 2), (1, 3))
    group_ids = []
    for group in packet.expected_packet_splits:
        ids = [
            {box.text for box in packet.pages[i].gold_boxes if box.label == L.ID_NUM}
            for i in group
        ]
        assert all(value == ids[0] for value in ids)
        group_ids.append(ids[0])
    assert group_ids[0].isdisjoint(group_ids[1])


def test_reprs_hide_all_gold_values_and_pixels(cases):
    for case in cases:
        rendered = repr(case)
        assert case.sha256 in rendered
        for page in case.pages:
            assert "pgm_bytes=" not in rendered
            assert "text=" not in rendered
            assert page.text not in rendered
            for box in page.gold_boxes:
                assert box.text not in rendered
                assert box.text not in repr(box)
                assert box.text.startswith(("SYNTH", "СИНТ", "2099-"))
        assert case.case_id.startswith("vision-v1-")
        assert all(page.page_id.startswith("page-") for page in case.pages)


def test_loaded_pixels_are_detached_and_case_is_immutable(cases):
    page = cases[0].pages[0]
    image = page.to_image()
    image.paste(0, (0, 0, *page.size))
    assert page.to_image().getbbox() is not None
    with pytest.raises(FrozenInstanceError):
        page.text = "changed"


@pytest.mark.parametrize("target", ["font", "catalog"])
def test_resource_digest_drift_fails_closed(monkeypatch, target):
    actual_files = fixtures.files

    class Resource:
        def __init__(self, path):
            self.path = path

        def joinpath(self, *parts):
            return Resource(self.path.joinpath(*parts))

        def read_bytes(self):
            payload = self.path.read_bytes()
            suffix = "catalog.json" if target == "catalog" else ".ttf"
            return payload + b"DRIFT" if str(self.path).endswith(suffix) else payload

    monkeypatch.setattr(
        fixtures, "files", lambda package: Resource(actual_files(package))
    )
    with pytest.raises(ValueError, match=f"vision fixture {target} digest mismatch"):
        tuple(fixtures.load_openphi_vision_fixtures())


@pytest.mark.parametrize("target", ["box", "pixels", "gold", "split", "critical"])
def test_case_and_rendering_drift_fails_closed(catalog, target):
    case = catalog["cases"][0]
    item = case["pages"][0]["items"][0]
    if target == "box":
        item["bbox"][2] += 1
    elif target == "pixels":
        case["pages"][0]["sha256"] = "0" * 64
    elif target == "gold":
        item["text"] = "DO-NOT-ECHO-THIS-VALUE"
    elif target == "critical":
        item["critical"] = False
    else:
        case["expected_packet_splits"] = [[1]]
    font_bytes = ROOT.joinpath("IBMPlexMono-Regular.ttf").read_bytes()
    with pytest.raises(ValueError, match="case digest mismatch") as caught:
        fixtures._load_case(case, font_bytes)
    assert "DO-NOT-ECHO" not in str(caught.value)
    if target in ("box", "pixels"):
        # Even after a declaration digest is updated, independently pinned ink
        # bounds and pixel digests catch a broken renderer or changed font.
        case["sha256"] = fixtures._case_digest(case)
        with pytest.raises(ValueError, match="rendering mismatch|page digest mismatch"):
            fixtures._load_case(case, font_bytes)
