from __future__ import annotations

import hashlib
import importlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.core.labels import DATE, ID_NUM, ID_SUBTYPE_MRN, PERSON
from openmed.training.synthetic import BurnedInGenerator, generate_burned_in_example
from openmed.training.synthetic.burned_in import (
    COMPLEX_LAYOUT_SCRIPTS,
    DEFAULT_FONT_LICENSE,
    DEFAULT_FONT_NAMES,
    FONT_SCRIPTS,
    PILLOW_DEFAULT_FONT_LICENSE,
    PILLOW_DEFAULT_FONT_NAME,
    BurnedInFont,
    BurnedInFontError,
    BurnedInFontUnavailableError,
    BurnedInGlyphCoverageError,
    BurnedInShapingUnavailableError,
    FontCharacterMap,
    read_font_character_map,
)

Image = pytest.importorskip("PIL.Image")
ImageChops = pytest.importorskip("PIL.ImageChops")
ImageFont = pytest.importorskip("PIL.ImageFont")
PILFeatures = pytest.importorskip("PIL.features")

REPOSITORY_ROOT = Path(__file__).parents[3]
# Repository-owned OFL-1.1 brand font: covers Latin, Cyrillic, and Greek only.
BUNDLED_FONT = REPOSITORY_ROOT / "docs/brand/assets/fonts/IBMPlexSans-Regular.ttf"
requires_bundled_font = pytest.mark.skipif(
    not BUNDLED_FONT.is_file(), reason="repository brand font is not checked out"
)
_FIELD = r"[a-z0-9_]+=[A-Za-z0-9_.:-]+"
_VALUE_FREE_MESSAGE = re.compile(rf"^[a-z_]+(: {_FIELD}(, {_FIELD})*)?$")


def _default_fonts_installed() -> bool:
    try:
        for name in DEFAULT_FONT_NAMES:
            ImageFont.truetype(name, 12)
    except OSError:
        return False
    return True


def _pillow_default_digest() -> str:
    return hashlib.sha256(ImageFont.load_default(size=12).font_bytes).hexdigest()


def _bundled_font() -> BurnedInFont:
    return BurnedInFont(BUNDLED_FONT, license="OFL-1.1")


class _EveryCodePoint:
    def __contains__(self, code_point: object) -> bool:
        return True


def _cover_everything(data: bytes, index: int) -> _EveryCodePoint:
    return _EveryCodePoint()


@pytest.fixture
def hide_default_fonts(monkeypatch):
    """Make the DejaVu defaults unavailable on every platform."""

    real_truetype = ImageFont.truetype

    def truetype_without_defaults(font, *args, **kwargs):
        if isinstance(font, (str, os.PathLike)) and (
            os.path.basename(os.fspath(font)) in DEFAULT_FONT_NAMES
        ):
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", truetype_without_defaults)


def _synthetic_person(monkeypatch, value: str) -> None:
    """Replace locale PHI values with a synthetic PERSON string."""

    def values(language, case_seed):
        return (
            (value, PERSON, {"source": "test"}),
            ("2024-01-02", DATE, {"source": "test"}),
            ("MRN-1234567", ID_NUM, {"source": "test"}),
        )

    monkeypatch.setattr(BurnedInGenerator, "_synthetic_values", staticmethod(values))


def _text_mask(example):
    background = Image.new(
        "L", example.image.size, color=example.metadata["background_value"]
    )
    return ImageChops.difference(example.image, background)


def _assert_boxes_bound_text(example) -> None:
    changed = _text_mask(example)
    assert changed.getbbox() is not None
    for span, box in zip(example.gold_spans, example.gold_boxes, strict=True):
        assert example.text[box.start : box.end] == box.text == span.text
        assert (box.start, box.end) == (span.start, span.end)
        x0, y0, x1, y1 = box.bbox
        assert changed.crop(box.bbox).getbbox() == (0, 0, x1 - x0, y1 - y0)
    outside = changed.copy()
    for box in example.gold_boxes:
        outside.paste(0, box.bbox)
    assert outside.getbbox() is None


def test_rendered_case_has_aligned_canonical_spans_and_controlled_boxes():
    positions = ((20, 20), (40, 120), (60, 220))
    example = BurnedInGenerator(seed=1790, font_fallback=True).generate(
        "en",
        positions=positions,
        font_sizes=(24, 22, 20),
        font_names=("DejaVuSans.ttf",) * 3,
        background="solid",
    )

    assert example.image.mode == "L"
    assert example.image.size == (640, 320)
    assert example.metadata["synthetic"] is True
    assert example.metadata["contains_real_phi"] is False
    assert example.metadata["augmentation_only"] is True
    assert example.metadata["synthetic_source"] == "burned_in"
    assert example.metadata["font_fallback_allowed"] is True
    assert example.image.info["synthetic"] is True
    assert tuple(span.label for span in example.gold_spans) == (
        PERSON,
        DATE,
        ID_NUM,
    )
    assert len(example.gold_boxes) == len(example.gold_spans) == 3

    for position, span, box in zip(
        positions, example.gold_spans, example.gold_boxes, strict=True
    ):
        assert example.text[span.start : span.end] == span.text
        assert (box.start, box.end, box.label, box.text) == (
            span.start,
            span.end,
            span.label,
            span.text,
        )
        assert box.bbox[:2] == position
        assert box.canonical_label == span.label
        assert span.metadata["synthetic"] is True
        assert box.metadata["synthetic"] is True
        assert re.fullmatch(r"[0-9a-f]{64}", box.metadata["font_sha256"])
        assert box.metadata["font_license"] in {
            DEFAULT_FONT_LICENSE,
            PILLOW_DEFAULT_FONT_LICENSE,
        }

    mrn_span = example.gold_spans[-1]
    assert mrn_span.text.startswith("MRN-")
    assert mrn_span.metadata["id_subtype"] == ID_SUBTYPE_MRN
    assert example.gold_boxes[-1].metadata["source"] == "clinical_ids"

    item = example.to_training_item()
    assert item["is_synthetic"] is True
    assert item["synthetic_source"] == "burned_in"
    assert item["labels"] == [span.to_dict() for span in example.gold_spans]
    assert item["pixel_boxes"] == [box.to_dict() for box in example.gold_boxes]


def test_gold_boxes_bound_every_drawn_text_pixel():
    example = BurnedInGenerator(seed=41, font_fallback=True).generate(
        positions=((16, 16), (16, 112), (16, 208)),
        font_sizes=(26, 26, 26),
        font_names=("DejaVuSansMono.ttf",) * 3,
        background="solid",
    )
    background = Image.new(
        "L", example.image.size, color=example.metadata["background_value"]
    )
    changed = ImageChops.difference(example.image, background)

    assert changed.getbbox() is not None
    for box in example.gold_boxes:
        x0, y0, x1, y1 = box.bbox
        assert 0 <= x0 < x1 <= example.image.width
        assert 0 <= y0 < y1 <= example.image.height
        assert changed.crop(box.bbox).getbbox() == (0, 0, x1 - x0, y1 - y0)

    for y in range(example.image.height):
        for x in range(example.image.width):
            if changed.getpixel((x, y)):
                assert any(
                    x0 <= x < x1 and y0 <= y < y1
                    for x0, y0, x1, y1 in (
                        annotation.bbox for annotation in example.gold_boxes
                    )
                )


def test_generation_is_deterministic_per_seed_and_varies_across_seeds():
    first = generate_burned_in_example(seed=101, font_fallback=True)
    second = generate_burned_in_example(seed=101, font_fallback=True)
    different = generate_burned_in_example(seed=102, font_fallback=True)

    assert first.text == second.text
    assert first.gold_spans == second.gold_spans
    assert first.gold_boxes == second.gold_boxes
    assert first.metadata == second.metadata
    assert first.image.tobytes() == second.image.tobytes()
    assert (
        first.text,
        first.gold_boxes,
        first.image.tobytes(),
    ) != (
        different.text,
        different.gold_boxes,
        different.image.tobytes(),
    )


@pytest.mark.skipif(
    not _default_fonts_installed(), reason="DejaVu default fonts are not installed"
)
def test_default_english_output_is_unchanged_when_default_fonts_are_installed(
    monkeypatch,
):
    module = importlib.import_module("openmed.training.synthetic.burned_in")
    example = generate_burned_in_example(seed=101)

    assert example.metadata["font_fallback_allowed"] is False
    assert example.metadata["font_fallback_used"] is False
    for box in example.gold_boxes:
        assert box.font_name in DEFAULT_FONT_NAMES
        assert box.metadata["font_fallback"] is False
        assert box.metadata["font_license"] == DEFAULT_FONT_LICENSE

    def load_by_name(font_mod, resolved, size):
        # The pre-check renderer loaded default fonts by name, not from bytes.
        return font_mod.truetype(resolved.display_name, size)

    monkeypatch.setattr(module, "_load_face", load_by_name)
    legacy = generate_burned_in_example(seed=101)

    assert legacy.image.tobytes() == example.image.tobytes()
    assert legacy.gold_boxes == example.gold_boxes


def test_missing_default_font_is_never_replaced_without_opt_in(hide_default_fonts):
    with pytest.raises(BurnedInFontUnavailableError) as raised:
        BurnedInGenerator(seed=3).generate("en")

    error = raised.value
    assert isinstance(error, ValueError)
    assert error.reason == "font_unavailable"
    assert error.font_ref in DEFAULT_FONT_NAMES
    assert error.label == PERSON
    assert _VALUE_FREE_MESSAGE.match(str(error))


def test_opt_in_fallback_is_recorded_with_font_digest_and_license(
    hide_default_fonts,
):
    example = BurnedInGenerator(seed=3, font_fallback=True).generate("en")
    digest = _pillow_default_digest()

    for box in example.gold_boxes:
        assert box.font_name == PILLOW_DEFAULT_FONT_NAME
        assert box.metadata["font_fallback"] is True
        assert box.metadata["font_sha256"] == digest
        assert box.metadata["font_license"] == PILLOW_DEFAULT_FONT_LICENSE
    assert example.metadata["font_fallback_allowed"] is True
    assert example.metadata["font_fallback_used"] is True
    assert example.metadata["fonts"] == [
        {
            "font_fallback": True,
            "font_license": PILLOW_DEFAULT_FONT_LICENSE,
            "font_name": PILLOW_DEFAULT_FONT_NAME,
            "font_sha256": digest,
        }
    ]


@requires_bundled_font
def test_fallback_prefers_an_installed_default_and_flags_each_box(monkeypatch):
    real_truetype = ImageFont.truetype

    def only_dejavu_sans(font, *args, **kwargs):
        # Stand in the bundled font for DejaVuSans; the other defaults are missing.
        if font == "DejaVuSans.ttf":
            font = str(BUNDLED_FONT)
        elif isinstance(font, str) and font in DEFAULT_FONT_NAMES:
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", only_dejavu_sans)
    example = BurnedInGenerator(seed=3, font_fallback=True).generate(
        "en", font_names=("DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf")
    )

    person, date, mrn = example.gold_boxes
    digest = hashlib.sha256(BUNDLED_FONT.read_bytes()).hexdigest()
    assert {box.metadata["font_sha256"] for box in example.gold_boxes} == {digest}
    assert {box.font_name for box in example.gold_boxes} == {"DejaVuSans.ttf"}
    assert [box.metadata["font_fallback"] for box in (person, date, mrn)] == [
        False,
        True,
        False,
    ]
    assert example.metadata["font_fallback_used"] is True
    assert [font["font_fallback"] for font in example.metadata["fonts"]] == [
        False,
        True,
    ]


def test_unloadable_caller_font_is_reported_without_its_path(tmp_path):
    missing = tmp_path / "private-dir" / "NotInstalled.ttf"
    font = BurnedInFont(missing, license="OFL-1.1")

    with pytest.raises(BurnedInFontUnavailableError) as raised:
        BurnedInGenerator(seed=3, font_names=(font,)).generate("en")

    message = str(raised.value)
    assert raised.value.font_ref.startswith("name-sha256:")
    assert "private-dir" not in message and "NotInstalled" not in message
    assert _VALUE_FREE_MESSAGE.match(message)


@pytest.mark.parametrize(
    ("language", "script"),
    [("th", "Thai"), ("hi", "Devanagari"), ("he", "Hebrew"), ("fa", "Arabic")],
)
def test_uncovered_text_raises_value_free_error_naming_script_and_digest(
    hide_default_fonts, language, script
):
    # The pre-check renderer silently drew these spans with Pillow's Latin-only
    # default font, so distinct names rendered as identical missing-glyph boxes.
    rendered = BurnedInGenerator(
        seed=9,
        font_fallback=True,
        character_map_provider=_cover_everything,
        layout_capability_provider=lambda font: True,
    ).generate(language)

    with pytest.raises(BurnedInGlyphCoverageError) as raised:
        BurnedInGenerator(seed=9, font_fallback=True).generate(language)

    error = raised.value
    message = str(error)
    assert error.reason == "glyph_coverage_missing"
    assert error.script == script
    assert error.label == PERSON
    assert error.font_sha256 == _pillow_default_digest()
    assert error.missing_count > 0
    assert script in message and error.font_sha256 in message
    assert _VALUE_FREE_MESSAGE.match(message)
    assert message.isascii()
    for span in rendered.gold_spans:
        assert span.text not in message
    assert all(
        span.text not in repr(arg) for arg in error.args for span in rendered.gold_spans
    )


@requires_bundled_font
@pytest.mark.parametrize(("language", "script"), [("ru", "Cyrillic"), ("el", "Greek")])
def test_covering_font_renders_distinct_strings_with_aligned_boxes(
    hide_default_fonts, language, script
):
    controls = {
        "positions": ((16, 16), (16, 112), (16, 208)),
        "font_sizes": (22, 22, 22),
        "background": "solid",
    }
    first = BurnedInGenerator(seed=11, font_names=(_bundled_font(),)).generate(
        language, **controls
    )
    second = BurnedInGenerator(seed=12, font_names=(_bundled_font(),)).generate(
        language, **controls
    )

    digest = hashlib.sha256(BUNDLED_FONT.read_bytes()).hexdigest()
    first_person, second_person = first.gold_boxes[0], second.gold_boxes[0]
    assert first_person.text != second_person.text
    assert first_person.metadata["scripts"] == [script]
    assert first_person.font_name == BUNDLED_FONT.name
    assert first_person.metadata["font_sha256"] == digest
    assert first_person.metadata["font_license"] == "OFL-1.1"
    assert first_person.metadata["font_fallback"] is False

    first_glyphs = _text_mask(first).crop(first_person.bbox).point(lambda v: v and 255)
    second_glyphs = (
        _text_mask(second).crop(second_person.bbox).point(lambda v: v and 255)
    )
    assert first_glyphs.tobytes() != second_glyphs.tobytes()
    _assert_boxes_bound_text(first)
    _assert_boxes_bound_text(second)


@requires_bundled_font
def test_same_seed_and_font_digest_produce_byte_identical_images(
    hide_default_fonts,
):
    first = BurnedInGenerator(seed=21, font_names=(_bundled_font(),)).generate("ru")
    second = BurnedInGenerator(seed=21, font_names=(_bundled_font(),)).generate("ru")

    assert first.metadata["fonts"] == second.metadata["fonts"]
    assert [font["font_sha256"] for font in first.metadata["fonts"]] == [
        hashlib.sha256(BUNDLED_FONT.read_bytes()).hexdigest()
    ]
    assert first.image.tobytes() == second.image.tobytes()
    assert first.gold_boxes == second.gold_boxes


@requires_bundled_font
def test_script_fonts_route_spans_and_record_per_box_provenance(hide_default_fonts):
    example = BurnedInGenerator(
        seed=5,
        font_fallback=True,
        script_fonts={"Cyrillic": _bundled_font()},
    ).generate("ru", background="solid")

    person, date, mrn = example.gold_boxes
    bundled_digest = hashlib.sha256(BUNDLED_FONT.read_bytes()).hexdigest()
    assert person.metadata["scripts"] == ["Cyrillic"]
    assert person.metadata["font_sha256"] == bundled_digest
    assert person.metadata["font_fallback"] is False
    assert date.metadata["font_sha256"] == _pillow_default_digest()
    assert mrn.metadata["font_fallback"] is True
    assert [font["font_sha256"] for font in example.metadata["fonts"]] == [
        bundled_digest,
        _pillow_default_digest(),
    ]
    assert example.metadata["font_fallback_used"] is True
    _assert_boxes_bound_text(example)


# Synthetic strings exercising joining, conjuncts, reordering, and stacking.
_COMPLEX_SAMPLES = [
    ("Arabic", "ببب متحن"),
    ("Devanagari", "क्ष त्र"),
    ("Thai", "ทดสอบ่"),
    ("Myanmar", "ကြာ္က"),
    ("Ethiopic", "ሀለመ"),
    ("Hebrew", "אבג"),
]


@pytest.mark.parametrize(("script", "value"), _COMPLEX_SAMPLES)
def test_complex_scripts_fail_closed_without_a_shaping_layout_engine(
    monkeypatch, hide_default_fonts, script, value
):
    _synthetic_person(monkeypatch, value)
    generator = BurnedInGenerator(
        seed=1,
        font_fallback=True,
        character_map_provider=_cover_everything,
        layout_capability_provider=lambda font: False,
    )

    with pytest.raises(BurnedInShapingUnavailableError) as raised:
        generator.generate("en")

    error = raised.value
    assert script in COMPLEX_LAYOUT_SCRIPTS
    assert error.reason == "complex_layout_unavailable"
    assert error.script == script
    assert error.label == PERSON
    assert error.font_sha256 == _pillow_default_digest()
    assert error.layout_engine in {"basic", "raqm", "bitmap"}
    assert value not in str(error)
    assert _VALUE_FREE_MESSAGE.match(str(error))


@pytest.mark.parametrize(("script", "value"), _COMPLEX_SAMPLES)
def test_complex_scripts_render_when_the_layout_engine_shapes(
    monkeypatch, hide_default_fonts, script, value
):
    _synthetic_person(monkeypatch, value)
    seen_fonts = []

    def shapes(font):
        seen_fonts.append(font)
        return True

    example = BurnedInGenerator(
        seed=1,
        font_fallback=True,
        character_map_provider=_cover_everything,
        layout_capability_provider=shapes,
    ).generate("en")

    person = example.gold_boxes[0]
    assert person.text == value
    assert person.metadata["scripts"] == [script]
    assert person.metadata["layout_engine"] in {"basic", "raqm"}
    # The capability provider is asked only for spans that need shaping.
    assert len(seen_fonts) == 1


def test_default_layout_capability_requires_raqm():
    module = importlib.import_module("openmed.training.synthetic.burned_in")
    basic = ImageFont.load_default(size=12)

    assert basic.layout_engine == ImageFont.Layout.BASIC
    assert module.font_supports_complex_layout(basic) is False
    assert module.font_supports_complex_layout(object()) is False


def test_latin_text_never_requires_shaping(monkeypatch, hide_default_fonts):
    calls = []
    example = BurnedInGenerator(
        seed=1,
        font_fallback=True,
        layout_capability_provider=lambda font: calls.append(font) or False,
    ).generate("en")

    assert not calls
    assert {box.metadata["layout_engine"] for box in example.gold_boxes} == {"basic"}


def test_pillow_default_character_map_is_latin_only():
    data = ImageFont.load_default(size=12).font_bytes
    coverage = read_font_character_map(data)

    assert isinstance(coverage, FontCharacterMap)
    assert all(code_point in coverage for code_point in range(0x20, 0x7F))
    for code_point in (0x0E01, 0x0915, 0x05D0, 0x0628, 0x1200, 0x1000, 0x0416):
        assert code_point not in coverage
    assert 90 < len(coverage) < 200


@requires_bundled_font
def test_bundled_font_character_map_matches_its_declared_scripts():
    coverage = read_font_character_map(BUNDLED_FONT.read_bytes())

    assert ord("Ж") in coverage and ord("Ω") in coverage and ord("ệ") in coverage
    assert 0x0E01 not in coverage and 0x0915 not in coverage


def test_font_collection_faces_are_read_by_index(tmp_path):
    single = ImageFont.load_default(size=12).font_bytes
    collection = _as_collection(single)

    assert len(read_font_character_map(collection, 0)) == len(
        read_font_character_map(single)
    )
    with pytest.raises(BurnedInFontError) as raised:
        read_font_character_map(collection, 1)
    assert raised.value.reason == "font_unparseable"

    path = tmp_path / "Collection.ttc"
    path.write_bytes(collection)
    font = BurnedInFont(path, license="CC0-1.0", index=0)
    example = BurnedInGenerator(seed=3, font_names=(font,)).generate("en")
    assert {box.font_name for box in example.gold_boxes} == {"Collection.ttc"}
    assert {box.metadata["font_sha256"] for box in example.gold_boxes} == {
        hashlib.sha256(collection).hexdigest()
    }


def test_character_map_follows_freetype_subtable_choice_and_skips_notdef():
    bmp = _cmap_format4({0x41: 5, 0x42: 6})
    ucs4 = _cmap_format12([(0x40, 0x43, 0), (0x1F600, 0x1F610, 8)])
    font = _sfnt(
        {
            b"maxp": struct.pack(">IH", 0x00005000, 12),
            b"cmap": _cmap([((3, 1), bmp), ((3, 10), ucs4)]),
        }
    )

    coverage = read_font_character_map(font)

    # The UCS-4 subtable wins; code 0x40 maps to .notdef and glyph ids at or
    # beyond numGlyphs (12) are not real glyphs.
    assert 0x40 not in coverage
    assert all(code_point in coverage for code_point in (0x41, 0x42, 0x43))
    assert 0x1F600 in coverage and 0x1F603 in coverage
    assert 0x1F604 not in coverage
    assert len(coverage) == 7


def test_unreadable_font_bytes_fail_closed():
    for data in (b"", b"not a font at all", b"\x00\x01\x00\x00\x00\x09"):
        with pytest.raises(BurnedInFontError) as raised:
            read_font_character_map(data)
        assert raised.value.reason == "font_unparseable"

    symbol_only = _sfnt({b"cmap": _cmap([((3, 0), _cmap_format4({0x41: 1}))])})
    assert len(read_font_character_map(symbol_only)) == 0


def test_burned_in_font_requires_a_permissive_declared_license(tmp_path):
    path = tmp_path / "Font.ttf"
    assert BurnedInFont(path, license=" ofl-1.1 ").license == "OFL-1.1"
    assert BurnedInFont(path, license="apache-2.0").license == "Apache-2.0"

    for value in ("GPL-3.0-only", "LicenseRef-Vendor-Font", "", None, "OFL-1.1+"):
        with pytest.raises(BurnedInFontError) as raised:
            BurnedInFont(path, license=value)
        assert raised.value.reason == "font_license_not_permissive"
        assert str(raised.value) == "font_license_not_permissive"

    with pytest.raises(ValueError, match="font index"):
        BurnedInFont(path, license="MIT", index=-1)
    with pytest.raises(ValueError, match="font path"):
        BurnedInFont("  ", license="MIT")


def test_script_fonts_and_providers_are_validated(tmp_path):
    font = BurnedInFont(tmp_path / "Font.ttf", license="MIT")

    assert "Thai" in FONT_SCRIPTS and "Inherited" not in FONT_SCRIPTS
    with pytest.raises(ValueError, match="unsupported font script"):
        BurnedInGenerator(script_fonts={"Klingon": font})
    with pytest.raises(ValueError, match="BurnedInFont"):
        BurnedInGenerator(script_fonts={"Thai": "NotoSansThai-Regular.ttf"})
    with pytest.raises(ValueError, match="font_fallback"):
        BurnedInGenerator(font_fallback="yes")
    with pytest.raises(TypeError, match="callable"):
        BurnedInGenerator(character_map_provider=object())


def _thai_system_font() -> Path | None:
    candidates = (
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansThai-Regular.ttf",
    )
    return next((Path(path) for path in candidates if Path(path).is_file()), None)


@pytest.mark.skipif(
    _thai_system_font() is None or not PILFeatures.check_feature("raqm"),
    reason="needs an installed Noto Sans Thai font and Pillow Raqm layout",
)
def test_system_thai_font_with_raqm_renders_distinct_shaped_names(hide_default_fonts):
    font = BurnedInFont(_thai_system_font(), license="OFL-1.1")
    controls = {
        "positions": ((16, 16), (16, 112), (16, 208)),
        "font_sizes": (22, 22, 22),
        "background": "solid",
    }
    examples = [
        BurnedInGenerator(
            seed=seed, font_fallback=True, script_fonts={"Thai": font}
        ).generate("th", **controls)
        for seed in (31, 32)
    ]

    people = [example.gold_boxes[0] for example in examples]
    assert people[0].text != people[1].text
    assert {person.metadata["layout_engine"] for person in people} == {"raqm"}
    crops = [
        _text_mask(example).crop(person.bbox).point(lambda v: v and 255).tobytes()
        for example, person in zip(examples, people, strict=True)
    ]
    assert crops[0] != crops[1]
    for example in examples:
        _assert_boxes_bound_text(example)


def test_importing_core_and_burned_in_module_does_not_import_pillow():
    code = (
        "import sys\n"
        "import openmed\n"
        "import openmed.training.synthetic.burned_in\n"
        "print(any(name == 'PIL' or name.startswith('PIL.') "
        "for name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_renderer_names_multimodal_extra_when_pillow_is_missing(monkeypatch):
    module = importlib.import_module("openmed.training.synthetic.burned_in")
    real_import_module = module.importlib.import_module

    def import_without_pillow(name: str):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", import_without_pillow)

    with pytest.raises(ImportError, match=r"openmed\[multimodal\]"):
        BurnedInGenerator(seed=7).generate()


def test_multimodal_extra_owns_pillow_dependency():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
        import tomli as tomllib

    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    core_dependencies = project["dependencies"]
    multimodal_dependencies = project["optional-dependencies"]["multimodal"]
    assert not any(
        dependency.lower().startswith("pillow") for dependency in core_dependencies
    )
    assert any(
        dependency.lower().startswith("pillow")
        for dependency in multimodal_dependencies
    )


def _sfnt(tables: dict[bytes, bytes]) -> bytes:
    tags = sorted(tables)
    offset = 12 + (16 * len(tags))
    directory = bytearray(struct.pack(">IHHHH", 0x00010000, len(tags), 0, 0, 0))
    body = bytearray()
    for tag in tags:
        data = tables[tag]
        directory += struct.pack(">4sIII", tag, 0, offset + len(body), len(data))
        body += data + (b"\x00" * (-len(data) % 4))
    return bytes(directory + body)


def _as_collection(font: bytes) -> bytes:
    header_size = 16
    shifted = bytearray(font)
    (table_count,) = struct.unpack_from(">H", shifted, 4)
    for record in range(table_count):
        position = 12 + (16 * record) + 8
        (table_offset,) = struct.unpack_from(">I", shifted, position)
        struct.pack_into(">I", shifted, position, table_offset + header_size)
    return struct.pack(">4sIII", b"ttcf", 0x00010000, 1, header_size) + bytes(shifted)


def _cmap(subtables: list[tuple[tuple[int, int], bytes]]) -> bytes:
    header = bytearray(struct.pack(">HH", 0, len(subtables)))
    body = bytearray()
    start = 4 + (8 * len(subtables))
    for (platform, encoding), data in subtables:
        header += struct.pack(">HHI", platform, encoding, start + len(body))
        body += data
    return bytes(header + body)


def _cmap_format4(mapping: dict[int, int]) -> bytes:
    segments = [
        (code, code, (glyph - code) & 0xFFFF) for code, glyph in sorted(mapping.items())
    ]
    segments.append((0xFFFF, 0xFFFF, 1))
    count = len(segments)
    ends = b"".join(struct.pack(">H", end) for _, end, _ in segments)
    starts = b"".join(struct.pack(">H", start) for start, _, _ in segments)
    deltas = b"".join(struct.pack(">H", delta) for _, _, delta in segments)
    range_offsets = b"\x00\x00" * count
    body = ends + b"\x00\x00" + starts + deltas + range_offsets
    return struct.pack(">HHHHHHH", 4, 14 + len(body), 0, 2 * count, 0, 0, 0) + body


def _cmap_format12(groups: list[tuple[int, int, int]]) -> bytes:
    body = b"".join(struct.pack(">III", *group) for group in groups)
    return struct.pack(">HHIII", 12, 0, 16 + len(body), 0, len(groups)) + body
