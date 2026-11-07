"""Offline sweep: burned-in renders draw only covered, shaped PHI or fail closed."""

from __future__ import annotations

import os

import pytest

from openmed.training.synthetic import SUPPORTED_LOCALE_PHI_LANGUAGES
from openmed.training.synthetic.burned_in import (
    COMPLEX_LAYOUT_SCRIPTS,
    DEFAULT_FONT_NAMES,
    BurnedInFontError,
    BurnedInGenerator,
    read_font_character_map,
)

ImageFont = pytest.importorskip("PIL.ImageFont")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:OpenMed. language:UserWarning"),
]

_SEEDS = (0, 1, 2)


class _EveryCodePoint:
    def __contains__(self, code_point: object) -> bool:
        return True


@pytest.fixture
def platform_independent_fonts(monkeypatch):
    """Hide the DejaVu defaults so every platform uses Pillow's embedded font."""

    real_truetype = ImageFont.truetype

    def truetype_without_defaults(font, *args, **kwargs):
        if isinstance(font, (str, os.PathLike)) and (
            os.path.basename(os.fspath(font)) in DEFAULT_FONT_NAMES
        ):
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", truetype_without_defaults)


def _span_texts(language: str, seed: int) -> list[str]:
    # Render with permissive providers only to learn the synthetic values that
    # a refused render must never echo.
    example = BurnedInGenerator(
        seed=seed,
        font_fallback=True,
        character_map_provider=lambda data, index: _EveryCodePoint(),
        layout_capability_provider=lambda font: True,
    ).generate(language)
    return [span.text for span in example.gold_spans]


@pytest.mark.parametrize("language", SUPPORTED_LOCALE_PHI_LANGUAGES)
def test_every_locale_renders_covered_shaped_text_or_fails_closed(
    platform_independent_fonts, language
):
    coverage = read_font_character_map(ImageFont.load_default(size=12).font_bytes)

    for seed in _SEEDS:
        try:
            example = BurnedInGenerator(seed=seed, font_fallback=True).generate(
                language
            )
        except BurnedInFontError as error:
            message = str(error)
            assert message.isascii()
            assert error.font_sha256 is not None
            for text in _span_texts(language, seed):
                assert text not in message
            with pytest.raises(type(error)) as repeated:
                BurnedInGenerator(seed=seed, font_fallback=True).generate(language)
            assert str(repeated.value) == message
            continue

        repeat = BurnedInGenerator(seed=seed, font_fallback=True).generate(language)
        assert repeat.image.tobytes() == example.image.tobytes()
        assert repeat.metadata == example.metadata
        for box in example.gold_boxes:
            assert all(ord(char) in coverage for char in box.text)
            assert box.metadata["font_fallback"] is True
            complex_scripts = set(box.metadata["scripts"]) & COMPLEX_LAYOUT_SCRIPTS
            assert not complex_scripts or box.metadata["layout_engine"] == "raqm"
