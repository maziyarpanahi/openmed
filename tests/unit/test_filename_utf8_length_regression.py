"""Sanitized Unicode names must fit byte-limited filesystem components."""

import pytest

from openmed.utils.validation import sanitize_filename


@pytest.mark.parametrize(
    "name",
    ["\u754c" * 100, "\U0001f680" * 100, "a" * 254 + "\u754c", "\u00e9" * 200],
    ids=["cjk", "emoji", "ascii-plus-cjk", "accented"],
)
def test_unicode_filename_is_bounded_without_split_codepoints(name):
    actual = sanitize_filename(name)
    assert actual
    assert len(actual.encode("utf-8")) <= 255
    assert actual == name.encode("utf-8")[:255].decode("utf-8", errors="ignore")
    assert "\ufffd" not in actual


def test_short_surrogate_escaped_name_is_preserved():
    name = "synthetic-\udcff"
    assert sanitize_filename(name) == name


def test_multibyte_filename_can_be_created(tmp_path):
    name = sanitize_filename("\u754c" * 100)
    assert len(name.encode("utf-8")) <= 255
    path = tmp_path / name
    path.write_bytes(b"synthetic")
    assert path.read_bytes() == b"synthetic"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a" * 300, "a" * 255),
        ("short.txt", "short.txt"),
        ("\u00e9.txt", "\u00e9.txt"),
        ("a/b\\c", "a_b_c"),
        ("  ", "output"),
        ("a\x00b", "ab"),
        (123, "123"),
    ],
    ids=[
        "long-ascii",
        "short-ascii",
        "short-unicode",
        "separators",
        "whitespace",
        "null-character",
        "stringify",
    ],
)
def test_existing_sanitization_behaviors_remain(name, expected):
    assert sanitize_filename(name) == expected
