"""Exercise the line boundaries already recognized by str.splitlines()."""

import pytest

from openmed.multimodal.documents_text import extract_text, write_redacted_text

ENDINGS = [
    "\r\n",
    "\r",
    "\n",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
]


def write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("ending", ENDINGS)
def test_line_columns_exclude_the_recognized_ending(tmp_path, ending):
    text = "abc" + ending + "de"
    document = extract_text(write(tmp_path / "source.txt", text))
    first, second = document.metadata["lines"]
    assert document.text == text
    assert first["columns"] == 3
    assert first["content_end"] == 3
    assert first["newline"] == ending
    assert second["start"] == 3 + len(ending)
    assert second["columns"] == 2
    assert document.metadata["max_columns"] == 3


@pytest.mark.parametrize("ending", ENDINGS)
def test_replacement_cannot_consume_a_line_boundary(tmp_path, ending):
    source = write(tmp_path / "source.txt", "abc" + ending + "de")
    with pytest.raises(ValueError, match="line endings"):
        write_redacted_text(source, tmp_path / "out.txt", [(2, 4 + len(ending), "X")])
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.parametrize("ending", ENDINGS)
def test_replacement_cannot_introduce_a_line_boundary(tmp_path, ending):
    source = write(tmp_path / "source.txt", "abcdef")
    with pytest.raises(ValueError, match="line endings"):
        write_redacted_text(source, tmp_path / "out.txt", [(0, 3, "X" + ending)])


@pytest.mark.parametrize("ending", ENDINGS)
def test_valid_write_keeps_original_line_bytes(tmp_path, ending):
    source = write(tmp_path / "source.txt", "abc" + ending + "de")
    output = write_redacted_text(source, tmp_path / "out.txt", [(0, 3, "X")])
    assert output.read_bytes() == ("X  " + ending + "de").encode("utf-8")


@pytest.mark.parametrize("text,count", [("", 0), ("abc", 1), ("abc\n", 1)])
def test_empty_and_unterminated_controls(tmp_path, text, count):
    assert (
        extract_text(write(tmp_path / "source.txt", text)).metadata["line_count"]
        == count
    )
