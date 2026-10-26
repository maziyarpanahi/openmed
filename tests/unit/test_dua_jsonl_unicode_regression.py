import json

import pytest

from openmed.eval.datasets._dua import load_json_rows


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_literal_unicode_separators_remain_inside_json_strings(
    tmp_path, separator, suffix
):
    rows = [{"text": "synthetic" + separator + "note"}, {"text": "second note"}]
    path = tmp_path / ("synthetic" + suffix)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    assert load_json_rows(path, dataset="synthetic", authority="test") == rows


def test_crlf_blank_lines_and_final_record_without_newline(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_bytes(b'\r\n{"text":"one"}\r\n\r\n{"text":"two"}')
    assert load_json_rows(path, dataset="synthetic", authority="test") == [
        {"text": "one"},
        {"text": "two"},
    ]


def test_error_uses_physical_line_number_after_unicode_separator(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text(
        json.dumps({"text": "one\u2028two"}, ensure_ascii=False) + "\n\nnot-json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSONL row 3 is invalid"):
        load_json_rows(path, dataset="synthetic", authority="test")


def test_regular_json_loading_unchanged(tmp_path):
    path = tmp_path / "synthetic.json"
    rows = [{"text": "one\u2028two"}]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert load_json_rows(path, dataset="synthetic", authority="test") == rows


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson", ".json"])
def test_object_pairs_hook_is_preserved_for_each_format(tmp_path, suffix):
    path = tmp_path / ("synthetic" + suffix)
    rows = [{"text": "synthetic", "nested": {"flag": 1}}]
    payload = rows if suffix == ".json" else rows[0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed = []

    def hook(pairs):
        observed.append(tuple(key for key, _ in pairs))
        return dict(pairs)

    result = load_json_rows(
        path, dataset="synthetic", authority="test", object_pairs_hook=hook
    )
    assert result == rows
    assert observed == [("flag",), ("text", "nested")]


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_unicode_content_does_not_bypass_duplicate_key_hook(tmp_path, suffix):
    path = tmp_path / ("synthetic" + suffix)
    path.write_text('{"text":"synthetic\u2028value","key":1,"key":2}', encoding="utf-8")

    class DuplicateKey(ValueError):
        pass

    def reject_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise DuplicateKey("duplicate_key")
        return dict(pairs)

    with pytest.raises(DuplicateKey, match="^duplicate_key$"):
        load_json_rows(
            path,
            dataset="synthetic",
            authority="test",
            object_pairs_hook=reject_duplicates,
        )
