"""Single-line configuration comment handling, not full TOML conformance."""

import pytest

from openmed.core import config


def test_hashes_inside_quoted_values_are_preserved(subtests, tmp_path):
    case_root = tmp_path
    for line, expected in [
        ('cache_dir = "cache#synthetic"', "cache#synthetic"),
        ("cache_dir = 'cache#synthetic'", "cache#synthetic"),
        ('cache_dir = "cache#synthetic" # trailing', "cache#synthetic"),
        ("cache_dir = '#first#second' # trailing", "#first#second"),
        ('cache_dir = "a=b#c"', "a=b#c"),
        ('cache_dir = "synthetic路径#目录"', "synthetic路径#目录"),
        (r'cache_dir = "a\"#b" # trailing', r"a\"#b"),
        (r'cache_dir = "a\\" # trailing', r"a\\"),
        (r"cache_dir = 'a\' # trailing", "a\\"),
    ]:
        with subtests.test(line=line, expected=expected):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()
            path = tmp_path / "synthetic.toml"
            path.write_text(line + "\n", encoding="utf-8")
            assert config._load_toml(path) == {"cache_dir": expected}


def test_existing_scalars_and_unquoted_comments_are_preserved(subtests, tmp_path):
    case_root = tmp_path
    for value, expected in [
        ("10 # comment", 10),
        ("0.5 # comment", 0.5),
        ("true # comment", True),
        ("false # comment", False),
        ("null # comment", None),
        ('"plain" # comment', "plain"),
    ]:
        with subtests.test(value=value, expected=expected):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()
            path = tmp_path / "synthetic.toml"
            path.write_text("# heading\n\nkey = " + value + "\n", encoding="utf-8")
            assert config._load_toml(path) == {"key": expected}


def test_generated_scalar_config_roundtrips_hashes(tmp_path):
    values = {
        "cache_dir": "cache#synthetic",
        "device": "cpu",
        "timeout": 10,
        "local_only": True,
        "pii_model_revision": None,
    }
    path = tmp_path / "synthetic.toml"
    path.write_text(config._dump_toml(values), encoding="utf-8")
    assert config._load_toml(path) == values
