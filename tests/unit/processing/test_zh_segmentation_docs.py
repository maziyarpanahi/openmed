"""Source-of-truth guards for the Chinese segmentation operations guide.

Every operational number, environment variable, and behavioural claim in
``docs/chinese-segmentation-operations.md`` is asserted against the code it
describes, so the guide cannot silently drift from the implementation.
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

from openmed.core import config as config_module
from openmed.core.config import OpenMedConfig
from openmed.processing import tokenization
from openmed.processing import zh_segmentation as segmentation

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs" / "chinese-segmentation-operations.md"
ANONYMIZATION = ROOT / "docs" / "anonymization.md"
MKDOCS = ROOT / "mkdocs.yml"
PUBLICATION = ROOT / "docs" / "brand" / "system" / "publication.yml"
PYPROJECT = ROOT / "pyproject.toml"
SEGMENTATION_TESTS = Path(__file__).with_name("test_zh_segmentation.py")

_UNITS = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _tables(text: str) -> list[list[list[str]]]:
    """Split markdown into tables, each a list of cell rows."""

    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if set("".join(cells)) <= {"-", ":", " "}:
                continue  # alignment row
            current.append(cells)
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _table_rows(text: str, *header_terms: str) -> dict[str, list[str]]:
    """Return the row map of the one table whose header holds every term.

    Several tables in the guide share a first-column value (``jieba`` appears
    in both the backend and deployment tables), so lookups must be scoped to a
    single table rather than merged across the page.
    """

    matches = [
        table
        for table in _tables(text)
        if table and all(term in table[0] for term in header_terms)
    ]
    assert len(matches) == 1, (
        f"expected exactly one table with header {header_terms!r}, found {len(matches)}"
    )
    return {row[0]: row[1:] for row in matches[0][1:]}


@contextmanager
def _stub_module(name: str, module: object):
    """Temporarily install a stand-in for an uninstalled optional backend."""

    previous = sys.modules.get(name)
    sys.modules[name] = module  # type: ignore[assignment]
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _StubJiebaTokenizer:
    """Records exactly how JiebaSegmenter feeds it dictionary entries."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.added: list[tuple[str, object, object]] = []

    def load_userdict(self, path: str) -> None:
        self.loaded.append(str(path))

    def add_word(self, word: str, freq: object = None, tag: object = None) -> None:
        self.added.append((word, freq, tag))

    def tokenize(self, text: str, HMM: bool = True):  # noqa: N803 - jieba's name
        for index, char in enumerate(text):
            yield (char, index, index + 1)


def _flat(text: str) -> str:
    """Collapse whitespace so wrapped prose still matches a one-line claim."""

    return " ".join(text.split())


def _quantity(cell: str) -> float:
    """Parse ``4096``, ``100.0``, or ``16 MiB`` into a number."""

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s+(KiB|MiB|GiB))?", cell.strip())
    if match is None:
        raise AssertionError(f"Documented quantity is unparseable: {cell!r}")
    value = float(match.group(1))
    if match.group(2):
        value *= _UNITS[match.group(2)]
    return value


def test_guide_is_published_in_the_nav_and_cross_linked() -> None:
    nav = MKDOCS.read_text(encoding="utf-8")
    publication = PUBLICATION.read_text(encoding="utf-8")
    anonymization = ANONYMIZATION.read_text(encoding="utf-8")

    assert "Chinese Segmentation Operations: chinese-segmentation-operations.md" in nav
    assert "- chinese-segmentation-operations.md" in publication
    assert "(chinese-segmentation-operations.md)" in anonymization
    assert "(anonymization.md#chinese-word-segmentation)" in _guide()


def test_documented_backends_match_the_supported_set() -> None:
    guide = _guide()
    documented = {
        backend for backend in ("jieba", "pkuseg", "hanlp") if f"`{backend}`" in guide
    }

    assert documented == set(segmentation.SUPPORTED_CHINESE_SEGMENTATION_BACKENDS)


def test_documented_version_ranges_match_pyproject() -> None:
    with PYPROJECT.open("rb") as handle:
        project = _toml.load(handle)["project"]
    optional = project["optional-dependencies"]

    expected = {
        "jieba": next(
            requirement
            for requirement in project["dependencies"]
            if requirement.startswith("jieba")
        ),
        "pkuseg": optional["zh-pkuseg"][0],
        "hanlp": optional["zh-hanlp"][0],
    }

    rows = _table_rows(_guide(), "Backend", "Install", "License", "Declared range")
    for backend, requirement in expected.items():
        cells = rows[f"`{backend}`"]
        assert cells[-1] == f"`{requirement}`", backend

    # The install commands must name extras that actually exist.
    documented_extras = set(re.findall(r"openmed\[([^\]]+)\]", _guide()))
    assert documented_extras == {"zh", "zh-pkuseg", "zh-hanlp"}
    assert documented_extras <= set(optional)

    # The guide claims jieba is pinned twice; both pins must be real, and the
    # base pin must be the narrower one.
    zh_jieba = next(req for req in optional["zh"] if req.startswith("jieba"))
    assert zh_jieba != expected["jieba"]
    assert f"`{zh_jieba}`" in _guide()


def test_documented_licenses_match_the_import_error_metadata() -> None:
    """Each backend's documented license is the one its ImportError reports."""

    rows = _table_rows(_guide(), "Backend", "Install", "License", "Declared range")
    sources = {
        "jieba": ("jieba", None),
        "pkuseg": ("pkuseg", "zh-pkuseg"),
        "hanlp": ("hanlp", "zh-hanlp"),
    }
    licenses = {"jieba": "MIT", "pkuseg": "MIT", "hanlp": "Apache-2.0"}

    for backend, (module_name, extra) in sources.items():
        assert rows[f"`{backend}`"][1] == licenses[backend], backend
        with pytest.raises(ImportError) as excinfo:
            segmentation._import_optional_dependency(
                f"{module_name}_absent_for_this_test",
                extra=extra,
                license_name=licenses[backend],
            )
        assert f"Backend license: {licenses[backend]}." in str(excinfo.value)
        if extra is not None:
            assert f"openmed[{extra}]" in str(excinfo.value)


def test_documented_configuration_surface_matches_config() -> None:
    rows = _table_rows(_guide(), "Setting", "Environment variable", "Default")
    defaults = OpenMedConfig()

    expected = {
        "`chinese_segmentation_backend`": (
            config_module.CHINESE_SEGMENTATION_BACKEND_ENV_VAR,
            defaults.chinese_segmentation_backend,
        ),
        "`chinese_user_dict_path`": (
            config_module.CHINESE_USER_DICT_ENV_VAR,
            defaults.chinese_user_dict_path,
        ),
        "`chinese_pkuseg_domain`": (
            config_module.CHINESE_PKUSEG_DOMAIN_ENV_VAR,
            defaults.chinese_pkuseg_domain,
        ),
    }

    for setting, (env_var, default) in expected.items():
        documented_env, documented_default = rows[setting]
        assert documented_env == f"`{env_var}`", setting
        if default is None:
            assert documented_default == "unset", setting
        else:
            assert documented_default == f"`{default}`", setting


def test_documented_config_validation_matches_the_implementation() -> None:
    guide = _guide()
    assert "lower-cased" in guide

    assert (
        OpenMedConfig(
            chinese_segmentation_backend="  PKUSEG "
        ).chinese_segmentation_backend
        == "pkuseg"
    )
    with pytest.raises(ValueError):
        OpenMedConfig(chinese_segmentation_backend="spacy")
    with pytest.raises(ValueError):
        OpenMedConfig(chinese_pkuseg_domain="   ")


def test_documented_dictionary_limits_match_the_defaults() -> None:
    limits = tokenization.DEFAULT_DICTIONARY_LIMITS
    rows = _table_rows(_guide(), "Limit", "Value")

    expected = {
        "Max compressed source bytes (`.zip`)": limits.max_compressed_bytes,
        "Max decompressed bytes": limits.max_decompressed_bytes,
        "Max entries": limits.max_entries,
        "Max records": limits.max_records,
        "Max bytes per entry": limits.max_entry_bytes,
        "Max characters per term": limits.max_term_characters,
        "Max archive expansion ratio": limits.max_expansion_ratio,
    }

    for label, value in expected.items():
        assert _quantity(rows[label][0]) == float(value), label


def test_documented_validation_example_accepts_and_rejects_as_described(
    tmp_path: Path,
) -> None:
    """Run the guide's validation snippet, including its privacy claim."""

    from openmed.processing import DictionaryIngestionError, load_user_dictionary

    guide = _guide()
    dictionary = tmp_path / "zh_terms.txt"
    secret_term = "临床路径"
    dictionary.write_text(f"心脏超声 90000 nz\n{secret_term}\n", encoding="utf-8")

    entries = load_user_dictionary(str(dictionary))
    assert [(entry.term, entry.frequency, entry.pos) for entry in entries] == [
        ("心脏超声", 90000, "nz"),
        (secret_term, None, None),
    ]

    with pytest.raises(DictionaryIngestionError) as excinfo:
        load_user_dictionary(str(tmp_path / "missing.txt"))

    # The guide promises neither raw paths nor dictionary content leak out.
    message = str(excinfo.value)
    assert secret_term not in message
    assert str(tmp_path) not in message
    assert "never logged or included in raised exceptions" in guide


def test_documented_dictionary_errors_exist_and_share_a_base() -> None:
    documented = set(re.findall(r"`(Dictionary\w*Error)`", _guide()))

    assert "DictionaryIngestionError" in documented
    subclasses = documented - {"DictionaryIngestionError"}
    assert subclasses, "the guide must name the concrete rejection reasons"
    for name in documented:
        error = getattr(tokenization, name)
        assert issubclass(error, tokenization.DictionaryIngestionError), name


def test_documented_pkuseg_resolution_rule_is_executed(tmp_path: Path) -> None:
    """Execute the guide's pkuseg resolution table, row by row.

    The table is the normative statement of the rule, so inverting it — or
    inverting the prose that summarises it — must fail here. Presence checks
    would not catch that, so every row is run against the implementation and
    the observed outcome is compared to the documented one.
    """

    rows = _table_rows(
        _guide(), "`chinese_pkuseg_domain`", "Known model name", "Resolution"
    )
    assert rows, "the pkuseg resolution table disappeared"

    home = tmp_path / "pkuseg_home"
    home.mkdir()

    for raw_value, (known_cell, resolution_cell) in rows.items():
        value = raw_value.strip("`")
        known = {"yes": True, "no": False}[known_cell.strip().lower()]
        resolution = resolution_cell.strip("` ")

        # Build a pkuseg whose available_models agrees with the documented column.
        available = ("medicine", "default")
        assert (value in available) is known, (
            f"row {value!r} claims known={known}, but the fixture's "
            f"available_models says otherwise"
        )
        module = SimpleNamespace(
            config=SimpleNamespace(available_models=available, pkuseg_home=str(home))
        )

        if resolution == "verbatim":
            assert segmentation._local_pkuseg_model(module, value) == value, value
            continue

        # Documented as resolved under pkuseg_home.
        expected_suffix = resolution.removeprefix("pkuseg_home/")
        assert expected_suffix == value, resolution

        # Missing directory must raise rather than silently pass through.
        with pytest.raises(FileNotFoundError) as excinfo:
            segmentation._local_pkuseg_model(module, value)
        assert "does not download model files implicitly" in str(excinfo.value)

        # Present directory resolves to exactly the documented location.
        (home / value).mkdir()
        assert segmentation._local_pkuseg_model(module, value) == str(home / value)

    # The rule must actually discriminate, or the table proves nothing.
    outcomes = {cells[1].strip("` ") for cells in rows.values()}
    assert {"verbatim"} < outcomes, "table must contain both resolution outcomes"


def test_documented_failure_timing_table_is_executed(tmp_path: Path) -> None:
    """Dictionaries fail at construction; model assets fail on first segment()."""

    rows = _table_rows(_guide(), "Input", "Read at", "Failure surfaces as")
    timing = {key.strip("`"): cells[0].strip() for key, cells in rows.items()}

    assert timing["bundled dictionary"] == "construction"
    assert timing["chinese_user_dict_path"] == "construction"
    assert timing["pkuseg domain or path"] == "first `segment()`"
    assert timing["HanLP model path"] == "first `segment()`"

    module = SimpleNamespace(
        config=SimpleNamespace(available_models=(), pkuseg_home=str(tmp_path))
    )

    # A bad user dictionary really does fail during __init__.
    with pytest.raises(tokenization.DictionaryIngestionError):
        with _stub_module("pkuseg", module):
            segmentation.PkusegSegmenter(
                model_name="medicine",
                user_dict_path=str(tmp_path / "missing-dictionary.txt"),
            )

    # Constructing with a good dictionary succeeds even though the model path
    # is bogus, proving model resolution is deferred past __init__.
    with _stub_module("pkuseg", module):
        segmenter = segmentation.PkusegSegmenter(
            model_name=str(tmp_path / "no-such-model")
        )
    assert segmenter is not None

    # And construction does read the bundled dictionary from disk.
    assert segmentation.DEFAULT_MEDICAL_USER_DICTIONARY.is_file()
    assert "does **not** touch the filesystem" not in _guide()
    assert "**does** touch the filesystem" in _guide()


def test_documented_hanlp_provisioning_matches_the_implementation() -> None:
    guide = _guide()
    assert "`ValueError`" in guide
    assert "`FileNotFoundError`" in guide

    hanlp = SimpleNamespace(load=lambda path: lambda text: [text])

    without_model = segmentation.HanLPSegmenter.__new__(segmentation.HanLPSegmenter)
    without_model._hanlp = hanlp
    without_model._model = None
    without_model._model_source = None
    with pytest.raises(ValueError) as value_error:
        without_model._get_model()
    assert "does not download model weights implicitly" in str(value_error.value)

    missing_path = segmentation.HanLPSegmenter.__new__(segmentation.HanLPSegmenter)
    missing_path._hanlp = hanlp
    missing_path._model = None
    missing_path._model_source = str(ROOT / "does-not-exist-hanlp-model")
    with pytest.raises(FileNotFoundError):
        missing_path._get_model()


def test_documented_dictionary_precedence_rule_is_executed(tmp_path: Path) -> None:
    """Execute the guide's precedence table by observing each backend.

    Inverting the table, or the prose restating it, must fail here. So the
    behaviour is measured per backend and compared to the documented cells
    rather than checked for the presence of a word like "overrides".
    """

    rows = _table_rows(
        _guide(),
        "Backend",
        "Duplicate term in your file",
        "`frequency` and `POS` columns",
    )
    documented = {
        backend.strip("`"): (cells[0].strip(), cells[1].strip())
        for backend, cells in rows.items()
    }

    # A user dictionary that repeats a bundled term with a different frequency.
    bundled_term = "王芳"
    user_dict = tmp_path / "zh_terms.txt"
    user_dict.write_text(f"{bundled_term} 12345 nz\n", encoding="utf-8")
    assert bundled_term in segmentation.DEFAULT_MEDICAL_USER_DICTIONARY.read_text(
        encoding="utf-8"
    )

    observed: dict[str, tuple[str, str]] = {}

    # jieba: bundled file loaded first, then each entry applied with freq/tag.
    tokenizer = _StubJiebaTokenizer()
    jieba_stub = SimpleNamespace(Tokenizer=lambda: tokenizer)
    with _stub_module("jieba", jieba_stub):
        segmentation.JiebaSegmenter(user_dict_path=str(user_dict))
    assert tokenizer.loaded == [str(segmentation.DEFAULT_MEDICAL_USER_DICTIONARY)]
    applied = {word: (freq, tag) for word, freq, tag in tokenizer.added}
    jieba_overrides = applied.get(bundled_term) == (12345, "nz")
    jieba_uses_columns = any(
        freq is not None or tag is not None for _, freq, tag in tokenizer.added
    )
    observed["jieba"] = (
        "overrides" if jieba_overrides else "no effect",
        "used" if jieba_uses_columns else "discarded",
    )

    # pkuseg and HanLP: merged entries collapse to de-duplicated bare strings.
    pkuseg_stub = SimpleNamespace(
        config=SimpleNamespace(available_models=(), pkuseg_home=str(tmp_path))
    )
    with _stub_module("pkuseg", pkuseg_stub):
        pkuseg_segmenter = segmentation.PkusegSegmenter(
            model_name=str(tmp_path / "model"), user_dict_path=str(user_dict)
        )
    with _stub_module("hanlp", SimpleNamespace(load=lambda path: None)):
        hanlp_segmenter = segmentation.HanLPSegmenter(
            model=lambda text: [], user_dict_path=str(user_dict)
        )

    for name, segmenter in (
        ("pkuseg", pkuseg_segmenter),
        ("hanlp", hanlp_segmenter),
    ):
        terms = segmenter._user_terms
        assert all(isinstance(term, str) for term in terms), name
        observed[name] = (
            "no effect" if terms.count(bundled_term) == 1 else "overrides",
            "discarded",
        )

    assert observed == documented

    # The table must discriminate between the backends, or it proves nothing.
    assert len(set(documented.values())) > 1


def test_documented_boundary_f1_example_holds() -> None:
    """The guide's worked example must produce the scores it asserts."""

    text = "患者张伟因高血压入院"
    gold = [
        tokenization.SpanToken("患者", 0, 2),
        tokenization.SpanToken("张伟", 2, 4),
        tokenization.SpanToken("因", 4, 5),
        tokenization.SpanToken("高血压", 5, 8),
        tokenization.SpanToken("入院", 8, 10),
    ]
    predicted = [
        tokenization.SpanToken("患者", 0, 2),
        tokenization.SpanToken("张伟", 2, 4),
        tokenization.SpanToken("因高血压", 4, 8),
        tokenization.SpanToken("入院", 8, 10),
    ]

    # The example text and offsets must agree with each other.
    assert text in _guide()
    for token in gold:
        assert text[token.start : token.end] == token.text
    segmentation.validate_segmentation(text, gold)
    segmentation.validate_segmentation(text, predicted)

    assert segmentation.segmentation_boundary_f1(gold, gold) == 1.0
    assert segmentation.segmentation_boundary_f1(gold, predicted) < 1.0


def test_documented_regression_gate_matches_the_shipped_threshold() -> None:
    shipped = SEGMENTATION_TESTS.read_text(encoding="utf-8")
    threshold = re.search(r"scores\)\s*>=\s*(\d+\.\d+)", shipped)
    assert threshold is not None, "the shipped boundary-F1 gate moved"

    documented = re.search(r"mean boundary F1 of at least (\d+\.\d+)", _guide())
    assert documented is not None, "the guide must state the shipped F1 gate"
    assert float(documented.group(1)) == float(threshold.group(1))

    assert str(SEGMENTATION_TESTS.relative_to(ROOT)) in _guide()


CONFORMANCE_TESTS = Path(__file__).with_name("test_zh_segmentation_conformance.py")
CONFORMANCE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "processing" / "zh_segmentation_conformance.json"
)


def test_documented_skip_reasons_are_quoted_verbatim() -> None:
    """Every skip reason in the guide must appear in the shipped suite."""

    shipped = _flat(CONFORMANCE_TESTS.read_text(encoding="utf-8")).replace('" "', "")

    rows = _table_rows(_guide(), "Emitted when", "Reason")
    documented = {cells[0].strip("` ") for cells in rows.values()}
    assert documented, "the skip-reason table disappeared"

    # Every documented reason must exist verbatim in the shipped suite.
    for reason in documented:
        assert _flat(reason) in shipped, reason

    # The jieba row must be described as a broken install, not a missing extra,
    # because jieba is a base dependency and that reason cannot appear on a
    # correct installation.
    jieba_row = [
        condition for condition, cells in rows.items() if "jieba dependency" in cells[0]
    ]
    assert len(jieba_row) == 1
    assert "broken" in jieba_row[0]
    flat_guide = _flat(_guide())
    assert "the first row never appears" in flat_guide
    assert "exactly these four reasons" not in flat_guide


def test_documented_hanlp_env_var_and_pkuseg_asymmetry_are_real() -> None:
    guide = _guide()
    shipped = CONFORMANCE_TESTS.read_text(encoding="utf-8")

    shipped_name = re.search(r'HANLP_MODEL_ENV_VAR\s*=\s*"([A-Z_]+)"', shipped)
    assert shipped_name is not None
    env_var = shipped_name.group(1)

    # Bind the table CELL to the shipped constant, not merely its presence
    # somewhere on the page: a wrong name in the table must fail.
    opt_in = _table_rows(_guide(), "Backend", "Opt-in", "Why")
    assert opt_in["`hanlp`"][0].strip("` ") == env_var
    assert "pkuseg_home/medicine" in opt_in["`pkuseg`"][0]

    # Every environment variable the guide spells must be real. The one
    # deliberate counter-example is written as a glob and excluded here, but
    # its non-existence is asserted below.
    real = {
        env_var,
        config_module.CHINESE_SEGMENTATION_BACKEND_ENV_VAR,
        config_module.CHINESE_USER_DICT_ENV_VAR,
        config_module.CHINESE_PKUSEG_DOMAIN_ENV_VAR,
    }
    for candidate in set(re.findall(r"\bOPENMED_[A-Z_]+\*?", guide)):
        if candidate.endswith("*"):
            continue
        assert candidate in real, candidate

    # The documented asymmetry: no pkuseg model-path variable exists anywhere.
    assert "OPENMED_PKUSEG_" not in shipped
    assert not [
        name
        for name in vars(config_module)
        if name.endswith("_ENV_VAR") and "PKUSEG" in name and "DOMAIN" not in name
    ]

    # The guide claims there is deliberately no pkuseg counterpart.
    assert "OPENMED_PKUSEG_" not in shipped
    assert not [
        name
        for name in dir(config_module)
        if name.startswith("PKUSEG") or "PKUSEG_MODEL" in name
    ]
    assert "there\nis none" in guide or "there is none" in guide


def test_documented_integration_marker_is_declared_and_used() -> None:
    with PYPROJECT.open("rb") as handle:
        markers = _toml.load(handle)["tool"]["pytest"]["ini_options"]["markers"]

    assert any(marker.startswith("integration:") for marker in markers)
    assert "@pytest.mark.integration" in CONFORMANCE_TESTS.read_text(encoding="utf-8")
    assert "`integration`" in _guide()


def test_documented_conformance_corpus_shape_matches_the_fixture() -> None:
    import json

    document = json.loads(CONFORMANCE_FIXTURE.read_text(encoding="utf-8"))
    metadata = document["metadata"]
    guide = _guide()

    assert str(CONFORMANCE_FIXTURE.relative_to(ROOT)) in guide
    assert metadata["synthetic"] is True
    assert metadata["case_count"] == 200
    assert len(document["names"]) == 40
    assert len(document["conditions"]) == 5
    assert len(document["templates"]) == 4
    assert (
        len(document["names"]) * len(document["conditions"]) == metadata["case_count"]
    )

    flat = _flat(guide)
    for count, noun in (
        (40, "synthetic names"),
        (5, "conditions"),
        (4, "word templates"),
        (200, "cases"),
    ):
        assert f"{count} {noun}" in flat, noun

    assert metadata["sha256"].startswith("sha256:")
    assert "metadata.sha256" in guide


def test_documented_conformance_checks_match_the_declared_set() -> None:
    rows = _table_rows(_guide(), "Check", "Fails when the backend")
    documented = {row.strip("`") for row in rows}

    assert documented == set(segmentation.SEGMENTATION_CONFORMANCE_CHECKS)


def test_documented_harness_example_gates_on_ok_not_on_evidence() -> None:
    """Run the guide's harness snippet and its stated gating semantics."""

    guide = _guide()
    case = segmentation.SegmentationConformanceCase(
        text="患者王芳因心房颤动入院",
        gold_words=("患者", "王芳", "因", "心房颤动", "入院"),
        required_terms=("王芳",),
    )

    def _spans(words):
        tokens = []
        cursor = 0
        for word in words:
            tokens.append(tokenization.SpanToken(word, cursor, cursor + len(word)))
            cursor += len(word)
        return tokens

    class _Conforming:
        def segment(self, text):
            return _spans(("患者", "王芳", "因", "心房颤动", "入院"))

    class _SplitsRequiredTerm:
        def segment(self, text):
            return _spans(("患者", "王", "芳", "因", "心房颤动", "入院"))

    passing = segmentation.run_segmenter_conformance(
        _Conforming(), [case], backend="stub"
    )
    assert passing.ok is True
    assert passing.issues == ()

    failing = segmentation.run_segmenter_conformance(
        _SplitsRequiredTerm(), [case], backend="stub"
    )
    assert failing.ok is False
    assert failing.checks_triggered == frozenset({"dictionary"})
    assert failing.issues[0].check == "dictionary"
    assert failing.issues[0].detail

    # Evidence fields are reported but must not decide report.ok. Assert the
    # documented independence by construction, in both directions.
    evidence = passing.to_evidence()
    for field in ("boundary_f1", "dictionary_hit_rate", "chars_per_second"):
        assert field in evidence, field
        assert f"`{field}`" in guide, field

    report_type = type(passing)
    poor_metrics_no_defect = report_type(
        backend="stub",
        case_count=1,
        issues=(),
        boundary_f1=0.0,
        dictionary_hit_rate=0.0,
        chars_per_second=0.0,
    )
    perfect_metrics_one_defect = report_type(
        backend="stub",
        case_count=1,
        issues=failing.issues,
        boundary_f1=1.0,
        dictionary_hit_rate=1.0,
        chars_per_second=1e9,
    )
    assert poor_metrics_no_defect.ok is True
    assert perfect_metrics_one_defect.ok is False

    assert "not part of `report.ok`" in _flat(guide)
    assert "`report.ok` is the gate" in guide
    assert "do not gate" not in _flat(guide)


def test_documented_metric_gating_table_matches_the_shipped_suite() -> None:
    """Which metrics the suite gates, and which are hardware-dependent."""

    shipped = CONFORMANCE_TESTS.read_text(encoding="utf-8")
    rows = _table_rows(_guide(), "Metric", "Suite asserts", "Role", "Determinism")
    documented = {
        metric.strip("`"): tuple(cell.strip() for cell in cells)
        for metric, cells in rows.items()
    }

    for metric, (asserts_cell, role, determinism) in documented.items():
        # The suite asserts each metric in more than one layer: the in-repo
        # reference segmenter is held to an exact value, installed backends to
        # the floor. Collect every assertion and require the documented cell to
        # be one the suite actually makes.
        actual = {
            found.strip().rstrip(")").replace("pytest.approx(", "").strip()
            for found in re.findall(rf"assert report\.{metric}\s*([^\n,\[]+)", shipped)
        }
        assert actual, metric
        assert asserts_cell.strip("` ") in actual, (metric, sorted(actual))

        # Only throughput may be described as hardware-dependent, and it is the
        # only one that must not be presented as a quality gate.
        if metric == "chars_per_second":
            assert determinism == "hardware-dependent"
            assert role == "liveness only"
        else:
            assert determinism == "deterministic", metric
            assert role == "quality gate", metric

    assert "SEGMENTATION_BOUNDARY_F1_FLOOR" in documented["boundary_f1"][0]
    floor = segmentation.SEGMENTATION_BOUNDARY_F1_FLOOR
    assert f"is **{floor:.2f}**" in _guide()
    assert re.search(
        r"assert report\.boundary_f1 >= SEGMENTATION_BOUNDARY_F1_FLOOR", shipped
    )

    # The floor is exported from the package alongside the other conformance
    # names, so the guide must show the package path.
    import openmed.processing as processing

    assert processing.SEGMENTATION_BOUNDARY_F1_FLOOR == floor
    assert "SEGMENTATION_BOUNDARY_F1_FLOOR" in processing.__all__
    assert "from openmed.processing import SEGMENTATION_BOUNDARY_F1_FLOOR" in _guide()

    # All seven conformance names must stay uniformly exported; the floor was
    # once the only one missing, and that asymmetry should not return.
    for name in (
        "SEGMENTATION_BOUNDARY_F1_FLOOR",
        "SEGMENTATION_CONFORMANCE_CHECKS",
        "SegmentationAlignmentError",
        "SegmentationConformanceCase",
        "SegmentationConformanceIssue",
        "SegmentationConformanceReport",
        "run_segmenter_conformance",
    ):
        assert name in processing.__all__, name


def test_documented_two_threshold_split_matches_the_suite() -> None:
    """The reference self-check and the conformance floor are distinct bars."""

    shipped = CONFORMANCE_TESTS.read_text(encoding="utf-8")
    rows = _table_rows(_guide(), "Subject", "Threshold", "Question it answers")

    reference = next(cells for subject, cells in rows.items() if "reference" in subject)
    installed = next(cells for subject, cells in rows.items() if "installed" in subject)

    floor = segmentation.SEGMENTATION_BOUNDARY_F1_FLOOR
    assert reference[0].strip("` ") == "exactly `1.0`".strip("` ")
    assert installed[0].strip("` ") == f">= {floor:.2f}"

    # Both thresholds must really exist in the suite.
    assert re.search(r"assert report\.boundary_f1 == pytest\.approx\(1\.0\)", shipped)
    assert re.search(
        r"assert report\.boundary_f1 >= SEGMENTATION_BOUNDARY_F1_FLOOR", shipped
    )

    # Only the installed-backend row is a bar a user's backend must clear.
    flat = _flat(_guide())
    assert "Only the second row is a bar your backend must clear." in flat
    assert "judges nothing about the number" in flat


def test_documented_floor_scope_limit_is_pinned_by_measurement() -> None:
    """The floor's documented blind spot must sit correctly around the floor."""

    guide = _guide()

    assert (
        "validates the span protocol and dictionary-term survival, plus a 0.90 "
        "floor that rejects character-level output; it does not validate "
        "general segmentation quality." in _flat(guide)
    )

    rows = _table_rows(guide, "Stub", "`boundary_f1`", "Outcome")
    documented = {stub: (float(cells[0]), cells[1]) for stub, cells in rows.items()}
    floor = segmentation.SEGMENTATION_BOUNDARY_F1_FLOOR

    character_row = next(
        value for stub, value in documented.items() if "one character" in stub
    )
    blob_row = next(value for stub, value in documented.items() if "blob" in stub)

    assert character_row[0] < floor
    assert "rejected" in character_row[1]
    assert blob_row[0] >= floor
    assert "passes" in blob_row[1]

    assert "not as evidence of good segmentation" in _flat(guide)


def test_documented_unicode_folding_caveat_still_holds() -> None:
    """The caveat says no folding exists upstream of segmentation today.

    If that ever changes, this fails so the caveat can be rewritten rather
    than quietly becoming false.
    """

    from openmed.utils import gateway

    flat = _flat(_guide())
    assert "applies no Unicode normalization" in flat
    assert "not as a present-day defect" in flat

    # The gateway must remain a size and encoding guardrail only.
    gateway_source = Path(gateway.__file__).read_text(encoding="utf-8")
    assert "unicodedata" not in gateway_source
    for form in ("NFC", "NFKC", "NFD", "NFKD"):
        assert form not in gateway_source, form

    # Full-width text must survive the gateway byte-for-byte.
    probe = "患者ＣＴ检查"
    assert gateway.normalize_text(probe) == probe

    # And segmentation must not route through the gateway at all.
    segmentation_source = Path(segmentation.__file__).read_text(encoding="utf-8")
    assert "normalize_text" not in segmentation_source
    assert "utils.gateway" not in segmentation_source


def test_documented_protocol_edge_cases_match_the_harness() -> None:
    """None is a protocol defect; a generator is accepted and scored."""

    case = segmentation.SegmentationConformanceCase(
        text="患者王芳因心房颤动入院",
        gold_words=("患者", "王芳", "因", "心房颤动", "入院"),
        required_terms=(),
    )

    class _ReturnsNone:
        def segment(self, text):
            return None

    class _EmptyGenerator:
        def segment(self, text):
            return (token for token in ())

    none_report = segmentation.run_segmenter_conformance(
        _ReturnsNone(), [case], backend="stub"
    )
    assert none_report.checks_triggered == frozenset({"protocol"})

    generator_report = segmentation.run_segmenter_conformance(
        _EmptyGenerator(), [case], backend="stub"
    )
    assert generator_report.checks_triggered == frozenset({"coverage"})

    flat = _flat(_guide())
    assert "returns `None` is reported as a `protocol` defect" in flat
    assert "generator is accepted" in flat
    # The superseded claim that these crash the harness must be gone.
    assert "raises `TypeError` out of the" not in flat


def test_documented_symbols_are_part_of_the_public_processing_api() -> None:
    import openmed.processing as processing

    referenced = {
        "create_chinese_segmenter",
        "load_user_dictionary",
        "segmentation_boundary_f1",
        "validate_segmentation",
        "DictionaryIngestionError",
    }
    guide = _guide()
    for name in referenced:
        assert name in guide, name
        assert name in processing.__all__, name

    # SpanToken is not re-exported from openmed.processing, so the guide must
    # import it from the module that does export it.
    assert "SpanToken" not in processing.__all__
    assert "from openmed.processing.tokenization import SpanToken" in guide
