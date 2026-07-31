"""Acceptance checks for the Chinese segmentation operations guide."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs" / "chinese-segmentation-operations.md"
NOTE = "患者王芳因心房颤动入院"
GOLD_CUTS = (
    ["患者", "王芳", "因", "心房颤动", "入院"],
    ["患者", "李雷", "因", "高血压", "入院"],
    ["患者", "张伟", "诊断", "为", "糖尿病"],
)


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _python_snippets(markdown: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL)


def _section(title: str) -> str:
    sections = [
        section
        for section in re.split(r"(?m)^## ", _guide())
        if section.startswith(title)
    ]
    assert len(sections) == 1, f"expected exactly one '{title}' section"
    return sections[0]


def _run_snippet(snippet: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(snippet, "chinese-segmentation-operations", "exec"), namespace)
    return namespace


def _fake_module(monkeypatch, module_name: str, module: Any) -> None:
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module if name == module_name else real_import(name),
    )


def test_guide_is_published_in_the_nav_and_linked_from_anonymization() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    publication = (ROOT / "docs" / "brand" / "system" / "publication.yml").read_text(
        encoding="utf-8"
    )
    anonymization = (ROOT / "docs" / "anonymization.md").read_text(encoding="utf-8")

    assert "Chinese Segmentation Operations: chinese-segmentation-operations.md" in nav
    assert "    - chinese-segmentation-operations.md\n" in publication
    assert (
        "[Chinese segmentation model operations]"
        "(chinese-segmentation-operations.md)" in anonymization
    )


def test_guide_states_supported_versions_and_deployment_requirements() -> None:
    guide = _guide()

    for required in (
        "openmed[zh-pkuseg]",
        "openmed[zh-hanlp]",
        "pkuseg>=0.0.25,<0.1",
        "hanlp>=2.1,<3",
        "PKUSEG_HOME",
        "HANLP_HOME",
        "`~/.pkuseg`",
        "`~/.hanlp`",
        "meta.json",
        "OPENMED_CHINESE_PKUSEG_DOMAIN",
        "model_hash",
    ):
        assert required in guide


def test_guide_documents_dictionary_governance_and_regression_checks() -> None:
    guide = _guide()

    for required in (
        "provenance",
        "100,000 entries",
        "load_user_dictionary",
        "DictionaryIngestionError",
        "segmentation_boundary_f1",
        "validate_segmentation",
        "0.90",
        "## Upgrade and rollback",
    ):
        assert required in guide


def test_guide_python_examples_compile() -> None:
    snippets = _python_snippets(_guide())

    assert snippets
    for index, snippet in enumerate(snippets, start=1):
        compile(snippet, f"chinese-segmentation-operations-python-{index}", "exec")


def test_documented_pkuseg_example_uses_a_locally_provisioned_domain_model(
    monkeypatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "pkuseg" / "medicine-v0.0.16"
    model_dir.mkdir(parents=True)
    dictionary = tmp_path / "institution.txt"
    dictionary.write_text("心脏超声 90000 nz\n", encoding="utf-8")
    observed: dict[str, Any] = {}

    class LocalDomainModel:
        def cut(self, text: str) -> list[str]:
            assert text == NOTE
            return ["患者", "王芳", "因", "心房颤动", "入院"]

    def build_pkuseg(**kwargs: Any) -> LocalDomainModel:
        observed.update(kwargs)
        return LocalDomainModel()

    _fake_module(
        monkeypatch,
        "pkuseg",
        SimpleNamespace(
            pkuseg=build_pkuseg,
            config=SimpleNamespace(
                available_models=["default", "medicine", "tourism", "web", "news"],
                pkuseg_home=str(tmp_path / "pkuseg-home"),
            ),
        ),
    )
    monkeypatch.setenv("OPENMED_CHINESE_SEGMENTATION_BACKEND", "pkuseg")
    monkeypatch.setenv("OPENMED_CHINESE_PKUSEG_DOMAIN", str(model_dir))
    monkeypatch.setenv("OPENMED_CHINESE_USER_DICT", str(dictionary))

    namespace = _run_snippet(
        _python_snippets(_section("Provision a pkuseg domain model"))[0]
    )
    tokens = namespace["tokens"]

    assert observed["model_name"] == str(model_dir)
    assert "心脏超声" in observed["user_dict"]
    assert [token.text for token in tokens] == [
        "患者",
        "王芳",
        "因",
        "心房颤动",
        "入院",
    ]
    assert all(token.text == NOTE[token.start : token.end] for token in tokens)


def test_documented_hanlp_example_loads_a_locally_provisioned_tokenizer(
    monkeypatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "hanlp" / "fine_electra_small_zh"
    model_dir.mkdir(parents=True)
    (model_dir / "meta.json").write_text("{}", encoding="utf-8")
    loaded: list[str] = []

    def load(path: str):
        loaded.append(path)
        return lambda text: {
            "tok/fine": [["患者", "王", "芳", "因", "心房", "颤动", "入院"]]
        }

    _fake_module(monkeypatch, "hanlp", SimpleNamespace(load=load))
    monkeypatch.setenv("OPENMED_CHINESE_SEGMENTATION_BACKEND", "hanlp")
    monkeypatch.setenv("OPENMED_HANLP_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("OPENMED_CHINESE_USER_DICT", raising=False)

    namespace = _run_snippet(
        _python_snippets(_section("Provision a HanLP tokenizer"))[0]
    )
    tokens = namespace["tokens"]

    assert loaded == [str(model_dir)]
    assert [token.text for token in tokens] == [
        "患者",
        "王芳",
        "因",
        "心房颤动",
        "入院",
    ]
    assert all(token.text == NOTE[token.start : token.end] for token in tokens)


def test_documented_regression_check_scores_a_local_gold_set(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    gold_set = tmp_path / "zh_gold.jsonl"
    gold_set.write_text(
        "".join(
            f"{json.dumps({'words': words}, ensure_ascii=False)}\n"
            for words in GOLD_CUTS
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENMED_CHINESE_SEGMENTATION_BACKEND", "jieba")
    monkeypatch.setenv("OPENMED_ZH_GOLD_SET", str(gold_set))
    monkeypatch.delenv("OPENMED_CHINESE_USER_DICT", raising=False)
    monkeypatch.delenv("OPENMED_HANLP_MODEL_DIR", raising=False)

    namespace = _run_snippet(
        _python_snippets(_section("Run a segmentation regression check"))[0]
    )

    assert namespace["mean_f1"] == 1.0
    assert "mean boundary F1" in capsys.readouterr().out
