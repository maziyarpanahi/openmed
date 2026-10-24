"""Offline checks for the decision-model guide and the MedDecider NLI recipe."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "decision-models.md"
LOCAL_BACKENDS = ROOT / "docs" / "clinical" / "local-backends.md"
NOT_SUPPORTED = "Not supported by default (license)"


def _python_examples(markdown: str) -> list[tuple[bool, str]]:
    examples = re.findall(r"(?ms)^```python\n(.*?)^```\s*$", markdown)
    if not examples:
        raise ValueError("decision-model guide has no Python examples")
    classified: list[tuple[bool, str]] = []
    for source in examples:
        marker = source.splitlines()[0]
        if not marker.startswith(("# Runnable:", "# Fragment:")):
            raise ValueError("decision-model example needs a Runnable or Fragment")
        compile(source, "decision_models_example.py", "exec")
        classified.append((marker.startswith("# Runnable:"), source))
    return classified


def _section(markdown: str, heading: str) -> str:
    assert heading in markdown, f"missing section: {heading.strip()}"
    return markdown.split(heading, 1)[1].split("\n## ", 1)[0]


def _support_rows(markdown: str) -> list[list[str]]:
    section = _section(markdown, "## Support matrix\n")
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5:
            rows.append(cells)
    return rows


def _shell_command(markdown: str, program: str) -> list[str]:
    blocks = re.findall(r"(?ms)^```bash\n(.*?)^```\s*$", markdown)
    for block in blocks:
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            if line.strip().startswith(program):
                return shlex.split(line)
    raise AssertionError(f"no {program!r} command in the recipe")


def test_guide_examples_are_classified_and_compile() -> None:
    examples = _python_examples(GUIDE.read_text(encoding="utf-8"))
    assert any(runnable for runnable, _ in examples)
    assert any(not runnable for runnable, _ in examples)


@pytest.mark.parametrize(
    "markdown",
    ["No examples.", "```python\nprint('unclassified')\n```\n"],
)
def test_guide_example_check_requires_a_classification(markdown: str) -> None:
    with pytest.raises(ValueError):
        _python_examples(markdown)


def test_guide_runnable_examples_run_offline_without_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from openmed.clinical.decision_support import (
        CLINICAL_DECISION_SUPPORT_DISCLAIMER,
        GuardedSuggestion,
    )
    from openmed.core.models import ModelLoader
    from openmed.core.offline import network_blocked_if_offline

    def reject_model_load(*args: Any, **kwargs: Any) -> None:
        pytest.fail("decision-model examples must not load a model")

    monkeypatch.setattr(ModelLoader, "load_model", reject_model_load)
    monkeypatch.setattr(
        ModelLoader, "load_local_sequence_classifier", reject_model_load
    )
    namespaces = []
    with network_blocked_if_offline(local_only=True):
        for runnable, source in _python_examples(GUIDE.read_text(encoding="utf-8")):
            if runnable:
                namespace: dict[str, Any] = {"__name__": "__decision_docs__"}
                exec(compile(source, "decision_models_example.py", "exec"), namespace)
                namespaces.append(namespace)

    assert len(namespaces) == 1
    result = namespaces[0]["result"]
    suggestion = namespaces[0]["suggestion"]
    assert result.state.value == "success"
    assert result.autonomous_action is False
    assert result.review["required"] is True
    assert isinstance(suggestion, GuardedSuggestion)
    assert suggestion.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert suggestion.autonomous_decision is False
    serialized = json.dumps(suggestion.provenance)
    assert namespaces[0]["note"] not in serialized
    assert capsys.readouterr().out == ""


def test_support_matrix_states_the_check_date_and_license_rule() -> None:
    from openmed.structured.decision import PERMISSIVE_LICENSES

    markdown = GUIDE.read_text(encoding="utf-8")
    assert "Checked against each model repository on 2026-10-24." in _section(
        markdown, "## Support matrix\n"
    )
    rows = _support_rows(markdown)
    models = {row[0].strip("`") for row in rows}
    assert {
        "LiquidAI/d1-omni-600M",
        "LiquidAI/d1-3B",
        "lion-ai/MedDecider-4B-NLI",
        "knowledgator/gliclass-v4-edge",
        "fastino/gliner2.5-small-v1",
    } <= models
    assert len(models) == len(rows)
    for model, size, license_id, inputs, path in rows:
        assert size and inputs, f"incomplete row: {model}"
        if license_id in PERMISSIVE_LICENSES:
            assert path != NOT_SUPPORTED, (
                f"permissive model marked unsupported: {model}"
            )
        else:
            assert path == NOT_SUPPORTED, f"non-permissive model offered: {model}"


def test_support_matrix_check_rejects_an_offered_non_permissive_model() -> None:
    from openmed.structured.decision import PERMISSIVE_LICENSES

    markdown = GUIDE.read_text(encoding="utf-8").replace(
        f"| text, images | {NOT_SUPPORTED} |",
        "| text, images | Local server through the bridge (text only) |",
        1,
    )
    offered = [
        row
        for row in _support_rows(markdown)
        if row[2] not in PERMISSIVE_LICENSES and row[4] != NOT_SUPPORTED
    ]
    assert [row[0] for row in offered] == ["`LiquidAI/d1-3B`"]


def test_meddecider_recipe_uses_the_existing_torch_qualification_command() -> None:
    from openmed.cli.main import build_parser
    from openmed.clinical.nli_labels import normalize_nli_label

    markdown = _section(
        LOCAL_BACKENDS.read_text(encoding="utf-8"), "### Recipe: MedDecider-4B-NLI\n"
    )
    assert "lion-ai/MedDecider-4B-NLI" in markdown
    assert re.search(r"--revision [0-9a-f]{40}\b", markdown)
    command = _shell_command(markdown, "openmed nli-qualify")
    args = build_parser().parse_args(command[1:])
    assert args.runtime == "torch"
    assert args.json_output is True

    mapping_json = re.search(r"echo '(\{.*?\})' > ", markdown)
    assert mapping_json is not None
    mapping = json.loads(mapping_json.group(1))
    labels = {key: normalize_nli_label(value).value for key, value in mapping.items()}
    assert labels == {"0": "entailment", "1": "neutral", "2": "contradiction"}
    assert "```python" not in markdown
