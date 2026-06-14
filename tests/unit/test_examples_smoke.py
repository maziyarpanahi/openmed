from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "clinical_ner_families.py"


def test_clinical_ner_families_example_is_valid_python() -> None:
    ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_clinical_ner_families_example_runs_with_mocked_analyzer(
    capsys, monkeypatch
) -> None:
    from examples import clinical_ner_families

    calls: list[tuple[str, str]] = []

    def fake_analyze_text(text: str, *, model_name: str, **kwargs):
        calls.append((text, model_name))
        return SimpleNamespace(
            entities=[
                SimpleNamespace(label="DISEASE", text="leukemia", confidence=0.91),
                SimpleNamespace(label="DRUG", text="imatinib", confidence=0.89),
            ]
        )

    monkeypatch.setattr(clinical_ner_families, "analyze_text", fake_analyze_text)

    clinical_ner_families.main()

    output = capsys.readouterr().out
    assert "Disease" in output
    assert "Pharmaceutical" in output
    assert "Oncology" in output
    assert "leukemia" in output
    assert len(calls) == 3
