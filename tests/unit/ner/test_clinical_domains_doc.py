from pathlib import Path

from openmed.ner.labels import (
    generate_clinical_domains_markdown,
    load_default_label_map,
)


def test_doc_drift_clinical_domains_markdown():
    """Ensure there is no documentation drift in the clinical domains markdown file."""
    doc_path = Path("docs/clinical-domains.md")

    assert doc_path.exists(), "docs/clinical-domains.md does not exist."

    expected_content = doc_path.read_text(encoding="utf-8")
    actual_content = generate_clinical_domains_markdown()

    assert actual_content == expected_content, (
        "docs/clinical-domains.md is out of date. "
        "Run 'python openmed/ner/labels.py' to update it."
    )


def test_clinical_domains_markdown_covers_label_map_metadata():
    markdown = generate_clinical_domains_markdown()

    assert "SDOH cue normalization" not in markdown
    assert "not clinical guidance" in markdown
    for domain, labels in load_default_label_map().items():
        heading = domain.replace("_", " ").title()
        assert f"## {heading}" in markdown
        for label in labels:
            assert f"| {label} |" in markdown

    for fixture_path in (
        "tests/fixtures/clinical/anesthesia.jsonl",
        "tests/fixtures/clinical/endocrinology.jsonl",
        "tests/fixtures/clinical/gastroenterology.jsonl",
        "tests/fixtures/clinical/genomic_variant.jsonl",
        "tests/fixtures/clinical/immunization.jsonl",
        "tests/fixtures/clinical/nutrition_diet.jsonl",
        "tests/fixtures/clinical/pediatrics_growth.jsonl",
    ):
        assert Path(fixture_path).exists()
        assert fixture_path in markdown

    assert "Not shipped" in markdown
    assert "OM-138 FHIR Immunization exporter" in markdown
    assert "VaccineLot to lotNumber" in markdown
