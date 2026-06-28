from pathlib import Path

from openmed.ner.labels import generate_clinical_domains_markdown


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
