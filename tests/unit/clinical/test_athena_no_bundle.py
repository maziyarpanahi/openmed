from __future__ import annotations

import os
from pathlib import Path

import openmed


def test_openmed_package_contains_no_athena_or_restricted_vocabulary_files() -> None:
    package_root = Path(openmed.__file__).parent
    forbidden_names = {
        "CONCEPT.csv",
        "CONCEPT_RELATIONSHIP.csv",
        "CONCEPT_SYNONYM.csv",
        "CONCEPT_ANCESTOR.csv",
        "VOCABULARY.csv",
        "DOMAIN.csv",
        "CONCEPT_CLASS.csv",
        "DRUG_STRENGTH.csv",
        "CPT4.csv",
    }

    found = []
    for root, _dirs, files in os.walk(package_root):
        found.extend(
            os.path.join(root, file_name)
            for file_name in files
            if file_name in forbidden_names
        )

    assert found == []
