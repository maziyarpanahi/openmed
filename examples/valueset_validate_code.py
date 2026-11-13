"""Check codes against a local ValueSet, offline, with a synthetic vocabulary.

``validate_code()`` and ``validate_codeable_concept()`` evaluate FHIR ValueSet
membership from a ValueSet document you provide. They never contact a
terminology server. The system and codes below are invented for this example;
no real vocabulary content is used.
"""

from __future__ import annotations

import json
from typing import Any

from openmed.clinical.exporters.valueset import (
    validate_code,
    validate_codeable_concept,
)

SYSTEM = "urn:openmed:example:visit-priority"

# Three codes are included, and one of them is then excluded again.
VALUESET: dict[str, Any] = {
    "resourceType": "ValueSet",
    "url": "urn:openmed:example:valueset:visit-priority",
    "status": "draft",
    "compose": {
        "include": [
            {
                "system": SYSTEM,
                "concept": [
                    {"code": "routine"},
                    {"code": "soon"},
                    {"code": "immediate"},
                ],
            }
        ],
        "exclude": [{"system": SYSTEM, "concept": [{"code": "immediate"}]}],
    },
}

# One coding is a member of the ValueSet and one is not.
CONCEPT: dict[str, Any] = {
    "text": "Visit priority",
    "coding": [
        {"system": SYSTEM, "code": "soon"},
        {"system": SYSTEM, "code": "unlisted"},
    ],
}


def main() -> list[dict[str, Any]]:
    """Validate a member, a non-member and an excluded code, then each policy.

    Prints one JSON line per case and returns the same records.
    """

    records: list[dict[str, Any]] = []

    for case, code in (
        ("member", "routine"),
        ("non_member", "unlisted"),
        ("excluded", "immediate"),
    ):
        result = validate_code(SYSTEM, code, VALUESET)
        records.append(
            {
                "case": case,
                "code": code,
                "valid": result.valid,
                "message": result.message,
            }
        )

    # The policy decides what happens to a coding that is not a member:
    #   annotate  keeps it and adds a validation extension (the default)
    #   drop      removes it
    #   downgrade keeps it and reports the finding as a warning, not an error
    for policy in ("annotate", "drop", "downgrade"):
        validated = validate_codeable_concept(CONCEPT, VALUESET, policy=policy)
        records.append(
            {
                "case": f"codeable_concept_{policy}",
                "valid": validated.valid,
                "codes": [coding["code"] for coding in validated["coding"]],
                "annotated_codes": [
                    coding["code"]
                    for coding in validated["coding"]
                    if "extension" in coding
                ],
                "severities": [issue["severity"] for issue in validated.issues],
            }
        )

    for record in records:
        print(json.dumps(record, sort_keys=True))
    return records


if __name__ == "__main__":
    main()
