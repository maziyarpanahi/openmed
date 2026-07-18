"""Synthetic India onboarding walkthrough for policy-aware de-identification.

The Aadhaar-format number, ABHA-format number, name, and clinical note in this
example are fabricated test data. They do not identify a real person or account.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from openmed import deidentify
from openmed.core.anonymizer.providers import validate_abha_number
from openmed.core.pii_i18n import validate_aadhaar
from openmed.core.policy import list_policies, load_policy

POLICY_NAME = "india_dpdp_act"
HINDI_MODEL_ID = "OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1"

SYNTHETIC_PERSON = "Asha Verma"
SYNTHETIC_AADHAAR = "2467 7832 5484"
SYNTHETIC_ABHA = "91-0000-0000-0000"
SYNTHETIC_HINGLISH_NOTE = (
    "Synthetic Hinglish note: Patient Asha Verma ka follow-up aaj hai. "
    "Aadhaar 2467 7832 5484 aur ABHA 91-0000-0000-0000 record mein hain. "
    "Patient ko do din se halka bukhar hai."
)

SYNTHETIC_NAME_RECOGNIZER = {
    "case_sensitive": False,
    "deny": {
        "terms": [
            {"term": SYNTHETIC_PERSON, "label": "PERSON"},
        ],
    },
}

Deidentifier = Callable[..., Any]


def validate_synthetic_inputs() -> None:
    """Assert that the fabricated identifiers have the documented shapes."""

    assert validate_aadhaar(SYNTHETIC_AADHAAR)
    assert re.fullmatch(r"\d{2}-\d{4}-\d{4}-\d{4}", SYNTHETIC_ABHA)
    assert validate_abha_number(SYNTHETIC_ABHA)


def assert_synthetic_pii_is_deidentified(deidentified_text: str) -> None:
    """Fail closed if any fabricated direct identifier remains in output."""

    assert SYNTHETIC_AADHAAR not in deidentified_text, (
        "Synthetic Aadhaar-format identifier was not de-identified"
    )
    assert SYNTHETIC_ABHA not in deidentified_text, (
        "Synthetic ABHA-format identifier was not de-identified"
    )
    assert SYNTHETIC_PERSON not in deidentified_text, (
        "Synthetic person name was not de-identified"
    )


def run(*, deidentifier: Deidentifier = deidentify) -> Any:
    """De-identify the synthetic Hinglish note with a shipped policy."""

    validate_synthetic_inputs()
    assert POLICY_NAME in list_policies()

    policy = load_policy(POLICY_NAME)
    assert policy.default_action == "replace"
    assert policy.action_for("PERSON") == "replace"
    assert policy.action_for("ID_NUM") == "replace"
    assert policy.safety_sweep_mandatory
    assert policy.keep_mapping is False
    assert policy.reversible_id is False

    result = deidentifier(
        SYNTHETIC_HINGLISH_NOTE,
        method="replace",
        model_name=HINDI_MODEL_ID,
        lang="hi",
        locale="en_IN",
        policy=POLICY_NAME,
        use_safety_sweep=True,
        custom_recognizer=SYNTHETIC_NAME_RECOGNIZER,
        consistent=True,
        seed=707,
    )
    assert_synthetic_pii_is_deidentified(result.deidentified_text)
    return result


if __name__ == "__main__":
    deidentified = run()
    print("Synthetic de-identified output:")
    print(deidentified.deidentified_text)
