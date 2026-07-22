# OpenMed onboarding for users in India

[हिन्दी संस्करण](onboarding-india.hi.md)

This guide shows an India-oriented, local-first de-identification setup: start
with a policy that OpenMed ships today, de-identify synthetic Aadhaar- and
ABHA-format identifiers, process a synthetic code-mixed Hinglish note, and
select a compact Hindi checkpoint for a CPU-only machine.

For a complete synthetic Hindi and code-mixed Hinglish walkthrough, run the
[end-to-end example](https://github.com/maziyarpanahi/openmed/blob/master/examples/deid_hindi_hinglish_note.py)
or follow the
[Chinese and Hindi notebook tour](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb).

!!! important "Synthetic walkthrough only"
    Every name and identifier on this page and in
    `examples/onboarding_india_dpdp.py` is fabricated test data. The values are
    format examples only; they are not issued, verified, or associated with a
    real person. Never paste production personal or health data into a tutorial.

## Install the local runtime

Create a virtual environment, then install OpenMed with the Hugging Face model
dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "openmed[hf]"
```

Model download needs a network connection once. After the checkpoint is in the
local Hugging Face cache, OpenMed inference runs on the machine and does not
send the clinical note to a hosted API.

## DPDP-oriented policy quickstart

The Digital Personal Data Protection (DPDP) framework is broader than text
masking. Purpose and lawful processing, notices and consent where applicable,
retention, access controls, incident response, contracts, and human governance
remain organizational responsibilities. Consult the official
[DPDP Act, 2023](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
and [DPDP Rules, 2025](https://www.meity.gov.in/documents/act-and-policies/digital-personal-data-protection-rules-2025-gDOxUjMtQWa?pageTitle=Digital-Personal-Data-Protection-Rules-2025686cadad39.pdf),
and obtain appropriate legal advice. A software policy profile is one technical
control, not a claim of DPDP compliance.

OpenMed ships the assist-only `india_dpdp_act` profile. It replaces direct
identifiers, masks quasi-identifiers, keeps clinical concepts, requires the
safety sweep, and disables both reversible IDs and retained mappings:

```python
from openmed.core.policy import list_policies, load_policy

policy_name = "india_dpdp_act"
assert policy_name in list_policies()

policy = load_policy(policy_name)
assert policy.default_action == "replace"
assert policy.action_for("PERSON") == "replace"
assert policy.action_for("ID_NUM") == "replace"
assert policy.safety_sweep_mandatory
assert policy.keep_mapping is False
assert policy.reversible_id is False

print(policy.name, policy.default_action)
```

This code loads the shipped
`openmed/core/policies/india_dpdp_act.json` profile. Its metadata records the
official sources and makes clear that the profile is assist-only, not legal
advice or an autonomous compliance determination.

## De-identify synthetic Aadhaar and ABHA identifiers

[UIDAI describes Aadhaar](https://www.uidai.gov.in/en/my-aadhaar/about-your-aadhaar.html)
as a 12-digit number, while the
[Ayushman Bharat Digital Mission describes ABHA](https://abdm.gov.in/FAQ) as a
14-digit number. The example uses these deliberately fabricated, display-form
values:

- Synthetic Aadhaar-format value: `2467 7832 5484` (also passes OpenMed's
  Verhoeff checksum validator)
- Synthetic ABHA-format value: `91-0000-0000-0000`

The `india_dpdp_act` profile automatically enables OpenMed's ABDM recognizer,
which validates these Aadhaar and ABHA display forms and normalizes them to the
canonical `ID_NUM` label. The small custom recognizer below only makes the
fabricated person's name deterministic for this offline tutorial. The policy
then controls the replacement action:

```python
from openmed import deidentify

synthetic_note = (
    "Synthetic Hinglish note: Patient Asha Verma ka follow-up aaj hai. "
    "Aadhaar 2467 7832 5484 aur ABHA 91-0000-0000-0000 record mein hain. "
    "Patient ko do din se halka bukhar hai."
)

india_recognizer = {
    "case_sensitive": False,
    "deny": {
        "terms": [{"term": "Asha Verma", "label": "PERSON"}],
    },
}

result = deidentify(
    synthetic_note,
    method="replace",
    model_name="OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1",
    lang="hi",
    locale="en_IN",
    policy="india_dpdp_act",
    use_safety_sweep=True,
    custom_recognizer=india_recognizer,
    consistent=True,
    seed=707,
)

assert "2467 7832 5484" not in result.deidentified_text
assert "91-0000-0000-0000" not in result.deidentified_text
assert "Asha Verma" not in result.deidentified_text
print(result.deidentified_text)
```

The Hindi language hint selects Hindi-aware normalization and patterns. The
compact Hindi model handles the code-mixed note, while the explicit local rules
make the tutorial deterministic for the synthetic name and identifier formats.
For a real deployment, derive custom rules from your documented data contracts,
test direct-identifier recall on representative synthetic fixtures, and review
every residual leak before release.

Run the complete example from a source checkout:

```bash
python examples/onboarding_india_dpdp.py
```

## Low-RAM, CPU-only setup

Start with one of the small Hindi PII checkpoints already listed in OpenMed's
model manifest. Raw FP32 weight size is approximately four bytes per parameter;
the Python runtime, tokenizer, temporary tensors, and input length add overhead.

| Checkpoint | Parameters | Approximate raw FP32 weights | Use |
| --- | ---: | ---: | --- |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | 33M | 132 MB | Lowest weight footprint; Latin-script evaluation only |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | 44M | 176 MB | Compact default; supports Devanagari |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | 66M | 264 MB | Latin-script evaluation only |

These small checkpoints map to OpenMed's **Tiny** device tier, whose release
target is at most 350 MB resident RAM. Current tokenizer-script coverage marks
only the 44M checkpoint as supporting Devanagari; the 33M and 66M checkpoints
must not be selected for Devanagari notes without a new passing script audit.
Treat the memory target as a release budget, not a guarantee for every Python
environment; measure peak RSS on the exact machine and input lengths you will
deploy. See [Device Tiers and SLOs](tiers.md) and
[Tokenizer Script Coverage](model-tokenizer-script-coverage.md).

For a constrained clinic workstation or startup VM:

1. Start with the 44M checkpoint, CPU execution, and `batch_size=1` when notes
   can contain Devanagari. Consider the 33M checkpoint only for a tested,
   Latin-script-only Hinglish corpus.
2. Split long notes at sentence boundaries before batching. Keep adjacent
   sentences in order and never split a structured identifier across chunks.
3. Increase to batches of 2, then 4, only after observing enough peak-memory
   headroom with representative notes. Larger batches improve throughput but
   raise temporary tensor memory.
4. Reuse one loaded model or one `BatchProcessor` instead of constructing a
   loader for every sentence. Avoid running several model-worker processes on
   a low-RAM host.
5. Keep `india_dpdp_act`, the safety sweep, and the local identifier rules
   enabled when reducing model size. Re-run leakage tests before accepting a
   smaller checkpoint.

For multiple short notes, `BatchProcessor(operation="deidentify")` exposes the
document `batch_size` control and reuses the underlying loader. See
[Batch Processing](batch-processing.md) for the API.

## Production checklist

- Keep inference and cached artifacts inside the approved device or network
  boundary; OpenMed does not add telemetry to the de-identification path.
- Do not retain `result.original_text`, raw entity values, or reversible
  mappings in logs or audit exports.
- Add locally relevant identifier patterns and synthetic regression fixtures;
  this walkthrough is not an exhaustive India identifier catalogue.
- Validate Hindi, Latin-script Hinglish, and code-mixed notes separately.
- Treat DPDP governance and ABDM participation requirements as independent of
  the selected OpenMed policy profile.
