# India Health-ID De-identification

OpenMed's `india_health_id` policy protects Indian health and adjacent
identifiers found in clinical documents:

- 14-digit ABHA numbers;
- ABHA Addresses ending in `@abdm` or the sandbox-style `@sbx`;
- context-confirmed UPI virtual payment addresses;
- context-confirmed ration-card identifiers.

ABDM describes ABHA as a randomly generated 14-digit identifier and documents
ABHA Addresses such as `ajay@abdm`. NPCI describes a UPI ID as a virtual payment
address. OpenMed performs local structural recognition only; it does not query
ABDM, a payment service provider, or a ration-card registry.

## Use the policy

```python
from openmed import deidentify

result = deidentify(
    clinical_note,
    policy="india_health_id",
    lang="hi",  # English, Hindi, and Telugu code-mixed cues are supported
)

print(result.deidentified_text)
```

The policy uses high-recall arbitration, makes the deterministic safety sweep
mandatory, classifies the recognized values as high-risk direct identifiers,
and masks them by default. Supplying a weaker global method does not weaken the
policy action.

## Masking behavior

UIDAI's Masked Aadhaar convention hides the first eight Aadhaar digits and
retains the last four. ABHA numbers, ABHA Addresses, UPI IDs, and ration-card
numbers are different identifier classes, and OpenMed does not claim that the
Aadhaar display rule is an ABDM rule. The `india_health_id` policy applies the
more conservative behavior: it fully replaces every recognized surface with a
typed mask and never retains a visible suffix.

The ABHA checksum used by OpenMed is an offline validation contract for
synthetic fixtures and surrogates. A value that passes it is structurally valid
for OpenMed testing; it is not proof that the value exists in ABDM.

## Surrogates and audit safety

With `method="replace"` outside the masking policy, ABHA numbers, ABHA
Addresses, UPI IDs, and ration-card identifiers receive validator-compatible
synthetic replacements. Pass a `SurrogateVault` to reuse the same replacement
for the same source across documents. Vault keys use HMAC source hashes and do
not persist raw source identifiers.

Audit evidence records offsets, hashes, labels, and actions. Do not write raw
ABHA or UPI values to application logs, exception messages, metrics, or custom
audit payloads.

## FHIR export

FHIR bundle assembly automatically redacts matching values in `Identifier`
containers and PatientID-style fields before emission. The sanitizer leaves
clinical narrative and non-identifier `value` fields unchanged and does not
mutate the caller's input resources.

## Validation gate

The bundled synthetic gate covers valid ABHA, ABHA Address, UPI, and ration-card
surfaces plus invalid-checksum and ordinary-email negatives:

```python
from openmed.eval.suites import assert_india_health_id_leakage_gate

gate = assert_india_health_id_leakage_gate()
assert gate.entity_leakage == 0.0
assert gate.false_accept_count == 0
```

The gate contains no real PHI and reports failures with offsets and hashes only.

## Official references

- [ABDM FAQ](https://abdm.gov.in/FAQ)
- [ABDM citizen information](https://abdm.gov.in/citizens)
- [NPCI UPI overview](https://www.npci.org.in/product/upi/about-upi)
- [UIDAI Masked Aadhaar FAQ](https://uidai.gov.in/en/contact-support/have-any-question/921-english-uk/faqs/aadhaar-online-services.html)
