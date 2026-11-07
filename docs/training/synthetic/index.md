# Synthetic training data

`openmed.training.synthetic` holds the deterministic, offline generators that
OpenMed uses to build augmentation data for de-identification, document
structure, and social-determinant models. The examples here use repository-owned
templates and synthetic seeds without network access or restricted clinical
records. Caller-supplied seeds and translators retain their own provenance
and runtime requirements.

**Synthetic outputs are augmentation data, not clinical ground truth.** The
spans, boxes, and section labels are correct *by construction* for the
synthetic text they came from. They say nothing about how a model will behave
on real notes, and they must never be used to report clinical accuracy. Keep
them clearly separated from held-out evaluation data; the generators that
touch evaluation fixtures include a leakage guard for exactly that reason.

## Generators

| Page | Public symbols | Produces |
|---|---|---|
| [Locale PHI notes](locale-phi.md) | `LocalePhiGenerator`, `generate_locale_phi_examples`, `LocalePhiExample`, `SyntheticPhiSpan`, `SUPPORTED_LOCALE_PHI_LANGUAGES`, `LOCALE_PHI_LABELS` | One-paragraph notes in the declared supported languages with aligned PHI spans |
| [Burned-in PHI images](burned-in.md) | `BurnedInGenerator`, `generate_burned_in_example`, `BurnedInExample`, `BurnedInTextBox`, `BACKGROUND_MODES`, `BURNED_IN_LABELS`, `DEFAULT_CANVAS_SIZE`, `DEFAULT_FONT_NAMES`, `DEFAULT_FONT_SIZES` | Images with rendered PHI text plus pixel boxes and text spans |
| [Social-history sections](social-history.md) | `generate_social_history_examples`, `SyntheticSocialHistory`, `SDOHEvent`, `SOCIAL_HISTORY_CATEGORIES`, `SYNTHETIC_SOCIAL_HISTORY_SOURCE` | Social History sections with SDOH gold events |
| [Section-label datasets](section-labels.md) | `build_section_dataset`, `load_section_dataset`, `load_section_manifest`, `validate_section_labels`, `validate_section_record`, `assert_no_eval_overlap`, `SectionDatasetBuildResult`, `SectionDatasetLeakageError`, and the section constants | JSONL notes with BIO section tags and a hash manifest |
| [Span offset projection](offset-projection.md) | `SpanAnnotation`, `SpanProjectionError`, `normalize_span_annotations`, `validate_span_integrity`, `span_corruption_count`, `realign_translated_spans` | Checked span offsets, used by translation augmentation |
| [Translation augmentation](translation-augmentation.md) | `augment_span_annotated_examples`, `DictionaryTranslator`, `ModelBackedTranslator`, `TranslationAugmentedExample`, `load_span_jsonl`, `write_augmented_jsonl`, `DEFAULT_TARGET_LANGUAGES`, `SYNTHETIC_SOURCE` | Translated and back-translated span-annotated examples |
| [SPDX license identifiers](license-identifiers.md) | `normalize_spdx_identifier`, `SpdxIdentifier`, `SpdxIdentifierStatus`, `PERMISSIVE_SPDX_IDENTIFIERS`, `SPDX_IDENTIFIER_SCHEMA_VERSION` | Normalized license identifiers for lineage manifests |

## Conventions shared by every page

- **Offline.** Each example on these pages runs without network access.
  Pillow is only needed for the burned-in generator (`pip install
  "openmed[multimodal]"`); the other generators need nothing beyond the core
  install.
- **Synthetic values only.** Names, identifiers, phone numbers, and addresses
  in the examples come from the generators themselves. They are not real
  people and must not be copied into issues or logs as if they were.
- **Seeded determinism.** The same seed gives the same output within one
  environment. Values that come from [Faker](https://faker.readthedocs.io/)
  can change when the Faker version changes, so pin it, and record the seed
  and package versions next to any dataset you keep.
- **Provenance flags.** Preserve the source and synthetic metadata described
  on each generator page. Their keys differ: social-history records use
  `metadata.synthetic` and `metadata.source`, while locale PHI training items
  also expose `is_synthetic`, `synthetic_source`, and `contains_real_phi`.
  Keep these declarations when composing datasets.

## Additional public exports

The package also re-exports these APIs added since the original generator set.
The linked guide or schema section owns their contract.

- HL7 v2 message generation and leakage:
  [`DEFAULT_HL7V2_TIMEZONE_OFFSETS`](hl7v2-messages.md),
  [`HL7V2_ADT_EVENTS`](hl7v2-messages.md),
  [`HL7V2_SYNTHETIC_MESSAGE_TYPES`](hl7v2-messages.md),
  [`HL7V2_SYNTHETIC_SCHEMA_VERSION`](hl7v2-messages.md),
  [`HL7V2_TIMESTAMP_PRECISIONS`](hl7v2-messages.md),
  [`MAX_HL7V2_MESSAGES`](hl7v2-messages.md),
  [`MAX_HL7V2_NTE_SEGMENTS`](hl7v2-messages.md),
  [`MAX_HL7V2_OBX_SEGMENTS`](hl7v2-messages.md),
  [`MAX_HL7V2_REPETITIONS`](hl7v2-messages.md),
  [`SYNTHETIC_HL7V2_SOURCE`](hl7v2-messages.md),
  [`HL7V2LeakageReport`](hl7v2-messages.md),
  [`HL7V2MessageGenerator`](hl7v2-messages.md),
  [`HL7V2ResidualPosition`](hl7v2-messages.md),
  [`HL7V2TruthValue`](hl7v2-messages.md),
  [`SyntheticHL7V2Message`](hl7v2-messages.md),
  [`HL7V2_SURROGATE_PROVIDERS`](hl7v2-messages.md),
  [`HL7V2LeakageError`](hl7v2-messages.md),
  [`HL7V2SynthesisConfig`](hl7v2-messages.md),
  [`HL7V2SynthesisError`](hl7v2-messages.md),
  [`evaluate_hl7v2_leakage`](hl7v2-messages.md),
  [`generate_hl7v2_messages`](hl7v2-messages.md).
- Synthetic license and lineage validation:
  [`DEFAULT_LICENSE_LINEAGE_POLICY`](license-lineage-manifests.md),
  [`DEFAULT_MAX_DEPENDENCIES`](license-lineage-manifests.md),
  [`LICENSE_LINEAGE_FIELDS`](license-lineage-manifests.md),
  [`LICENSE_LINEAGE_SCHEMA_VERSION`](license-lineage-manifests.md),
  [`LICENSE_LINEAGE_SOURCE`](license-lineage-manifests.md),
  [`MAX_LICENSE_EXPRESSION_LENGTH`](license-lineage-manifests.md),
  [`RESTRICTED_LICENSE_IDENTIFIERS`](license-lineage-manifests.md),
  [`LicenseExpressionAssessment`](license-lineage-manifests.md),
  [`LicenseExpressionStatus`](license-lineage-manifests.md),
  [`LicenseLineageFinding`](license-lineage-manifests.md),
  [`LicenseLineageReasonCode`](license-lineage-manifests.md),
  [`LicenseLineageReport`](license-lineage-manifests.md),
  [`LicenseLineageVerdict`](license-lineage-manifests.md),
  [`SyntheticLicenseLineageError`](license-lineage-manifests.md),
  [`SyntheticModelProvenance`](license-lineage-manifests.md),
  [`load_license_lineage_manifest`](license-lineage-manifests.md),
  [`MAX_DEPENDENCIES`](license-lineage-manifests.md),
  [`LicenseLineageSourceClass`](license-lineage-manifests.md),
  [`RedistributionDecision`](license-lineage-manifests.md),
  [`SyntheticDatasetDependency`](license-lineage-manifests.md),
  [`SyntheticDatasetLicenseManifest`](license-lineage-manifests.md),
  [`SyntheticGeneratorProvenance`](license-lineage-manifests.md),
  [`SyntheticLicenseLineagePolicy`](license-lineage-manifests.md),
  [`assess_license_expression`](license-lineage-manifests.md),
  [`assert_redistribution_allowed`](license-lineage-manifests.md),
  [`manifest_digest`](license-lineage-manifests.md),
  [`validate_license_lineage`](license-lineage-manifests.md).
- Synthetic record schemas:
  [`SYNTHETIC_RECORD_SCHEMA_IDS`](#record-schemas),
  [`SYNTHETIC_RECORD_SCHEMA_NAMES`](#record-schemas),
  [`SYNTHETIC_RECORD_SCHEMA_VERSION`](#record-schemas),
  [`SyntheticRecordSchemaError`](#record-schemas),
  [`export_record_schema`](#record-schemas),
  [`export_record_schema_fingerprints`](#record-schemas),
  [`export_record_schema_json`](#record-schemas),
  [`export_record_schemas`](#record-schemas),
  [`export_record_schemas_json`](#record-schemas),
  [`record_schema_fingerprint`](#record-schemas),
  [`record_schema_id`](#record-schemas),
  [`validate_record`](#record-schemas),
  [`write_record_schemas`](#record-schemas).

## Record schemas

The schema helpers describe the synthetic records already emitted by the
package. `validate_record` checks the declared schema and span/box invariants;
it does not certify clinical correctness or prove that arbitrary input is
synthetic. `SyntheticRecordSchemaError` reports a rejected record.

```python
from openmed.training.synthetic import (
    LocalePhiGenerator, export_record_schema, record_schema_id, validate_record,
)

name = "locale_phi_example"
schema = export_record_schema(name)
assert schema["$id"] == record_schema_id(name)
validate_record(name, LocalePhiGenerator(seed=7).generate("en").to_training_item())
```

`SYNTHETIC_RECORD_SCHEMA_NAMES`, `SYNTHETIC_RECORD_SCHEMA_IDS` and
`SYNTHETIC_RECORD_SCHEMA_VERSION` describe the available schema contracts.
The JSON and fingerprint export helpers return canonical content;
`write_record_schemas(directory)` writes them only to the directory you choose.
