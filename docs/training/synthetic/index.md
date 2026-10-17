# Synthetic training data

`openmed.training.synthetic` holds the deterministic, offline generators that
OpenMed uses to build augmentation data for de-identification, document
structure, and social-determinant models. Every generator is repository-owned:
templates and value pools ship with the package, no network call is made, and
no restricted or real clinical record is read.

**Synthetic outputs are augmentation data, not clinical ground truth.** The
spans, boxes, and section labels are correct *by construction* for the
synthetic text they came from. They say nothing about how a model will behave
on real notes, and they must never be used to report clinical accuracy. Keep
them clearly separated from held-out evaluation data; the generators that
touch evaluation fixtures include a leakage guard for exactly that reason.

## Generators

| Page | Public symbols | Produces |
|---|---|---|
| [Locale PHI notes](locale-phi.md) | `LocalePhiGenerator`, `generate_locale_phi_examples`, `LocalePhiExample`, `SyntheticPhiSpan`, `SUPPORTED_LOCALE_PHI_LANGUAGES`, `LOCALE_PHI_LABELS` | One-paragraph notes in 42 languages with aligned PHI spans |
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
- **Provenance flags.** Records carry `synthetic: True` (or `is_synthetic`) and
  `contains_real_phi: False`, and training items name their `synthetic_source`.
  Filter on these when mixing synthetic and real data, and never strip them.
