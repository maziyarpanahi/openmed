# Pull Request

## Description
This PR implements the full Korean (ko) PII language pack, resolving issue #262.

Key additions:
- `validate_korean_rrn()` — validates the 13-digit YYMMDD-Sxxxxxx Resident Registration Number using the weighted mod-11 checksum
- `_KOREAN_PII_PATTERNS` — regex patterns for Korean dates (YYYY년 MM월 DD일 and numeric formats), +82/010 phone numbers, RRN national ID, 5-digit postcode, and 시/구/동/로/길 addresses
- `_generate_korean_rrn_surrogate()` in `anonymizer/registry.py` — generates checksum-valid Korean RRN surrogates. This was necessary because Faker's `ko_KR` `ssn()` provider uses random digit templates (`ssn_formats`) without computing the mod-11 checksum, unlike `fr_FR` which calls `calculate_checksum()`. As a result `faker.ssn()` for Korean produces RRNs that fail `validate_korean_rrn()`. The custom generator bypasses this.
- Korean `LANGUAGE_MONTH_NAMES`, `LANGUAGE_FAKE_DATA`, `SUPPORTED_LANGUAGES` entry
- `ko.jsonl` golden fixture with a clinical multilingual example and a checksum-ids example (with hard negative)
- Tests covering the validator, patterns, surrogate round-trip, and fixture offsets

Note: `"ko"` was previously in `NATIONAL_ID_ONLY_LANGUAGES` since only a validator existed. This PR removes it from that set since Korean now has a full language pack.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [x] Test addition/improvement

## Changes Made - added the full korean PII language pack, 
-  Files Edited:   
    openmed/core/pii_i18n.py
    openmed/core/anonymizer/locales.py
    openmed/core/anonymizer/registry.py
    openmed/eval/golden/fixtures/i18n/ko.jsonl
    tests/unit/test_pii_i18n.py
    openmed/CHANGELOG.md

## Testing
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
- [x] I have tested this change with different models/inputs

## Documentation
- [ ] I have updated the documentation accordingly
- [x] I have added docstrings to new functions/classes
- [x] I have updated the CHANGELOG.md

## Code Quality
- [x] I ran `make format`, `make lint`, and `make format-check`
- [ ] For Swift/OpenMedKit changes, I ran `make format-swift` and `make lint-swift`
- [x] I have performed a self-review of my own code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] My changes generate no new warnings

## Dependencies
- [x] I have not added any new dependencies
- [ ] OR I have added new dependencies and they are justified because: ____

## Checklist
- [x] I have read the contributing guidelines
- [x] My commits have clear, descriptive messages
- [x] I have squashed/organized my commits appropriately

## Related Issues
Closes #262

## Screenshots/Examples
Sample output of passing testcases:
(env) ankitha@Mac openmed % python -m pytest tests/unit/test_pii_i18n.py -v 2>&1 | tail -40
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_spanish_phones_have_country_code PASSED [ 90%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_portuguese_phones_have_country_code PASSED [ 91%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_dutch_phones_have_country_code PASSED [ 91%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_hindi_phones_have_country_code PASSED [ 91%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_telugu_phones_have_country_code PASSED [ 91%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_arabic_phones_have_country_code PASSED [ 92%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_hebrew_phones_have_country_code PASSED [ 92%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_japanese_phones_have_country_code PASSED [ 92%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_turkish_phones_have_country_code PASSED [ 93%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_thai_phones_have_country_code PASSED [ 93%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_indonesian_phones_have_country_code PASSED [ 93%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_malay_phones_have_country_code PASSED [ 94%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_tagalog_phones_have_country_code PASSED [ 94%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_danish_phones_have_country_code PASSED [ 94%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_korean_names_are_korean PASSED [ 94%]
tests/unit/test_pii_i18n.py::TestLanguageFakeData::test_korean_phones_have_country_code_or_local PASSED [ 95%]
tests/unit/test_pii_i18n.py::TestIndonesianLocaleAndFixture::test_locale_and_surrogate_nik_round_trip PASSED [ 95%]
tests/unit/test_pii_i18n.py::TestIndonesianLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [ 95%]
tests/unit/test_pii_i18n.py::TestMalayLocaleAndFixture::test_locale_and_surrogate_mykad_round_trip PASSED [ 96%]
tests/unit/test_pii_i18n.py::TestMalayLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [ 96%]
tests/unit/test_pii_i18n.py::TestTagalogLocaleAndFixture::test_locale_and_surrogate_philsys_round_trip PASSED [ 96%]
tests/unit/test_pii_i18n.py::TestTagalogLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [ 97%]
tests/unit/test_pii_i18n.py::TestDanishLocaleAndFixture::test_locale_and_surrogate_cpr_round_trip PASSED [ 97%]
tests/unit/test_pii_i18n.py::TestDanishLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [ 97%]
tests/unit/test_pii_i18n.py::test_validate_latvian_personas_kods PASSED  [ 97%]
tests/unit/test_pii_i18n.py::test_generated_latvian_surrogate_passes_validator PASSED [ 98%]
tests/unit/test_pii_i18n.py::test_latvian_clinical_sample_expected_spans PASSED [ 98%]
tests/unit/test_pii_i18n.py::test_latvian_i18n_golden_fixture_offsets PASSED [ 98%]
tests/unit/test_pii_i18n.py::TestSlovakLocaleAndFixture::test_locale_and_surrogate_rodne_cislo_round_trip PASSED [ 99%]
tests/unit/test_pii_i18n.py::TestSlovakLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [ 99%]
tests/unit/test_pii_i18n.py::TestKoreanLocaleAndFixture::test_locale_and_surrogate_rrn_round_trip PASSED [ 99%]
tests/unit/test_pii_i18n.py::TestKoreanLocaleAndFixture::test_i18n_golden_fixture_offsets PASSED [100%]

=============================== warnings summary ===============================
tests/unit/test_pii_i18n.py::TestValidateMalaysianMyKad::test_generated_mykad_surrogate_passes_validator
  /Users/ankitha/Documents/openmed/tests/unit/test_pii_i18n.py:639: UserWarning: OpenMed: language 'ms' has no native Faker locale; using 'ms_MY' backed by 'id_ID'. Pass locale=... to override.
    surrogate = anonymizer.surrogate("850817-14-5678", "national_id")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 337 passed, 1 warning in 0.52s ========================


Happy to make any changes or adjustments based on review feedback.