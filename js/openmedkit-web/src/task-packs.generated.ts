// GENERATED FILE. Do not edit by hand.
//
// Source: openmed/core/data/task_packs.json and the web ONNX model catalog
// (models.jsonl), through scripts/web/build_web_task_packs.py.
// Regenerate: python scripts/web/build_web_task_packs.py
// Check:      python scripts/web/build_web_task_packs.py --check
//
// The web runtime entry of every portable task pack, with the fields the
// Python resolver (openmed.core.task_packs.resolve) returns and each
// candidate's catalog reproducibility hash. Use the typed API in "./packs"
// instead of this file.

import type { WebTaskPackData } from "./packs";

export const TASK_PACKS_SCHEMA = "openmed.task_packs";
export const TASK_PACKS_SCHEMA_VERSION = "1.0.0";

/** SHA-256 of the task_packs.json bytes this module was built from. */
export const TASK_PACKS_SOURCE_DIGEST = "sha256:dbeca2fa67b6331d33af455b69899972b8e0adcb0a60f36c8ca027d79a36f937";

/** Default English notice text by key. Kits localize by key. */
export const TASK_PACK_NOTICES: Readonly<Record<string, string>> = {
  "openmed.notice.clinical_entity.extraction_only": "Reports condition mentions and their offsets only. It does not diagnose, rank or recommend care.",
  "openmed.notice.medication.extraction_only": "Reports medication mentions and their offsets only. It never gives dosing advice and never triggers an action.",
  "openmed.notice.pii.review_required": "Detected identifiers are redaction candidates. Automated detection can miss identifiers, so review output before it leaves the device."
};

/** The web runtime entry of each pack, sorted by pack id. */
export const WEB_TASK_PACKS: readonly WebTaskPackData[] = [
  {
    "pack_id": "clinical_entity",
    "pack_version": "1.0.0",
    "task": "clinical_entity_extraction",
    "runtime": "web",
    "format": "onnx",
    "status": "resolved",
    "candidates": [
      {
        "rank": 1,
        "repo_id": "OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M-v1-onnx-android",
        "param_count": 33000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:5f22790e0683fee106df70d08690575e75d905f5417588f0e2b46ed1e35614b0"
      },
      {
        "rank": 2,
        "repo_id": "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-65M-v1-onnx-android",
        "param_count": 65000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:60d2ca44f81080002de836ad330b36d4373407e9c714faae26b8b8a7d852e589"
      },
      {
        "rank": 3,
        "repo_id": "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-66M-v1-onnx-android",
        "param_count": 66000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:f633d776365ba0cb493939c8f0fadcb0cd4e588eef1ab4ca064b454d7be742ab"
      }
    ],
    "canonical_labels": [
      "CONDITION",
      "DISEASE",
      "PATHOLOGY"
    ],
    "notice_key": "openmed.notice.clinical_entity.extraction_only"
  },
  {
    "pack_id": "medication",
    "pack_version": "1.0.0",
    "task": "medication_mention_extraction",
    "runtime": "web",
    "format": "onnx",
    "status": "resolved",
    "candidates": [
      {
        "rank": 1,
        "repo_id": "OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-33M-v1-onnx-android",
        "param_count": 33000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:626cfa5ed3ccdb21c5cd57285608737ed224b872bbe79055439bef34d217ca88"
      },
      {
        "rank": 2,
        "repo_id": "OpenMed/OpenMed-NER-PharmaDetect-TinyMed-65M-v1-onnx-android",
        "param_count": 65000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:ac58106bfa579dd7c8ca4a5eacc9dc6777535454386c7edaaf919e8b71cf2eca"
      },
      {
        "rank": 3,
        "repo_id": "OpenMed/OpenMed-NER-PharmaDetect-TinyMed-66M-v1-onnx-android",
        "param_count": 66000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:a32d47c9f9e914d9095600df4983915df50b8a19f8e66954f9291843083f15b1"
      }
    ],
    "canonical_labels": [
      "CHEMICAL"
    ],
    "notice_key": "openmed.notice.medication.extraction_only"
  },
  {
    "pack_id": "pii",
    "pack_version": "1.0.0",
    "task": "pii_detection",
    "runtime": "web",
    "format": "onnx",
    "status": "resolved",
    "candidates": [
      {
        "rank": 1,
        "repo_id": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android",
        "param_count": 33000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:aa416cb6773092132bbcbd6fbf2b322b11fbaf138c90aed76e0a48d72bc9618e"
      },
      {
        "rank": 2,
        "repo_id": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android",
        "param_count": 44000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:ece4789f35a450ede1d4c32249bb4c4864502af4274c2fcced0bb1240fdf8798"
      },
      {
        "rank": 3,
        "repo_id": "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-onnx-android",
        "param_count": 66000000,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:db486bcfc184fb35e523860e239e521d968be8206803da74a73986bb19622d11"
      }
    ],
    "canonical_labels": [
      "ACCOUNT_NUMBER",
      "DATE_OF_BIRTH",
      "EMAIL",
      "FIRST_NAME",
      "ID_NUM",
      "IP_ADDRESS",
      "LAST_NAME",
      "PERSON",
      "PHONE",
      "SSN",
      "STREET_ADDRESS",
      "URL"
    ],
    "notice_key": "openmed.notice.pii.review_required"
  }
];
