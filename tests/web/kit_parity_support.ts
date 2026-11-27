// Shared helpers for tests/fixtures/parity/kit_parity_vectors.json: the
// fixture's types and the projection of a vector into the web package's own
// UTF-16 result shape. Used by test_kit_parity.spec.ts and the contract-gate
// reporter.
import {
  scalarToUtf16Offset,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
} from "../../js/openmedkit-web/src/index";

export interface VectorSpan {
  label: string;
  canonical_label: string;
  action: string;
  start: number;
  end: number;
  confidence_ppm?: number;
  replacement?: string;
  utf8_start: number;
  utf8_end: number;
  utf16_start: number;
  utf16_end: number;
}

export interface Vector {
  id: string;
  categories: string[];
  text: string;
  spans: VectorSpan[];
  deidentified_text: string;
  deidentified_sha256: string;
  result_json: string;
  result_sha256: string;
}

export interface Rejection {
  id: string;
  input: string;
  reason: string;
  field_path: string;
  text?: string;
}

export interface Vectors {
  schema: string;
  version: number;
  profile: {
    name: string;
    version: number;
    actions: string[];
    canonical_labels: string[];
    offset_contract_versions: number[];
    decoder_contract_versions: number[];
    privacy_sentinel: string;
  };
  vectors: Vector[];
  rejections: Rejection[];
  source_rejections: Required<Rejection>[];
  offset_rejections: {
    id: string;
    text: string;
    unit: "scalar" | "utf8" | "utf16";
    offset: number;
    reason: string;
  }[];
  compatibility: {
    id: string;
    offset_contract_version: number;
    decoder_contract_version: number;
    outcome: string;
  }[];
}

export function webSpan(text: string, span: VectorSpan): OpenMedSpan {
  return {
    schema_version: 1,
    doc_id: "kit-parity",
    start: scalarToUtf16Offset(text, span.start),
    end: scalarToUtf16Offset(text, span.end),
    text_hash: "hmac-sha256:00",
    entity_type: span.label,
    canonical_label: span.canonical_label as OpenMedSpan["canonical_label"],
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score:
      span.confidence_ppm === undefined ? null : span.confidence_ppm / 1_000_000,
    detector: "kit-parity-fixture",
    evidence: {},
    action: span.action as OpenMedSpan["action"],
    replacement: span.replacement ?? null,
    reversible_id: null,
    section: null,
    metadata: {},
  };
}

export function webResult(vector: Vector): OpenMedDeidentifyResult {
  return {
    text: vector.text,
    deidentifiedText: vector.deidentified_text,
    // Reverse the order so the projection's own sort is exercised.
    spans: vector.spans.map((span) => webSpan(vector.text, span)).reverse(),
  };
}
