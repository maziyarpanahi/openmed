import { decodeBioTokenSpans } from "./decoder";
import { loadTokenClassificationPipeline } from "./model-loader";
import {
  OPENMED_SPAN_SCHEMA_VERSION,
  type DeidentifyOptions,
  type ExtractPiiOptions,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
  type SpanAction,
} from "./types";

export {
  CANONICAL_LABELS,
  OPENMED_SPAN_SCHEMA_VERSION,
  POLICY_LABELS,
  SPAN_ACTIONS,
} from "./types";
export type {
  CanonicalLabel,
  DecodedEntitySpan,
  DeidentifyOptions,
  ExtractPiiOptions,
  LoadModelOptions,
  ModelLoader,
  OpenMedDeidentifyResult,
  OpenMedSpan,
  PolicyLabel,
  SpanAction,
  TextHash,
  TokenClassificationCallOptions,
  TokenClassificationEntity,
  TokenClassificationOutput,
  TokenClassificationPipeline,
  TransformersRuntime,
} from "./types";
export {
  decodeBioTokenSpans,
  flattenTokenClassificationOutput,
  normalizeLabel,
  policyLabelFor,
  refinePrivacyFilterSpan,
  trimSpanWhitespace,
} from "./decoder";
export {
  isLocalModelReference,
  loadTokenClassificationPipeline,
} from "./model-loader";

const DEFAULT_MODEL_ID = "OpenMed/privacy-filter-transformersjs";
const DEFAULT_HASH_SECRET = "openmedkit-web";

export async function extractPii(
  text: string,
  options: ExtractPiiOptions = {},
): Promise<OpenMedSpan[]> {
  const pipeline =
    options.pipeline ??
    (await (options.modelLoader ?? loadTokenClassificationPipeline)(
      options.model ?? DEFAULT_MODEL_ID,
      options.loaderOptions,
    ));
  const rawOutput = await pipeline(text, {
    aggregation_strategy: "none",
    ...(options.pipelineOptions ?? {}),
  });
  const decoded = decodeBioTokenSpans(text, rawOutput, {
    threshold: options.threshold ?? 0,
  });

  const spans: OpenMedSpan[] = [];
  for (const entity of decoded) {
    const surface = text.slice(entity.start, entity.end);
    spans.push({
      schema_version: OPENMED_SPAN_SCHEMA_VERSION,
      doc_id: options.docId ?? "document",
      start: entity.start,
      end: entity.end,
      text_hash: await hmacTextHash(
        surface,
        options.hashSecret ?? DEFAULT_HASH_SECRET,
      ),
      entity_type: entity.entity_type,
      canonical_label: entity.canonical_label,
      policy_label: entity.policy_label,
      regulatory_tags: [...(options.regulatoryTags ?? [])],
      score: entity.score,
      detector: options.detector ?? "transformersjs",
      evidence: {
        token_count: entity.token_count,
      },
      action: "keep",
      replacement: null,
      reversible_id: null,
      section: options.section ?? null,
      metadata: {
        ...(options.model ? { model: options.model } : {}),
        ...(options.metadata ?? {}),
      },
    });
  }
  return spans;
}

export async function deidentify(
  text: string,
  options: DeidentifyOptions = {},
): Promise<OpenMedDeidentifyResult> {
  const spans = await extractPii(text, options);
  const redactionSpans = spans.map((span) => {
    const replacement = options.replacement?.(span) ?? `[${span.canonical_label}]`;
    return {
      ...span,
      action: "redact" as SpanAction,
      replacement,
    };
  });
  return {
    text,
    deidentifiedText: spansToRedactedText(text, redactionSpans),
    spans: redactionSpans,
  };
}

export function spansToRedactedText(text: string, spans: OpenMedSpan[]): string {
  let redacted = text;
  for (const span of [...spans].sort((left, right) => right.start - left.start)) {
    redacted =
      redacted.slice(0, span.start) +
      (span.replacement ?? `[${span.canonical_label}]`) +
      redacted.slice(span.end);
  }
  return redacted;
}

export async function hmacTextHash(
  surface: string | Uint8Array,
  secret: string | Uint8Array,
): Promise<`hmac-sha256:${string}`> {
  const payload = toBytes(surface);
  const key = toBytes(secret);
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const cryptoKey = await subtle.importKey(
      "raw",
      toArrayBuffer(key),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"],
    );
    const signature = await subtle.sign("HMAC", cryptoKey, toArrayBuffer(payload));
    return `hmac-sha256:${toHex(new Uint8Array(signature))}`;
  }

  const { createHmac } = await import("node:crypto");
  const digest = createHmac("sha256", key).update(payload).digest("hex");
  return `hmac-sha256:${digest}`;
}

function toBytes(value: string | Uint8Array): Uint8Array {
  return typeof value === "string" ? new TextEncoder().encode(value) : value;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join(
    "",
  );
}
