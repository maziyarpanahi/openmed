import catalog from "./safety-notices.json";
import { effectiveRegulatoryTier, regulatoryObligations } from "./intended-use";
import type { RegulatoryTier } from "./types";

/** Bundled notice with the Python catalog's version binding. */
export interface OpenMedDisclaimer {
  readonly id: string;
  readonly version: string;
  readonly version_digest: string;
  readonly text: string;
  readonly locale: string;
  readonly tier: RegulatoryTier;
  readonly requires_acknowledgement: boolean;
}

export interface NoticeAcknowledgement {
  readonly notice_id: string;
  readonly version_digest: string;
  readonly acknowledged_at: string;
}

export interface ReviewOptions {
  readonly clinicianMode?: boolean;
  readonly acknowledgement?: NoticeAcknowledgement | null;
  /** A deployment may require stricter safeguards than the capability's default. */
  readonly regulatoryTier?: RegulatoryTier;
}

export class CapabilityReviewError extends Error {
  constructor(readonly code: "clinician_mode_required" | "acknowledgement_required" | "invalid_acknowledgement") {
    super(code);
    this.name = "CapabilityReviewError";
  }
}

const notices = new Map<string, OpenMedDisclaimer>(
  Object.entries(catalog.notices).map(([id, row]) => [id, Object.freeze({ ...row, tier: effectiveRegulatoryTier(row.tier) })]),
);
const capabilities = new Map(Object.entries({ ...catalog.capabilities, ...catalog.overrides }));

export function disclaimerById(id: string): OpenMedDisclaimer | undefined {
  return notices.get(id);
}

export function disclaimerForTier(tier: RegulatoryTier): OpenMedDisclaimer {
  return notices.get(catalog.tiers[effectiveRegulatoryTier(tier)])!;
}

export function disclaimerForCapability(capability: string): OpenMedDisclaimer {
  const id = capabilities.get(capability);
  return (id && notices.get(id)) || disclaimerForTier("restricted");
}

/** Record the host's explicit acknowledgement without accessing browser storage. */
export function acknowledgeNotice(notice: OpenMedDisclaimer, at: Date): NoticeAcknowledgement | null {
  const published = notices.get(notice.id);
  if (!published || published.version_digest !== notice.version_digest) {
    throw new CapabilityReviewError("invalid_acknowledgement");
  }
  if (!published.requires_acknowledgement) return null;
  const utc = at.toISOString().replace(/\.(\d{3})Z$/, (_, ms: string) => ms === "000" ? "+00:00" : `.${ms}000+00:00`);
  return Object.freeze({ notice_id: published.id, version_digest: published.version_digest, acknowledged_at: utc });
}

export function requireCapabilityAccess(capability: string, review: ReviewOptions = {}): OpenMedDisclaimer {
  let notice = disclaimerForCapability(capability);
  if (review.regulatoryTier !== undefined) {
    const requested = disclaimerForTier(review.regulatoryTier);
    if (regulatoryObligations(requested.tier).length > regulatoryObligations(notice.tier).length) notice = requested;
  }
  if (notice.tier === "restricted" && review.clinicianMode !== true) {
    throw new CapabilityReviewError("clinician_mode_required");
  }
  if (notice.requires_acknowledgement) {
    const ack = review.acknowledgement;
    if (!ack) throw new CapabilityReviewError("acknowledgement_required");
    if (ack.notice_id !== notice.id || ack.version_digest !== notice.version_digest ||
        typeof ack.acknowledged_at !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/.test(ack.acknowledged_at) ||
        !Number.isFinite(Date.parse(ack.acknowledged_at))) {
      throw new CapabilityReviewError("invalid_acknowledgement");
    }
  }
  return notice;
}
