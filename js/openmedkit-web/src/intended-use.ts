import declarations from "./intended-use.json";
import type { IntendedUse, RegulatoryTier } from "./types";

type WireDeclaration = Omit<IntendedUse, "regulatory_tier" | "autonomous_clinical_action" | "obligations"> & {
  readonly regulatory_tier: string;
  readonly autonomous_clinical_action: boolean;
};

const obligations: Record<RegulatoryTier, readonly string[]> = {
  non_clinical: Object.freeze([]),
  informational: Object.freeze(["notice_required"]),
  assistive_review: Object.freeze(["notice_required", "acknowledgement_required"]),
  restricted: Object.freeze([
    "notice_required", "acknowledgement_required", "clinician_mode_required", "crisis_routing_required",
  ]),
};

/** Resolve future or absent tier codes to the full restricted safeguards. */
export function effectiveRegulatoryTier(value: unknown): RegulatoryTier {
  switch (value) {
    case "non_clinical":
    case "informational":
    case "assistive_review":
      return value;
    default:
      return "restricted";
  }
}

export function regulatoryObligations(tier: unknown): readonly string[] {
  return obligations[effectiveRegulatoryTier(tier)];
}

function decode(row: WireDeclaration): IntendedUse {
  if (row.autonomous_clinical_action !== false) {
    throw new Error("autonomous_clinical_action_forbidden");
  }
  const tier = effectiveRegulatoryTier(row.regulatory_tier);
  return Object.freeze({
    ...row,
    regulatory_tier: tier,
    intended_users: Object.freeze([...row.intended_users]),
    excluded_uses: Object.freeze([...row.excluded_uses]),
    input_modalities: Object.freeze([...row.input_modalities]),
    languages: Object.freeze([...row.languages]),
    autonomous_clinical_action: false,
    obligations: regulatoryObligations(tier),
  });
}

const registry = new Map(Object.entries(declarations).map(([code, row]) => [code, decode(row)]));

/** Resolve an exact capability code using data bundled with the kit. */
export function getIntendedUse(capability: string): IntendedUse | undefined {
  return registry.get(capability);
}
