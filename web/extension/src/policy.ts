import type { OpenMedSpan, SpanAction } from "openmed";

import clinicalMinimalRedaction from "../../../openmed/core/policies/clinical_minimal_redaction.json";
import hipaaSafeHarbor from "../../../openmed/core/policies/hipaa_safe_harbor.json";
import strictNoLeak from "../../../openmed/core/policies/strict_no_leak.json";

type PolicyAction = SpanAction | "remove";

export interface PolicyProfile {
  name: string;
  default_action: PolicyAction;
  policy_label_actions: Record<string, PolicyAction>;
  actions: Record<string, PolicyAction>;
}

export const BUNDLED_POLICIES = {
  hipaa_safe_harbor: hipaaSafeHarbor as PolicyProfile,
  clinical_minimal_redaction: clinicalMinimalRedaction as PolicyProfile,
  strict_no_leak: strictNoLeak as PolicyProfile,
} as const;

export type PolicyName = keyof typeof BUNDLED_POLICIES;

export const DEFAULT_POLICY: PolicyName = "hipaa_safe_harbor";

export const POLICY_OPTIONS: ReadonlyArray<{
  name: PolicyName;
  label: string;
}> = [
  { name: "hipaa_safe_harbor", label: "HIPAA Safe Harbor" },
  {
    name: "clinical_minimal_redaction",
    label: "Clinical minimal redaction",
  },
  { name: "strict_no_leak", label: "Strict no-leak" },
];

export function isPolicyName(value: unknown): value is PolicyName {
  return typeof value === "string" && value in BUNDLED_POLICIES;
}

export function applyPolicy(
  spans: OpenMedSpan[],
  policyName: PolicyName,
): OpenMedSpan[] {
  const profile = BUNDLED_POLICIES[policyName];
  const selected: OpenMedSpan[] = [];

  for (const span of spans) {
    const configuredAction =
      profile.actions[span.canonical_label] ??
      profile.policy_label_actions[span.policy_label] ??
      profile.default_action;
    if (configuredAction === "keep") {
      continue;
    }
    selected.push({
      ...span,
      action: extensionAction(configuredAction),
      replacement: `[${span.canonical_label}]`,
      metadata: {
        ...span.metadata,
        policy_profile: policyName,
        policy_action: configuredAction,
      },
    });
  }
  return selected;
}

function extensionAction(action: PolicyAction): SpanAction {
  return action === "remove" ? "redact" : action;
}
