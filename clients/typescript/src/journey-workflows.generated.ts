// Generated from the canonical Journey workflow registry. Do not edit.

export type JourneyResourceType =
  | "artifact"
  | "job"
  | "fact"
  | "conflict"
  | "journey"
  | "cohort"
  | "dataset"
  | "registry"
  | "measure"
  | "trial_review"
  | "evidence"
  | "current_fact"
  | "journey_event"
  | "mapping"
  | "cohort_run"
  | "dataset_manifest";

export type JourneyResourceState =
  | "success"
  | "partial"
  | "empty"
  | "unknown"
  | "conflict"
  | "unsupported"
  | "denied"
  | "failure";

export type JourneyWorkflowName =
  | "journey"
  | "cohort"
  | "dataset"
  | "registry"
  | "measure"
  | "trial_review";

export interface JourneyResourceQuery {
  resource_type: JourneyResourceType;
  namespace?: string;
  purpose?: string;
  role?: string;
  attributes?: string[];
  consent_state?: "active" | "unknown" | "withdrawn";
  export_policy?: string;
  first?: number;
  after?: string | null;
  fields?: string[];
}

export type JourneyWorkflowQuery = Omit<JourneyResourceQuery, "resource_type">;

export interface JourneyResourcePage {
  state: JourneyResourceState;
  code: string | null;
  resources: Array<{
    resource_type: JourneyResourceType;
    resource_id: string;
    namespace: string;
    data: Record<string, unknown>;
    state: JourneyResourceState;
    version: number;
    revision: number;
    schema_version: string;
    compatibility_policy: "same_major";
    extensions: Record<string, unknown>;
  }>;
  page_info: {
    has_next_page: boolean;
    end_cursor: string | null;
    page_size: number;
    snapshot_digest: string;
  };
  policy: {
    state: "success" | "denied";
    namespace: string;
    purpose: string;
    role: string;
    attributes: string[];
    consent_state: "active" | "unknown" | "withdrawn";
    export_policy: string;
    decision_id: string;
    request_digest: string;
    allowed_fields: string[];
    code: string | null;
    policy_version: string;
  };
  schema_version: string;
  compatibility_policy: "same_major";
}

export const JOURNEY_WORKFLOW_RESOURCE_TYPES = {
  journey: "journey",
  cohort: "cohort",
  dataset: "dataset",
  registry: "registry",
  measure: "measure",
  trial_review: "trial_review",
} as const satisfies Record<JourneyWorkflowName, JourneyResourceType>;
