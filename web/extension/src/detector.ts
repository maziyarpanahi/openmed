import type {
  TokenClassificationEntity,
  TokenClassificationPipeline,
} from "openmed";

interface DetectorRule {
  label: string;
  pattern: RegExp;
  captureGroup?: number;
}

interface Candidate extends TokenClassificationEntity {
  priority: number;
}

const DETECTOR_RULES: DetectorRule[] = [
  {
    label: "PERSON",
    pattern:
      /\b(?:Patient|Dr\.?|Doctor)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,2})\b/g,
    captureGroup: 1,
  },
  {
    label: "EMAIL",
    pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
  },
  {
    label: "SSN",
    pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
  },
  {
    label: "PHONE",
    pattern:
      /\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]\d{4}\b/g,
  },
  {
    label: "DATE_OF_BIRTH",
    pattern:
      /\b(?:DOB|Date of birth|Born)[:\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{4})\b/gi,
    captureGroup: 1,
  },
  {
    label: "DATE",
    pattern: /\b(?:19|20)\d{2}-\d{2}-\d{2}\b/g,
  },
];

/** Compact, deterministic detector bundled with the offline extension. */
export const bundledPhiPipeline: TokenClassificationPipeline = (text) => {
  const candidates: Candidate[] = [];

  for (const [priority, rule] of DETECTOR_RULES.entries()) {
    rule.pattern.lastIndex = 0;
    for (const match of text.matchAll(rule.pattern)) {
      const surface = match[rule.captureGroup ?? 0];
      if (surface === undefined || match.index === undefined) {
        continue;
      }
      const fullMatch = match[0];
      const start =
        match.index +
        (rule.captureGroup === undefined ? 0 : fullMatch.indexOf(surface));
      candidates.push({
        entity: `S-${rule.label}`,
        word: surface,
        score: 0.99,
        start,
        end: start + surface.length,
        priority,
      });
    }
  }

  const accepted: Candidate[] = [];
  for (const candidate of candidates.sort(
    (left, right) =>
      left.start - right.start ||
      left.priority - right.priority ||
      right.end - right.start - (left.end - left.start),
  )) {
    if (
      accepted.some(
        (span) => candidate.start < span.end && candidate.end > span.start,
      )
    ) {
      continue;
    }
    accepted.push(candidate);
  }

  return accepted
    .sort((left, right) => left.start - right.start)
    .map(({ priority: _priority, ...entity }, index) => ({
      ...entity,
      index,
    }));
};
