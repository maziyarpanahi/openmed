import {
  CANONICAL_LABELS,
  type CanonicalLabel,
  type DecodedEntitySpan,
  type PolicyLabel,
  type TokenClassificationEntity,
  type TokenClassificationOutput,
} from "./types";

const CANONICAL_LABEL_SET = new Set<string>(CANONICAL_LABELS);

const POLICY_LABEL_BY_CANONICAL: Record<CanonicalLabel, PolicyLabel> = {
  ACCOUNT_NUMBER: "DIRECT_IDENTIFIER",
  AGE: "QUASI_IDENTIFIER",
  AMOUNT: "QUASI_IDENTIFIER",
  API_KEY: "DIRECT_IDENTIFIER",
  BIC: "DIRECT_IDENTIFIER",
  BITCOIN_ADDRESS: "DIRECT_IDENTIFIER",
  BUILDING_NUMBER: "DIRECT_IDENTIFIER",
  CREDIT_CARD: "DIRECT_IDENTIFIER",
  CREDIT_CARD_ISSUER: "QUASI_IDENTIFIER",
  CURRENCY: "QUASI_IDENTIFIER",
  CVV: "DIRECT_IDENTIFIER",
  DATE: "QUASI_IDENTIFIER",
  DATE_OF_BIRTH: "DIRECT_IDENTIFIER",
  EMAIL: "DIRECT_IDENTIFIER",
  ETHEREUM_ADDRESS: "DIRECT_IDENTIFIER",
  EYE_COLOR: "QUASI_IDENTIFIER",
  FIRST_NAME: "DIRECT_IDENTIFIER",
  GENDER: "QUASI_IDENTIFIER",
  GPS_COORDINATES: "DIRECT_IDENTIFIER",
  HEIGHT: "QUASI_IDENTIFIER",
  IBAN: "DIRECT_IDENTIFIER",
  ID_NUM: "DIRECT_IDENTIFIER",
  IMEI: "DIRECT_IDENTIFIER",
  IP_ADDRESS: "DIRECT_IDENTIFIER",
  JOB_DEPARTMENT: "QUASI_IDENTIFIER",
  JOB_TITLE: "QUASI_IDENTIFIER",
  LAST_NAME: "DIRECT_IDENTIFIER",
  LITECOIN_ADDRESS: "DIRECT_IDENTIFIER",
  LOCATION: "QUASI_IDENTIFIER",
  MAC_ADDRESS: "DIRECT_IDENTIFIER",
  MASKED_NUMBER: "DIRECT_IDENTIFIER",
  MIDDLE_NAME: "DIRECT_IDENTIFIER",
  OCCUPATION: "QUASI_IDENTIFIER",
  ORDINAL_DIRECTION: "QUASI_IDENTIFIER",
  ORGANIZATION: "QUASI_IDENTIFIER",
  OTHER: "CLINICAL_CONCEPT",
  PASSWORD: "DIRECT_IDENTIFIER",
  PERSON: "DIRECT_IDENTIFIER",
  PHONE: "DIRECT_IDENTIFIER",
  PIN: "DIRECT_IDENTIFIER",
  PREFIX: "DIRECT_IDENTIFIER",
  SSN: "DIRECT_IDENTIFIER",
  STREET_ADDRESS: "DIRECT_IDENTIFIER",
  TIME: "QUASI_IDENTIFIER",
  URL: "DIRECT_IDENTIFIER",
  USER_AGENT: "DIRECT_IDENTIFIER",
  USERNAME: "DIRECT_IDENTIFIER",
  VEHICLE_REGISTRATION: "DIRECT_IDENTIFIER",
  VIN: "DIRECT_IDENTIFIER",
  ZIPCODE: "QUASI_IDENTIFIER",
};

const ALIAS_MAP: Record<string, CanonicalLabel> = {
  accountname: "ACCOUNT_NUMBER",
  accountnumber: "ACCOUNT_NUMBER",
  address: "STREET_ADDRESS",
  aadhaar: "ID_NUM",
  age: "AGE",
  amount: "AMOUNT",
  apikey: "API_KEY",
  bankaccount: "ACCOUNT_NUMBER",
  bic: "BIC",
  birthdate: "DATE_OF_BIRTH",
  bitcoinaddress: "BITCOIN_ADDRESS",
  buildingnumber: "BUILDING_NUMBER",
  city: "LOCATION",
  cnpj: "ID_NUM",
  codicefiscale: "ID_NUM",
  company: "ORGANIZATION",
  country: "LOCATION",
  county: "LOCATION",
  cpf: "ID_NUM",
  creditcard: "CREDIT_CARD",
  creditcardissuer: "CREDIT_CARD_ISSUER",
  creditcardnumber: "CREDIT_CARD",
  creditdebitcard: "CREDIT_CARD",
  currency: "CURRENCY",
  currencycode: "CURRENCY",
  currencyname: "CURRENCY",
  currencysymbol: "CURRENCY",
  cvv: "CVV",
  date: "DATE",
  dateofbirth: "DATE_OF_BIRTH",
  department: "JOB_DEPARTMENT",
  dni: "ID_NUM",
  dob: "DATE_OF_BIRTH",
  doctor: "PERSON",
  email: "EMAIL",
  emailaddress: "EMAIL",
  employer: "ORGANIZATION",
  ethereumaddress: "ETHEREUM_ADDRESS",
  eyecolor: "EYE_COLOR",
  familyname: "LAST_NAME",
  fax: "PHONE",
  firstname: "FIRST_NAME",
  fullname: "PERSON",
  gender: "GENDER",
  givenname: "FIRST_NAME",
  gps: "GPS_COORDINATES",
  gpscoordinates: "GPS_COORDINATES",
  height: "HEIGHT",
  iban: "IBAN",
  id: "ID_NUM",
  identifier: "ID_NUM",
  idnum: "ID_NUM",
  imei: "IMEI",
  ip: "IP_ADDRESS",
  ipaddress: "IP_ADDRESS",
  jobdepartment: "JOB_DEPARTMENT",
  jobtitle: "JOB_TITLE",
  lastname: "LAST_NAME",
  licenseplate: "VEHICLE_REGISTRATION",
  litecoinaddress: "LITECOIN_ADDRESS",
  location: "LOCATION",
  macaddress: "MAC_ADDRESS",
  maskednumber: "MASKED_NUMBER",
  medicalrecordnumber: "ID_NUM",
  middlename: "MIDDLE_NAME",
  mrn: "ID_NUM",
  name: "PERSON",
  nationalid: "ID_NUM",
  nhs: "ID_NUM",
  nhsnumber: "ID_NUM",
  nie: "ID_NUM",
  npi: "ID_NUM",
  occupation: "OCCUPATION",
  ordinaldirection: "ORDINAL_DIRECTION",
  organization: "ORGANIZATION",
  password: "PASSWORD",
  patient: "PERSON",
  person: "PERSON",
  personalurl: "URL",
  phone: "PHONE",
  phonenumber: "PHONE",
  pin: "PIN",
  place: "LOCATION",
  postalcode: "ZIPCODE",
  postcode: "ZIPCODE",
  prefix: "PREFIX",
  profession: "OCCUPATION",
  region: "LOCATION",
  secondaryaddress: "STREET_ADDRESS",
  sex: "GENDER",
  socialsecuritynumber: "SSN",
  ssn: "SSN",
  state: "LOCATION",
  steuerid: "ID_NUM",
  street: "STREET_ADDRESS",
  streetaddress: "STREET_ADDRESS",
  surname: "LAST_NAME",
  swift: "BIC",
  telephone: "PHONE",
  teudatzehut: "ID_NUM",
  time: "TIME",
  title: "PREFIX",
  tz: "ID_NUM",
  url: "URL",
  urlpersonal: "URL",
  useragent: "USER_AGENT",
  userhandle: "USERNAME",
  username: "USERNAME",
  vin: "VIN",
  vrm: "VEHICLE_REGISTRATION",
  website: "URL",
  zip: "ZIPCODE",
  zipcode: "ZIPCODE",
};

type BoundaryTag = "B" | "I" | "E" | "S" | null;

interface PreparedToken {
  label: string;
  boundary: BoundaryTag;
  baseLabel: string;
  canonicalLabel: CanonicalLabel;
  policyLabel: PolicyLabel;
  start: number;
  end: number;
  score: number | null;
  index: number;
}

interface ActiveSpan {
  baseLabel: string;
  canonicalLabel: CanonicalLabel;
  policyLabel: PolicyLabel;
  start: number;
  end: number;
  scores: number[];
  tokenCount: number;
}

export function normalizeLabel(label: string): CanonicalLabel {
  if (!label) {
    return "OTHER";
  }
  const key = labelKey(label);
  if (!key) {
    return "OTHER";
  }
  const alias = ALIAS_MAP[key];
  if (alias) {
    return alias;
  }
  const upper = label
    .toUpperCase()
    .replace(/[-\s]+/g, "_")
    .replace(/[^A-Z0-9_]/g, "");
  return CANONICAL_LABEL_SET.has(upper) ? (upper as CanonicalLabel) : "OTHER";
}

export function policyLabelFor(label: string): PolicyLabel {
  return POLICY_LABEL_BY_CANONICAL[normalizeLabel(label)];
}

export function decodeBioTokenSpans(
  text: string,
  output: TokenClassificationOutput,
  options: { threshold?: number } = {},
): DecodedEntitySpan[] {
  const threshold = options.threshold ?? 0;
  const tokens = flattenTokenClassificationOutput(output)
    .map((token, fallbackIndex) => prepareToken(token, fallbackIndex))
    .filter((token): token is PreparedToken => token !== null)
    .sort((left, right) => left.index - right.index || left.start - right.start);

  const spans: DecodedEntitySpan[] = [];
  let current: ActiveSpan | null = null;
  let previousIndex: number | null = null;

  for (const token of tokens) {
    if (previousIndex !== null && token.index !== previousIndex + 1) {
      if (current !== null) {
        pushDecodedSpan(spans, current, text, threshold);
        current = null;
      }
    }

    if (token.boundary === null) {
      if (current !== null) {
        pushDecodedSpan(spans, current, text, threshold);
        current = null;
      }
      previousIndex = token.index;
      continue;
    }

    if (token.boundary === "S") {
      if (current !== null) {
        pushDecodedSpan(spans, current, text, threshold);
      }
      pushDecodedSpan(spans, activeSpanFromToken(token), text, threshold);
      current = null;
    } else if (token.boundary === "B") {
      if (current !== null) {
        pushDecodedSpan(spans, current, text, threshold);
      }
      current = activeSpanFromToken(token);
    } else if (
      token.boundary === "I" &&
      current !== null &&
      current.baseLabel === token.baseLabel
    ) {
      appendTokenToActiveSpan(current, token);
    } else if (
      token.boundary === "E" &&
      current !== null &&
      current.baseLabel === token.baseLabel
    ) {
      appendTokenToActiveSpan(current, token);
      pushDecodedSpan(spans, current, text, threshold);
      current = null;
    } else {
      if (current !== null) {
        pushDecodedSpan(spans, current, text, threshold);
      }
      current = activeSpanFromToken(token);
      if (token.boundary === "E") {
        pushDecodedSpan(spans, current, text, threshold);
        current = null;
      }
    }

    previousIndex = token.index;
  }

  if (current !== null) {
    pushDecodedSpan(spans, current, text, threshold);
  }
  return mergeAdjacentSpans(spans, text);
}

export function flattenTokenClassificationOutput(
  output: TokenClassificationOutput,
): TokenClassificationEntity[] {
  if (output.length === 0) {
    return [];
  }
  const first = output[0];
  if (Array.isArray(first)) {
    return (output as TokenClassificationEntity[][]).flat();
  }
  return output as TokenClassificationEntity[];
}

export function trimSpanWhitespace(
  start: number,
  end: number,
  text: string,
): { start: number; end: number } {
  let nextStart = Math.max(0, Math.min(start, text.length));
  let nextEnd = Math.max(nextStart, Math.min(end, text.length));
  while (nextStart < nextEnd && /\s/.test(text[nextStart] ?? "")) {
    nextStart += 1;
  }
  while (nextEnd > nextStart && /\s/.test(text[nextEnd - 1] ?? "")) {
    nextEnd -= 1;
  }
  return { start: nextStart, end: nextEnd };
}

export function refinePrivacyFilterSpan(
  label: string,
  start: number,
  end: number,
  text: string,
): { start: number; end: number } {
  const trimmed = trimSpanWhitespace(start, end, text);
  const spanText = text.slice(trimmed.start, trimmed.end);
  const normalized = label.toLowerCase();
  const structuredPatterns: Array<[string, RegExp]> = [
    ["email", /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/],
    ["url", /\b(?:https?:\/\/|www\.)[^\s,;)\]]+/],
    ["phone", /(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}/],
  ];

  for (const [hint, pattern] of structuredPatterns) {
    if (!normalized.includes(hint)) {
      continue;
    }
    const match = pattern.exec(spanText);
    if (match?.index !== undefined) {
      return {
        start: trimmed.start + match.index,
        end: trimmed.start + match.index + match[0].length,
      };
    }
  }

  let nextEnd = trimmed.end;
  const lowerSpan = spanText.toLowerCase();
  for (const suffix of [" and", " or"]) {
    if (lowerSpan.endsWith(suffix)) {
      nextEnd -= suffix.length;
      break;
    }
  }
  return trimSpanWhitespace(trimmed.start, nextEnd, text);
}

function prepareToken(
  token: TokenClassificationEntity,
  fallbackIndex: number,
): PreparedToken | null {
  const label = token.entity_group ?? token.entity ?? "";
  const start = Number(token.start);
  const end = Number(token.end);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null;
  }
  const parsed = parseBoundaryLabel(label);
  const canonicalLabel = normalizeLabel(parsed.baseLabel);
  return {
    label,
    boundary: parsed.boundary,
    baseLabel: parsed.baseLabel,
    canonicalLabel,
    policyLabel: POLICY_LABEL_BY_CANONICAL[canonicalLabel],
    start,
    end,
    score: token.score === undefined ? null : Number(token.score),
    index: token.index ?? fallbackIndex,
  };
}

function activeSpanFromToken(token: PreparedToken): ActiveSpan {
  return {
    baseLabel: token.baseLabel,
    canonicalLabel: token.canonicalLabel,
    policyLabel: token.policyLabel,
    start: token.start,
    end: token.end,
    scores: token.score === null ? [] : [token.score],
    tokenCount: 1,
  };
}

function appendTokenToActiveSpan(span: ActiveSpan, token: PreparedToken): void {
  span.end = token.end;
  if (token.score !== null) {
    span.scores.push(token.score);
  }
  span.tokenCount += 1;
}

function pushDecodedSpan(
  spans: DecodedEntitySpan[],
  span: ActiveSpan,
  text: string,
  threshold: number,
): void {
  const refined = refinePrivacyFilterSpan(span.baseLabel, span.start, span.end, text);
  if (refined.end <= refined.start) {
    return;
  }
  const score =
    span.scores.length > 0
      ? span.scores.reduce((total, value) => total + value, 0) /
        span.scores.length
      : null;
  if (score !== null && score < threshold) {
    return;
  }
  spans.push({
    entity_type: span.baseLabel,
    canonical_label: span.canonicalLabel,
    policy_label: span.policyLabel,
    start: refined.start,
    end: refined.end,
    score,
    token_count: span.tokenCount,
  });
}

function parseBoundaryLabel(label: string): {
  boundary: BoundaryTag;
  baseLabel: string;
} {
  if (label === "O") {
    return { boundary: null, baseLabel: "O" };
  }
  if (/^[BIES]-/.test(label)) {
    return {
      boundary: label[0] as Exclude<BoundaryTag, null>,
      baseLabel: label.slice(2),
    };
  }
  return { boundary: "B", baseLabel: label };
}

function labelKey(label: string): string {
  const stripped = label.trim().replace(/^[BIES]-/, "");
  return stripped.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function mergeAdjacentSpans(
  spans: DecodedEntitySpan[],
  text: string,
): DecodedEntitySpan[] {
  const merged: DecodedEntitySpan[] = [];
  for (const span of spans) {
    const previous = merged.at(-1);
    if (
      previous &&
      previous.entity_type === span.entity_type &&
      previous.end <= span.start &&
      text.slice(previous.end, span.start).trim() === ""
    ) {
      previous.end = span.end;
      previous.token_count += span.token_count;
      if (previous.score !== null && span.score !== null) {
        previous.score =
          (previous.score + span.score) / 2;
      } else {
        previous.score = previous.score ?? span.score;
      }
      continue;
    }
    merged.push({ ...span });
  }
  return merged;
}
