/**
 * Portable identifier checksum validators for the `openmed/validators` subpath.
 *
 * Each function is a port of the named Python reference in
 * `openmed.core.anonymizer.providers.clinical_ids` or `openmed.core.pii_i18n`
 * and returns the same boolean for the shared vectors in
 * `tests/fixtures/parity/identifier_validators_v1.json`. Python's string
 * semantics are kept where a reference relies on them: its whitespace set for
 * `\s` and `str.strip()`, Unicode decimal digits for `\d`, `int()` and
 * `str.isdigit()`, and ASCII-only `[0-9]` classes elsewhere.
 *
 * Validators never throw: anything that is not a string, or does not have the
 * identifier's shape, returns `false`. Nothing here performs I/O or logs.
 */

// Python's `str.isspace()` set, which `\s` and `str.strip()` use for text.
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_SPACE_RE = new RegExp(`[${PY_SPACE}]`, "gu");
const PY_STRIP_RE = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");
const DECIMAL_DIGIT = /^\p{Nd}$/u;

const ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹";
const GURMUKHI_DIGITS = "੦੧੨੩੪੫੬੭੮੯";

const VERHOEFF_D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
  [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
  [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
  [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
  [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
] as const;
const VERHOEFF_P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
  [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
  [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
] as const;

const DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE";
const CHINESE_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2] as const;
const CHINESE_ID_CHECKS = "10X98765432";
const CHINESE_ID_REGION_PREFIXES: ReadonlySet<string> = new Set([
  "11", "12", "13", "14", "15", "21", "22", "23", "31", "32", "33", "34", "35", "36", "37",
  "41", "42", "43", "44", "45", "46", "50", "51", "52", "53", "54", "61", "62", "63", "64",
  "65",
]);

const IBAN_LENGTHS: Readonly<Record<string, number>> = {
  AD: 24, AE: 23, AL: 28, AT: 20, AZ: 28, BA: 20, BE: 16, BG: 22, BH: 22, BR: 29,
  BY: 28, CH: 21, CR: 22, CY: 28, CZ: 24, DE: 22, DK: 18, DO: 28, EE: 20, EG: 29,
  ES: 24, FI: 18, FO: 18, FR: 27, GB: 22, GE: 22, GI: 23, GL: 18, GR: 27, GT: 28,
  HR: 21, HU: 28, IE: 22, IL: 23, IS: 26, IT: 27, JO: 30, KW: 30, KZ: 20, LB: 28,
  LC: 32, LI: 21, LT: 20, LU: 20, LV: 21, MC: 27, MD: 24, ME: 22, MK: 19, MR: 27,
  MT: 31, MU: 30, NL: 18, NO: 15, PK: 24, PL: 28, PS: 29, PT: 25, QA: 29, RO: 24,
  RS: 22, SA: 24, SC: 31, SE: 24, SI: 19, SK: 24, SM: 27, TN: 24, TR: 26, UA: 29,
  VA: 22, VG: 24, XK: 20,
};

function pyStrip(text: string): string {
  return text.replace(PY_STRIP_RE, "");
}

/** `re.sub(r"[^0-9]", "", text)`: ASCII digits only. */
function asciiDigits(text: string): string {
  return text.replace(/[^0-9]/g, "");
}

/**
 * The value of a Unicode decimal digit (`\p{Nd}`), as Python's `int()` reads
 * it. Every Nd run is a sequence of complete 0-9 blocks, so the value is the
 * position of the code point within its run, modulo 10.
 */
function decimalValue(char: string): number {
  let codePoint = char.codePointAt(0)!;
  let offset = 0;
  while (codePoint > 0 && DECIMAL_DIGIT.test(String.fromCodePoint(codePoint - 1))) {
    codePoint -= 1;
    offset += 1;
  }
  return offset % 10;
}

/** Python `int()` of a string of Unicode decimal digits, as a BigInt. */
function decimalBigInt(digits: string): bigint {
  let value = 0n;
  for (const char of digits) {
    value = value * 10n + BigInt(decimalValue(char));
  }
  return value;
}

function luhnCheckDigit(digits: readonly number[]): number {
  let total = 0;
  for (let index = 0; index < digits.length; index += 1) {
    let digit = digits[digits.length - 1 - index]!;
    if (index % 2 === 0) {
      digit *= 2;
      if (digit > 9) {
        digit -= 9;
      }
    }
    total += digit;
  }
  return (10 - (total % 10)) % 10;
}

function toNumbers(digits: string): number[] {
  return Array.from(digits, (digit) => digit.charCodeAt(0) - 48);
}

/** Port of `clinical_ids.validate_luhn`: at least 13 digits, Luhn check. */
export function validateLuhn(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const digits = asciiDigits(text);
  if (digits.length < 13) {
    return false;
  }
  const numbers = toNumbers(digits);
  return luhnCheckDigit(numbers.slice(0, -1)) === numbers.at(-1);
}

/** Port of `clinical_ids.validate_ssn`: nine digits, no never-issued parts. */
export function validateSsn(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const digits = asciiDigits(text);
  if (digits.length !== 9) {
    return false;
  }
  const area = digits.slice(0, 3);
  if (area === "000" || area === "666" || area[0] === "9") {
    return false;
  }
  return digits.slice(3, 5) !== "00" && digits.slice(5, 9) !== "0000";
}

/** Port of `clinical_ids.validate_npi`: ten digits, Luhn over `80840` + body. */
export function validateNpi(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const digits = asciiDigits(text);
  if (digits.length !== 10) {
    return false;
  }
  const numbers = toNumbers(digits);
  return luhnCheckDigit([8, 0, 8, 4, 0, ...numbers.slice(0, -1)]) === numbers.at(-1);
}

/** Port of `clinical_ids.validate_iban`: ISO 13616 length and mod-97 check. */
export function validateIban(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const cleaned = text.replace(new RegExp(`[${PY_SPACE}-]`, "gu"), "").toUpperCase();
  if (!/^[A-Z]{2}\p{Nd}{2}[A-Z0-9]{11,30}$/u.test(cleaned)) {
    return false;
  }
  const length = [...cleaned].length;
  const expected = Object.hasOwn(IBAN_LENGTHS, cleaned.slice(0, 2))
    ? IBAN_LENGTHS[cleaned.slice(0, 2)]
    : undefined;
  if (expected !== undefined ? length !== expected : length < 15 || length > 34) {
    return false;
  }
  const characters = [...cleaned];
  const rearranged = [...characters.slice(4), ...characters.slice(0, 4)];
  let remainder = 0;
  for (const char of rearranged) {
    const values = DECIMAL_DIGIT.test(char)
      ? String(decimalValue(char))
      : String(char.charCodeAt(0) - 55);
    for (const digit of values) {
      remainder = (remainder * 10 + Number(digit)) % 97;
    }
  }
  return remainder === 1;
}

/** Port of `clinical_ids.validate_canadian_sin`: nine digits, Luhn, no leading 0. */
export function validateCanadianSin(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const candidate = pyStrip(text);
  if (!/^(?:\p{Nd}{9}|\p{Nd}{3}([ -])\p{Nd}{3}\1\p{Nd}{3})$/u.test(candidate)) {
    return false;
  }
  const digits = asciiDigits(candidate);
  if (digits.length !== 9 || digits[0] === "0") {
    return false;
  }
  const numbers = toNumbers(digits);
  return luhnCheckDigit(numbers.slice(0, -1)) === numbers.at(-1);
}

/** Port of `clinical_ids.validate_uk_nhs_number`: ten digits, modulus 11. */
export function validateUkNhsNumber(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const digits = asciiDigits(text);
  if (digits.length !== 10) {
    return false;
  }
  const numbers = toNumbers(digits);
  let total = 0;
  for (let index = 0; index < 9; index += 1) {
    total += (10 - index) * numbers[index]!;
  }
  let check = 11 - (total % 11);
  if (check === 11) {
    check = 0;
  }
  return check !== 10 && numbers[9] === check;
}

function foldDigits(text: string, digits: string): string {
  return Array.from(text, (char) => {
    const index = digits.indexOf(char);
    return index === -1 ? char : String(index % 10);
  }).join("");
}

/**
 * Port of `pii_i18n.validate_aadhaar`: twelve digits (or the 4-4-4 display
 * form) starting 2-9, Verhoeff check. Gurmukhi and Arabic-Indic digits are
 * folded to ASCII first, as in Python.
 */
export function validateAadhaar(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const candidate = foldDigits(foldDigits(pyStrip(text), GURMUKHI_DIGITS), ARABIC_INDIC_DIGITS);
  if (!/^(?:[2-9][0-9]{11}|[2-9][0-9]{3} [0-9]{4} [0-9]{4})$/.test(candidate)) {
    return false;
  }
  const digits = candidate.replaceAll(" ", "");
  let check = 0;
  for (let index = 0; index < digits.length; index += 1) {
    const digit = digits.charCodeAt(digits.length - 1 - index) - 48;
    check = VERHOEFF_D[check]![VERHOEFF_P[index % 8]![digit]!]!;
  }
  return check === 0;
}

export interface ChineseResidentIdOptions {
  /** The date a birth date may not be after; defaults to today (local time). */
  today?: Date;
}

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function isCalendarDate(year: number, month: number, day: number): boolean {
  const lengths = [31, isLeapYear(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= lengths[month - 1]!;
}

/**
 * Port of `pii_i18n.validate_chinese_resident_id`: eighteen characters with a
 * province prefix, a real birth date not after today, a non-zero sequence and
 * the MOD 11-2 check character (`X` in any case).
 */
export function validateChineseResidentId(
  text: string,
  options: ChineseResidentIdOptions = {},
): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const cleaned = text.toUpperCase();
  if (!/^[0-9]{17}[0-9X]$/.test(cleaned)) {
    return false;
  }
  const prefecture = Number(cleaned.slice(2, 4));
  if (
    !CHINESE_ID_REGION_PREFIXES.has(cleaned.slice(0, 2)) ||
    (prefecture > 70 && prefecture !== 90) ||
    prefecture === 0 ||
    cleaned.slice(4, 6) === "00"
  ) {
    return false;
  }
  const year = Number(cleaned.slice(6, 10));
  const month = Number(cleaned.slice(10, 12));
  const day = Number(cleaned.slice(12, 14));
  if (!isCalendarDate(year, month, day)) {
    return false;
  }
  const today = options.today ?? new Date();
  const todayKey = today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate();
  if (year * 10000 + month * 100 + day > todayKey || cleaned.slice(14, 17) === "000") {
    return false;
  }
  let total = 0;
  for (let index = 0; index < 17; index += 1) {
    total += (cleaned.charCodeAt(index) - 48) * CHINESE_ID_WEIGHTS[index]!;
  }
  return cleaned[17] === CHINESE_ID_CHECKS[total % 11];
}

/** Port of `pii_i18n.validate_dutch_bsn`: 8 or 9 digits, elfproef. */
export function validateDutchBsn(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  let digits = asciiDigits(text);
  if (digits.length !== 8 && digits.length !== 9) {
    return false;
  }
  digits = digits.padStart(9, "0");
  if (digits === "000000000") {
    return false;
  }
  const weights = [9, 8, 7, 6, 5, 4, 3, 2, -1];
  const checksum = toNumbers(digits).reduce((sum, digit, index) => sum + digit * weights[index]!, 0);
  return checksum % 11 === 0;
}

/**
 * Port of `pii_i18n.validate_french_nir`: fifteen characters starting 1 or 2,
 * Corsican `2A`/`2B` folded to `19`/`18`, key = 97 - (body mod 97).
 */
export function validateFrenchNir(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const cleaned = [...text.replace(new RegExp(`[${PY_SPACE}.-]`, "gu"), "").toUpperCase()];
  if (cleaned.length !== 15 || (cleaned[0] !== "1" && cleaned[0] !== "2")) {
    return false;
  }
  // Python's int() reads the two-character key, which may carry a "+".
  const keyText = cleaned.slice(13, 15).join("");
  const keyMatch = /^\+?(\p{Nd}+)$/u.exec(keyText);
  if (keyMatch === null) {
    return false;
  }
  let body = cleaned.slice(0, 13).join("");
  if (body.includes("A") || body.includes("B")) {
    if (!/^[12]\p{Nd}{4}2[AB]\p{Nd}{6}$/u.test(body)) {
      return false;
    }
    body = body.replaceAll("2A", "19").replaceAll("2B", "18");
  } else if (!/^\p{Nd}+$/u.test(body)) {
    return false;
  }
  const number = decimalBigInt(body);
  return decimalBigInt(keyMatch[1]!) === 97n - (number % 97n);
}

/** Port of `pii_i18n.validate_spanish_dni`: eight digits and the mod-23 letter. */
export function validateSpanishDni(text: string): boolean {
  if (typeof text !== "string") {
    return false;
  }
  const cleaned = text.replace(PY_SPACE_RE, "").toUpperCase();
  if ([...cleaned].length !== 9) {
    return false;
  }
  const match = /^(\p{Nd}{8})([A-Z])$/u.exec(cleaned);
  if (match === null) {
    return false;
  }
  return match[2] === DNI_LETTERS[Number(decimalBigInt(match[1]!) % 23n)];
}

/** A portable identifier validator. */
export type IdentifierValidator = (text: string) => boolean;

/** Every portable validator, keyed by the name of its Python reference. */
export const IDENTIFIER_VALIDATORS: Readonly<Record<string, IdentifierValidator>> = Object.freeze({
  validate_aadhaar: validateAadhaar,
  validate_canadian_sin: validateCanadianSin,
  validate_chinese_resident_id: (text: string) => validateChineseResidentId(text),
  validate_dutch_bsn: validateDutchBsn,
  validate_french_nir: validateFrenchNir,
  validate_iban: validateIban,
  validate_luhn: validateLuhn,
  validate_npi: validateNpi,
  validate_spanish_dni: validateSpanishDni,
  validate_ssn: validateSsn,
  validate_uk_nhs_number: validateUkNhsNumber,
});
