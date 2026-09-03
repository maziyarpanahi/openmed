import type { TokenClassificationEntity } from "./types";

const CONTINUATION_PREFIX = "##";
const WORD_START_MARKERS = /^[▁Ġ\s]+/;
const SPECIAL_TOKEN_WORDS = new Set([
  "[CLS]",
  "[SEP]",
  "[PAD]",
  "[MASK]",
  "<s>",
  "</s>",
  "<pad>",
  "<mask>",
  "<cls>",
  "<sep>",
  "<bos>",
  "<eos>",
]);
const CONTIGUOUS_SEARCH_WINDOW = 16;

interface NormalizedText {
  chars: string;
  toOriginal: number[];
  toOriginalEnd: number[];
  fromOriginal: number[];
}

/**
 * Attach character offsets to token-classification output that lacks them.
 *
 * Transformers.js emits `{ entity, score, index, word }` without `start`/`end`.
 * Tokens that already carry finite offsets are kept as-is and advance the
 * cursor; the remaining tokens are located sequentially in `text`. Matching is
 * case- and accent-insensitive so lowercasing tokenizers (BERT-style) still
 * align, and it tolerates WordPiece (`##`), SentencePiece (`▁`) and byte-level
 * BPE (`Ġ`) markers. Unalignable tokens fail with a content-free error rather
 * than silently returning incomplete redaction spans. Supply explicit source
 * offsets for unknown tokens or tokenizers whose words cannot be aligned.
 */
export function alignTokenOffsets(
  text: string,
  tokens: TokenClassificationEntity[],
): TokenClassificationEntity[] {
  const normalized = normalizeText(text);
  const aligned: TokenClassificationEntity[] = [];
  let cursor = 0;
  let previousIndex: number | null = null;

  for (const token of tokens) {
    const index = typeof token.index === "number" ? token.index : undefined;
    if (hasFiniteOffsets(token)) {
      aligned.push(token);
      const end = Math.max(0, Math.min(Math.trunc(Number(token.end)), text.length));
      cursor = Math.max(cursor, normalized.fromOriginal[end] ?? cursor);
      previousIndex = index ?? previousIndex;
      continue;
    }

    const word = token.word ?? "";
    if (SPECIAL_TOKEN_WORDS.has(word.trim())) {
      previousIndex = index ?? previousIndex;
      continue;
    }
    const continuation = word.startsWith(CONTINUATION_PREFIX);
    const piece = normalizeString(stripTokenMarkers(word));
    if (!piece) {
      throw alignmentError();
    }

    // Tokens dropped upstream (ignored labels, [UNK]) leave index gaps, in
    // which case the next token may sit anywhere after the cursor.
    const contiguous =
      index === undefined ||
      (previousIndex === null ? index <= 1 : index === previousIndex + 1);
    previousIndex = index ?? previousIndex;

    const match = locatePiece(normalized.chars, piece, cursor, {
      continuation,
      contiguous,
    });
    const start = match === null ? undefined : normalized.toOriginal[match];
    const end =
      match === null ? undefined : normalized.toOriginalEnd[match + piece.length - 1];
    if (match === null || start === undefined || end === undefined) {
      throw alignmentError();
    }
    aligned.push({ ...token, start, end });
    cursor = match + piece.length;
  }

  return aligned;
}

function alignmentError(): Error {
  return new Error("Token offset alignment failed; provide source offsets.");
}

function hasFiniteOffsets(token: TokenClassificationEntity): boolean {
  const start = Number(token.start);
  const end = Number(token.end);
  return Number.isFinite(start) && Number.isFinite(end) && end > start;
}

function stripTokenMarkers(word: string): string {
  let piece = word;
  if (piece.startsWith(CONTINUATION_PREFIX)) {
    piece = piece.slice(CONTINUATION_PREFIX.length);
  }
  return piece.replace(WORD_START_MARKERS, "").trim();
}

function normalizeString(value: string): string {
  return value.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
}

function normalizeText(text: string): NormalizedText {
  const chars: string[] = [];
  const toOriginal: number[] = [];
  const toOriginalEnd: number[] = [];
  const fromOriginal: number[] = new Array<number>(text.length + 1);
  let offset = 0;
  // Iterate code points for Unicode case folding, but return JS UTF-16 offsets.
  for (const character of text) {
    const end = offset + character.length;
    for (let unit = offset; unit < end; unit += 1) {
      fromOriginal[unit] = chars.length;
    }
    const normalizedChar = normalizeString(character);
    // Include decomposed marks in the preceding character's source span.
    if (!normalizedChar && toOriginalEnd.length > 0) {
      toOriginalEnd[toOriginalEnd.length - 1] = end;
    }
    for (let unit = 0; unit < normalizedChar.length; unit += 1) {
      chars.push(normalizedChar[unit] ?? "");
      toOriginal.push(offset);
      toOriginalEnd.push(end);
    }
    offset = end;
  }
  fromOriginal[text.length] = chars.length;
  return { chars: chars.join(""), toOriginal, toOriginalEnd, fromOriginal };
}

function locatePiece(
  chars: string,
  piece: string,
  cursor: number,
  options: { continuation: boolean; contiguous: boolean },
): number | null {
  let position = cursor;
  while (position < chars.length && /\s/.test(chars[position] ?? "")) {
    position += 1;
  }
  if (chars.startsWith(piece, position)) {
    return position;
  }

  const limit = options.contiguous
    ? Math.min(chars.length, position + CONTIGUOUS_SEARCH_WINDOW)
    : chars.length;
  let from = position;
  while (from <= limit) {
    const found = chars.indexOf(piece, from);
    if (found === -1 || found > limit) {
      return null;
    }
    if (options.continuation || atWordBoundary(chars, found, piece)) {
      return found;
    }
    from = found + 1;
  }
  return null;
}

function atWordBoundary(chars: string, position: number, piece: string): boolean {
  if (position === 0) {
    return true;
  }
  return !isAlphanumeric(chars[position - 1]) || !isAlphanumeric(piece[0]);
}

function isAlphanumeric(value: string | undefined): boolean {
  return value !== undefined && /[\p{L}\p{N}]/u.test(value);
}
