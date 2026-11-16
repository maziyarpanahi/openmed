/**
 * Strict JSON reader and canonical writer for multimodal packets.
 *
 * `JSON.parse` cannot be used for admission: it silently keeps the last of
 * duplicate keys and rounds integers above 2^53. This reader follows Python's
 * `json.loads` grammar (RFC 8259 plus lone surrogate escapes), refuses
 * duplicate keys, `NaN`/`Infinity` constants and trailing data, and keeps every
 * number as its source text so integers stay exact as `bigint`.
 *
 * The writer reproduces Python's `json.dumps(sort_keys=True,
 * separators=(",", ":"), ensure_ascii=True)`, which is the canonical form the
 * contract hashes and compares byte for byte.
 */

/** A JSON number kept as its exact source text. */
export class JsonNumber {
  readonly text: string;

  constructor(text: string) {
    this.text = text;
  }

  /** Whether the literal is an integer (Python `int`), not a float. */
  get isInteger(): boolean {
    return INTEGER_LITERAL.test(this.text);
  }
}

export type JsonValue =
  | null
  | boolean
  | string
  | JsonNumber
  | readonly JsonValue[]
  | JsonObject;

export type JsonObject = ReadonlyMap<string, JsonValue>;

/** Thrown for any input that is not strict JSON. Carries no input text. */
export class JsonSyntaxError extends Error {
  constructor() {
    super("invalid_json");
    this.name = "JsonSyntaxError";
  }
}

const INTEGER_LITERAL = /^-?(?:0|[1-9][0-9]*)$/;
const NUMBER_LITERAL = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?/y;
const HEX4 = /^[0-9a-fA-F]{4}$/;
// Python's recursion limit makes deeper input fail; this bound keeps the
// reader well inside any JavaScript stack while accepting any real packet.
const MAX_DEPTH = 512;

const SHORT_ESCAPES: Readonly<Record<string, string>> = {
  '"': '"',
  "\\": "\\",
  "/": "/",
  b: "\b",
  f: "\f",
  n: "\n",
  r: "\r",
  t: "\t",
};

export function isJsonObject(value: JsonValue | undefined): value is JsonObject {
  return value instanceof Map;
}

export function isJsonArray(
  value: JsonValue | undefined,
): value is readonly JsonValue[] {
  return Array.isArray(value);
}

/**
 * Parse strict JSON text.
 *
 * @throws JsonSyntaxError for malformed JSON, duplicate keys, non-finite
 *   constants, excessive nesting or trailing data.
 */
export function parseStrictJson(text: string): JsonValue {
  let index = 0;

  const fail = (): never => {
    throw new JsonSyntaxError();
  };

  const skipWhitespace = () => {
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (code !== 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) {
        return;
      }
      index += 1;
    }
  };

  const parseString = (): string => {
    // The opening quote is at text[index].
    index += 1;
    let result = "";
    let chunkStart = index;
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (code === 0x22) {
        result += text.slice(chunkStart, index);
        index += 1;
        return result;
      }
      if (code < 0x20) {
        fail();
      }
      if (code === 0x5c) {
        result += text.slice(chunkStart, index);
        const escape = text[index + 1];
        if (escape === undefined) {
          fail();
        } else if (escape === "u") {
          const digits = text.slice(index + 2, index + 6);
          if (!HEX4.test(digits)) {
            fail();
          }
          result += String.fromCharCode(Number.parseInt(digits, 16));
          index += 6;
        } else {
          const replacement = SHORT_ESCAPES[escape];
          if (replacement === undefined) {
            fail();
          }
          result += replacement;
          index += 2;
        }
        chunkStart = index;
        continue;
      }
      index += 1;
    }
    return fail();
  };

  const parseValue = (depth: number): JsonValue => {
    if (depth > MAX_DEPTH) {
      fail();
    }
    skipWhitespace();
    const char = text[index];
    if (char === "{") {
      index += 1;
      const object = new Map<string, JsonValue>();
      skipWhitespace();
      if (text[index] === "}") {
        index += 1;
        return object;
      }
      for (;;) {
        skipWhitespace();
        if (text[index] !== '"') {
          fail();
        }
        const key = parseString();
        skipWhitespace();
        if (text[index] !== ":") {
          fail();
        }
        index += 1;
        const value = parseValue(depth + 1);
        if (object.has(key)) {
          fail();
        }
        object.set(key, value);
        skipWhitespace();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "}") {
          index += 1;
          return object;
        }
        fail();
      }
    }
    if (char === "[") {
      index += 1;
      const array: JsonValue[] = [];
      skipWhitespace();
      if (text[index] === "]") {
        index += 1;
        return array;
      }
      for (;;) {
        array.push(parseValue(depth + 1));
        skipWhitespace();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "]") {
          index += 1;
          return array;
        }
        fail();
      }
    }
    if (char === '"') {
      return parseString();
    }
    if (text.startsWith("true", index)) {
      index += 4;
      return true;
    }
    if (text.startsWith("false", index)) {
      index += 5;
      return false;
    }
    if (text.startsWith("null", index)) {
      index += 4;
      return null;
    }
    NUMBER_LITERAL.lastIndex = index;
    const match = NUMBER_LITERAL.exec(text);
    if (match === null) {
      // Includes NaN, Infinity and -Infinity, which Python only accepts
      // through parse_constant and the contract refuses.
      return fail();
    }
    index += match[0].length;
    return new JsonNumber(match[0]);
  };

  const value = parseValue(0);
  skipWhitespace();
  if (index !== text.length) {
    fail();
  }
  return value;
}

function compareCodePoints(left: string, right: string): number {
  const a = [...left];
  const b = [...right];
  const length = Math.min(a.length, b.length);
  for (let position = 0; position < length; position += 1) {
    const difference =
      (a[position] as string).codePointAt(0)! - (b[position] as string).codePointAt(0)!;
    if (difference !== 0) {
      return difference;
    }
  }
  return a.length - b.length;
}

function escapeString(value: string): string {
  let result = '"';
  for (let position = 0; position < value.length; position += 1) {
    const code = value.charCodeAt(position);
    if (code === 0x22) {
      result += '\\"';
    } else if (code === 0x5c) {
      result += "\\\\";
    } else if (code === 0x0a) {
      result += "\\n";
    } else if (code === 0x0d) {
      result += "\\r";
    } else if (code === 0x09) {
      result += "\\t";
    } else if (code === 0x08) {
      result += "\\b";
    } else if (code === 0x0c) {
      result += "\\f";
    } else if (code >= 0x20 && code <= 0x7e) {
      result += value[position];
    } else {
      // ensure_ascii escapes every other UTF-16 code unit, including each half
      // of a surrogate pair, as lowercase \uXXXX.
      result += `\\u${code.toString(16).padStart(4, "0")}`;
    }
  }
  return `${result}"`;
}

/**
 * Serialize a value in the contract's canonical form.
 *
 * @throws JsonSyntaxError for a non-integer number: no admitted packet field
 *   holds one, so it never reaches the canonical form.
 */
export function canonicalJson(value: JsonValue): string {
  if (value === null) {
    return "null";
  }
  if (value === true) {
    return "true";
  }
  if (value === false) {
    return "false";
  }
  if (typeof value === "string") {
    return escapeString(value);
  }
  if (value instanceof JsonNumber) {
    if (!value.isInteger) {
      throw new JsonSyntaxError();
    }
    return BigInt(value.text).toString();
  }
  if (isJsonArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const keys = [...value.keys()].sort(compareCodePoints);
  return `{${keys
    .map((key) => `${escapeString(key)}:${canonicalJson(value.get(key) as JsonValue)}`)
    .join(",")}}`;
}
