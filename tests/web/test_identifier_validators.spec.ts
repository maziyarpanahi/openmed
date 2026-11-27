// The portable identifier validators of the `openmed/validators` subpath,
// replayed from the shared synthetic vectors in
// tests/fixtures/parity/identifier_validators_v1.json (Python references in
// openmed.core.anonymizer.providers.clinical_ids and openmed.core.pii_i18n)
// and from web edge cases whose expected values a Python test checks.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import * as mainEntry from "../../js/openmedkit-web/src/index";
import * as validators from "../../js/openmedkit-web/src/validators";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "identifier_validators_v1.json");
const EDGES_PATH = join(rootDir, "tests", "web", "fixtures", "identifier_validator_edge_cases.json");
const PACKAGE_DIR = join(rootDir, "js", "openmedkit-web");
const DIST_DIR = join(PACKAGE_DIR, "dist");
const SOURCE_PATH = join(PACKAGE_DIR, "src", "validators.ts");

const NAMED: Record<string, (text: string) => boolean> = {
  validate_aadhaar: validators.validateAadhaar,
  validate_canadian_sin: validators.validateCanadianSin,
  validate_chinese_resident_id: validators.validateChineseResidentId,
  validate_dutch_bsn: validators.validateDutchBsn,
  validate_french_nir: validators.validateFrenchNir,
  validate_iban: validators.validateIban,
  validate_luhn: validators.validateLuhn,
  validate_npi: validators.validateNpi,
  validate_spanish_dni: validators.validateSpanishDni,
  validate_ssn: validators.validateSsn,
  validate_uk_nhs_number: validators.validateUkNhsNumber,
};

interface VectorCase {
  validator: string;
  reference: string;
  valid: string[];
  invalid: string[];
}

interface Vectors {
  version: number;
  fixture: string;
  synthetic: boolean;
  cases: VectorCase[];
}

interface Edges {
  synthetic: boolean;
  cases: { reference: string; value: string; expected: boolean }[];
}

async function vectors(): Promise<Vectors> {
  return JSON.parse(await readFile(VECTORS_PATH, "utf8")) as Vectors;
}

function referenceName(reference: string): string {
  return reference.slice(reference.lastIndexOf(".") + 1);
}

test("the shared vectors are synthetic and name a web port for every reference", async () => {
  const fixture = await vectors();
  assert.equal(fixture.fixture, "identifier_validators_v1");
  assert.equal(fixture.version, 1);
  assert.equal(fixture.synthetic, true);
  assert.deepEqual(
    fixture.cases.map((item) => referenceName(item.reference)).sort(),
    Object.keys(validators.IDENTIFIER_VALIDATORS).sort(),
  );
  assert.deepEqual(Object.keys(NAMED).sort(), Object.keys(validators.IDENTIFIER_VALIDATORS).sort());
});

test("every shared vector matches the Python result", async () => {
  const fixture = await vectors();
  const mismatches: string[] = [];
  let checked = 0;
  for (const item of fixture.cases) {
    const name = referenceName(item.reference);
    for (const [expected, values] of [
      [true, item.valid],
      [false, item.invalid],
    ] as const) {
      values.forEach((value, index) => {
        for (const validate of [validators.IDENTIFIER_VALIDATORS[name]!, NAMED[name]!]) {
          checked += 1;
          if (validate(value) !== expected) {
            mismatches.push(`${item.validator}-${expected ? "valid" : "invalid"}-${index + 1}`);
          }
        }
      });
    }
  }
  assert.deepEqual([...new Set(mismatches)], []);
  assert.ok(checked > 100);
});

test("edge inputs give the Python result, Unicode digits and Python whitespace included", async () => {
  const edges = JSON.parse(await readFile(EDGES_PATH, "utf8")) as Edges;
  assert.equal(edges.synthetic, true);
  const mismatches: string[] = [];
  edges.cases.forEach((item, index) => {
    if (validators.IDENTIFIER_VALIDATORS[item.reference]!(item.value) !== item.expected) {
      mismatches.push(`${item.reference}-${index}`);
    }
  });
  assert.deepEqual(mismatches, []);
  assert.ok(edges.cases.some((item) => item.expected) && edges.cases.some((item) => !item.expected));
});

test("a Chinese resident ID born after the reference date is rejected", async () => {
  const fixture = await vectors();
  const valid = fixture.cases.find((item) => referenceName(item.reference) === "validate_chinese_resident_id")!
    .valid[0]!;
  const birth = valid.slice(6, 14);
  const born = new Date(Number(birth.slice(0, 4)), Number(birth.slice(4, 6)) - 1, Number(birth.slice(6, 8)));
  assert.equal(validators.validateChineseResidentId(valid, { today: born }), true);
  const before = new Date(born);
  before.setDate(before.getDate() - 1);
  assert.equal(validators.validateChineseResidentId(valid, { today: before }), false);
});

test("validators never throw, whatever they are given", () => {
  const sentinel = "Quorble-Zyxwick-0042";
  const inputs: unknown[] = [
    undefined,
    null,
    42,
    4992737877893288n,
    {},
    [],
    { toString: () => sentinel },
    Symbol("synthetic"),
    "",
    sentinel,
    "\u{1F600}".repeat(20),
    "9".repeat(10_000),
  ];
  for (const [name, validate] of Object.entries(validators.IDENTIFIER_VALIDATORS)) {
    for (const input of inputs) {
      let result: unknown;
      assert.doesNotThrow(() => {
        result = validate(input as string);
      }, name);
      assert.equal(typeof result, "boolean", name);
      if (typeof input !== "string") {
        assert.equal(result, false, name);
      }
    }
  }
});

test("the validators ship on the openmed/validators subpath only", async () => {
  const packageJson = JSON.parse(await readFile(join(PACKAGE_DIR, "package.json"), "utf8")) as {
    exports: Record<string, unknown>;
  };
  assert.deepEqual(packageJson.exports["./validators"], {
    import: { types: "./dist/validators.d.ts", default: "./dist/validators.js" },
    require: { types: "./dist/validators.d.cts", default: "./dist/validators.cjs" },
  });
  for (const name of [...Object.keys(validators), "IDENTIFIER_VALIDATORS"]) {
    assert.equal(Object.hasOwn(mainEntry, name), false, name);
  }

  const vectorsDoc = await vectors();
  const sample = vectorsDoc.cases[0]!;
  const esm = await import(pathToFileURL(join(DIST_DIR, "validators.js")).href);
  const cjs = createRequire(import.meta.url)(join(DIST_DIR, "validators.cjs"));
  for (const build of [esm, cjs]) {
    const validate = build.IDENTIFIER_VALIDATORS[referenceName(sample.reference)];
    assert.equal(validate(sample.valid[0]), true);
    assert.equal(validate(sample.invalid[0]), false);
  }
  // The built main entry carries none of the validator code.
  const mainBundle = await readFile(join(DIST_DIR, "index.js"), "utf8");
  const mainCjs = await readFile(join(DIST_DIR, "index.cjs"), "utf8");
  for (const marker of ["TRWAGMYFPDXBNJZSQVHLCKE", "validateAadhaar", "IDENTIFIER_VALIDATORS"]) {
    assert.equal(mainBundle.includes(marker), false, marker);
    assert.equal(mainCjs.includes(marker), false, marker);
  }
  assert.doesNotMatch(mainBundle, /validators/);

  const source = await readFile(SOURCE_PATH, "utf8");
  assert.doesNotMatch(source, /^import |require\(|import\(/m);
});
