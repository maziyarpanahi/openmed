// Offline checks for the `openmed/catalog` subpath. Synthetic rows use
// synthetic repo ids; the bundled rows come from the committed generated file.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  MODEL_CATALOG_DIGEST,
  MODEL_CATALOG_FIELDS,
  MODEL_CATALOG_FORMAT,
  ModelCatalogError,
  createModelCatalog,
  getModel,
  getModelCatalog,
  listModels,
  parseModelCatalogJsonl,
  type ModelCatalogEntry,
  type ModelCatalogFilter,
} from "../../js/openmedkit-web/src/catalog";
import * as mainEntry from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const distDir = join(rootDir, "js", "openmedkit-web", "dist");

// Mirrors PERMISSIVE_LICENSES in scripts/android/build_android_catalog.py.
const PERMISSIVE_LICENSES = new Set([
  "apache-2.0",
  "bsd-2-clause",
  "bsd-3-clause",
  "cc-by-3.0",
  "cc-by-4.0",
  "cc0-1.0",
  "isc",
  "mit",
  "unlicense",
]);

const hash = (digit: string) => `sha256:${digit.repeat(64)}`;

const SYNTHETIC_ROWS = [
  {
    repo_id: "Synthetic/alpha-tiny-onnx",
    formats: ["onnx", "onnx-int8"],
    tier: "Tiny",
    param_count: 22_000_000,
    languages: ["en", "es"],
    license: "apache-2.0",
    reproducibility_hash: hash("1"),
  },
  {
    repo_id: "Synthetic/bravo-small-onnx",
    formats: ["ONNX"],
    tier: "small",
    param_count: 44_000_000,
    languages: ["FR"],
    license: "MIT",
    reproducibility_hash: hash("2"),
  },
  {
    repo_id: "Synthetic/charlie-null-tier-onnx",
    formats: ["onnx-fp16"],
    tier: null,
    param_count: 109_000_000,
    languages: ["en"],
    license: "apache-2.0",
    reproducibility_hash: hash("3"),
  },
  {
    repo_id: "Synthetic/delta-null-params-onnx",
    formats: ["onnx"],
    tier: "Large",
    param_count: null,
    languages: ["es"],
    license: "bsd-3-clause",
    reproducibility_hash: hash("4"),
  },
] as const;

function ids(entries: readonly ModelCatalogEntry[]): string[] {
  return entries.map((entry) => entry.repo_id);
}

function catalogError(code: string, fieldPath?: string) {
  return (error: unknown) => {
    assert.ok(error instanceof ModelCatalogError);
    assert.equal(error.code, code);
    if (fieldPath !== undefined) {
      assert.equal(error.fieldPath, fieldPath);
    }
    return true;
  };
}

test("bundled catalog holds only permissive ONNX rows in repo id order", () => {
  const entries = listModels();
  assert.ok(entries.length > 0);
  assert.equal(MODEL_CATALOG_FORMAT, "openmed.web-model-catalog.v1");
  const seen = new Set<string>();
  for (const entry of entries) {
    assert.deepEqual(Object.keys(entry), [...MODEL_CATALOG_FIELDS]);
    assert.ok(!seen.has(entry.repo_id), "duplicate repo id");
    seen.add(entry.repo_id);
    assert.ok(entry.formats.length > 0);
    assert.ok(entry.formats.every((format) => /^onnx/i.test(format)));
    assert.ok(PERMISSIVE_LICENSES.has(entry.license?.toLowerCase() ?? ""));
    assert.match(entry.reproducibility_hash, /^sha256:[0-9a-f]{64}$/);
    assert.ok(
      entry.param_count === null || Number.isSafeInteger(entry.param_count),
    );
    assert.ok(Object.isFrozen(entry));
    assert.ok(Object.isFrozen(entry.formats));
  }
  const folded = ids(entries).map((repoId) => repoId.toLowerCase());
  assert.deepEqual(folded, [...folded].sort());
});

test("bundled digest matches the rows as Android-format compact JSONL", () => {
  const payload = getModelCatalog()
    .entries.map((entry) => `${JSON.stringify(entry)}\n`)
    .join("");
  const digest = createHash("sha256").update(payload, "utf8").digest("hex");
  assert.equal(MODEL_CATALOG_DIGEST, `sha256:${digest}`);
});

test("getModel matches exact repo ids from the bundled catalog", () => {
  const [first] = listModels();
  assert.ok(first !== undefined);
  assert.equal(getModel(first.repo_id), first);
  assert.equal(getModel(first.repo_id.toUpperCase()), undefined);
  assert.equal(getModel("Synthetic/not-in-the-catalog"), undefined);
  assert.equal(getModelCatalog(), getModelCatalog());
});

test("bundled entries cannot be mutated through the returned arrays", () => {
  const first = listModels()[0];
  assert.ok(first !== undefined);
  const tier = first.tier;
  assert.equal(Reflect.set(first, "tier", "mutated"), false);
  assert.equal(Reflect.set(first.languages, 0, "mutated"), false);
  assert.equal(getModel(first.repo_id)?.tier, tier);
  const listed = listModels();
  listed.length = 0;
  assert.ok(listModels().length > 0);
});

test("filters over a synthetic manifest return the expected repo ids", () => {
  const catalog = createModelCatalog(SYNTHETIC_ROWS);
  const cases: Array<[ModelCatalogFilter | undefined, string[]]> = [
    [undefined, ids(catalog.entries)],
    [{}, ids(catalog.entries)],
    [
      { format: "onnx" },
      ["Synthetic/alpha-tiny-onnx", "Synthetic/bravo-small-onnx", "Synthetic/delta-null-params-onnx"],
    ],
    [{ format: "ONNX-INT8" }, ["Synthetic/alpha-tiny-onnx"]],
    [{ format: "onnx-q4" }, []],
    [{ tier: "SMALL" }, ["Synthetic/bravo-small-onnx"]],
    [{ tier: "tiny" }, ["Synthetic/alpha-tiny-onnx"]],
    [
      { maxParamCount: 44_000_000 },
      ["Synthetic/alpha-tiny-onnx", "Synthetic/bravo-small-onnx"],
    ],
    [{ maxParamCount: 0 }, []],
    [
      { maxParamCount: Number.POSITIVE_INFINITY },
      [
        "Synthetic/alpha-tiny-onnx",
        "Synthetic/bravo-small-onnx",
        "Synthetic/charlie-null-tier-onnx",
      ],
    ],
    [
      { language: "ES" },
      ["Synthetic/alpha-tiny-onnx", "Synthetic/delta-null-params-onnx"],
    ],
    [{ language: "fr" }, ["Synthetic/bravo-small-onnx"]],
    [{ language: "de" }, []],
    [{ license: "mit" }, ["Synthetic/bravo-small-onnx"]],
    [
      { license: "Apache-2.0" },
      ["Synthetic/alpha-tiny-onnx", "Synthetic/charlie-null-tier-onnx"],
    ],
    [
      { language: "en", maxParamCount: 50_000_000, license: "apache-2.0" },
      ["Synthetic/alpha-tiny-onnx"],
    ],
    [{ format: "onnx", language: "en", tier: "small" }, []],
  ];
  for (const [filter, expected] of cases) {
    assert.deepEqual(
      ids(catalog.listModels(filter)),
      expected,
      JSON.stringify(filter),
    );
  }
  assert.equal(
    catalog.getModel("Synthetic/delta-null-params-onnx")?.tier,
    "Large",
  );
  assert.equal(catalog.getModel("synthetic/delta-null-params-onnx"), undefined);
});

test("JSONL parsing reads Android asset rows and ignores unknown fields", () => {
  const text = [
    JSON.stringify({ ...SYNTHETIC_ROWS[0], task: "token-classification" }),
    "",
    JSON.stringify({
      repo_id: "Synthetic/minimal-onnx",
      reproducibility_hash: hash("5"),
    }),
    "",
  ].join("\r\n");
  const catalog = parseModelCatalogJsonl(text);
  assert.deepEqual(ids(catalog.entries), [
    "Synthetic/alpha-tiny-onnx",
    "Synthetic/minimal-onnx",
  ]);
  assert.equal(Object.hasOwn(catalog.entries[0] ?? {}, "task"), false);
  assert.deepEqual(catalog.getModel("Synthetic/minimal-onnx"), {
    repo_id: "Synthetic/minimal-onnx",
    formats: [],
    tier: null,
    param_count: null,
    languages: [],
    license: null,
    reproducibility_hash: hash("5"),
  });
  assert.deepEqual(catalog.listModels({ license: "apache-2.0" }).length, 1);
});

test("malformed rows and filters fail with codes, not row contents", () => {
  const secretLike = "Synthetic/row-value-that-must-not-leak";
  assert.throws(
    () => createModelCatalog([{ ...SYNTHETIC_ROWS[0], formats: secretLike }]),
    catalogError("invalid_row", "$[0].formats"),
  );
  assert.throws(
    () => createModelCatalog([{ repo_id: secretLike }]),
    (error: unknown) =>
      catalogError("invalid_row", "$[0].reproducibility_hash")(error) &&
      !(error as Error).message.includes(secretLike),
  );
  assert.throws(
    () => createModelCatalog([{ ...SYNTHETIC_ROWS[0], param_count: 1.5 }]),
    catalogError("invalid_row", "$[0].param_count"),
  );
  assert.throws(
    () => createModelCatalog([{ ...SYNTHETIC_ROWS[0], param_count: 2 ** 53 }]),
    catalogError("invalid_row", "$[0].param_count"),
  );
  assert.throws(
    () => createModelCatalog([SYNTHETIC_ROWS[0], SYNTHETIC_ROWS[0]]),
    catalogError("duplicate_repo_id", "$[1].repo_id"),
  );
  assert.throws(
    () => createModelCatalog(["not a row"]),
    catalogError("invalid_row", "$[0]"),
  );
  assert.throws(
    () => parseModelCatalogJsonl(`${JSON.stringify(SYNTHETIC_ROWS[0])}\n{${secretLike}`),
    (error: unknown) =>
      catalogError("invalid_json", "line 2")(error) &&
      !(error as Error).message.includes(secretLike),
  );

  const catalog = createModelCatalog(SYNTHETIC_ROWS);
  const badFilters: Array<[unknown, string]> = [
    [{ lang: "en" }, "lang"],
    [{ maxParamCount: -1 }, "maxParamCount"],
    [{ maxParamCount: Number.NaN }, "maxParamCount"],
    [{ maxParamCount: "44000000" }, "maxParamCount"],
    [{ language: "" }, "language"],
    [{ format: 1 }, "format"],
    ["onnx", "$"],
  ];
  for (const [filter, fieldPath] of badFilters) {
    assert.throws(
      () => catalog.listModels(filter as ModelCatalogFilter),
      catalogError("invalid_filter", fieldPath),
    );
  }
});

test("catalog stays off the main entry and builds as its own subpath", async () => {
  assert.equal(Object.hasOwn(mainEntry, "listModels"), false);
  assert.equal(Object.hasOwn(mainEntry, "getModel"), false);

  const built = await import(pathToFileURL(join(distDir, "catalog.js")).href);
  assert.equal(built.MODEL_CATALOG_DIGEST, MODEL_CATALOG_DIGEST);
  assert.deepEqual(ids(built.listModels()), ids(listModels()));

  // The default entry must not carry the generated rows.
  const mainBundle = await readFile(join(distDir, "index.js"), "utf8");
  assert.equal(mainBundle.includes(MODEL_CATALOG_DIGEST), false);
  assert.equal(mainBundle.includes(MODEL_CATALOG_FORMAT), false);
  const mainDeclarations = await readFile(join(distDir, "index.d.ts"), "utf8");
  assert.equal(mainDeclarations.includes("ModelCatalogEntry"), false);
});
