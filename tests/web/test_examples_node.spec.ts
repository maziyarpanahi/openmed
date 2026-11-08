import assert from "node:assert/strict";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import type { TokenClassificationPipeline } from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const exampleUrl = pathToFileURL(
  join(rootDir, "examples", "web", "node_deidentify.mjs"),
).href;

const stubPipeline: TokenClassificationPipeline = (text) => {
  const entities: Array<[string, string, string]> = [
    ["Alice Nguyen", "B-NAME", "Alice"],
    ["1979-04-12", "S-DATE_OF_BIRTH", "1979-04-12"],
    ["alice@example.org", "S-EMAIL", "alice@example.org"],
  ];
  return entities.map(([surface, entity, word], index) => {
    const start = text.indexOf(surface);
    return {
      entity,
      score: 0.99,
      word,
      start,
      end: start + surface.length,
      index,
    };
  });
};

async function loadExample() {
  return (await import(exampleUrl)) as {
    run: (options: {
      pipeline: TokenClassificationPipeline;
      print?: (line: string) => void;
    }) => Promise<{
      deidentifiedText: string;
      spans: Array<{ canonical_label: string }>;
    }>;
  };
}

test("node deidentify example redacts a synthetic note with a stub pipeline", async () => {
  const example = await loadExample();

  const printed: string[] = [];
  const result = await example.run({
    pipeline: stubPipeline,
    print: (line) => printed.push(line),
  });

  assert.equal(
    result.deidentifiedText,
    "Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].",
  );
  assert.deepEqual(
    result.spans.map((span) => span.canonical_label),
    ["PERSON", "DATE_OF_BIRTH", "EMAIL"],
  );
  assert.equal(printed[0], result.deidentifiedText);
  assert.equal(printed[1], "PERSON, DATE_OF_BIRTH, EMAIL");

  const output = printed.join("\n");
  assert.ok(!output.includes("Alice Nguyen"));
  assert.ok(!output.includes("1979-04-12"));
  assert.ok(!output.includes("alice@example.org"));
});

test("the example prints nothing if the synthetic identifiers survive", async () => {
  const example = await loadExample();
  const printed: string[] = [];
  await assert.rejects(
    example.run({ pipeline: () => [], print: (line) => printed.push(line) }),
    /synthetic note still contains an identifier/,
  );
  assert.deepEqual(printed, []);
});
