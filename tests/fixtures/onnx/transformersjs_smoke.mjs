#!/usr/bin/env node

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const bundleDir = process.argv[2];

if (!bundleDir) {
  throw new Error("usage: node transformersjs_smoke.mjs <bundle-dir>");
}

const requiredFiles = [
  "onnx/model.onnx",
  "onnx/model_quantized.onnx",
  "tokenizer.json",
  "tokenizer_config.json",
  "config.json",
  "quantize_config.json",
  "transformersjs-contract.json",
];

for (const fileName of requiredFiles) {
  const filePath = join(bundleDir, fileName);
  if (!existsSync(filePath)) {
    throw new Error(`missing required bundle file: ${fileName}`);
  }
}

const config = JSON.parse(
  await readFile(join(bundleDir, "config.json"), "utf8"),
);
if (!config.id2label || Object.keys(config.id2label).length === 0) {
  throw new Error("config.json must contain id2label");
}

const quantizeConfig = JSON.parse(
  await readFile(join(bundleDir, "quantize_config.json"), "utf8"),
);
if (quantizeConfig.format !== "transformersjs") {
  throw new Error("quantize_config.json must identify transformersjs format");
}

const contract = JSON.parse(
  await readFile(join(bundleDir, "transformersjs-contract.json"), "utf8"),
);
const inputNames = contract.inputs.map((input) => input.name);
for (const expectedName of ["input_ids", "attention_mask"]) {
  if (!inputNames.includes(expectedName)) {
    throw new Error(`missing pipeline input: ${expectedName}`);
  }
}

if (
  contract.outputs.length !== 1 ||
  contract.outputs[0].name !== "logits"
) {
  throw new Error("token-classification output must be logits");
}

for (const input of contract.inputs) {
  const axes = input.axes.join(",");
  const dynamicAxes = input.shape
    .slice(0, 2)
    .every((axis) => axis.kind === "dynamic");
  if (axes !== "batch,sequence" || !dynamicAxes) {
    throw new Error(`${input.name} must be [batch, sequence]`);
  }
}

const logits = contract.outputs[0];
const logitsAxes = logits.axes.join(",");
if (
  logitsAxes !== "batch,sequence,labels" ||
  logits.shape[0].kind !== "dynamic" ||
  logits.shape[1].kind !== "dynamic" ||
  logits.shape[2].kind !== "static"
) {
  throw new Error("logits must be [batch, sequence, labels]");
}

console.log("Transformers.js token-classification contract ok");
