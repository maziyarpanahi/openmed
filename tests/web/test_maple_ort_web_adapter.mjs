import assert from "node:assert/strict";
import test from "node:test";

import {
  OpenMedMapleOrtWebRuntime,
  resolveBundleAssetUrl,
  resolveSameOriginHttpUrl,
  sampleTokenId,
  validateOpenMedMapleBundleManifest,
} from "../../docs/demo/web/maple-ort-web-adapter.mjs";

const VOCAB_SIZE = 151_936;
const EOS_TOKEN_ID = 151_645;
const MODEL_BASE_URL = new URL("https://openmed.test/docs/demo/web/models/maple/");

test("enforces exact same-origin URLs and contained bundle paths", () => {
  const pageUrl = new URL("https://openmed.test/docs/demo/web/");
  assert.equal(
    resolveSameOriginHttpUrl("./models/maple/", "model", {
      baseUrl: pageUrl,
      directory: true,
    }).href,
    MODEL_BASE_URL.href,
  );
  for (const value of [
    "https://other.test/model/",
    "https://openmed.test:444/model/",
    "https://user@openmed.test/model/",
    "https://openmed.test/model/?token=secret",
    "https://openmed.test/model/#fragment",
  ]) {
    assert.throws(
      () =>
        resolveSameOriginHttpUrl(value, "model", {
          baseUrl: pageUrl,
          directory: true,
        }),
      /origin|credentials|slash/,
    );
  }
  assert.throws(
    () =>
      resolveSameOriginHttpUrl("./models/maple", "model", {
        baseUrl: pageUrl,
        directory: true,
      }),
    /end with a slash/,
  );
  for (const path of ["../model.onnx", "%2e%2e/model.onnx", "/model.onnx"]) {
    assert.throws(() => resolveBundleAssetUrl(MODEL_BASE_URL, path), /POSIX/);
  }
});

test("validates the unified cached-decoder bundle contract", () => {
  const contract = createContract();

  assert.equal(contract.graphPath, "model.onnx");
  assert.equal(contract.graphUrl.href, `${MODEL_BASE_URL.href}model.onnx`);
  assert.equal(contract.externalData.length, 1);
  assert.deepEqual(contract.externalData[0], {
    data: `${MODEL_BASE_URL.href}model.onnx.data`,
    path: "model.onnx.data",
  });
  assert.equal(contract.pastNames.length, 48);
  assert.equal(contract.pastNames[0], "past_key_values.0.key");
  assert.equal(contract.pastNames.at(-1), "past_key_values.23.value");
  assert.equal(contract.presentNames[0], "present.0.key");
  assert.equal(contract.presentNames.at(-1), "present.23.value");
  assert.equal(contract.eosTokenIds.has(EOS_TOKEN_ID), true);

  const wrongGraph = structuredClone(createManifest());
  wrongGraph.graphs.decode_path = "decode.onnx";
  assert.throws(
    () => validateOpenMedMapleBundleManifest(wrongGraph, MODEL_BASE_URL),
    /unified/,
  );

  const wrongQuantization = structuredClone(createManifest());
  wrongQuantization.quantization = "ternary-2bit-packed";
  assert.throws(
    () => validateOpenMedMapleBundleManifest(wrongQuantization, MODEL_BASE_URL),
    /does not run the 2-bit MLX checkpoint/,
  );
});

test("prefills, reuses KV outputs for decode, and disposes resources", async () => {
  const contract = createContract();
  const ort = { Tensor: MockTensor };
  const session = createSession(contract, [65, EOS_TOKEN_ID]);
  const tokenizer = createTokenizer();
  const runtime = new OpenMedMapleOrtWebRuntime({
    cache: false,
    contextTokens: 32,
    contract,
    ort,
    session,
    tokenizer,
  });

  const events = await Array.fromAsync(
    runtime.generate([{ role: "user", content: "synthetic" }], {
      maxNewTokens: 4,
      minP: 0.03,
      temperature: 0.2,
    }),
  );

  assert.equal(events.length, 1);
  assert.deepEqual(events[0], {
    index: 0,
    text: "A",
    token: 65,
    tokenCount: 1,
  });
  assert.equal(session.calls.length, 2);

  const prefill = session.calls[0];
  assert.deepEqual(prefill.feeds.input_ids.dims, [1, 2]);
  assert.deepEqual([...prefill.feeds.input_ids.data], [151643n, 10n]);
  assert.deepEqual(prefill.feeds.attention_mask.dims, [1, 2]);
  assert.deepEqual([...prefill.feeds.attention_mask.data], [1n, 1n]);
  assert.equal("position_ids" in prefill.feeds, false);
  assert.deepEqual(prefill.feeds[contract.pastNames[0]].dims, [1, 4, 0, 128]);

  const decode = session.calls[1];
  assert.deepEqual(decode.feeds.input_ids.dims, [1, 1]);
  assert.deepEqual([...decode.feeds.input_ids.data], [65n]);
  assert.deepEqual(decode.feeds.attention_mask.dims, [1, 3]);
  assert.deepEqual([...decode.feeds.attention_mask.data], [1n, 1n, 1n]);
  assert.equal(
    decode.feeds[contract.pastNames[0]],
    prefill.outputs[contract.presentNames[0]],
  );
  assert.deepEqual(decode.feeds[contract.pastNames[0]].dims, [1, 4, 2, 128]);
  assert.equal(prefill.runOptions.terminate, false);

  assert.match(runtime.details().Validation, /real Maple WebGPU inference unvalidated/);
  await runtime.dispose();
  assert.equal(session.releaseCalls, 1);
  assert.equal(tokenizer.disposeCalls, 1);
  await runtime.dispose();
  assert.equal(session.releaseCalls, 1);
});

test("propagates generation cancellation into the active ORT run", async () => {
  const contract = createContract();
  const pending = createPendingSession(contract);
  const runtime = new OpenMedMapleOrtWebRuntime({
    cache: false,
    contextTokens: 32,
    contract,
    ort: { Tensor: MockTensor },
    session: pending.session,
    tokenizer: createTokenizer(),
  });
  const controller = new AbortController();
  const iterator = runtime.generate([{ role: "user", content: "synthetic" }], {
    maxNewTokens: 1,
    signal: controller.signal,
    temperature: 0,
  });
  const next = iterator.next();
  await pending.started;

  controller.abort();
  assert.equal(pending.runOptions.terminate, true);
  pending.reject(new Error("terminated by mock ORT"));
  await assert.rejects(next, { name: "AbortError" });

  await runtime.dispose();
  assert.equal(pending.session.releaseCalls, 1);
});

test("samples argmax and min-p candidates without full normalization", () => {
  const logits = new Float32Array([-10, 3, 2, 1]);
  assert.equal(sampleTokenId(logits, { size: 4, temperature: 0 }), 1);
  assert.equal(
    sampleTokenId(logits, {
      minP: 0.5,
      random: () => 0.99,
      size: 4,
      temperature: 1,
    }),
    1,
  );
  assert.equal(
    sampleTokenId(logits, {
      minP: 0.1,
      random: () => 0.99,
      size: 4,
      temperature: 1,
    }),
    3,
  );
});

class MockTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = [...dims];
    this.disposeCalls = 0;
  }

  dispose() {
    this.disposeCalls += 1;
  }
}

function createContract() {
  return validateOpenMedMapleBundleManifest(createManifest(), MODEL_BASE_URL);
}

function createManifest() {
  return {
    schema_version: 1,
    source_model: "deepgrove/maple-preview",
    source_revision: "a".repeat(40),
    architecture: "MapleForCausalLM",
    quantization: "qmoe-4bit-blockwise-128",
    runtime: "onnxruntime-web",
    tokenizer_path: "tokenizer.json",
    graphs: {
      prefill_path: "model.onnx",
      decode_path: "model.onnx",
      input_ids_name: "input_ids",
      attention_mask_name: "attention_mask",
      position_ids_name: "position_ids",
      logits_name: "logits",
    },
    cache: {
      past_input_prefix: "past_key_values.",
      present_output_prefix: "present.",
    },
    generation: {
      eos_token_ids: [EOS_TOKEN_ID],
      max_context_tokens: 4096,
      max_input_tokens: 3072,
    },
    files: [
      { path: "model.onnx", size_bytes: 10, sha256: "1".repeat(64) },
      { path: "model.onnx.data", size_bytes: 20, sha256: "2".repeat(64) },
      { path: "tokenizer.json", size_bytes: 30, sha256: "3".repeat(64) },
    ],
  };
}

function requiredNames(contract) {
  return {
    inputNames: ["input_ids", "attention_mask", ...contract.pastNames],
    outputNames: [contract.logitsName, ...contract.presentNames],
  };
}

function createSession(contract, tokens) {
  const names = requiredNames(contract);
  return {
    ...names,
    calls: [],
    releaseCalls: 0,
    async run(feeds, outputNames, runOptions) {
      const token = tokens[this.calls.length];
      const totalLength = feeds.attention_mask.dims[1];
      const outputs = decoderOutputs(
        contract,
        feeds.input_ids.dims[1],
        totalLength,
        token,
      );
      this.calls.push({ feeds: { ...feeds }, outputNames, outputs, runOptions });
      return outputs;
    },
    release() {
      this.releaseCalls += 1;
    },
  };
}

function createPendingSession(contract) {
  const names = requiredNames(contract);
  let reject;
  let startedResolve;
  const started = new Promise((resolve) => {
    startedResolve = resolve;
  });
  const holder = {
    runOptions: null,
    session: {
      ...names,
      releaseCalls: 0,
      run(_feeds, _outputNames, runOptions) {
        holder.runOptions = runOptions;
        startedResolve();
        return new Promise((_resolve, rejectRun) => {
          reject = rejectRun;
        });
      },
      release() {
        this.releaseCalls += 1;
      },
    },
    started,
    reject(error) {
      reject(error);
    },
  };
  return holder;
}

function decoderOutputs(contract, inputLength, totalLength, token) {
  const logits = new Uint16Array(inputLength * VOCAB_SIZE);
  logits.fill(0xfc00);
  logits[(inputLength - 1) * VOCAB_SIZE + token] = 0;
  const outputs = {
    [contract.logitsName]: new MockTensor(
      "float16",
      logits,
      [1, inputLength, VOCAB_SIZE],
    ),
  };
  for (const name of contract.presentNames) {
    outputs[name] = new MockTensor(
      "float16",
      new Uint16Array(0),
      [1, 4, totalLength, 128],
    );
  }
  return outputs;
}

function createTokenizer() {
  return {
    disposeCalls: 0,
    async encodeMessages() {
      return [151_643, 10];
    },
    async decode(tokenIds) {
      return String.fromCodePoint(...tokenIds);
    },
    dispose() {
      this.disposeCalls += 1;
    },
  };
}
