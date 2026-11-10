import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_BYTES,
  MODEL_VARIANT,
  OpenMedLfm25ChatRuntime,
  REQUIRED_GRAPH_FILES,
  SOURCE_MODEL,
  SOURCE_REVISION,
  closeTrailingReasoningPrompt,
  resolveModelAssetUrl,
  resolveSameOriginHttpUrl,
  validateLfm25Config,
} from "../../docs/demo/web/lfm25-chat-web-adapter.mjs";

const PAGE_URL = new URL("https://openmed.test/docs/demo/web/");
const MODEL_BASE_URL = new URL(
  "https://openmed.test/docs/demo/web/models/lfm2.5-2.6b-onnx-q4f16/",
);

test("enforces same-origin local model URLs and contained asset paths", () => {
  assert.equal(
    resolveSameOriginHttpUrl("./models/lfm2.5-2.6b-onnx-q4f16/", "model", {
      baseUrl: PAGE_URL,
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
          baseUrl: PAGE_URL,
          directory: true,
        }),
      /origin|credentials|slash/,
    );
  }
  assert.throws(
    () =>
      resolveSameOriginHttpUrl("./models/lfm", "model", {
        baseUrl: PAGE_URL,
        directory: true,
      }),
    /end with a slash/,
  );

  assert.equal(
    resolveModelAssetUrl(MODEL_BASE_URL, "onnx/model_q4f16.onnx").href,
    `${MODEL_BASE_URL.href}onnx/model_q4f16.onnx`,
  );
  for (const path of ["../config.json", "%2e%2e/config.json", "/config.json"]) {
    assert.throws(
      () => resolveModelAssetUrl(MODEL_BASE_URL, path),
      /POSIX|escaped/,
    );
  }
});

test("pins the official Q4F16 graph and validates the LFM2.5 config", () => {
  assert.equal(SOURCE_MODEL, "LiquidAI/LFM2.5-2.6B-ONNX");
  assert.equal(SOURCE_REVISION, "66826372fd4fa166f53be0371c9315745c07cace");
  assert.equal(MODEL_VARIANT, "q4f16");
  assert.equal(MODEL_BYTES, 1_534_153_680);
  assert.deepEqual(REQUIRED_GRAPH_FILES, [
    {
      path: "onnx/model_q4f16.onnx",
      sha256: "871354d43abc0d718a7a089d2fe10ad5d1f83e08acbfb9f45e4d137b41a1e9a4",
      sizeBytes: 222_160,
    },
    {
      path: "onnx/model_q4f16.onnx_data",
      sha256: "34537bf4a6d70ddf1627bd3709d31c8b7db8d5bcaee2098c45661be59476fbec",
      sizeBytes: 1_063_972_864,
    },
    {
      path: "onnx/model_q4f16.onnx_data_1",
      sha256: "1b645b44902caccd406098c4dbef5724927c5fb2a2be4a087ae328989b111a7f",
      sizeBytes: 469_958_656,
    },
  ]);
  assert.equal(
    REQUIRED_GRAPH_FILES.reduce((total, file) => total + file.sizeBytes, 0),
    MODEL_BYTES,
  );

  const config = createConfig();
  assert.equal(validateLfm25Config(config), config);

  for (const [key, value] of [
    ["model_type", "llama"],
    ["hidden_size", 1024],
    ["num_hidden_layers", 29],
    ["vocab_size", 32_000],
  ]) {
    assert.throws(
      () => validateLfm25Config({ ...config, [key]: value }),
      /not an Lfm2ForCausalLM bundle|config changed/,
    );
  }
});

test("closes the chat-template reasoning prompt before direct generation", () => {
  assert.equal(
    closeTrailingReasoningPrompt("<|start|>assistant<|channel|>analysis<think>"),
    "<|start|>assistant<|channel|>analysis<think></think>\n",
  );
  assert.throws(
    () => closeTrailingReasoningPrompt("<|start|>assistant"),
    /missing its reasoning prompt suffix/,
  );
  assert.throws(
    () => closeTrailingReasoningPrompt(new Uint32Array([1, 2])),
    /unsupported prompt/,
  );
});

test("streams direct output with bounded deterministic generation", async () => {
  const mock = createGenerator([
    '{"answer":"No drug interaction is documented",',
    '"evidence":["Synthetic medication list"],"uncertainty":"low"}',
  ]);
  const runtime = createRuntime(mock.generator, 128);
  const messages = [{ role: "user", content: "Use only the synthetic note." }];

  const events = await Array.fromAsync(
    runtime.generate(messages, {
      maxNewTokens: 200,
      reasoning: false,
      temperature: 0,
    }),
  );

  assert.equal(
    events.map((event) => event.delta).join(""),
    '{"answer":"No drug interaction is documented","evidence":["Synthetic medication list"],"uncertainty":"low"}',
  );
  assert.deepEqual(
    events.map(({ index, tokenCount }) => ({ index, tokenCount })),
    [
      { index: 0, tokenCount: 1 },
      { index: 1, tokenCount: 2 },
    ],
  );
  assert.deepEqual(mock.templateCalls, [
    {
      messages,
      options: {
        add_generation_prompt: true,
        return_dict: false,
        return_tensor: false,
        tokenize: false,
      },
    },
  ]);
  assert.equal(mock.calls[0].prompt, "synthetic prompt<think></think>\n");
  assert.deepEqual(
    {
      doSample: mock.calls[0].options.do_sample,
      maxNewTokens: mock.calls[0].options.max_new_tokens,
      repetitionPenalty: mock.calls[0].options.repetition_penalty,
      returnFullText: mock.calls[0].options.return_full_text,
    },
    {
      doSample: false,
      maxNewTokens: 127,
      repetitionPenalty: 1.05,
      returnFullText: false,
    },
  );
  assert.equal("temperature" in mock.calls[0].options, false);
  assert.equal(runtime.details().Revision, SOURCE_REVISION);

  await runtime.dispose();
  await runtime.dispose();
  assert.equal(mock.disposeCalls(), 1);
});

test("rejects streamed hidden-reasoning markers", async () => {
  const mock = createGenerator(["safe output <", "think>hidden reasoning"]);
  const runtime = createRuntime(mock.generator);

  await assert.rejects(
    Array.fromAsync(
      runtime.generate([{ role: "user", content: "synthetic" }], {
        maxNewTokens: 16,
        reasoning: false,
        temperature: 0,
      }),
    ),
    /attempted to emit hidden reasoning/,
  );
  assert.equal(mock.stoppingCriteria.at(-1).interruptCalls, 1);
  await runtime.dispose();
});

test("propagates abort into stopping criteria and releases the pipeline", async () => {
  const started = Promise.withResolvers();
  const release = Promise.withResolvers();
  const mock = createGenerator([], async () => {
    started.resolve();
    await release.promise;
  });
  const runtime = createRuntime(mock.generator);
  const controller = new AbortController();
  const next = runtime
    .generate([{ role: "user", content: "synthetic" }], {
      maxNewTokens: 16,
      reasoning: false,
      signal: controller.signal,
      temperature: 0,
    })
    .next();
  await started.promise;

  controller.abort("Stopped by test");
  assert.equal(mock.stoppingCriteria.at(-1).interruptCalls, 1);
  release.resolve();
  await assert.rejects(next, { name: "AbortError" });

  await runtime.dispose();
  assert.equal(mock.disposeCalls(), 1);
});

function createRuntime(generator, contextTokens = 64) {
  return new OpenMedLfm25ChatRuntime({
    InterruptableStoppingCriteria: MockInterruptableStoppingCriteria,
    TextStreamer: MockTextStreamer,
    config: createConfig(),
    contextTokens,
    generator,
    modelBaseUrl: MODEL_BASE_URL,
  });
}

function createGenerator(chunks, beforeChunks = async () => {}) {
  const calls = [];
  const templateCalls = [];
  let releases = 0;
  const generator = async (prompt, options) => {
    calls.push({ options, prompt });
    await beforeChunks(options);
    for (const chunk of chunks) options.streamer.callbackFunction(chunk);
    return [];
  };
  generator.tokenizer = {
    apply_chat_template(messages, options) {
      templateCalls.push({ messages, options });
      return "synthetic prompt<think>";
    },
  };
  generator.dispose = async () => {
    releases += 1;
  };
  return {
    calls,
    disposeCalls: () => releases,
    generator,
    stoppingCriteria: MockInterruptableStoppingCriteria.instances,
    templateCalls,
  };
}

class MockTextStreamer {
  constructor(_tokenizer, options) {
    this.callbackFunction = options.callback_function;
  }
}

class MockInterruptableStoppingCriteria {
  static instances = [];

  constructor() {
    this.interruptCalls = 0;
    MockInterruptableStoppingCriteria.instances.push(this);
  }

  interrupt() {
    this.interruptCalls += 1;
  }
}

function createConfig() {
  return {
    architectures: ["Lfm2ForCausalLM"],
    eos_token_id: 124_900,
    hidden_size: 2048,
    model_type: "lfm2",
    num_attention_heads: 32,
    num_hidden_layers: 30,
    num_key_value_heads: 8,
    vocab_size: 128_000,
  };
}
