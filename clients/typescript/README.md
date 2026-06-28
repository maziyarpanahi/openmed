# OpenMed TypeScript REST Client

Dependency-light TypeScript client for the OpenMed REST service. It uses the
global `fetch` implementation by default and also accepts an injected fetch for
Node runtimes and tests.

## Install

From a checkout of this repository:

```bash
npm install ./clients/typescript
```

For local SDK development:

```bash
cd clients/typescript
npm run typecheck
```

## Usage

```ts
import { OpenMedApiError, OpenMedClient } from "@openmed/rest-client";

const client = new OpenMedClient({
  baseUrl: "http://localhost:8080",
});

const health = await client.health();
const analysis = await client.analyze({
  text: "Patient started imatinib for CML.",
  model_name: "disease_detection_superclinical",
  confidence_threshold: 0.25,
  aggregation_strategy: "simple",
  keep_alive: "5m",
});

const pii = await client.extractPii({
  text: "Paciente: Maria Garcia, DNI: 12345678Z",
  lang: "es",
  use_smart_merging: true,
});

const deidentified = await client.deidentify({
  text: "Paciente: Maria Garcia, DNI: 12345678Z",
  method: "mask",
  lang: "es",
  keep_mapping: true,
});

await client.unloadModels({ model_name: "disease_detection_superclinical" });
await client.unloadModels({ all: true });
```

Use `loadedModels()` to inspect cached model resources:

```ts
const loaded = await client.loadedModels();
for (const [modelName, stats] of Object.entries(loaded.models ?? {})) {
  console.log(modelName, stats.pipelines ?? 0);
}
```

Inject `fetch` in tests or runtimes that do not expose it globally:

```ts
const client = new OpenMedClient({
  baseUrl: "http://localhost:8080",
  fetch: async (input, init) => fetch(input, init),
});
```

## Error Handling

Non-2xx responses throw `OpenMedApiError`. The error preserves the REST
service envelope, including `error.code`, `error.message`, and `error.details`.

```ts
try {
  await client.deidentify({ text: "   ", method: "mask" });
} catch (error) {
  if (error instanceof OpenMedApiError) {
    console.error(error.status);
    console.error(error.code);
    console.error(error.message);
    console.error(error.details);
    console.error(error.envelope.error.code);
    console.error(error.envelope.error.message);
  }
}
```

The underlying service envelope has this shape:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "body.text",
        "message": "Text must not be blank",
        "type": "value_error"
      }
    ]
  }
}
```
