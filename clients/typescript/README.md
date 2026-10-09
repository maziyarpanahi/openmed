# OpenMed TypeScript REST Client

Dependency-light TypeScript client for the OpenMed REST service. It uses the
global `fetch` implementation by default and also accepts an injected fetch for
Node runtimes and tests.

## Install

Workflow methods (`workflowPreflight`, `workflowPreview`, `workflowStatus`,
`workflowSubmitReceipt`, `workflowCancel`) validate the existing HTTP contract
and return immutable `WorkflowSnapshot` metadata. They reject unknown fields,
versions, malformed JSON, mismatched action/run/workflow bindings and invalid
preview or effect commitments. Server custody still owns authorization and
approval verification. `pollWorkflow` stops at review, completion or abortion.

Inspection makes one request by default. `maxAttempts` explicitly permits at
most three inspection attempts for transport failures or declared temporary
service failures. Receipt submission and cancellation intent always make one
attempt; an unknown outcome requires inspecting status before an explicit next
step. `WorkflowClientError` preserves a controlled code, HTTP status, safe
correlation ID and `mutationOutcome`; it discards raw messages, details and
credential-bearing transport errors. Generic service calls retain their existing
`OpenMedApiError` behavior.

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

const grounded = await client.ground({
  text: "Aspirin 81 mg daily",
  systems: ["rxnorm"],
  source_language: "en",
  offline: true,
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

const ndjson = await client.deidentifyStream({
  text: "Paciente: Maria Garcia, DNI: 12345678Z",
  method: "mask",
  lang: "es",
  chunk_size: 1024,
});
for (const line of ndjson.split("\n")) {
  if (!line) continue;
  const event = JSON.parse(line) as {
    type: string;
    redacted_text?: string;
  };
  if (event.type === "chunk") {
    consumeRedactedText(event.redacted_text ?? "");
  }
}

const job = await client.createJob({
  documents: [
    { id: "note-1", text: "Paciente: Maria Garcia, DNI: 12345678Z" },
  ],
  method: "mask",
  webhook: {
    url: "https://pipeline.example.com/openmed/jobs",
    secret: "replace-with-shared-secret",
  },
});
const jobStatus = await client.getJob(job.id);

const facts = await client.journeyResources({
  resource_type: "fact",
  purpose: "care_review",
  first: 20,
  fields: ["subject_id", "concept", "assertion"],
});
if (facts.state === "success") {
  for (const fact of facts.resources) consumeStructuredFact(fact.data);
}

// Fixed workflow methods use the same generated resource contract.
const journey = await client.journey({ first: 10 });
const cohort = await client.cohort({ purpose: "analytics" });
const dataset = await client.dataset();
const registry = await client.registry();
const measure = await client.measure();
const trialReview = await client.trialReview();

const decision = await client.decision({
  mode: "fixed_choice",
  input_text: "Synthetic review priority is urgent.",
  options: ["urgent", "routine"],
});
if (decision.state === "success") {
  console.log(decision.choice, decision.confidence);
} else {
  console.log(decision.state, decision.code);
}

await client.unloadModels({ model_name: "disease_detection_superclinical" });
await client.unloadModels({ all: true });
```

Start and inspect a SMART backend-services bulk ingestion job:

```ts
const job = await client.startSmartBackendIngestion({
  fhir_base_url: "https://fhir.example.org",
  token_url: "https://auth.example.org/token",
  client_id: "openmed-backend-client",
  private_key_pem: process.env.SMART_PRIVATE_KEY_PEM ?? "",
  output_dir: "/secure/openmed/deidentified",
  max_inflight_downloads: 2,
});

const status = await client.smartBackendIngestionStatus(job.job_id);
const summary = await client.smartBackendIngestionSummary(job.job_id);
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

Streaming methods return the NDJSON response as text without JSON-decoding the
whole body. Split it into lines as above; each non-empty line is one event.

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

## Governed workflow inspection

```ts
import { OpenMedClient, WorkflowClientError } from "@openmed/rest-client";

const client = new OpenMedClient({ baseUrl: "http://127.0.0.1:8080" });
const reference = {
  schema_version: "openmed.service.workflow_request.v1" as const,
  run_id: "run_" + "1".repeat(32),
  workflow_id: "workflow:test.example/review@1.0.0",
  action_digest: "sha256:" + "2".repeat(64),
  expected_state_digest: null,
  request_id: null,
};
const stop = new AbortController();
try {
  const snapshot = await client.pollWorkflow(reference, {
    maxRequests: 20, timeoutMs: 60000, intervalMs: 1000, signal: stop.signal,
  });
  console.log(snapshot.phase, snapshot.committed_effect_count);
  // A waiting-review snapshot requires an external reviewer; it is no approval.
} catch (error) {
  if (error instanceof WorkflowClientError) console.log(error.code);
  else throw error;
}
// stop.abort() stops local waiting and observation. Request server cancellation
// separately with workflowCancel and a fresh state digest + random request ID.
```

Per-request deadlines are at most 30 seconds; polling allows at most 100 actual
status requests and a five-minute deadline. `AbortSignal` interrupts waiting,
body reads and pending fetch observation, including an injected fetch that ignores
the signal. It cannot retract a request already sent. Cancellation intent does
not roll back committed effects. A parsed receipt cannot prove its custody;
only submit an already issued, consumed receipt from the trusted application.
Receipt timestamps use safe integer `number` or `bigint` up to signed Int64;
`bigint` serializes as an exact JSON integer rather than a rounded number.

Successful replies are limited to 256 KiB, eight container levels, 4,096 values
and 128 effects. Requests are limited to 64 KiB. SHA-256 checks use standard
Web Crypto (`crypto.subtle`), available in modern browsers and Node runtimes.
No dependency, server, issuer or clinical dispatcher is added.

Offline native SDK controls after building the SDK:

```bash
node --test clients/typescript/tests/workflow-client.test.mjs
```

Run from the repository root. An external build directory can be selected with
`OPENMED_WORKFLOW_CLIENT_DIST`. The tests use the same frozen synthetic JSON
vectors as Python and make no network requests.
