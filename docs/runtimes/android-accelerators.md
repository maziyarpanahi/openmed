# Android QNN and NNAPI Acceleration

OpenMedKit can prefer Qualcomm QNN or Android NNAPI for local ONNX token
classification while retaining ONNX Runtime's CPU execution provider as the
deterministic fallback. The selection order is QNN, NNAPI, then CPU by default.
Inference remains on device; accelerator fallback never invokes a network or
cloud runtime.

## Runtime Requirements

| Provider | Runtime requirement | OpenMedKit behavior |
| --- | --- | --- |
| QNN | A custom ONNX Runtime Android build compiled with QNN plus app-supplied Qualcomm AI Runtime libraries | Probed first; session creation uses `libQnnHtp.so` by default |
| NNAPI | Android 8.1 (API 27) or newer and an ONNX Runtime build with NNAPI | Probed after QNN; FP16 relaxation is disabled by default |
| CPU | The standard ONNX Runtime CPU provider | Always appended as the final deterministic attempt |

OpenMed does not bundle the Qualcomm SDK, QNN backend libraries, or a
proprietary QNN-enabled AAR. Applications that opt into QNN are responsible for
supplying compatible runtime libraries and complying with their licenses. The
standard OpenMedKit dependency can still select NNAPI where the packaged ONNX
Runtime and device expose it.

## Configure a Session

`OpenMedBackend` uses `AcceleratorConfig()` by default, so the normal
`OpenMedKit` path automatically probes the provider order above. Use a CPU-only
configuration for a baseline run:

```kotlin
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.onnx.AcceleratorConfig

val cpuBackend = OpenMedBackend(
    modelDirectory = modelDirectory,
    acceleratorConfig = AcceleratorConfig.cpuOnly(),
)
```

For a QNN-enabled application, keep the backend library path explicit when it
differs from the Android default:

```kotlin
import com.openmed.openmedkit.onnx.AcceleratorConfig
import com.openmed.openmedkit.onnx.AcceleratorProvider

val acceleratorConfig = AcceleratorConfig(
    preferredProviders = listOf(
        AcceleratorProvider.QNN,
        AcceleratorProvider.NNAPI,
        AcceleratorProvider.CPU,
    ),
    qnnOptions = mapOf("backend_path" to "libQnnHtp.so"),
)
```

`BackendOnnxTokenClassifier.acceleratorSelection` and
`AcceleratorSession.selection` expose the selected provider, privacy-safe
attempt outcomes, and the operator partition. They never contain input text or
span surface text.

## Capability Probe and Fallback

OpenMedKit first asks ONNX Runtime which execution providers were compiled into
the active Android package. NNAPI is excluded below API 27. Each eligible
delegate gets its own session-creation attempt:

1. If the provider is absent, OpenMedKit records `NOT_AVAILABLE` and tries the
   next provider.
2. If measured coverage shows that the provider supports none of the model's
   operators, it records `NO_SUPPORTED_OPERATORS` without creating a session.
3. If delegate registration or session creation fails, it records
   `SESSION_CREATION_FAILED` and tries the next provider.
4. CPU is attempted last. Failure to create the CPU session is a typed
   `InferenceError.SessionCreation` instead of an accelerator-specific crash.

NNAPI is registered without `CPU_DISABLED`, and the QNN path does not set
`session.disable_cpu_ep_fallback`. ONNX Runtime can therefore assign supported
subgraphs to the delegate and unsupported subgraphs to its CPU provider.

## Per-Family Operator Coverage

Android exports record the model family and graph operators in
`openmed-onnx.json`. `AcceleratorSession` reads that metadata automatically when
the manifest is next to the model. Hardware support still varies by device,
driver, model precision, and operator attributes, so a release device matrix
should supply observed support for each family:

```kotlin
import com.openmed.openmedkit.onnx.AcceleratorConfig
import com.openmed.openmedkit.onnx.AcceleratorProvider
import com.openmed.openmedkit.onnx.ModelFamilyOperatorCoverage
import java.io.File

val coverage = ModelFamilyOperatorCoverage.fromManifest(
    manifestFile = File(modelDirectory, "openmed-onnx.json"),
    supportedOperators = mapOf(
        AcceleratorProvider.QNN to setOf(
            "Gather",
            "MatMul",
            "Gelu",
        ),
    ),
)
val config = AcceleratorConfig(modelCoverage = coverage)
```

A missing provider entry means coverage is unknown: the runtime is allowed to
perform its own capability and graph-partitioning pass. A present empty set is
measured zero coverage and skips that delegate. Partial coverage selects the
delegate and exposes the remainder through
`selection.operatorCoverage.cpuFallbackOperators`.

Do not treat a static operator name alone as certification. NNAPI and QNN may
place constraints on shapes, data types, quantization, and constant inputs.
Populate the support set from the same model artifact and device tier used by
the parity run.

## Device-Tier Parity and Recall Gate

Every accelerated artifact should be compared with a CPU session over the same
synthetic evaluation cases. Store only labels, offsets, aggregate recall, and
latency:

```kotlin
val evidence = AcceleratorValidationRecord(
    latency = DeviceTierLatencyRecord(
        deviceTier = AndroidDeviceTier.MID_RANGE,
        provider = acceleratorSession.selection.selectedProvider,
        cpuP50Milliseconds = cpuP50,
        delegateP50Milliseconds = delegateP50,
        sampleCount = 100,
    ),
    cpuSpans = cpuPredictions.map {
        AcceleratorSpanSignature(it.label, it.startOffset, it.endOffset)
    },
    delegateSpans = delegatePredictions.map {
        AcceleratorSpanSignature(it.label, it.startOffset, it.endOffset)
    },
    cpuRecall = cpuRecall,
    delegateRecall = delegateRecall,
    boundaryToleranceCharacters = 0,
    maxRecallDrop = 0.0,
).requirePassing()
```

`requirePassing()` rejects label/boundary drift and recall loss beyond the
configured CPU-relative budget. `DeviceTierLatencyRecord` reports the CPU and
delegate p50 values, sample count, and computed speedup. The committed JVM test
uses `DEVICE_FARM_STUB` to assert delegate selection, partial CPU partitioning,
exact span parity, and zero recall delta without requiring Qualcomm hardware in
CI.

For INT8 artifacts, use the certified recall-delta budget recorded by the
Android ONNX export rather than increasing the tolerance just to make a device
run pass. All test inputs and fixtures must remain synthetic.

## Operational Notes

- QNN HTP generally expects a compatible quantized graph; session creation can
  fall through to NNAPI or CPU when the model/backend combination is rejected.
- NNAPI performance is device- and model-specific. Excessive CPU/delegate
  partition boundaries can be slower than CPU-only execution, so retain the
  per-tier latency record.
- `nnapiAllowFp16` is off by default. Enabling it requires a fresh span and
  recall comparison because lower precision can change outputs.
- OpenMedKit never logs input text, detected spans, or redacted output while
  selecting or falling back between providers.

## See Also

- [Android ONNX Export](../export-onnx-android.md)
- [Android Span Parity](../android-parity.md)
- [Android Quickstart](../android-quickstart.md)
