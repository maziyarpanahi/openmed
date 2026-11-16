# OpenMedKit App Intents

The `OpenMedIntents` library product exposes two OpenMed actions to Shortcuts,
Siri and Spotlight. Both run entirely on the device:

| Intent | Input | Output |
|---|---|---|
| `RedactTextIntent` ("Redact Text with OpenMed") | text and a bundled policy name | the redacted text |
| `PIISummaryIntent` ("Summarize PII with OpenMed") | text and a bundled policy name | an `OpenMedPIISummary`: redacted text, identifier count, `label: count` entries |

The policy defaults to `hipaa_safe_harbor`. Any name from
`Policy.bundledProfileNames` works.

## On-device boundary

- **The model is already on the device.** The intents redact with a model
  that is already in the `OpenMedModelStore` cache. The model is loaded with
  network access disabled. An intent never downloads a model and has no cloud
  fallback. If the model is not cached, the intent fails with a localized
  message asking the person to download it in the app, and it makes no network
  request.
- **The app stays closed.** Both intents set `openAppWhenRun` to `false`.
- **Nothing is logged.** The intent sources contain no logging or networking
  calls, and a test enforces this.
- **Errors carry no values.** Every failure surfaces as `OpenMedIntentError`
  with a message from the package's string catalog: `emptyInput`,
  `inputTooLarge` (over 64 KiB of UTF-8), `unknownPolicy`, `noCachedModel`,
  `runtimeUnavailable` and `redactionFailed`. Underlying errors, which could
  carry a path or text, are never shown.
- **MLX limits.** MLX cannot run on the iOS Simulator, and OpenMedKit does not
  run it on watchOS or visionOS. There the default redactor reports
  `runtimeUnavailable`. Supply a Core ML pipeline instead (see below).

## Why detected values are never returned

Shortcuts can pass an intent's output to any other app or web action. A
"what PII is in this text" shortcut that returned the detected names, numbers
or their offsets would move protected values out of OpenMed's control, for
example into a messaging or web request action.

The intents therefore return only:

- text that the selected policy has already redacted;
- canonical label names with counts, such as `first_name: 2` and
  `phone_number: 1`, plus the total.

The output never includes original values, character offsets, confidences or
replacement spans. Unit tests feed sentinel values through a fake pipeline and
check that none of them, and no span field, appears anywhere in the result.

## Add the intents to an app

Add the product and make its intents visible from your app:

```swift
// Package or Xcode target dependency
.product(name: "OpenMedIntents", package: "openmed")
```

```swift
import AppIntents
import OpenMedIntents
import SwiftUI

@main
struct MyApp: App {
    init() {
        // Use the model the app downloads with OpenMedModelStore.
        OpenMedIntentsConfiguration.redactor = .cachedMLXModel(
            repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
        )
    }

    var body: some Scene { WindowGroup { ContentView() } }
}

struct MyAppIntents: AppIntentsPackage {
    static var includedPackages: [any AppIntentsPackage.Type] { [OpenMedIntentsPackage.self] }
}

struct MyShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: RedactTextIntent(),
            phrases: ["Redact text with \(.applicationName)"],
            shortTitle: "Redact Text",
            systemImageName: "eye.slash"
        )
        AppShortcut(
            intent: PIISummaryIntent(),
            phrases: ["Summarize PII with \(.applicationName)"],
            shortTitle: "PII Summary",
            systemImageName: "list.bullet.rectangle"
        )
    }
}
```

The demo app in `swift/OpenMedDemo` registers exactly these shortcuts in
`OpenMedDemoApp.swift`.

### Use a different local pipeline

`OpenMedIntentRedactor` wraps any on-device pipeline that returns a
`PolicyDeidentificationResult`. Use it, for example, to run a Core ML model
that ships in the app bundle:

```swift
OpenMedIntentsConfiguration.redactor = OpenMedIntentRedactor { text, policy in
    let openmed = try await OpenMed.load(
        backend: .coreML(modelURL: modelURL, id2labelURL: labelsURL, tokenizerFolderURL: tokenizerURL)
    )
    return try openmed.deidentify(text, policy: policy)
}
```

Whatever the pipeline returns, the intents still reduce it to redacted text
and label counts before the result leaves them.

## Testing

```bash
cd swift/OpenMedKit
swift test --filter RedactTextIntentTests
swift test --filter PIISummaryIntentTests
```

The tests inject a fake pipeline through `OpenMedIntentsConfiguration.redactor`
and use only synthetic text. They run on any macOS host without a model.
