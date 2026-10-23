# iOS Share and Action Extensions

OpenMedKit includes reusable Swift targets for redacting selected text inside an
iOS Share extension or Action extension. Both paths use bundled policy profiles,
return the redacted text to the host app, and keep model loading and inference on
the device.

The package provides three products:

- `OpenMedExtensionSupport` contains the item handler, span result, local model
  guard, and memory budget.
- `OpenMedShareExtension` contains `RedactionShareViewController` and a policy
  menu backed by `Policy.bundledProfileNames`.
- `OpenMedActionExtension` contains `RedactionActionRequestHandler` for a non-UI
  Action extension.

Swift Package Manager supplies the reusable code. Create and embed the Share and
Action extension targets in the host Xcode project; App Store provisioning and
submission remain host-app responsibilities.

## Bundle a Nano model

Extensions must not download model or tokenizer files. Add this folder to both
extension targets as a folder reference:

```text
OpenMedPIINano/
├── OpenMedPIINano.mlmodelc
├── id2label.json
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

Use a precompiled, INT8 Core ML PII model. The extension Nano packaging profile
accepts no more than 40 MiB of model, label, and tokenizer assets, caps token
sequences at 256, and reserves 56 MiB for tokenizer, Core ML, input, and output
working memory. Its maximum estimated peak is therefore 96 MiB, below the
package's conservative 120 MiB extension envelope. Model loading fails before
inference when the asset budget is exceeded.

`NanoModelConfiguration` accepts only local `file:` URLs and verifies that the
compiled model and tokenizer are directories, the label map and required
tokenizer files are regular files, and no asset is a symbolic link. Hidden
files count toward the asset budget. The runtime uses the tokenizer folder with
`allowNetworkAccess: false`, so missing assets fail closed instead of falling
back to a download.

## Add the Share extension

1. In Xcode, add an iOS Share Extension target.
2. Link the `OpenMedShareExtension` package product to that target.
3. Replace the generated view controller with a thin host-target subclass and
   keep `$(PRODUCT_MODULE_NAME).ShareViewController` as the principal class:

   ```swift
   import OpenMedShareExtension

   final class ShareViewController: RedactionShareViewController {}
   ```

4. Restrict the activation rule to plain text with
   `NSExtensionActivationSupportsText = true`.
5. Add the `OpenMedPIINano` folder to the extension target's Copy Bundle
   Resources phase.

The controller reads `NSExtensionItem` plain-text attachments, displays the
selected text and every bundled policy profile, runs the Nano model after the
user confirms, and completes the request with a new plain-text attachment.

The share extension's button titles, status/error messages, accessibility labels
and hints, and policy display names come from its `Localizable.xcstrings`
resource catalog. English (`en`) is the base language; Chinese (`zh`) and Hindi
(`hi`) are included. Policy identifiers remain unchanged; the menu uses
explicit display names, including the HIPAA, GDPR, and PIPEDA acronyms.

Declare the supported languages in the containing app and extension's
localization configuration. Hosts without localized `.lproj` resources can
declare `en`, `zh`, and `hi` in `CFBundleLocalizations` in their Info.plist.
Foundation normally aligns resource-bundle language selection with the main
bundle. A host that intentionally allows its framework/resource bundles to use
different languages can opt into `CFBundleAllowMixedLocalizations` instead.
See Apple's [app language selection guide](https://developer.apple.com/library/archive/qa/qa1828/_index.html).

Fonts update while the extension remains open when the preferred text size
changes. The full-width action buttons are consistently stacked: a filled
**Redact and Return** primary action above an outlined **Cancel** secondary
action, following the demo's button hierarchy. They have at least 52-point touch
targets, padded wrapping titles, and adaptive light/dark colors. The
screen scrolls and the policy control grows to accommodate larger text. Changing
text size preserves the selected policy.

The selected-text preview has a 160-point minimum height and scrolls internally
for long selections. At accessibility text sizes, there is no extra stack spacing
between the preview and policy control. The control shows the selected profile,
a chevron, and a localized prompt with the number of available profiles. Tap it
to open the native policy menu; every bundled profile is listed and the selected
one has a checkmark. Selecting a profile updates the title and VoiceOver value.

Accessibility hints are static catalog strings and never include selected text.
The text preview itself remains readable through VoiceOver. Error presentation
maps input/model failures to controlled localized messages; it never interpolates
underlying error descriptions, filenames, private paths, or source text.

Run the catalog resource and policy display-name checks with:

```bash
cd swift/OpenMedKit
swift test --filter ShareExtensionLocalizationTests
```

The same suite includes iOS-only UIKit checks for live text-size changes,
selection preservation, and accessibility hints. Run it on an installed simulator
from the package directory (replace the destination with an available device):

```bash
xcodebuild -scheme OpenMedKit-Package \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -parallel-testing-enabled NO \
  -only-testing:ExtensionTests/ShareExtensionLocalizationTests test
```

For manual validation, open the share extension with synthetic text in each
supported language, check all policy names, and change Text Size while it is
open. At the largest accessibility size, scroll to both buttons and verify that
they remain usable. With VoiceOver, check the policy label, selected policy,
status, button labels, and hints. Without bundled Nano assets, **Redact and
Return** displays a localized missing-model message without exposing an asset
path; successful redaction still requires the bundled assets described above.

## Add the Action extension

1. In Xcode, add an iOS Action Extension target.
2. Link the `OpenMedActionExtension` package product to that target.
3. Replace the generated request handler with a thin host-target subclass and
   keep `$(PRODUCT_MODULE_NAME).ActionRequestHandler` as the principal class:

   ```swift
   import OpenMedActionExtension

   final class ActionRequestHandler: RedactionActionRequestHandler {}
   ```

   Configure the target with a text activation rule.
4. Add the same `OpenMedPIINano` folder to the Action target.

The request handler processes text attachments sequentially and returns one
redacted `NSExtensionItem` for each input. It defaults to
`hipaa_safe_harbor`. A host can choose another bundled profile by setting the
input item's `OpenMedPolicyProfile` user-info value.

## Privacy and failure behavior

- The extension source modules call no networking API and request no networking
  entitlement. A source guard test rejects additions such as `URLSession`,
  `NWConnection`, or a network-client entitlement, and a runtime test verifies
  that missing tokenizer assets fail closed.
- Remote model URLs are rejected even if the host app passes one accidentally.
- Each input is limited to 16,384 characters and 64 KiB of UTF-8, with at most
  eight text attachments and 128 KiB across one request. Image and PDF
  attachments are not accepted.
- Raw input and detected span text are not logged or written to audit artifacts.
- The model runtime is released after each request and OpenMedKit clears unused
  runtime buffers on both success and failure before the extension exits.

Run the extension contract tests with:

```bash
cd swift/OpenMedKit
swift test --filter ShareRedactionTests
```
