# iOS Share and Action Extensions

OpenMedKit includes reusable Swift targets for redacting selected text inside an
iOS Share extension or Action extension. Both paths use bundled policy profiles,
return the redacted text to the host app, and keep model loading and inference on
the device.

The package provides three products:

- `OpenMedExtensionSupport` contains the item handler, span result, local model
  guard, and memory budget.
- `OpenMedShareExtension` contains `RedactionShareViewController` and a policy
  picker backed by `Policy.bundledProfileNames`.
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
compiled model, label map, and required tokenizer files exist. The runtime uses
the tokenizer folder with `allowNetworkAccess: false`, so missing assets fail
closed instead of falling back to a download.

## Add the Share extension

1. In Xcode, add an iOS Share Extension target.
2. Link the `OpenMedShareExtension` package product to that target.
3. Replace the generated view controller with a thin host-target subclass and
   keep `$(PRODUCT_MODULE_NAME).ShareViewController` as the principal class:

   ```swift
   import ShareExtension

   final class ShareViewController: RedactionShareViewController {}
   ```

4. Restrict the activation rule to plain text with
   `NSExtensionActivationSupportsText = true`.
5. Add the `OpenMedPIINano` folder to the extension target's Copy Bundle
   Resources phase.

The controller reads `NSExtensionItem` plain-text attachments, displays the
selected text and every bundled policy profile, runs the Nano model after the
user confirms, and completes the request with a new plain-text attachment.

## Add the Action extension

1. In Xcode, add an iOS Action Extension target.
2. Link the `OpenMedActionExtension` package product to that target.
3. Replace the generated request handler with a thin host-target subclass and
   keep `$(PRODUCT_MODULE_NAME).ActionRequestHandler` as the principal class:

   ```swift
   import ActionExtension

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
- Input is limited to 16,384 characters. Image and PDF attachments are not
  accepted.
- Raw input and detected span text are not logged or written to audit artifacts.
- The model runtime is released after each request and OpenMedKit clears unused
  runtime buffers before the extension exits.

Run the extension contract tests with:

```bash
cd swift/OpenMedKit
swift test --filter ShareRedactionTests
```
