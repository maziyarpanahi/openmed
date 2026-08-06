#if canImport(UIKit)
    import OpenMedExtensionSupport
    import OpenMedKit
    import UIKit

    /// Share extension UI for policy-selectable, on-device text redaction.
    @MainActor
    open class RedactionShareViewController: UIViewController {
        private let textView = UITextView()
        private let policyPicker = UIPickerView()
        private let statusLabel = UILabel()
        private let redactButton = UIButton(type: .system)
        private let cancelButton = UIButton(type: .system)
        private let policies = Policy.bundledProfileNames
        private var inputText = ""

        public final override func viewDidLoad() {
            super.viewDidLoad()
            configureView()
            Task {
                await loadInputText()
            }
        }

        private func configureView() {
            view.backgroundColor = .systemBackground

            textView.isEditable = false
            textView.font = .preferredFont(forTextStyle: .body)
            textView.layer.borderColor = UIColor.separator.cgColor
            textView.layer.borderWidth = 1
            textView.layer.cornerRadius = 8

            policyPicker.dataSource = self
            policyPicker.delegate = self
            policyPicker.selectRow(
                policies.firstIndex(of: Policy.defaultName) ?? 0,
                inComponent: 0,
                animated: false
            )

            statusLabel.font = .preferredFont(forTextStyle: .footnote)
            statusLabel.numberOfLines = 0
            statusLabel.textColor = .secondaryLabel
            statusLabel.text = "Loading selected text…"

            redactButton.setTitle("Redact and Return", for: .normal)
            redactButton.titleLabel?.font = .preferredFont(forTextStyle: .headline)
            redactButton.isEnabled = false
            redactButton.addTarget(self, action: #selector(redactSelection), for: .touchUpInside)

            cancelButton.setTitle("Cancel", for: .normal)
            cancelButton.addTarget(self, action: #selector(cancelRequest), for: .touchUpInside)

            let buttonRow = UIStackView(arrangedSubviews: [cancelButton, redactButton])
            buttonRow.axis = .horizontal
            buttonRow.distribution = .equalSpacing

            let stack = UIStackView(arrangedSubviews: [textView, policyPicker, statusLabel, buttonRow])
            stack.axis = .vertical
            stack.spacing = 12
            stack.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(stack)

            NSLayoutConstraint.activate([
                stack.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
                stack.trailingAnchor.constraint(
                    equalTo: view.safeAreaLayoutGuide.trailingAnchor,
                    constant: -16
                ),
                stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
                stack.bottomAnchor.constraint(
                    equalTo: view.safeAreaLayoutGuide.bottomAnchor,
                    constant: -16
                ),
                policyPicker.heightAnchor.constraint(equalToConstant: 112),
            ])
        }

        private func loadInputText() async {
            do {
                let texts = try await ExtensionItemCodec.plainText(
                    from: extensionContext?.inputItems ?? []
                )
                inputText = texts.joined(separator: "\n\n")
                textView.text = inputText
                statusLabel.text = "Choose a bundled policy profile. Processing stays on this device."
                redactButton.isEnabled = true
            } catch {
                show(error)
            }
        }

        @objc private func redactSelection() {
            let policyName = policies[policyPicker.selectedRow(inComponent: 0)]
            let text = inputText
            redactButton.isEnabled = false
            statusLabel.text = "Redacting with the local Nano model…"

            Task {
                do {
                    let configuration = try NanoModelConfiguration.bundled()
                    let output = try await Task.detached(priority: .userInitiated) {
                        let result: ExtensionRedactionOutput
                        do {
                            let handler = try ExtensionRedactionHandler(configuration: configuration)
                            result = try handler.redact(text, policyName: policyName)
                        }
                        OpenMed.clearRuntimeMemoryCache()
                        return result
                    }.value
                    textView.text = output.redactedText
                    extensionContext?.completeRequest(
                        returningItems: ExtensionItemCodec.extensionItems(for: [output.redactedText])
                    )
                } catch {
                    show(error)
                    redactButton.isEnabled = true
                }
            }
        }

        @objc private func cancelRequest() {
            extensionContext?.cancelRequest(withError: ExtensionRedactionError.emptyInput)
        }

        private func show(_ error: Error) {
            statusLabel.text = error.localizedDescription
            statusLabel.textColor = .systemRed
        }
    }

    extension RedactionShareViewController: UIPickerViewDataSource, UIPickerViewDelegate {
        public func numberOfComponents(in pickerView: UIPickerView) -> Int {
            1
        }

        public func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
            policies.count
        }

        public func pickerView(
            _ pickerView: UIPickerView,
            titleForRow row: Int,
            forComponent component: Int
        ) -> String? {
            policies[row]
                .replacingOccurrences(of: "_", with: " ")
                .capitalized
        }
    }
#endif
