#if os(iOS) && canImport(UIKit)
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
        private let buttonRow = UIStackView()
        private var pickerHeight: NSLayoutConstraint?
        private let policies = Policy.bundledProfileNames
        private var inputTexts: [String] = []
        private var redactionTask: Task<Void, Never>?

        public final override func viewDidLoad() {
            super.viewDidLoad()
            configureView()
            registerForTraitChanges([UITraitPreferredContentSizeCategory.self]) {
                (controller: RedactionShareViewController, _: UITraitCollection) in
                controller.updateTypography()
            }
            Task {
                await loadInputText()
            }
        }

        private func configureView() {
            view.backgroundColor = .systemBackground

            textView.isEditable = false
            textView.font = .preferredFont(forTextStyle: .body)
            textView.adjustsFontForContentSizeCategory = true
            textView.accessibilityLabel = ShareExtensionStrings.text(.textLabel)
            textView.accessibilityHint = ShareExtensionStrings.text(.textHint)
            textView.layer.borderColor = UIColor.separator.cgColor
            textView.layer.borderWidth = 1
            textView.layer.cornerRadius = 8

            policyPicker.dataSource = self
            policyPicker.delegate = self
            policyPicker.accessibilityLabel = ShareExtensionStrings.text(.policyLabel)
            policyPicker.accessibilityHint = ShareExtensionStrings.text(.policyHint)
            policyPicker.accessibilityValue = ShareExtensionStrings.policyDisplayName(Policy.defaultName)
            policyPicker.selectRow(
                policies.firstIndex(of: Policy.defaultName) ?? 0,
                inComponent: 0,
                animated: false
            )

            statusLabel.font = .preferredFont(forTextStyle: .footnote)
            statusLabel.adjustsFontForContentSizeCategory = true
            statusLabel.accessibilityLabel = ShareExtensionStrings.text(.statusLabel)
            statusLabel.numberOfLines = 0
            statusLabel.textColor = .secondaryLabel
            setStatus(ShareExtensionStrings.text(.loading))

            redactButton.setTitle(ShareExtensionStrings.text(.redact), for: .normal)
            redactButton.titleLabel?.font = .preferredFont(forTextStyle: .headline)
            redactButton.titleLabel?.adjustsFontForContentSizeCategory = true
            redactButton.titleLabel?.numberOfLines = 0
            redactButton.titleLabel?.lineBreakMode = .byWordWrapping
            redactButton.titleLabel?.textAlignment = .center
            redactButton.accessibilityLabel = ShareExtensionStrings.text(.redact)
            redactButton.accessibilityHint = ShareExtensionStrings.text(.redactHint)
            redactButton.isEnabled = false
            redactButton.addTarget(self, action: #selector(redactSelection), for: .touchUpInside)

            cancelButton.setTitle(ShareExtensionStrings.text(.cancel), for: .normal)
            cancelButton.titleLabel?.font = .preferredFont(forTextStyle: .headline)
            cancelButton.titleLabel?.adjustsFontForContentSizeCategory = true
            cancelButton.titleLabel?.numberOfLines = 0
            cancelButton.titleLabel?.lineBreakMode = .byWordWrapping
            cancelButton.titleLabel?.textAlignment = .center
            cancelButton.accessibilityLabel = ShareExtensionStrings.text(.cancel)
            cancelButton.accessibilityHint = ShareExtensionStrings.text(.cancelHint)
            cancelButton.addTarget(self, action: #selector(cancelRequest), for: .touchUpInside)

            buttonRow.addArrangedSubview(cancelButton)
            buttonRow.addArrangedSubview(redactButton)
            buttonRow.spacing = 12
            buttonRow.distribution = .fillEqually

            let stack = UIStackView(arrangedSubviews: [textView, policyPicker, statusLabel, buttonRow])
            stack.axis = .vertical
            stack.spacing = 12
            stack.translatesAutoresizingMaskIntoConstraints = false
            let scrollView = UIScrollView()
            scrollView.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(scrollView)
            scrollView.addSubview(stack)

            pickerHeight = policyPicker.heightAnchor.constraint(equalToConstant: 112)
            pickerHeight?.isActive = true

            NSLayoutConstraint.activate([
                scrollView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
                scrollView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
                scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
                scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
                stack.leadingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.leadingAnchor, constant: 16),
                stack.trailingAnchor.constraint(
                    equalTo: scrollView.contentLayoutGuide.trailingAnchor,
                    constant: -16
                ),
                stack.widthAnchor.constraint(equalTo: scrollView.frameLayoutGuide.widthAnchor, constant: -32),
                stack.topAnchor.constraint(equalTo: scrollView.contentLayoutGuide.topAnchor, constant: 16),
                stack.bottomAnchor.constraint(
                    equalTo: scrollView.contentLayoutGuide.bottomAnchor,
                    constant: -16
                ),
                textView.heightAnchor.constraint(equalToConstant: 160),
                cancelButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 44),
                redactButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 44),
            ])
            updateTypography()
        }

        private func updateTypography() {
            textView.font = .preferredFont(forTextStyle: .body, compatibleWith: traitCollection)
            statusLabel.font = .preferredFont(forTextStyle: .footnote, compatibleWith: traitCollection)
            for button in [cancelButton, redactButton] {
                button.titleLabel?.font = .preferredFont(forTextStyle: .headline, compatibleWith: traitCollection)
            }
            buttonRow.axis = traitCollection.preferredContentSizeCategory.isAccessibilityCategory ? .vertical : .horizontal
            let selectedRow = policyPicker.selectedRow(inComponent: 0)
            pickerHeight?.constant = max(112, policyRowHeight * 3)
            policyPicker.reloadAllComponents()
            policyPicker.selectRow(selectedRow, inComponent: 0, animated: false)
        }

        private var policyRowHeight: CGFloat {
            let font = UIFont.preferredFont(forTextStyle: .body, compatibleWith: traitCollection)
            return max(44, ceil(font.lineHeight * 3 + 12))
        }

        private func setStatus(_ text: String) {
            statusLabel.text = text
            statusLabel.accessibilityValue = text
        }

        private func loadInputText() async {
            do {
                let texts = try await ExtensionItemCodec.plainText(
                    from: extensionContext?.inputItems ?? []
                )
                inputTexts = texts
                textView.text = texts.joined(separator: "\n\n")
                setStatus(ShareExtensionStrings.text(.ready))
                redactButton.isEnabled = true
            } catch {
                show(error)
            }
        }

        @objc private func redactSelection() {
            let policyName = policies[policyPicker.selectedRow(inComponent: 0)]
            let texts = inputTexts
            redactButton.isEnabled = false
            setStatus(ShareExtensionStrings.text(.redacting))

            redactionTask = Task {
                defer { redactionTask = nil }
                do {
                    let configuration = try NanoModelConfiguration.bundled()
                    let worker = Task.detached(priority: .userInitiated) {
                        try ExtensionRedactionHandler.withRuntime(configuration: configuration) {
                            handler in
                            try handler.redact(texts, policyName: policyName).map(\.redactedText)
                        }
                    }
                    let redactedTexts = try await withTaskCancellationHandler {
                        try await worker.value
                    } onCancel: {
                        worker.cancel()
                    }
                    try Task.checkCancellation()
                    let redactedText = redactedTexts.joined(separator: "\n\n")
                    inputTexts.removeAll(keepingCapacity: false)
                    textView.text = redactedText
                    extensionContext?.completeRequest(
                        returningItems: ExtensionItemCodec.extensionItems(for: [redactedText])
                    )
                } catch {
                    guard !Task.isCancelled else { return }
                    show(error)
                    redactButton.isEnabled = true
                }
            }
        }

        @objc private func cancelRequest() {
            redactionTask?.cancel()
            inputTexts.removeAll(keepingCapacity: false)
            textView.text = ""
            extensionContext?.cancelRequest(withError: ExtensionRedactionError.emptyInput)
        }

        private func show(_ error: Error) {
            setStatus(ShareExtensionStrings.errorMessage(error))
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
            ShareExtensionStrings.policyDisplayName(policies[row])
        }

        public func pickerView(
            _ pickerView: UIPickerView,
            viewForRow row: Int,
            forComponent component: Int,
            reusing view: UIView?
        ) -> UIView {
            let label = view as? UILabel ?? UILabel()
            label.text = ShareExtensionStrings.policyDisplayName(policies[row])
            label.font = .preferredFont(forTextStyle: .body, compatibleWith: traitCollection)
            label.adjustsFontForContentSizeCategory = true
            label.numberOfLines = 3
            label.textAlignment = .center
            label.accessibilityLabel = label.text
            return label
        }

        public func pickerView(_ pickerView: UIPickerView, rowHeightForComponent component: Int) -> CGFloat {
            policyRowHeight
        }

        public func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
            pickerView.accessibilityValue = ShareExtensionStrings.policyDisplayName(policies[row])
        }
    }
#endif
