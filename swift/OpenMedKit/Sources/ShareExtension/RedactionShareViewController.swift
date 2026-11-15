#if os(iOS) && canImport(UIKit)
    import OpenMedExtensionSupport
    import OpenMedKit
    import UIKit

    /// Share extension UI for policy-selectable, on-device text redaction.
    @MainActor
    open class RedactionShareViewController: UIViewController {
        private let textView = UITextView()
        private let policyButton = UIButton(type: .system)
        private let statusLabel = UILabel()
        private let redactButton = UIButton(type: .system)
        private let cancelButton = UIButton(type: .system)
        private let buttonRow = UIStackView()
        private let contentStack = UIStackView()
        private let policies = Policy.bundledProfileNames
        private var selectedPolicyIndex = Policy.bundledProfileNames.firstIndex(of: Policy.defaultName) ?? 0
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
            textView.isScrollEnabled = true
            textView.alwaysBounceVertical = true
            textView.font = .preferredFont(forTextStyle: .body)
            textView.adjustsFontForContentSizeCategory = true
            textView.accessibilityLabel = ShareExtensionStrings.text(.textLabel)
            textView.accessibilityHint = ShareExtensionStrings.text(.textHint)
            textView.layer.borderColor = UIColor.separator.cgColor
            textView.layer.borderWidth = 1
            textView.layer.cornerRadius = 8

            policyButton.showsMenuAsPrimaryAction = true
            policyButton.accessibilityLabel = ShareExtensionStrings.text(.policyLabel)
            policyButton.accessibilityHint = ShareExtensionStrings.text(.policyHint)
            policyButton.isAccessibilityElement = true
            policyButton.accessibilityTraits.insert(.button)

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

            buttonRow.addArrangedSubview(redactButton)
            buttonRow.addArrangedSubview(cancelButton)
            buttonRow.axis = .vertical
            buttonRow.alignment = .fill
            buttonRow.spacing = 12
            buttonRow.distribution = .fill

            let stack = contentStack
            for control in [textView, policyButton, statusLabel, buttonRow] {
                stack.addArrangedSubview(control)
            }
            stack.axis = .vertical
            stack.spacing = 12
            stack.setCustomSpacing(20, after: statusLabel)
            stack.translatesAutoresizingMaskIntoConstraints = false
            let scrollView = UIScrollView()
            scrollView.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(scrollView)
            scrollView.addSubview(stack)

            let preferredPreviewHeight = textView.heightAnchor.constraint(equalToConstant: 160)
            preferredPreviewHeight.priority = .defaultHigh

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
                textView.heightAnchor.constraint(greaterThanOrEqualToConstant: 160),
                preferredPreviewHeight,
                policyButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
                cancelButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
                redactButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 52),
            ])
            updateTypography()
        }

        private func updateTypography() {
            textView.font = .preferredFont(forTextStyle: .body, compatibleWith: traitCollection)
            statusLabel.font = .preferredFont(forTextStyle: .footnote, compatibleWith: traitCollection)
            updateActionStyle(redactButton, title: .redact, primary: true)
            updateActionStyle(cancelButton, title: .cancel, primary: false)
            let accessibilitySize = traitCollection.preferredContentSizeCategory.isAccessibilityCategory
            contentStack.setCustomSpacing(accessibilitySize ? 0 : 12, after: textView)
            updatePolicyButton()
        }

        private func updatePolicyButton() {
            let name = ShareExtensionStrings.policyDisplayName(policies[selectedPolicyIndex])
            var configuration = UIButton.Configuration.plain()
            configuration.title = name
            configuration.subtitle = ShareExtensionStrings.policyOptionsText(count: policies.count)
            configuration.titleAlignment = .leading
            configuration.titleLineBreakMode = .byWordWrapping
            configuration.subtitleLineBreakMode = .byWordWrapping
            configuration.titlePadding = 6
            configuration.image = UIImage(systemName: "chevron.up.chevron.down")
            configuration.imagePlacement = .trailing
            configuration.imagePadding = 12
            configuration.contentInsets = NSDirectionalEdgeInsets(top: 14, leading: 16, bottom: 14, trailing: 16)
            configuration.cornerStyle = .fixed
            configuration.background.cornerRadius = 8
            configuration.background.backgroundColor = .secondarySystemBackground
            configuration.background.strokeColor = .separator
            configuration.background.strokeWidth = 1
            configuration.baseForegroundColor = .label
            let titleFont = UIFont.preferredFont(forTextStyle: .body, compatibleWith: traitCollection)
            let subtitleFont = UIFont.preferredFont(forTextStyle: .footnote, compatibleWith: traitCollection)
            configuration.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { attributes in
                var attributes = attributes
                attributes.font = titleFont
                return attributes
            }
            configuration.subtitleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { attributes in
                var attributes = attributes
                attributes.font = subtitleFont
                attributes.foregroundColor = UIColor.secondaryLabel
                return attributes
            }
            policyButton.configuration = configuration
            policyButton.contentHorizontalAlignment = .fill
            policyButton.titleLabel?.adjustsFontForContentSizeCategory = true
            policyButton.subtitleLabel?.adjustsFontForContentSizeCategory = true
            policyButton.accessibilityValue = name
            let actions = policies.enumerated().map { index, profile in
                UIAction(
                    title: ShareExtensionStrings.policyDisplayName(profile),
                    identifier: UIAction.Identifier(profile),
                    state: index == selectedPolicyIndex ? .on : .off
                ) { [weak self] _ in
                    self?.selectPolicy(at: index)
                }
            }
            policyButton.menu = UIMenu(
                title: ShareExtensionStrings.text(.policyLabel),
                options: .singleSelection,
                children: actions
            )
        }

        private func selectPolicy(at index: Int) {
            guard policies.indices.contains(index) else { return }
            selectedPolicyIndex = index
            updatePolicyButton()
        }

        private func updateActionStyle(_ button: UIButton, title: ShareExtensionStrings.Key, primary: Bool) {
            // Follow the demo's filled-primary / outlined-secondary hierarchy using adaptive UIKit colors.
            var configuration: UIButton.Configuration = primary ? .filled() : .plain()
            configuration.title = ShareExtensionStrings.text(title)
            configuration.titleAlignment = .center
            configuration.titleLineBreakMode = .byWordWrapping
            configuration.contentInsets = NSDirectionalEdgeInsets(top: 14, leading: 20, bottom: 14, trailing: 20)
            configuration.cornerStyle = .fixed
            configuration.background.cornerRadius = 8
            configuration.baseForegroundColor = primary ? .systemBackground : .label
            if primary {
                configuration.baseBackgroundColor = .label
            } else {
                configuration.background.backgroundColor = .systemBackground
                configuration.background.strokeColor = .label
                configuration.background.strokeWidth = 1
            }
            let font = UIFont.preferredFont(forTextStyle: .headline, compatibleWith: traitCollection)
            configuration.titleTextAttributesTransformer = UIConfigurationTextAttributesTransformer { attributes in
                var attributes = attributes
                attributes.font = font
                return attributes
            }
            button.configuration = configuration
            button.titleLabel?.adjustsFontForContentSizeCategory = true
            button.isAccessibilityElement = true
            button.accessibilityTraits.insert(.button)
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
            let policyName = policies[selectedPolicyIndex]
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

    // Preserve the existing public picker delegate entry points for embedding hosts.
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
            selectPolicy(at: row)
            pickerView.accessibilityValue = ShareExtensionStrings.policyDisplayName(policies[selectedPolicyIndex])
        }
    }
#endif
